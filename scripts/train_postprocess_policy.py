#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import yaml
from PIL import Image


@dataclass(frozen=True)
class Action:
    conf: float
    iou: float
    max_det: int
    agnostic: bool
    min_area: float = 0.0
    top_k: int | None = None


@dataclass
class ImageRecord:
    image_id: int
    image_path: str
    width: int
    height: int
    gt_boxes: torch.Tensor
    gt_classes: torch.Tensor
    pred_boxes: torch.Tensor
    pred_scores: torch.Tensor
    pred_classes: torch.Tensor


class PostprocessPolicy(torch.nn.Module):
    def __init__(self, state_dim: int, action_count: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, action_count),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


STATE_DIM = 15


def parse_float_list(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def load_data_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset YAML: {path}")
    return data


def split_file(data: dict, split: str) -> Path:
    if split not in data:
        raise KeyError(f"Dataset YAML does not define split {split!r}.")
    split_value = data[split]
    if isinstance(split_value, list):
        raise ValueError("This script expects each split to be a text file or image directory, not a YAML list.")
    split_path = Path(str(split_value))
    if not split_path.is_absolute() and "path" in data:
        split_path = Path(str(data["path"])) / split_path
    return split_path


def image_paths_from_split(split_path: Path, limit: int | None) -> list[Path]:
    if split_path.is_file():
        paths = [Path(line.strip()) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif split_path.is_dir():
        suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = sorted(path for path in split_path.rglob("*") if path.suffix.lower() in suffixes)
    else:
        raise FileNotFoundError(f"Split path does not exist: {split_path}")
    return paths[:limit] if limit else paths


def label_path_for(image_path: Path) -> Path:
    text = str(image_path)
    if "/images/" in text:
        return Path(text.replace("/images/", "/labels/")).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def load_ground_truth(image_path: Path, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    label_path = label_path_for(image_path)
    boxes: list[list[float]] = []
    classes: list[int] = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = [float(value) for value in parts]
            x1 = max(0.0, (xc - bw / 2.0) * width)
            y1 = max(0.0, (yc - bh / 2.0) * height)
            x2 = min(float(width), (xc + bw / 2.0) * width)
            y2 = min(float(height), (yc + bh / 2.0) * height)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                classes.append(int(cls))
    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(classes, dtype=torch.int64),
    )


def tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def record_to_json(record: ImageRecord) -> dict:
    return {
        "image_id": record.image_id,
        "image_path": record.image_path,
        "width": record.width,
        "height": record.height,
        "gt_boxes": tensor_to_list(record.gt_boxes),
        "gt_classes": tensor_to_list(record.gt_classes),
        "pred_boxes": tensor_to_list(record.pred_boxes),
        "pred_scores": tensor_to_list(record.pred_scores),
        "pred_classes": tensor_to_list(record.pred_classes),
    }


def record_from_json(payload: dict) -> ImageRecord:
    return ImageRecord(
        image_id=int(payload["image_id"]),
        image_path=str(payload["image_path"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        gt_boxes=torch.tensor(payload["gt_boxes"], dtype=torch.float32).reshape(-1, 4),
        gt_classes=torch.tensor(payload["gt_classes"], dtype=torch.int64),
        pred_boxes=torch.tensor(payload["pred_boxes"], dtype=torch.float32).reshape(-1, 4),
        pred_scores=torch.tensor(payload["pred_scores"], dtype=torch.float32),
        pred_classes=torch.tensor(payload["pred_classes"], dtype=torch.int64),
    )


def load_cached_records(path: Path) -> list[ImageRecord]:
    return [
        record_from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_cached_records(path: Path, records: Iterable[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record_to_json(record), sort_keys=True) + "\n")


def cache_split_predictions(
    *,
    model_path: str,
    image_paths: list[Path],
    cache_path: Path,
    imgsz: int,
    device: str,
    batch: int,
    candidate_conf: float,
    candidate_iou: float,
    candidate_max_det: int,
) -> list[ImageRecord]:
    if cache_path.exists():
        return load_cached_records(cache_path)

    from ultralytics import YOLO

    model = YOLO(model_path, task="detect")
    records: list[ImageRecord] = []
    for start in range(0, len(image_paths), batch):
        batch_paths = image_paths[start : start + batch]
        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=imgsz,
            device=device,
            conf=candidate_conf,
            iou=candidate_iou,
            max_det=candidate_max_det,
            stream=False,
            verbose=False,
        )
        for image_path, result in zip(batch_paths, results):
            with Image.open(image_path) as image:
                width, height = image.size
            gt_boxes, gt_classes = load_ground_truth(image_path, width, height)
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                pred_boxes = torch.empty((0, 4), dtype=torch.float32)
                pred_scores = torch.empty((0,), dtype=torch.float32)
                pred_classes = torch.empty((0,), dtype=torch.int64)
            else:
                pred_boxes = boxes.xyxy.detach().cpu().float()
                pred_scores = boxes.conf.detach().cpu().float()
                pred_classes = boxes.cls.detach().cpu().long()
            records.append(
                ImageRecord(
                    image_id=len(records),
                    image_path=str(image_path),
                    width=width,
                    height=height,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes,
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_classes=pred_classes,
                )
            )
        print(f"cached={len(records)}/{len(image_paths)} split={cache_path.stem}", flush=True)

    save_cached_records(cache_path, records)
    return records


def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + 1e-9)


def nms_indices(boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor, iou: float, agnostic: bool) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    order = torch.argsort(scores, descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        overlaps = box_iou_one_to_many(boxes[current], boxes[rest])
        if agnostic:
            suppress = overlaps > iou
        else:
            suppress = (overlaps > iou) & (classes[rest] == classes[current])
        order = rest[~suppress]
    return torch.tensor(keep, dtype=torch.long)


def apply_action(record: ImageRecord, action: Action) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = record.pred_scores >= action.conf
    boxes = record.pred_boxes[keep]
    scores = record.pred_scores[keep]
    classes = record.pred_classes[keep]
    if boxes.numel() == 0:
        return boxes, scores, classes
    if action.min_area > 0:
        image_area = max(float(record.width * record.height), 1.0)
        areas = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)) / image_area
        area_keep = areas >= action.min_area
        boxes = boxes[area_keep]
        scores = scores[area_keep]
        classes = classes[area_keep]
        if boxes.numel() == 0:
            return boxes, scores, classes
    if action.top_k is not None and action.top_k > 0 and scores.numel() > action.top_k:
        top_indices = torch.argsort(scores, descending=True)[: action.top_k]
        boxes = boxes[top_indices]
        scores = scores[top_indices]
        classes = classes[top_indices]
    nms_keep = nms_indices(boxes, scores, classes, action.iou, action.agnostic)
    nms_keep = nms_keep[: action.max_det]
    return boxes[nms_keep], scores[nms_keep], classes[nms_keep]


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def evaluate_records(records: list[ImageRecord], actions: list[Action]) -> dict[str, float]:
    if len(records) != len(actions):
        raise ValueError("Number of records and actions must match.")

    ground_truths: dict[int, dict[int, list[torch.Tensor]]] = {}
    predictions: dict[int, list[tuple[int, float, torch.Tensor]]] = {}
    total_predictions = 0
    for record, action in zip(records, actions):
        for cls, box in zip(record.gt_classes, record.gt_boxes):
            ground_truths.setdefault(int(cls.item()), {}).setdefault(record.image_id, []).append(box)
        boxes, scores, classes = apply_action(record, action)
        total_predictions += int(scores.numel())
        for cls, score, box in zip(classes, scores, boxes):
            predictions.setdefault(int(cls.item()), []).append((record.image_id, float(score.item()), box))

    aps_by_iou: list[float] = []
    precision50_values: list[float] = []
    recall50_values: list[float] = []
    for threshold in [value / 100 for value in range(50, 100, 5)]:
        class_aps: list[float] = []
        class_precisions: list[float] = []
        class_recalls: list[float] = []
        for cls in sorted(set(ground_truths) | set(predictions)):
            gt_by_image = ground_truths.get(cls, {})
            n_gt = sum(len(boxes) for boxes in gt_by_image.values())
            if n_gt == 0:
                continue
            matched = {image_id: torch.zeros(len(boxes), dtype=torch.bool) for image_id, boxes in gt_by_image.items()}
            tp: list[float] = []
            fp: list[float] = []
            for image_id, _score, pred_box in sorted(predictions.get(cls, []), key=lambda item: item[1], reverse=True):
                gt_boxes = gt_by_image.get(image_id, [])
                if not gt_boxes:
                    tp.append(0.0)
                    fp.append(1.0)
                    continue
                ious = box_iou_one_to_many(pred_box, torch.stack(gt_boxes))
                best_iou, best_index = torch.max(ious, dim=0)
                if best_iou >= threshold and not matched[image_id][best_index]:
                    matched[image_id][best_index] = True
                    tp.append(1.0)
                    fp.append(0.0)
                else:
                    tp.append(0.0)
                    fp.append(1.0)
            if not tp:
                class_aps.append(0.0)
                class_precisions.append(0.0)
                class_recalls.append(0.0)
                continue
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = (tp_cum / max(n_gt, 1)).tolist()
            precisions = (tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)).tolist()
            class_aps.append(average_precision(recalls, precisions))
            class_precisions.append(float(precisions[-1]))
            class_recalls.append(float(recalls[-1]))
        aps_by_iou.append(float(np.mean(class_aps)) if class_aps else 0.0)
        if math.isclose(threshold, 0.5):
            precision50_values = class_precisions
            recall50_values = class_recalls

    return {
        "map50_95": float(np.mean(aps_by_iou)),
        "map50": aps_by_iou[0],
        "precision": float(np.mean(precision50_values)) if precision50_values else 0.0,
        "recall": float(np.mean(recall50_values)) if recall50_values else 0.0,
        "avg_predictions_per_image": total_predictions / max(len(records), 1),
    }


def image_reward(record: ImageRecord, action: Action, latency_weight: float) -> float:
    metrics = evaluate_records([record], [action])
    return (
        metrics["map50_95"]
        + 0.1 * metrics["map50"]
        - latency_weight * metrics["avg_predictions_per_image"]
    )


def scale_feature(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def top_pairwise_iou_mean(record: ImageRecord, top_k: int = 50) -> float:
    boxes = record.pred_boxes
    scores = record.pred_scores
    if boxes.shape[0] < 2:
        return 0.0
    order = torch.argsort(scores, descending=True)[: min(top_k, boxes.shape[0])]
    boxes = boxes[order]
    x1 = torch.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = torch.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = torch.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = torch.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    union = areas[:, None] + areas[None, :] - inter + 1e-9
    ious = inter / union
    mask = ~torch.eye(ious.shape[0], dtype=torch.bool)
    return float(ious[mask].mean().item()) if mask.any() else 0.0


def state_from_record(record: ImageRecord) -> np.ndarray:
    scores = record.pred_scores
    boxes = record.pred_boxes
    image_area = max(float(record.width * record.height), 1.0)
    if scores.numel() == 0:
        return np.zeros(STATE_DIM, dtype=np.float32)
    areas = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)) / image_area
    top_scores = scores[torch.argsort(scores, descending=True)[: min(50, scores.numel())]]
    class_count = max(float(torch.unique(record.pred_classes).numel()), 1.0)
    count = float(scores.numel())
    mean_conf = float(scores.mean().item())
    conf_std = float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0
    density = scale_feature(count, 25.0, 250.0)
    low_confidence = 1.0 - scale_feature(mean_conf, 0.05, 0.60)
    uncertainty = max(low_confidence, scale_feature(conf_std, 0.05, 0.30))
    overlap = scale_feature(top_pairwise_iou_mean(record), 0.05, 0.45)
    pressure = max(density, overlap, uncertainty)
    combined_conf_weight = 1.0 - (0.55 * pressure + 0.45 * uncertainty)
    combined_iou_weight = 0.65 * max(density, overlap) + 0.35 * (1.0 - uncertainty)
    combined_max_det_weight = 0.65 * density + 0.25 * uncertainty + 0.10 * overlap
    return np.array(
        [
            min(count / 500.0, 5.0),
            mean_conf,
            float(scores.max().item()),
            conf_std,
            float(areas.mean().item()) if areas.numel() else 0.0,
            float(areas.max().item()) if areas.numel() else 0.0,
            min(class_count / 80.0, 1.0),
            float(top_scores.mean().item()) if top_scores.numel() else 0.0,
            density,
            low_confidence,
            uncertainty,
            overlap,
            max(0.0, min(1.0, combined_conf_weight)),
            max(0.0, min(1.0, combined_iou_weight)),
            max(0.0, min(1.0, combined_max_det_weight)),
        ],
        dtype=np.float32,
    )


def make_actions(conf_values: list[float], iou_values: list[float], max_det_values: list[int], include_agnostic: bool) -> list[Action]:
    agnostic_values = [False, True] if include_agnostic else [False]
    return [
        Action(conf=conf, iou=iou, max_det=max_det, agnostic=agnostic)
        for conf in conf_values
        for iou in iou_values
        for max_det in max_det_values
        for agnostic in agnostic_values
    ]


def fixed_actions(records: list[ImageRecord], action: Action) -> list[Action]:
    return [action for _ in records]


def grid_search(records: list[ImageRecord], actions: list[Action], cache_path: Path | None = None) -> tuple[Action, dict[str, float]]:
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        action = Action(**payload["action"])
        return action, payload["metrics"]
    best_action = actions[0]
    best_metrics = evaluate_records(records, fixed_actions(records, best_action))
    print(
        f"grid_action=1/{len(actions)} map50_95={best_metrics['map50_95']:.6f} action={best_action}",
        flush=True,
    )
    for index, action in enumerate(actions[1:], start=2):
        metrics = evaluate_records(records, fixed_actions(records, action))
        if metrics["map50_95"] > best_metrics["map50_95"]:
            best_action = action
            best_metrics = metrics
        print(
            f"grid_action={index}/{len(actions)} current={metrics['map50_95']:.6f} "
            f"best={best_metrics['map50_95']:.6f}",
            flush=True,
        )
    if cache_path is not None:
        cache_path.write_text(
            json.dumps({"action": best_action.__dict__, "metrics": best_metrics}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return best_action, best_metrics


def oracle_action_indices(
    records: list[ImageRecord],
    actions: list[Action],
    latency_weight: float,
    cache_path: Path | None = None,
) -> tuple[list[int], dict[int, int]]:
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        indices = [int(index) for index in payload["indices"]]
        counts = {int(key): int(value) for key, value in payload["counts"].items()}
        return indices, counts
    indices: list[int] = []
    counts: dict[int, int] = {}
    for record_index, record in enumerate(records, start=1):
        rewards = [image_reward(record, action, latency_weight) for action in actions]
        best_index = int(np.argmax(rewards))
        indices.append(best_index)
        counts[best_index] = counts.get(best_index, 0) + 1
        if record_index % 10 == 0 or record_index == len(records):
            print(f"oracle_record={record_index}/{len(records)}", flush=True)
    if cache_path is not None:
        cache_path.write_text(
            json.dumps({"indices": indices, "counts": counts}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return indices, counts


def train_imitation_policy(
    records: list[ImageRecord],
    actions: list[Action],
    target_indices: list[int],
    epochs: int,
    seed: int,
    policy_lr: float,
    out_dir: Path,
) -> PostprocessPolicy:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state_dim = len(state_from_record(records[0])) if records else STATE_DIM
    policy = PostprocessPolicy(state_dim=state_dim, action_count=len(actions))
    optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    states = torch.stack([torch.tensor(state_from_record(record), dtype=torch.float32) for record in records])
    targets = torch.tensor(target_indices, dtype=torch.long)
    history_path = out_dir / "imitation_history.jsonl"
    history_path.unlink(missing_ok=True)
    for epoch in range(epochs):
        order = torch.randperm(len(records))
        losses: list[float] = []
        correct = 0
        for index in order:
            logits = policy(states[index])
            loss = torch.nn.functional.cross_entropy(logits.reshape(1, -1), targets[index].reshape(1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            correct += int(torch.argmax(logits).item() == int(targets[index].item()))
        selected = deterministic_policy_actions(records, actions, policy)
        metrics = evaluate_records(records, selected)
        row = {
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": correct / max(len(records), 1),
            "train_map50_95": metrics["map50_95"],
            "train_map50": metrics["map50"],
        }
        with history_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"imitation_epoch={epoch} accuracy={row['accuracy']:.4f} "
            f"train_map50_95={row['train_map50_95']:.6f}",
            flush=True,
        )
    torch.save({"policy_state_dict": policy.state_dict(), "actions": [action.__dict__ for action in actions]}, out_dir / "imitation_policy.pt")
    return policy


def train_policy(
    records: list[ImageRecord],
    actions: list[Action],
    epochs: int,
    seed: int,
    policy_lr: float,
    entropy_weight: float,
    latency_weight: float,
    out_dir: Path,
    initial_policy: PostprocessPolicy | None = None,
    reward_mode: str = "image",
    rl_batch_size: int = 16,
) -> PostprocessPolicy:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state_dim = len(state_from_record(records[0])) if records else STATE_DIM
    policy = initial_policy or PostprocessPolicy(state_dim=state_dim, action_count=len(actions))
    optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    baseline = 0.0
    history_path = out_dir / "policy_history.jsonl"
    history_path.unlink(missing_ok=True)
    states = [torch.tensor(state_from_record(record), dtype=torch.float32) for record in records]
    for epoch in range(epochs):
        order = list(range(len(records)))
        random.shuffle(order)
        rewards: list[float] = []
        losses: list[float] = []
        action_counts: dict[int, int] = {}
        if reward_mode == "image":
            for index in order:
                logits = policy(states[index])
                distribution = torch.distributions.Categorical(logits=logits)
                action_index = distribution.sample()
                reward = image_reward(records[index], actions[int(action_index.item())], latency_weight)
                advantage = reward - baseline
                baseline = 0.95 * baseline + 0.05 * reward
                loss = -(distribution.log_prob(action_index) * advantage + entropy_weight * distribution.entropy())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                rewards.append(float(reward))
                losses.append(float(loss.detach().item()))
                action_counts[int(action_index.item())] = action_counts.get(int(action_index.item()), 0) + 1
        elif reward_mode == "batch_map":
            for start in range(0, len(order), rl_batch_size):
                batch_indices = order[start : start + rl_batch_size]
                log_probs: list[torch.Tensor] = []
                entropies: list[torch.Tensor] = []
                selected_actions: list[Action] = []
                selected_records: list[ImageRecord] = []
                for index in batch_indices:
                    logits = policy(states[index])
                    distribution = torch.distributions.Categorical(logits=logits)
                    action_index = distribution.sample()
                    action_id = int(action_index.item())
                    selected_actions.append(actions[action_id])
                    selected_records.append(records[index])
                    log_probs.append(distribution.log_prob(action_index))
                    entropies.append(distribution.entropy())
                    action_counts[action_id] = action_counts.get(action_id, 0) + 1
                metrics = evaluate_records(selected_records, selected_actions)
                reward = (
                    metrics["map50_95"]
                    + 0.1 * metrics["map50"]
                    - latency_weight * metrics["avg_predictions_per_image"]
                )
                advantage = reward - baseline
                baseline = 0.95 * baseline + 0.05 * reward
                loss = -(
                    torch.stack(log_probs).sum() * advantage
                    + entropy_weight * torch.stack(entropies).mean()
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                rewards.append(float(reward))
                losses.append(float(loss.detach().item()))
        else:
            raise ValueError(f"Unsupported RL reward mode: {reward_mode}")
        selected = deterministic_policy_actions(records, actions, policy)
        metrics = evaluate_records(records, selected)
        row = {
            "epoch": epoch,
            "reward_mode": reward_mode,
            "rl_batch_size": rl_batch_size,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "baseline": baseline,
            "train_map50_95": metrics["map50_95"],
            "train_map50": metrics["map50"],
            "action_counts": action_counts,
        }
        with history_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"policy_epoch={epoch} mean_reward={row['mean_reward']:.4f} "
            f"train_map50_95={row['train_map50_95']:.6f}",
            flush=True,
        )
    torch.save({"policy_state_dict": policy.state_dict(), "actions": [action.__dict__ for action in actions]}, out_dir / "policy.pt")
    return policy


def deterministic_policy_actions(records: list[ImageRecord], actions: list[Action], policy: PostprocessPolicy) -> list[Action]:
    policy.eval()
    selected: list[Action] = []
    with torch.no_grad():
        for record in records:
            state = torch.tensor(state_from_record(record), dtype=torch.float32)
            action_index = int(torch.argmax(policy(state)).item())
            selected.append(actions[action_index])
    return selected


def random_actions(records: list[ImageRecord], actions: list[Action], seed: int) -> list[Action]:
    rng = random.Random(seed)
    return [rng.choice(actions) for _ in records]


def write_result(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serializable_args(args: argparse.Namespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def cache_name(split: str, limit: int | None) -> str:
    limit_text = "all" if limit is None else str(limit)
    return f"{split}_limit{limit_text}_predictions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-detector RL for adaptive YOLO post-processing.")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", type=Path, default=Path("datasets/globalwheat_subsets/pilot.yaml"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--candidate-conf", type=float, default=0.001)
    parser.add_argument("--candidate-iou", type=float, default=0.95)
    parser.add_argument("--candidate-max-det", type=int, default=1000)
    parser.add_argument("--conf-values", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--iou-values", default="0.40,0.50,0.60,0.70")
    parser.add_argument("--max-det-values", default="50,100,200,300")
    parser.add_argument("--include-agnostic", action="store_true")
    parser.add_argument("--imitation-epochs", type=int, default=0)
    parser.add_argument("--oracle-latency-weight", type=float, default=0.0)
    parser.add_argument("--eval-oracle-upper-bound", action="store_true")
    parser.add_argument("--policy-epochs", type=int, default=8)
    parser.add_argument("--policy-lr", type=float, default=3e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--latency-weight", type=float, default=0.001)
    parser.add_argument("--rl-reward-mode", choices=("image", "batch_map"), default="image")
    parser.add_argument("--rl-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/globalwheat_postprocess_rl"))
    parser.add_argument("--name", default="pilot")
    args = parser.parse_args()

    run_dir = args.project / args.name
    cache_dir = run_dir / "cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    data = load_data_yaml(args.data)
    train_paths = image_paths_from_split(split_file(data, args.train_split), args.train_limit)
    eval_paths = image_paths_from_split(split_file(data, args.eval_split), args.eval_limit)
    train_records = cache_split_predictions(
        model_path=args.model,
        image_paths=train_paths,
        cache_path=cache_dir / cache_name(args.train_split, args.train_limit),
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        candidate_conf=args.candidate_conf,
        candidate_iou=args.candidate_iou,
        candidate_max_det=args.candidate_max_det,
    )
    eval_records = cache_split_predictions(
        model_path=args.model,
        image_paths=eval_paths,
        cache_path=cache_dir / cache_name(args.eval_split, args.eval_limit),
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        candidate_conf=args.candidate_conf,
        candidate_iou=args.candidate_iou,
        candidate_max_det=args.candidate_max_det,
    )

    actions = make_actions(
        parse_float_list(args.conf_values),
        parse_float_list(args.iou_values),
        parse_int_list(args.max_det_values),
        args.include_agnostic,
    )
    standard_action = Action(conf=0.25, iou=0.70, max_det=300, agnostic=False)
    grid_action, grid_train_metrics = grid_search(train_records, actions, run_dir / "grid_search_train.json")
    train_oracle_indices, train_oracle_counts = oracle_action_indices(
        train_records,
        actions,
        args.oracle_latency_weight,
        run_dir / "train_oracle_indices.json",
    )
    eval_oracle_indices: list[int] = []
    if args.eval_oracle_upper_bound:
        eval_oracle_indices, _eval_oracle_counts = oracle_action_indices(
            eval_records,
            actions,
            args.oracle_latency_weight,
            run_dir / "eval_oracle_indices.json",
        )
    (run_dir / "train_oracle_actions.json").write_text(
        json.dumps(
            {
                "oracle_latency_weight": args.oracle_latency_weight,
                "action_counts": train_oracle_counts,
                "actions": [actions[index].__dict__ for index in train_oracle_indices],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    policy: PostprocessPolicy | None = None
    if args.imitation_epochs > 0:
        policy = train_imitation_policy(
            records=train_records,
            actions=actions,
            target_indices=train_oracle_indices,
            epochs=args.imitation_epochs,
            seed=args.seed,
            policy_lr=args.policy_lr,
            out_dir=run_dir,
        )
    if args.policy_epochs > 0:
        policy = train_policy(
            records=train_records,
            actions=actions,
            epochs=args.policy_epochs,
            seed=args.seed,
            policy_lr=args.policy_lr,
            entropy_weight=args.entropy_weight,
            latency_weight=args.latency_weight,
            out_dir=run_dir,
            initial_policy=policy,
            reward_mode=args.rl_reward_mode,
            rl_batch_size=args.rl_batch_size,
        )
    if policy is None:
        raise ValueError("Set --imitation-epochs or --policy-epochs to a positive value.")

    comparisons = [
        ("standard", fixed_actions(eval_records, standard_action), standard_action, None),
        ("grid_global", fixed_actions(eval_records, grid_action), grid_action, grid_train_metrics),
        ("random", random_actions(eval_records, actions, args.seed), None, None),
        ("rl_adaptive", deterministic_policy_actions(eval_records, actions, policy), None, None),
    ]
    if eval_oracle_indices:
        comparisons.insert(3, ("oracle_per_image", [actions[index] for index in eval_oracle_indices], None, None))
    rows = []
    for strategy, selected_actions, fixed_action, train_metrics in comparisons:
        metrics = evaluate_records(eval_records, selected_actions)
        row = {
            "strategy": strategy,
            "seed": args.seed,
            **metrics,
            "fixed_action": fixed_action.__dict__ if fixed_action else None,
            "train_metrics": train_metrics,
        }
        rows.append(row)
        write_result(run_dir / f"{strategy}_metrics.json", row)
        print(f"{strategy}: map50_95={metrics['map50_95']:.6f} map50={metrics['map50']:.6f}")

    csv_path = run_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "strategy",
                "seed",
                "map50_95",
                "map50",
                "precision",
                "recall",
                "avg_predictions_per_image",
                "fixed_action",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    metadata = {
        **serializable_args(args),
        "action_count": len(actions),
        "actions": [action.__dict__ for action in actions],
        "standard_action": standard_action.__dict__,
        "grid_action": grid_action.__dict__,
        "train_images": len(train_records),
        "eval_images": len(eval_records),
    }
    write_result(run_dir / "metadata.json", metadata)
    print(f"comparison={csv_path}")


if __name__ == "__main__":
    main()
