#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms import functional as F

from src.rl_finetune.controller import (
    ManualController,
    RandomController,
    ReinforceController,
    UniformController,
)

COCO80_TO_COCO91 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)


class YoloListDetectionDataset(Dataset):
    def __init__(self, list_path: Path, limit: int | None = None) -> None:
        image_paths = [Path(line.strip()) for line in list_path.read_text().splitlines() if line.strip()]
        self.image_paths = image_paths[:limit] if limit else image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")

        boxes: list[list[float]] = []
        labels: list[int] = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
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
                    labels.append(COCO80_TO_COCO91[int(cls)])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return F.to_tensor(image), target


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def load_split_paths(data_yaml: Path) -> dict[str, Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    return {split: Path(data[split]) for split in ("train", "val", "test") if split in data}


def make_model(model_name: str, pretrained: bool) -> torch.nn.Module:
    if model_name == "fasterrcnn_mobilenet":
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        return fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None)
    if model_name == "ssdlite320":
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        return ssdlite320_mobilenet_v3_large(weights=weights, weights_backbone=None)
    raise ValueError(f"Unsupported model: {model_name}")


def parameter_stage(name: str) -> str:
    if name.startswith("head.") or name.startswith("roi_heads.") or name.startswith("rpn."):
        return "head"
    if name.startswith("backbone.extra") or name.startswith("backbone.fpn"):
        return "neck"
    return "backbone"


def make_optimizer(model: torch.nn.Module, base_lr: float, weight_decay: float) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    groups: dict[str, list[torch.nn.Parameter]] = {"backbone": [], "neck": [], "head": []}
    counts = {stage: 0 for stage in groups}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        stage = parameter_stage(name)
        groups[stage].append(parameter)
        counts[stage] += parameter.numel()

    param_groups = [
        {"params": params, "lr": base_lr, "weight_decay": weight_decay, "rl_stage": stage}
        for stage, params in groups.items()
        if params
    ]
    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay), counts


def set_stage_lrs(optimizer: torch.optim.Optimizer, base_lr: float, multipliers: tuple[float, float, float]) -> dict[str, float]:
    stage_to_multiplier = {
        "backbone": multipliers[0],
        "neck": multipliers[1],
        "head": multipliers[2],
    }
    effective = {}
    for group in optimizer.param_groups:
        stage = str(group["rl_stage"])
        lr = base_lr * stage_to_multiplier[stage]
        group["lr"] = lr
        effective[stage] = lr
    return effective


def state_from_history(epoch: int, loss: float, current_map: float, previous_map: float) -> np.ndarray:
    return np.array(
        [
            float(epoch),
            float(loss),
            float(current_map),
            float(current_map - previous_map),
        ],
        dtype=np.float32,
    )


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    box_area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return inter / (box_area + boxes_area - inter + 1e-9)


def average_precision(recall: list[float], precision: list[float]) -> float:
    if not recall:
        return 0.0
    mrec = [0.0] + recall + [1.0]
    mpre = [0.0] + precision + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


@torch.no_grad()
def evaluate_map(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    max_detections: int,
) -> dict[str, float]:
    model.eval()
    ground_truths: dict[int, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    predictions: dict[int, list[tuple[int, float, torch.Tensor]]] = defaultdict(list)

    for images, targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            for label, box in zip(target["labels"], target["boxes"]):
                ground_truths[int(label.item())][image_id].append(box.cpu())

            scores = output["scores"].detach().cpu()
            keep = torch.argsort(scores, descending=True)[:max_detections]
            for idx in keep:
                score = float(scores[idx])
                if score < score_threshold:
                    continue
                label = int(output["labels"][idx].detach().cpu().item())
                box = output["boxes"][idx].detach().cpu()
                predictions[label].append((image_id, score, box))

    aps_by_iou = []
    precision50_values = []
    recall50_values = []
    for iou_threshold in [value / 100 for value in range(50, 100, 5)]:
        class_aps = []
        class_precisions = []
        class_recalls = []
        labels = sorted(set(ground_truths) | set(predictions))
        for label in labels:
            gt_by_image = ground_truths.get(label, {})
            n_gt = sum(len(boxes) for boxes in gt_by_image.values())
            if n_gt == 0:
                continue
            matched = {image_id: torch.zeros(len(boxes), dtype=torch.bool) for image_id, boxes in gt_by_image.items()}
            tp = []
            fp = []
            for image_id, _score, pred_box in sorted(predictions.get(label, []), key=lambda item: item[1], reverse=True):
                gt_boxes = gt_by_image.get(image_id, [])
                if not gt_boxes:
                    tp.append(0.0)
                    fp.append(1.0)
                    continue
                gt_tensor = torch.stack(gt_boxes)
                ious = box_iou(pred_box, gt_tensor)
                best_iou, best_index = torch.max(ious, dim=0)
                if best_iou >= iou_threshold and not matched[image_id][best_index]:
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
        if abs(iou_threshold - 0.5) < 1e-9:
            precision50_values = class_precisions
            recall50_values = class_recalls

    return {
        "map50_95": float(np.mean(aps_by_iou)),
        "map50": aps_by_iou[0],
        "precision": float(np.mean(precision50_values)) if precision50_values else 0.0,
        "recall": float(np.mean(recall50_values)) if recall50_values else 0.0,
    }


def controller_for(args: argparse.Namespace) -> Any:
    if args.strategy == "raw":
        return UniformController()
    if args.strategy == "uniform":
        return UniformController()
    if args.strategy == "manual":
        return ManualController()
    if args.strategy == "random":
        return RandomController(args.seed)
    controller = ReinforceController(
        seed=args.seed,
        deterministic=args.freeze_policy,
        manual_prior_strength=args.manual_prior_strength,
    )
    if args.policy_in:
        controller.load(args.policy_in)
    return controller


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-wise RL fine-tuning for TorchVision detectors.")
    parser.add_argument("--strategy", choices=["raw", "uniform", "manual", "random", "rl"], required=True)
    parser.add_argument("--model", choices=["fasterrcnn_mobilenet", "ssdlite320"], default="fasterrcnn_mobilenet")
    parser.add_argument("--data", type=Path, default=Path("datasets/coco_subsets/search_data.yaml"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--val-limit", type=int, default=100)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--project", type=Path, default=Path("runs/torchvision_lr"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--policy-in", type=Path)
    parser.add_argument("--policy-out", type=Path)
    parser.add_argument("--freeze-policy", action="store_true")
    parser.add_argument("--manual-prior-strength", type=float, default=2.0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = args.project / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    split_paths = load_split_paths(args.data)
    train_dataset = YoloListDetectionDataset(split_paths["train"], limit=args.train_limit)
    val_dataset = YoloListDetectionDataset(split_paths["val"], limit=args.val_limit)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate)

    device = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")
    model = make_model(args.model, pretrained=not args.no_pretrained).to(device)
    optimizer, parameter_counts = make_optimizer(model, args.base_lr, args.weight_decay)
    controller = controller_for(args)

    history_path = run_dir / "lr_history.jsonl"
    if history_path.exists():
        history_path.unlink()
    previous_map = 0.0
    current_map = 0.0
    last_loss = 0.0

    for epoch in range(args.epochs if args.strategy != "raw" else 0):
        state = state_from_history(epoch, last_loss, current_map, previous_map)
        decision = controller.select(state)
        effective_lrs = set_stage_lrs(optimizer, args.base_lr, decision.multipliers)

        model.train()
        losses = []
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items() if key != "image_id"} for target in targets]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        last_loss = float(np.mean(losses)) if losses else 0.0
        previous_map = current_map
        metrics = evaluate_map(model, val_loader, device, args.score_threshold, args.max_detections)
        current_map = metrics["map50_95"]
        reward = 100.0 * (current_map - previous_map)
        update_stats = controller.update(reward)
        record = {
            "epoch": epoch,
            "action_indices": decision.action_indices,
            "lr_multipliers": {
                "backbone": decision.multipliers[0],
                "neck": decision.multipliers[1],
                "head": decision.multipliers[2],
            },
            "effective_lrs": effective_lrs,
            "loss": last_loss,
            "reward": reward,
            **metrics,
            **update_stats,
        }
        with history_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    final_metrics = evaluate_map(model, val_loader, device, args.score_threshold, args.max_detections)
    strategy = "frozen_rl" if args.strategy == "rl" and args.freeze_policy else args.strategy
    strategy = "raw_pretrained" if args.strategy == "raw" else strategy
    result = {
        "strategy": strategy,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "base_lr": args.base_lr,
        "policy_frozen": bool(args.freeze_policy),
        "group_parameter_counts": parameter_counts,
        **final_metrics,
    }
    (run_dir / "final_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str, sort_keys=True), encoding="utf-8")
    if args.policy_out:
        controller.save(args.policy_out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
