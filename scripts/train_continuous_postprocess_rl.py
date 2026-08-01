#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from scripts.train_postprocess_policy import (
    Action,
    ImageRecord,
    STATE_DIM,
    cache_name,
    cache_split_predictions,
    evaluate_records,
    fixed_actions,
    image_paths_from_split,
    load_data_yaml,
    load_cached_records,
    random_actions,
    split_file,
    state_from_record,
    write_result,
)


class ContinuousPostprocessPolicy(torch.nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
        )
        self.mean = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(state)
        return self.mean(hidden), self.log_std.clamp(-4.0, 1.0).expand_as(self.mean(hidden))


def squash_action(
    raw: torch.Tensor,
    conf_range: tuple[float, float],
    iou_range: tuple[float, float],
    max_det_range: tuple[int, int],
    min_area_range: tuple[float, float] = (0.0, 0.0),
    top_k_range: tuple[int, int] | None = None,
) -> Action:
    values = torch.sigmoid(raw.detach().cpu())
    conf = conf_range[0] + float(values[0]) * (conf_range[1] - conf_range[0])
    iou = iou_range[0] + float(values[1]) * (iou_range[1] - iou_range[0])
    max_det_float = max_det_range[0] + float(values[2]) * (max_det_range[1] - max_det_range[0])
    min_area = 0.0
    top_k = None
    if values.numel() >= 5:
        min_area = min_area_range[0] + float(values[3]) * (min_area_range[1] - min_area_range[0])
        top_k_bounds = top_k_range or max_det_range
        top_k_float = top_k_bounds[0] + float(values[4]) * (top_k_bounds[1] - top_k_bounds[0])
        top_k = max(top_k_bounds[0], min(top_k_bounds[1], int(round(top_k_float))))
    return Action(
        conf=conf,
        iou=iou,
        max_det=max(max_det_range[0], min(max_det_range[1], int(round(max_det_float)))),
        agnostic=False,
        min_area=min_area,
        top_k=top_k,
    )


def _normalized_action(
    action: Action,
    conf_range: tuple[float, float],
    iou_range: tuple[float, float],
    max_det_range: tuple[int, int],
    min_area_range: tuple[float, float] = (0.0, 0.0),
    top_k_range: tuple[int, int] | None = None,
    action_dim: int = 3,
) -> torch.Tensor:
    values = [
        _scale(action.conf, conf_range[0], conf_range[1]),
        _scale(action.iou, iou_range[0], iou_range[1]),
        _scale(float(action.max_det), float(max_det_range[0]), float(max_det_range[1])),
    ]
    if action_dim >= 5:
        top_k_bounds = top_k_range or max_det_range
        values.extend(
            [
                _scale(action.min_area, min_area_range[0], min_area_range[1]),
                _scale(float(action.top_k if action.top_k is not None else action.max_det), float(top_k_bounds[0]), float(top_k_bounds[1])),
            ]
        )
    return torch.tensor(values, dtype=torch.float32)


def squash_residual_action(
    raw: torch.Tensor,
    base_action: Action,
    conf_range: tuple[float, float],
    iou_range: tuple[float, float],
    max_det_range: tuple[int, int],
    residual_scale: float,
    min_area_range: tuple[float, float] = (0.0, 0.0),
    top_k_range: tuple[int, int] | None = None,
) -> Action:
    base = _normalized_action(
        base_action,
        conf_range,
        iou_range,
        max_det_range,
        min_area_range,
        top_k_range,
        action_dim=int(raw.numel()),
    )
    delta = torch.tanh(raw.detach().cpu()) * residual_scale
    values = (base + delta).clamp(0.0, 1.0)
    conf = conf_range[0] + float(values[0]) * (conf_range[1] - conf_range[0])
    iou = iou_range[0] + float(values[1]) * (iou_range[1] - iou_range[0])
    max_det_float = max_det_range[0] + float(values[2]) * (max_det_range[1] - max_det_range[0])
    min_area = 0.0
    top_k = None
    if values.numel() >= 5:
        min_area = min_area_range[0] + float(values[3]) * (min_area_range[1] - min_area_range[0])
        top_k_bounds = top_k_range or max_det_range
        top_k_float = top_k_bounds[0] + float(values[4]) * (top_k_bounds[1] - top_k_bounds[0])
        top_k = max(top_k_bounds[0], min(top_k_bounds[1], int(round(top_k_float))))
    return Action(
        conf=conf,
        iou=iou,
        max_det=max(max_det_range[0], min(max_det_range[1], int(round(max_det_float)))),
        agnostic=False,
        min_area=min_area,
        top_k=top_k,
    )


def sample_action(
    policy: ContinuousPostprocessPolicy,
    state: torch.Tensor,
    conf_range: tuple[float, float],
    iou_range: tuple[float, float],
    max_det_range: tuple[int, int],
    min_area_range: tuple[float, float] = (0.0, 0.0),
    top_k_range: tuple[int, int] | None = None,
    base_action: Action | None = None,
    residual_scale: float = 0.25,
    freeze_extended: bool = False,
    deterministic: bool = False,
) -> tuple[Action, torch.Tensor, torch.Tensor]:
    mean, log_std = policy(state)
    std = torch.exp(log_std)
    distribution = torch.distributions.Normal(mean, std)
    raw = mean if deterministic else distribution.sample()
    active_dims = 3 if freeze_extended and raw.numel() >= 5 else raw.numel()
    log_prob = distribution.log_prob(raw)[:active_dims].sum()
    entropy = distribution.entropy()[:active_dims].sum()
    if freeze_extended and raw.numel() >= 5:
        raw = raw.clone()
        raw[3:] = 0.0
    if base_action is not None:
        return (
            squash_residual_action(
                raw,
                base_action,
                conf_range,
                iou_range,
                max_det_range,
                residual_scale,
                min_area_range,
                top_k_range,
            ),
            log_prob,
            entropy,
        )
    action = squash_action(raw, conf_range, iou_range, max_det_range, min_area_range, top_k_range)
    if freeze_extended and raw.numel() >= 5:
        action = Action(
            conf=action.conf,
            iou=action.iou,
            max_det=action.max_det,
            agnostic=action.agnostic,
            min_area=0.0,
            top_k=top_k_range[1] if top_k_range is not None else None,
        )
    return action, log_prob, entropy


def extended_base_actions(base_actions: list[Action], top_k_range: tuple[int, int] | None) -> list[Action]:
    top_k = top_k_range[1] if top_k_range is not None else None
    return [
        Action(
            conf=action.conf,
            iou=action.iou,
            max_det=action.max_det,
            agnostic=action.agnostic,
            min_area=0.0,
            top_k=top_k,
        )
        for action in base_actions
    ]


def deterministic_actions(
    records: list[ImageRecord],
    policy: ContinuousPostprocessPolicy,
    conf_range: tuple[float, float],
    iou_range: tuple[float, float],
    max_det_range: tuple[int, int],
    min_area_range: tuple[float, float] = (0.0, 0.0),
    top_k_range: tuple[int, int] | None = None,
    base_actions: list[Action] | None = None,
    residual_scale: float = 0.25,
    freeze_extended: bool = False,
) -> list[Action]:
    policy.eval()
    actions: list[Action] = []
    with torch.no_grad():
        for index, record in enumerate(records):
            state = torch.tensor(state_from_record(record), dtype=torch.float32)
            action, _log_prob, _entropy = sample_action(
                policy,
                state,
                conf_range,
                iou_range,
                max_det_range,
                min_area_range,
                top_k_range,
                base_action=base_actions[index] if base_actions is not None else None,
                residual_scale=residual_scale,
                freeze_extended=freeze_extended,
                deterministic=True,
            )
            actions.append(action)
    return actions


def train_policy(
    train_records: list[ImageRecord],
    eval_records: list[ImageRecord],
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    policy_lr: float,
    entropy_weight: float,
    latency_weight: float,
    conf_range: tuple[float, float],
    iou_range: tuple[float, float],
    max_det_range: tuple[int, int],
    out_dir: Path,
    min_area_range: tuple[float, float] = (0.0, 0.0),
    top_k_range: tuple[int, int] | None = None,
    action_mode: str = "direct",
    residual_scale: float = 0.25,
    action_space: str = "base",
    freeze_extended_epochs: int = 0,
) -> ContinuousPostprocessPolicy:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    states = [torch.tensor(state_from_record(record), dtype=torch.float32) for record in train_records]
    state_dim = int(states[0].numel()) if states else STATE_DIM
    action_dim = 5 if action_space == "extended" else 3
    policy = ContinuousPostprocessPolicy(state_dim=state_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    baseline = 0.0
    history_path = out_dir / "continuous_policy_history.jsonl"
    history_path.unlink(missing_ok=True)
    train_base_actions = None
    eval_base_actions = None
    if action_mode == "residual_combined":
        train_base_actions = heuristic_actions(train_records, conf_range, iou_range, max_det_range, rule="combined")
        eval_base_actions = heuristic_actions(eval_records, conf_range, iou_range, max_det_range, rule="combined")
        if action_space == "extended":
            train_base_actions = extended_base_actions(train_base_actions, top_k_range)
            eval_base_actions = extended_base_actions(eval_base_actions, top_k_range)

    for epoch in range(epochs):
        freeze_extended = action_space == "extended" and epoch < freeze_extended_epochs
        order = list(range(len(train_records)))
        random.shuffle(order)
        rewards: list[float] = []
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            selected_records: list[ImageRecord] = []
            selected_actions: list[Action] = []
            log_probs: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            for index in batch_indices:
                action, log_prob, entropy = sample_action(
                    policy,
                    states[index],
                    conf_range,
                    iou_range,
                    max_det_range,
                    min_area_range,
                    top_k_range,
                    base_action=train_base_actions[index] if train_base_actions is not None else None,
                    residual_scale=residual_scale,
                    freeze_extended=freeze_extended,
                )
                selected_records.append(train_records[index])
                selected_actions.append(action)
                log_probs.append(log_prob)
                entropies.append(entropy)

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

        train_actions = deterministic_actions(
            train_records,
            policy,
            conf_range,
            iou_range,
            max_det_range,
            min_area_range,
            top_k_range,
            base_actions=train_base_actions,
            residual_scale=residual_scale,
            freeze_extended=freeze_extended,
        )
        train_metrics = evaluate_records(train_records, train_actions)
        eval_actions = deterministic_actions(
            eval_records,
            policy,
            conf_range,
            iou_range,
            max_det_range,
            min_area_range,
            top_k_range,
            base_actions=eval_base_actions,
            residual_scale=residual_scale,
            freeze_extended=freeze_extended,
        )
        eval_metrics = evaluate_records(eval_records, eval_actions)
        row = {
            "epoch": epoch,
            "action_mode": action_mode,
            "action_space": action_space,
            "freeze_extended": freeze_extended,
            "freeze_extended_epochs": freeze_extended_epochs,
            "residual_scale": residual_scale,
            "baseline": baseline,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "train_map50_95": train_metrics["map50_95"],
            "eval_map50_95": eval_metrics["map50_95"],
            "eval_map50": eval_metrics["map50"],
            "eval_precision": eval_metrics["precision"],
            "eval_recall": eval_metrics["recall"],
            "eval_avg_predictions_per_image": eval_metrics["avg_predictions_per_image"],
        }
        with history_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"epoch={epoch} reward={row['mean_reward']:.4f} "
            f"train_map50_95={row['train_map50_95']:.6f} "
            f"eval_map50_95={row['eval_map50_95']:.6f}",
            flush=True,
        )

    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "conf_range": conf_range,
            "iou_range": iou_range,
            "max_det_range": max_det_range,
            "min_area_range": min_area_range,
            "top_k_range": top_k_range,
            "action_mode": action_mode,
            "action_space": action_space,
            "freeze_extended_epochs": freeze_extended_epochs,
            "residual_scale": residual_scale,
        },
        out_dir / "continuous_policy.pt",
    )
    return policy


def parse_range(text: str, cast=float) -> tuple:
    values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    if len(values) != 2:
        raise ValueError(f"Expected min,max range, got {text!r}")
    return tuple(values)


def summarize_actions(actions: list[Action]) -> dict[str, float]:
    summary = {
        "conf_mean": float(np.mean([action.conf for action in actions])),
        "conf_min": float(np.min([action.conf for action in actions])),
        "conf_max": float(np.max([action.conf for action in actions])),
        "iou_mean": float(np.mean([action.iou for action in actions])),
        "iou_min": float(np.min([action.iou for action in actions])),
        "iou_max": float(np.max([action.iou for action in actions])),
        "max_det_mean": float(np.mean([action.max_det for action in actions])),
        "max_det_min": float(np.min([action.max_det for action in actions])),
        "max_det_max": float(np.max([action.max_det for action in actions])),
    }
    if any(action.min_area > 0 for action in actions):
        summary.update(
            {
                "min_area_mean": float(np.mean([action.min_area for action in actions])),
                "min_area_min": float(np.min([action.min_area for action in actions])),
                "min_area_max": float(np.max([action.min_area for action in actions])),
            }
        )
    if any(action.top_k is not None for action in actions):
        top_ks = [action.top_k if action.top_k is not None else action.max_det for action in actions]
        summary.update(
            {
                "top_k_mean": float(np.mean(top_ks)),
                "top_k_min": float(np.min(top_ks)),
                "top_k_max": float(np.max(top_ks)),
            }
        )
    return summary


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _lerp(low: float, high: float, weight: float) -> float:
    return low + max(0.0, min(1.0, weight)) * (high - low)


def _top_pairwise_iou_mean(record: ImageRecord, top_k: int = 50) -> float:
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


def heuristic_actions(
    records: list[ImageRecord],
    conf_range: tuple[float, float],
    iou_range: tuple[float, float],
    max_det_range: tuple[int, int],
    *,
    rule: str,
) -> list[Action]:
    actions: list[Action] = []
    for record in records:
        scores = record.pred_scores
        count = int(scores.numel())
        mean_conf = float(scores.mean().item()) if count else 0.0
        conf_std = float(scores.std(unbiased=False).item()) if count > 1 else 0.0
        density = _scale(float(count), 25.0, 250.0)
        low_confidence = 1.0 - _scale(mean_conf, 0.05, 0.60)
        uncertainty = max(low_confidence, _scale(conf_std, 0.05, 0.30))
        overlap = _scale(_top_pairwise_iou_mean(record), 0.05, 0.45)

        if rule == "density":
            pressure = density
            conf_weight = 1.0 - pressure
            iou_weight = pressure
            max_det_weight = pressure
        elif rule == "confidence":
            pressure = uncertainty
            conf_weight = 1.0 - pressure
            iou_weight = 0.5
            max_det_weight = pressure
        elif rule == "combined":
            pressure = max(density, overlap, uncertainty)
            conf_weight = 1.0 - (0.55 * pressure + 0.45 * uncertainty)
            iou_weight = 0.65 * max(density, overlap) + 0.35 * (1.0 - uncertainty)
            max_det_weight = 0.65 * density + 0.25 * uncertainty + 0.10 * overlap
        else:
            raise ValueError(f"Unsupported heuristic rule: {rule}")

        actions.append(
            Action(
                conf=_lerp(conf_range[0], conf_range[1], conf_weight),
                iou=_lerp(iou_range[0], iou_range[1], iou_weight),
                max_det=int(round(_lerp(float(max_det_range[0]), float(max_det_range[1]), max_det_weight))),
                agnostic=False,
            )
        )
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous-action REINFORCE for YOLO post-processing.")
    parser.add_argument("--model", default="runs/globalwheat_yolo_supervised/yolov8n_pilot_e20/weights/best.pt")
    parser.add_argument("--data", type=Path, default=Path("datasets/globalwheat_subsets/pilot.yaml"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--candidate-conf", type=float, default=0.001)
    parser.add_argument("--candidate-iou", type=float, default=0.95)
    parser.add_argument("--candidate-max-det", type=int, default=1000)
    parser.add_argument("--conf-range", default="0.005,0.30")
    parser.add_argument("--iou-range", default="0.50,0.90")
    parser.add_argument("--max-det-range", default="50,400")
    parser.add_argument("--action-space", choices=("base", "extended"), default="base")
    parser.add_argument("--min-area-range", default="0.0,0.02")
    parser.add_argument("--top-k-range", default="50,1000")
    parser.add_argument("--policy-epochs", type=int, default=40)
    parser.add_argument("--rl-batch-size", type=int, default=16)
    parser.add_argument("--policy-lr", type=float, default=5e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--latency-weight", type=float, default=0.0)
    parser.add_argument("--action-mode", choices=("direct", "residual_combined"), default="direct")
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--freeze-extended-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/globalwheat_continuous_postprocess_rl"))
    parser.add_argument("--name", default="continuous_seed0")
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

    conf_range = parse_range(args.conf_range, float)
    iou_range = parse_range(args.iou_range, float)
    max_det_range = parse_range(args.max_det_range, int)
    min_area_range = parse_range(args.min_area_range, float)
    top_k_range = parse_range(args.top_k_range, int)

    policy = train_policy(
        train_records,
        eval_records,
        epochs=args.policy_epochs,
        batch_size=args.rl_batch_size,
        seed=args.seed,
        policy_lr=args.policy_lr,
        entropy_weight=args.entropy_weight,
        latency_weight=args.latency_weight,
        conf_range=conf_range,
        iou_range=iou_range,
        max_det_range=max_det_range,
        out_dir=run_dir,
        min_area_range=min_area_range,
        top_k_range=top_k_range,
        action_mode=args.action_mode,
        residual_scale=args.residual_scale,
        action_space=args.action_space,
        freeze_extended_epochs=args.freeze_extended_epochs,
    )

    standard_action = Action(conf=0.25, iou=0.70, max_det=300, agnostic=False)
    residual_base_actions = None
    if args.action_mode == "residual_combined":
        residual_base_actions = heuristic_actions(eval_records, conf_range, iou_range, max_det_range, rule="combined")
        if args.action_space == "extended":
            residual_base_actions = extended_base_actions(residual_base_actions, top_k_range)
    continuous_actions = deterministic_actions(
        eval_records,
        policy,
        conf_range,
        iou_range,
        max_det_range,
        min_area_range,
        top_k_range,
        base_actions=residual_base_actions,
        residual_scale=args.residual_scale,
    )
    random_baseline_actions = [
        Action(
            conf=random.uniform(*conf_range),
            iou=random.uniform(*iou_range),
            max_det=random.randint(*max_det_range),
            agnostic=False,
            min_area=random.uniform(*min_area_range) if args.action_space == "extended" else 0.0,
            top_k=random.randint(*top_k_range) if args.action_space == "extended" else None,
        )
        for _record in eval_records
    ]
    comparisons = [
        ("standard", fixed_actions(eval_records, standard_action), standard_action),
        ("random_continuous", random_baseline_actions, None),
        ("heuristic_density", heuristic_actions(eval_records, conf_range, iou_range, max_det_range, rule="density"), None),
        ("heuristic_confidence", heuristic_actions(eval_records, conf_range, iou_range, max_det_range, rule="confidence"), None),
        ("heuristic_combined", heuristic_actions(eval_records, conf_range, iou_range, max_det_range, rule="combined"), None),
        ("residual_heuristic_rl" if args.action_mode == "residual_combined" else "continuous_rl", continuous_actions, None),
    ]

    rows = []
    for strategy, actions, fixed_action in comparisons:
        metrics = evaluate_records(eval_records, actions)
        row = {
            "strategy": strategy,
            "seed": args.seed,
            **metrics,
            "fixed_action": fixed_action.__dict__ if fixed_action else None,
            "action_summary": summarize_actions(actions),
        }
        rows.append(row)
        write_result(run_dir / f"{strategy}_metrics.json", row)
        print(f"{strategy}: map50_95={metrics['map50_95']:.6f} map50={metrics['map50']:.6f}")

    with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as output:
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
                "action_summary",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    metadata = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "conf_range": conf_range,
        "iou_range": iou_range,
        "max_det_range": max_det_range,
        "min_area_range": min_area_range,
        "top_k_range": top_k_range,
        "action_space": args.action_space,
        "train_images": len(train_records),
        "eval_images": len(eval_records),
    }
    write_result(run_dir / "metadata.json", metadata)
    print(f"comparison={run_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
