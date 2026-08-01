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

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    ssdlite320_mobilenet_v3_large,
)

from scripts.train_continuous_postprocess_rl import (
    deterministic_actions,
    extended_base_actions,
    heuristic_actions,
    parse_range,
    summarize_actions,
    train_policy,
)
from scripts.train_postprocess_policy import (
    Action,
    ImageRecord,
    evaluate_records,
    fixed_actions,
    load_cached_records,
    save_cached_records,
    write_result,
)
from scripts.train_torchvision_layerwise_rl import YoloListDetectionDataset, collate, load_split_paths


def make_model(model_name: str, candidate_conf: float, candidate_iou: float, candidate_max_det: int) -> torch.nn.Module:
    if model_name == "fasterrcnn_mobilenet":
        return fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            weights_backbone=None,
            box_score_thresh=candidate_conf,
            box_nms_thresh=candidate_iou,
            box_detections_per_img=candidate_max_det,
        )
    if model_name == "ssdlite320":
        return ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
            weights_backbone=None,
            score_thresh=candidate_conf,
            nms_thresh=candidate_iou,
            detections_per_img=candidate_max_det,
        )
    raise ValueError(f"Unsupported TorchVision detector: {model_name}")


@torch.no_grad()
def cache_predictions(
    *,
    model_name: str,
    list_path: Path,
    cache_path: Path,
    limit: int | None,
    batch_size: int,
    device: torch.device,
    candidate_conf: float,
    candidate_iou: float,
    candidate_max_det: int,
) -> list[ImageRecord]:
    if cache_path.exists():
        return load_cached_records(cache_path)

    dataset = YoloListDetectionDataset(list_path, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    model = make_model(model_name, candidate_conf, candidate_iou, candidate_max_det).to(device)
    model.eval()

    records: list[ImageRecord] = []
    for images, targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        for image, target, output in zip(images, targets, outputs):
            height, width = image.shape[-2:]
            boxes = output["boxes"].detach().cpu().float()
            scores = output["scores"].detach().cpu().float()
            labels = output["labels"].detach().cpu().long()
            records.append(
                ImageRecord(
                    image_id=len(records),
                    image_path="",
                    width=int(width),
                    height=int(height),
                    gt_boxes=target["boxes"].detach().cpu().float(),
                    gt_classes=target["labels"].detach().cpu().long(),
                    pred_boxes=boxes,
                    pred_scores=scores,
                    pred_classes=labels,
                )
            )
        print(f"cached={len(records)}/{len(dataset)} split={cache_path.stem}", flush=True)

    save_cached_records(cache_path, records)
    return records


def random_continuous_actions(
    count: int,
    conf_range: tuple[float, float],
    iou_range: tuple[float, float],
    max_det_range: tuple[int, int],
    min_area_range: tuple[float, float],
    top_k_range: tuple[int, int],
    action_space: str,
    seed: int,
) -> list[Action]:
    rng = random.Random(seed)
    return [
        Action(
            conf=rng.uniform(*conf_range),
            iou=rng.uniform(*iou_range),
            max_det=rng.randint(*max_det_range),
            agnostic=False,
            min_area=rng.uniform(*min_area_range) if action_space == "extended" else 0.0,
            top_k=rng.randint(*top_k_range) if action_space == "extended" else None,
        )
        for _ in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous REINFORCE post-processing for TorchVision detectors.")
    parser.add_argument("--model", choices=("fasterrcnn_mobilenet", "ssdlite320"), default="fasterrcnn_mobilenet")
    parser.add_argument("--data", type=Path, default=Path("datasets/coco_subsets/search_data.yaml"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-conf", type=float, default=0.001)
    parser.add_argument("--candidate-iou", type=float, default=0.95)
    parser.add_argument("--candidate-max-det", type=int, default=300)
    parser.add_argument("--conf-range", default="0.005,0.30")
    parser.add_argument("--iou-range", default="0.50,0.90")
    parser.add_argument("--max-det-range", default="50,300")
    parser.add_argument("--action-space", choices=("base", "extended"), default="base")
    parser.add_argument("--min-area-range", default="0.0,0.02")
    parser.add_argument("--top-k-range", default="50,300")
    parser.add_argument("--policy-epochs", type=int, default=20)
    parser.add_argument("--rl-batch-size", type=int, default=16)
    parser.add_argument("--policy-lr", type=float, default=5e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--latency-weight", type=float, default=0.0)
    parser.add_argument("--action-mode", choices=("direct", "residual_combined"), default="direct")
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--freeze-extended-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/torchvision_continuous_postprocess_rl"))
    parser.add_argument("--name", default="fasterrcnn_seed0_e20")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")
    split_paths = load_split_paths(args.data)
    run_dir = args.project / args.name
    cache_dir = run_dir / "cache"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_records = cache_predictions(
        model_name=args.model,
        list_path=split_paths[args.train_split],
        cache_path=cache_dir / f"{args.train_split}_limit{args.train_limit}_predictions.jsonl",
        limit=args.train_limit,
        batch_size=args.batch,
        device=device,
        candidate_conf=args.candidate_conf,
        candidate_iou=args.candidate_iou,
        candidate_max_det=args.candidate_max_det,
    )
    eval_records = cache_predictions(
        model_name=args.model,
        list_path=split_paths[args.eval_split],
        cache_path=cache_dir / f"{args.eval_split}_limit{args.eval_limit}_predictions.jsonl",
        limit=args.eval_limit,
        batch_size=args.batch,
        device=device,
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
    random_actions = random_continuous_actions(
        len(eval_records),
        conf_range,
        iou_range,
        max_det_range,
        min_area_range,
        top_k_range,
        args.action_space,
        args.seed,
    )
    comparisons = [
        ("standard", fixed_actions(eval_records, standard_action), standard_action),
        ("random_continuous", random_actions, None),
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
        "device_resolved": str(device),
        "train_images": len(train_records),
        "eval_images": len(eval_records),
    }
    write_result(run_dir / "metadata.json", metadata)
    print(f"comparison={run_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
