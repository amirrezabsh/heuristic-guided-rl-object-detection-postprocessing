#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rl_finetune import (
    FOUR_GROUP_NAMES,
    GROUP_NAMES,
    LR_SPACES,
    LayerwiseLRCallback,
    ManualController,
    RandomController,
    ReinforceController,
    UniformController,
    neutral_action_indices,
    resolve_lr_multipliers,
)
from src.rl_finetune.ultralytics_patches import patch_validator_to_skip_loss_shape_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO with uniform, manual, random, or REINFORCE-controlled layer-wise LRs."
    )
    parser.add_argument("--strategy", choices=("standard", "uniform", "manual", "random", "rl"), required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", default="datasets/coco_subsets/search_data.yaml")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--skip-val-loss-errors", action="store_true")
    parser.add_argument(
        "--stage-mode",
        choices=("semantic3", "semantic4"),
        default="semantic3",
        help="semantic3=backbone/neck/head, semantic4=early_backbone/late_backbone/neck/head",
    )
    parser.add_argument(
        "--lr-space",
        choices=tuple(LR_SPACES),
        default="original",
        help=(
            "original=(0.25,0.5,1.0,1.5,2.0); "
            "baseline=(0.75,1.0,1.25,1.5) starts the manual/RL prior at standard fine-tuning"
        ),
    )
    parser.add_argument(
        "--test-split",
        choices=("val", "test"),
        default="val",
        help="evaluation split; use 'test' only once for the final thesis experiment",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/lr_finetuning"))
    parser.add_argument("--name")
    parser.add_argument("--policy-in", type=Path)
    parser.add_argument("--policy-out", type=Path)
    parser.add_argument("--manual-prior-strength", type=float, default=2.0)
    parser.add_argument("--keep-last", action="store_true")
    parser.add_argument(
        "--freeze-policy",
        action="store_true",
        help="use a loaded RL policy deterministically without policy updates",
    )
    return parser.parse_args()


def group_config(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if args.stage_mode == "semantic4":
        group_names = FOUR_GROUP_NAMES
        if args.lr_space == "original":
            return group_names, (0, 1, 1, 2)
        return group_names, neutral_action_indices(len(group_names), resolve_lr_multipliers(args.lr_space))
    group_names = GROUP_NAMES
    if args.lr_space == "original":
        return group_names, (0, 1, 2)
    return group_names, neutral_action_indices(len(group_names), resolve_lr_multipliers(args.lr_space))


def make_controller(args: argparse.Namespace):
    group_names, manual_action = group_config(args)
    if args.strategy == "uniform":
        return UniformController(group_names, lr_multipliers=args.lr_space)
    if args.strategy == "manual":
        return ManualController(group_names, manual_action, lr_multipliers=args.lr_space)
    if args.strategy == "random":
        return RandomController(args.seed, group_names, lr_multipliers=args.lr_space)
    controller = ReinforceController(
        seed=args.seed,
        deterministic=args.freeze_policy,
        manual_prior_strength=args.manual_prior_strength if not args.policy_in else 0.0,
        group_names=group_names,
        manual_action=manual_action,
        lr_multipliers=args.lr_space,
    )
    if args.policy_in:
        controller.load(args.policy_in)
    return controller


def main() -> None:
    args = parse_args()
    name = args.name or f"{args.strategy}_seed{args.seed}"
    project = args.project.resolve()
    run_dir = project / name
    os.environ.setdefault("MPLCONFIGDIR", str((project / "matplotlib").resolve()))

    if args.skip_val_loss_errors or "globalwheat" in str(args.data).lower():
        patch_validator_to_skip_loss_shape_errors()
    from ultralytics import YOLO

    group_names, _manual_action = group_config(args)
    lr_multipliers = resolve_lr_multipliers(args.lr_space)
    model = YOLO(args.model, task="detect")
    callback = None
    if args.strategy != "standard":
        controller = make_controller(args)
        callback = LayerwiseLRCallback(
            controller=controller,
            output_dir=run_dir,
            group_names=group_names,
            stage_mode=args.stage_mode,
        )
        callback.register(model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        lr0=args.base_lr,
        optimizer=args.optimizer,
        mosaic=args.mosaic,
        warmup_epochs=0.0,
        val=True,
        plots=False,
        save=True,
        save_period=-1,
        project=str(project),
        name=name,
        exist_ok=True,
    )

    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights were not produced: {best_weights}")

    test_model = YOLO(str(best_weights), task="detect")
    metrics = test_model.val(
        data=args.data,
        split=args.test_split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=False,
        project=str(project),
        name=f"{name}_final_test",
        exist_ok=True,
    )
    result_strategy = "frozen_rl" if args.strategy == "rl" and args.freeze_policy else args.strategy
    result = {
        "strategy": result_strategy,
        "policy_frozen": bool(args.freeze_policy),
        "seed": args.seed,
        "epochs": args.epochs,
        "base_lr": args.base_lr,
        "mosaic": args.mosaic,
        "skip_val_loss_errors": args.skip_val_loss_errors,
        "stage_mode": args.stage_mode,
        "lr_space": args.lr_space,
        "lr_multipliers": lr_multipliers,
        "test_split": args.test_split,
        "best_weights": str(best_weights),
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "group_parameter_counts": callback.group_parameter_counts if callback is not None else {},
    }
    result_path = run_dir / "final_metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    last_weights = run_dir / "weights" / "last.pt"
    if not args.keep_last:
        last_weights.unlink(missing_ok=True)

    if args.strategy == "rl" and not args.freeze_policy:
        policy_path = args.policy_out or run_dir / "policy.pt"
        controller.save(policy_path)
        print(f"policy={policy_path}")
    print(f"map50_95={result['map50_95']}")
    print(f"map50={result['map50']}")
    print(f"final_metrics={result_path}")


if __name__ == "__main__":
    main()
