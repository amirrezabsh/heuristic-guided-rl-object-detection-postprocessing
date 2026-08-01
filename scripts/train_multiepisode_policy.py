#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
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
    ReinforceController,
    neutral_action_indices,
    resolve_lr_multipliers,
)
from src.rl_finetune.ultralytics_patches import patch_validator_to_skip_loss_shape_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one shared REINFORCE LR policy across fresh YOLO fine-tuning episodes."
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", default="datasets/coco_subsets/search_data.yaml")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--epochs-per-episode", type=int, default=6)
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
            "baseline=(0.75,1.0,1.25,1.5) starts the policy prior at standard fine-tuning"
        ),
    )
    parser.add_argument("--policy-lr", type=float, default=3e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--manual-prior-strength", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("runs/lr_policy_development"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from policy.pt and completed records in episodes.jsonl",
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


def read_episode_history(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Episode history was not produced: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.epochs_per_episode < 2:
        raise ValueError("Use at least 1 episode and 2 epochs per episode.")

    output_dir = args.out.resolve()
    temporary_dir = output_dir / "temporary_episodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / "matplotlib").resolve()))

    if args.skip_val_loss_errors or "globalwheat" in str(args.data).lower():
        patch_validator_to_skip_loss_shape_errors()
    from ultralytics import YOLO

    group_names, manual_action = group_config(args)
    lr_multipliers = resolve_lr_multipliers(args.lr_space)
    controller = ReinforceController(
        seed=args.seed,
        learning_rate=args.policy_lr,
        entropy_weight=args.entropy_weight,
        manual_prior_strength=args.manual_prior_strength,
        group_names=group_names,
        manual_action=manual_action,
        lr_multipliers=args.lr_space,
    )
    episodes_path = output_dir / "episodes.jsonl"
    policy_path = output_dir / "policy.pt"
    start_episode = 0
    if args.resume:
        if not policy_path.exists() or not episodes_path.exists():
            raise FileNotFoundError(
                f"Resume requires both {policy_path} and {episodes_path}."
            )
        completed = read_episode_history(episodes_path)
        start_episode = len(completed)
        controller.load(policy_path)
        for stale_dir in temporary_dir.iterdir():
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir)
        print(f"resuming_from_episode={start_episode}")
    else:
        episodes_path.unlink(missing_ok=True)
        for stale_dir in temporary_dir.iterdir():
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir)

    for episode in range(start_episode, args.episodes):
        episode_seed = args.seed + episode
        episode_name = f"episode_{episode:02d}"
        episode_dir = temporary_dir / episode_name
        callback = LayerwiseLRCallback(
            controller=controller,
            output_dir=episode_dir,
            group_names=group_names,
            stage_mode=args.stage_mode,
        )
        model = YOLO(args.model, task="detect")
        callback.register(model)
        model.train(
            data=args.data,
            epochs=args.epochs_per_episode,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=episode_seed,
            lr0=args.base_lr,
            optimizer=args.optimizer,
            mosaic=args.mosaic,
            warmup_epochs=0.0,
            val=True,
            plots=False,
            save=False,
            project=str(temporary_dir),
            name=episode_name,
            exist_ok=True,
        )

        history = read_episode_history(episode_dir / "lr_history.jsonl")
        summary = {
            "episode": episode,
            "seed": episode_seed,
            "epochs": len(history),
            "first_val_map50_95": history[0]["map50_95"],
            "final_val_map50_95": history[-1]["map50_95"],
            "best_val_map50_95": max(record["map50_95"] for record in history),
            "total_reward": sum(record["reward"] for record in history),
            "actions": [record["action_indices"] for record in history],
        }
        with episodes_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(summary, sort_keys=True) + "\n")

        controller.save(policy_path)
        shutil.rmtree(episode_dir)
        print(
            f"episode={episode} seed={episode_seed} "
            f"best_val_map50_95={summary['best_val_map50_95']:.6f} "
            f"final_val_map50_95={summary['final_val_map50_95']:.6f}"
        )

    metadata = {
        "model": args.model,
        "data": args.data,
        "episodes": args.episodes,
        "epochs_per_episode": args.epochs_per_episode,
        "seed": args.seed,
        "base_lr": args.base_lr,
        "mosaic": args.mosaic,
        "skip_val_loss_errors": args.skip_val_loss_errors,
        "stage_mode": args.stage_mode,
        "lr_space": args.lr_space,
        "lr_multipliers": lr_multipliers,
        "policy_lr": args.policy_lr,
        "entropy_weight": args.entropy_weight,
        "manual_prior_strength": args.manual_prior_strength,
        "policy": str(policy_path),
        "development_split": "val",
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_dir.rmdir()
    print(f"policy={policy_path}")
    print(f"episodes={episodes_path}")


if __name__ == "__main__":
    main()
