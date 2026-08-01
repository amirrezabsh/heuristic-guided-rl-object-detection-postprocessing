#!/usr/bin/env python3
"""Colab-ready Global Wheat YOLO layer-wise RL experiment.

Upload this single file to Google Colab and run:

    !python colab_globalwheat_yolo_rl.py --preset quick

For a longer thesis-style run:

    !python colab_globalwheat_yolo_rl.py --preset full

Outputs are written under ./runs_globalwheat_colab by default.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def ensure_packages() -> None:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])


def patch_validator_to_skip_loss_shape_errors() -> None:
    """Skip only validation-loss shape errors; mAP metrics still run.

    This is mostly for safety. CUDA usually does not need it, but dense wheat
    labels can trigger validation-loss assignment issues on some backends.
    """

    import inspect
    import textwrap

    from ultralytics.engine import validator as validator_module

    if getattr(validator_module.BaseValidator, "_rl_skip_val_loss_patch", False):
        return

    source = textwrap.dedent(inspect.getsource(validator_module.BaseValidator.__call__))
    old = """if self.training:
                self.loss += model.loss(batch, preds)[1]"""
    new = """if self.training:
                try:
                    self.loss += model.loss(batch, preds)[1]
                except RuntimeError as exc:
                    message = str(exc)
                    if (
                        "shape mismatch" not in message
                        and "must match" not in message
                        and "broadcast" not in message
                    ):
                        raise"""
    if old in source:
        namespace = validator_module.__dict__
        exec(source.replace(old, new), namespace)
        validator_module.BaseValidator.__call__ = namespace["__call__"]
        validator_module.BaseValidator._rl_skip_val_loss_patch = True


LR_MULTIPLIERS = (0.25, 0.5, 1.0, 1.5, 2.0)
GROUP_NAMES = ("backbone", "neck", "head")
MANUAL_ACTION = (0, 1, 2)


@dataclass
class LRDecision:
    multipliers: tuple[float, ...]
    action_indices: tuple[int, ...]


class UniformController:
    def select(self, _state: Any) -> LRDecision:
        return LRDecision((1.0, 1.0, 1.0), (2, 2, 2))

    def update(self, _reward: float) -> dict[str, float]:
        return {}


class ManualController:
    def select(self, _state: Any) -> LRDecision:
        return LRDecision(tuple(LR_MULTIPLIERS[i] for i in MANUAL_ACTION), MANUAL_ACTION)

    def update(self, _reward: float) -> dict[str, float]:
        return {}


class RandomController:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def select(self, _state: Any) -> LRDecision:
        action = tuple(self.rng.randrange(len(LR_MULTIPLIERS)) for _ in GROUP_NAMES)
        return LRDecision(tuple(LR_MULTIPLIERS[i] for i in action), action)

    def update(self, _reward: float) -> dict[str, float]:
        return {}


class ReinforceController:
    def __init__(
        self,
        seed: int,
        learning_rate: float = 3e-3,
        entropy_weight: float = 0.01,
        manual_prior_strength: float = 2.0,
        deterministic: bool = False,
    ) -> None:
        import torch

        self.torch = torch
        torch.manual_seed(seed)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, len(GROUP_NAMES) * len(LR_MULTIPLIERS)),
        )
        with torch.no_grad():
            final = self.model[-1]
            assert isinstance(final, torch.nn.Linear)
            final.bias.zero_()
            for group_index, action_index in enumerate(MANUAL_ACTION):
                final.bias[group_index * len(LR_MULTIPLIERS) + action_index] = manual_prior_strength
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.entropy_weight = entropy_weight
        self.deterministic = deterministic
        self.reward_baseline = 0.0
        self.pending_log_prob = None
        self.pending_entropy = None

    def select(self, state: Any) -> LRDecision:
        torch = self.torch
        state_tensor = torch.as_tensor(state, dtype=torch.float32).view(1, -1)
        logits = self.model(state_tensor).view(len(GROUP_NAMES), len(LR_MULTIPLIERS))
        dist = torch.distributions.Categorical(logits=logits)
        if self.deterministic:
            action_tensor = torch.argmax(logits, dim=-1)
            self.pending_log_prob = None
            self.pending_entropy = None
        else:
            action_tensor = dist.sample()
            self.pending_log_prob = dist.log_prob(action_tensor).sum()
            self.pending_entropy = dist.entropy().sum()
        action = tuple(int(x) for x in action_tensor.tolist())
        return LRDecision(tuple(LR_MULTIPLIERS[i] for i in action), action)

    def update(self, reward: float) -> dict[str, float]:
        if self.pending_log_prob is None:
            return {}
        self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * reward
        advantage = reward - self.reward_baseline
        loss = -self.pending_log_prob * advantage - self.entropy_weight * self.pending_entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "reward_baseline": float(self.reward_baseline),
            "advantage": float(advantage),
            "policy_loss": float(loss.detach().cpu()),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "reward_baseline": self.reward_baseline,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = self.torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.reward_baseline = float(checkpoint.get("reward_baseline", 0.0))


def detect_index(model: Any) -> int:
    modules = getattr(model, "model", None)
    if modules is None:
        raise ValueError("Expected Ultralytics model.model sequence.")
    for index, module in enumerate(modules):
        if module.__class__.__name__.lower() == "detect":
            return index
    return len(modules) - 1


def parameter_stage(name: str, detect_idx: int) -> str:
    import re

    match = re.search(r"(?:^|\.)model\.(\d+)\.", name)
    if match is None:
        return "head"
    layer = int(match.group(1))
    backbone_end = max(1, int(round(detect_idx * 0.45)))
    if layer < backbone_end:
        return "backbone"
    if layer < detect_idx:
        return "neck"
    return "head"


def split_optimizer_by_stage(trainer: Any) -> dict[str, int]:
    detect_idx = detect_index(trainer.model)
    stage_by_param = {
        id(parameter): parameter_stage(name, detect_idx)
        for name, parameter in trainer.model.named_parameters()
    }
    new_groups = []
    new_lambdas = []
    counts = {name: 0 for name in GROUP_NAMES}
    old_lambdas = list(getattr(trainer.scheduler, "lr_lambdas", []))
    for old_index, old_group in enumerate(trainer.optimizer.param_groups):
        partitions = {name: [] for name in GROUP_NAMES}
        for parameter in old_group["params"]:
            partitions[stage_by_param.get(id(parameter), "head")].append(parameter)
        for stage, params in partitions.items():
            if not params:
                continue
            group = {key: value for key, value in old_group.items() if key != "params"}
            group["params"] = params
            group["rl_stage"] = stage
            group["rl_base_lr"] = float(old_group.get("initial_lr", trainer.args.lr0))
            group["initial_lr"] = group["rl_base_lr"]
            new_groups.append(group)
            if old_lambdas:
                new_lambdas.append(old_lambdas[min(old_index, len(old_lambdas) - 1)])
            counts[stage] += sum(parameter.numel() for parameter in params)
    if any(value == 0 for value in counts.values()):
        raise RuntimeError(f"Could not create all stage groups: {counts}")
    trainer.optimizer.param_groups[:] = new_groups
    trainer.scheduler.base_lrs = [group["initial_lr"] for group in new_groups]
    if new_lambdas:
        trainer.scheduler.lr_lambdas = new_lambdas
    if hasattr(trainer.scheduler, "_last_lr"):
        trainer.scheduler._last_lr = [group["lr"] for group in new_groups]
    return counts


class LayerwiseLRCallback:
    def __init__(self, controller: Any, output_dir: Path, reward_scale: float = 100.0) -> None:
        import numpy as np

        self.np = np
        self.controller = controller
        self.output_dir = output_dir
        self.reward_scale = reward_scale
        self.history_path = output_dir / "lr_history.jsonl"
        self.current_decision = LRDecision((1.0, 1.0, 1.0), (2, 2, 2))
        self.current_map = 0.0
        self.previous_map = None
        self.map_delta = 0.0
        self.current_loss = 0.0
        self.group_parameter_counts: dict[str, int] = {}

    def register(self, yolo: Any) -> None:
        yolo.add_callback("on_train_start", self.on_train_start)
        yolo.add_callback("on_train_epoch_start", self.on_train_epoch_start)
        yolo.add_callback("on_train_batch_start", self.on_train_batch_start)
        yolo.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)

    def state(self, trainer: Any) -> Any:
        progress = float(trainer.epoch) / max(1, int(trainer.epochs) - 1)
        return self.np.asarray(
            [
                progress,
                self.current_map,
                self.map_delta,
                math.log1p(max(0.0, self.current_loss)),
            ],
            dtype=self.np.float32,
        )

    def on_train_start(self, trainer: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path.unlink(missing_ok=True)
        self.group_parameter_counts = split_optimizer_by_stage(trainer)
        (self.output_dir / "group_parameter_counts.json").write_text(
            json.dumps(self.group_parameter_counts, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def on_train_epoch_start(self, trainer: Any) -> None:
        self.current_decision = self.controller.select(self.state(trainer))

    def on_train_batch_start(self, trainer: Any) -> None:
        stage_to_multiplier = dict(zip(GROUP_NAMES, self.current_decision.multipliers))
        schedule_factor = float(trainer.lf(trainer.epoch))
        for group in trainer.optimizer.param_groups:
            group["lr"] = float(group["rl_base_lr"]) * schedule_factor * stage_to_multiplier[group["rl_stage"]]

    def on_fit_epoch_end(self, trainer: Any) -> None:
        metrics = trainer.metrics or {}
        current_map = float(metrics.get("metrics/mAP50-95(B)", 0.0))
        if trainer.tloss is not None:
            self.current_loss = float(trainer.tloss.detach().float().mean().cpu())
        if self.previous_map is None:
            reward = 0.0
            update_stats = {}
        else:
            self.map_delta = current_map - self.previous_map
            reward = self.reward_scale * self.map_delta
            update_stats = self.controller.update(reward)
        record = {
            "epoch": int(trainer.epoch),
            "map50_95": current_map,
            "map_delta": self.map_delta,
            "reward": reward,
            "loss": self.current_loss,
            "action_indices": list(self.current_decision.action_indices),
            "lr_multipliers": dict(zip(GROUP_NAMES, self.current_decision.multipliers)),
            **update_stats,
        }
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self.previous_map = current_map
        self.current_map = current_map


def label_count(image_path: Path) -> int:
    label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not label_path.exists():
        return 0
    return len([line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()])


def collect_images(root: Path, parts: list[str], max_boxes: int | None) -> list[Path]:
    images: list[Path] = []
    for part in parts:
        for suffix in ("*.png", "*.jpg", "*.jpeg"):
            images.extend(sorted((root / "images" / part).glob(suffix)))
    output = []
    for image in images:
        count = label_count(image)
        if count <= 0:
            continue
        if max_boxes is not None and count > max_boxes:
            continue
        output.append(image)
    return output


def write_list(path: Path, images: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")


def prepare_globalwheat_splits(
    output_dir: Path,
    preset: str,
    max_boxes: int | None,
    seed: int,
) -> Path:
    from ultralytics.data.utils import check_det_dataset
    import yaml

    data = check_det_dataset("GlobalWheat2020.yaml")
    root = Path(data["path"])
    split_dir = output_dir / "data"
    rng = random.Random(seed)

    train_parts = ["arvalis_1", "arvalis_2", "arvalis_3", "rres_1", "inrae_1", "usask_1"]
    val_test_part = ["ethz_1"]

    train = collect_images(root, train_parts, max_boxes=max_boxes)
    val_test = collect_images(root, val_test_part, max_boxes=max_boxes)
    rng.shuffle(train)
    rng.shuffle(val_test)

    if preset == "quick":
        train = train[:800]
        val = val_test[:200]
        test = val_test[200:400]
    else:
        split = len(val_test) // 2
        val = val_test[:split]
        test = val_test[split:]

    if not train or not val or not test:
        raise RuntimeError(
            f"Empty split produced. train={len(train)} val={len(val)} test={len(test)} max_boxes={max_boxes}"
        )

    write_list(split_dir / "train.txt", train)
    write_list(split_dir / "val.txt", val)
    write_list(split_dir / "test.txt", test)

    yaml_path = split_dir / f"globalwheat_{preset}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": str((split_dir / "train.txt").resolve()),
                "val": str((split_dir / "val.txt").resolve()),
                "test": str((split_dir / "test.txt").resolve()),
                "names": {0: "wheat_head"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    metadata = {
        "preset": preset,
        "seed": seed,
        "max_boxes_per_image": max_boxes,
        "train_images": len(train),
        "val_images": len(val),
        "test_images": len(test),
        "train_boxes": sum(label_count(p) for p in train),
        "val_boxes": sum(label_count(p) for p in val),
        "test_boxes": sum(label_count(p) for p in test),
    }
    (split_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2))
    return yaml_path


def yolo_train_kwargs(args: argparse.Namespace, project: Path, name: str, epochs: int, seed: int) -> dict[str, Any]:
    return {
        "data": str(args.data_yaml),
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": 0,
        "seed": seed,
        "lr0": args.base_lr,
        "optimizer": "AdamW",
        "mosaic": args.mosaic,
        "warmup_epochs": 0.0,
        "val": True,
        "plots": False,
        "project": str(project),
        "name": name,
        "exist_ok": True,
    }


def train_policy(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    policy_dir = args.out / "policy_training"
    temporary_dir = policy_dir / "temporary_episodes"
    policy_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = policy_dir / "episodes.jsonl"
    episodes_path.unlink(missing_ok=True)

    controller = ReinforceController(seed=args.train_seed)
    for episode in range(args.episodes):
        name = f"episode_{episode:02d}"
        run_dir = temporary_dir / name
        callback = LayerwiseLRCallback(controller, run_dir)
        model = YOLO(args.model, task="detect")
        callback.register(model)
        model.train(
            **yolo_train_kwargs(
                args,
                temporary_dir,
                name,
                args.policy_epochs,
                args.train_seed + episode,
            ),
            save=False,
        )
        history = [
            json.loads(line)
            for line in (run_dir / "lr_history.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = {
            "episode": episode,
            "seed": args.train_seed + episode,
            "epochs": len(history),
            "first_val_map50_95": history[0]["map50_95"],
            "final_val_map50_95": history[-1]["map50_95"],
            "best_val_map50_95": max(r["map50_95"] for r in history),
            "total_reward": sum(r["reward"] for r in history),
            "actions": [r["action_indices"] for r in history],
        }
        with episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
        controller.save(policy_dir / "policy.pt")
        shutil.rmtree(run_dir, ignore_errors=True)
        print("POLICY_EPISODE", json.dumps(summary, sort_keys=True))

    shutil.rmtree(temporary_dir, ignore_errors=True)
    (policy_dir / "training_metadata.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    return policy_dir / "policy.pt"


def make_controller(strategy: str, seed: int, policy_path: Path | None) -> Any:
    if strategy == "uniform":
        return UniformController()
    if strategy == "manual":
        return ManualController()
    if strategy == "random":
        return RandomController(seed)
    if strategy == "frozen_rl":
        controller = ReinforceController(seed=seed, deterministic=True, manual_prior_strength=0.0)
        if policy_path is None:
            raise ValueError("policy_path required for frozen_rl")
        controller.load(policy_path)
        return controller
    raise ValueError(strategy)


def evaluate_strategy(args: argparse.Namespace, strategy: str, seed: int, policy_path: Path | None) -> dict[str, Any]:
    from ultralytics import YOLO

    project = args.out / "evaluation"
    name = f"{strategy}_seed{seed}"
    model = YOLO(args.model, task="detect")
    callback = None
    if strategy != "standard":
        callback = LayerwiseLRCallback(make_controller(strategy, seed, policy_path), project / name)
        callback.register(model)
    model.train(
        **yolo_train_kwargs(args, project, name, args.eval_epochs, seed),
        save=True,
        save_period=-1,
    )
    best = project / name / "weights" / "best.pt"
    metrics = YOLO(str(best), task="detect").val(
        data=str(args.data_yaml),
        split=args.test_split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        plots=False,
        project=str(project),
        name=f"{name}_final_test",
        exist_ok=True,
    )
    result = {
        "strategy": strategy,
        "seed": seed,
        "epochs": args.eval_epochs,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "best_weights": str(best),
        "group_parameter_counts": callback.group_parameter_counts if callback else {},
    }
    result_path = project / name / "final_metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (project / name / "weights" / "last.pt").unlink(missing_ok=True)
    print("RESULT", json.dumps(result, sort_keys=True))
    return result


def summarize(results: list[dict[str, Any]], out_dir: Path) -> None:
    evaluation_dir = out_dir / "evaluation"
    csv_path = evaluation_dir / "comparison.csv"
    fields = ["strategy", "seed", "map50_95", "map50", "precision", "recall", "best_weights"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fields})
    grouped: dict[str, list[float]] = {}
    for row in results:
        grouped.setdefault(row["strategy"], []).append(float(row["map50_95"]))
    summary = {}
    for strategy, values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        summary[strategy] = {"n": len(values), "mean_map50_95": mean, "std_pop": std, "values": values}
        print(f"{strategy}: n={len(values)} mean_map50_95={mean:.6f} std={std:.6f}")
    (evaluation_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"comparison={csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--out", type=Path, default=Path("runs_globalwheat_colab"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--policy-epochs", type=int)
    parser.add_argument("--eval-epochs", type=int)
    parser.add_argument("--eval-seeds", default=None)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--max-boxes", type=int, default=None)
    parser.add_argument("--test-split", choices=("val", "test"), default="test")
    parser.add_argument("--skip-val-loss-errors", action="store_true", default=True)
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def apply_preset_defaults(args: argparse.Namespace) -> None:
    if args.preset == "quick":
        args.imgsz = args.imgsz or 512
        args.batch = args.batch or 8
        args.episodes = args.episodes or 3
        args.policy_epochs = args.policy_epochs or 2
        args.eval_epochs = args.eval_epochs or 2
        args.eval_seeds = args.eval_seeds or "101"
        args.max_boxes = 80 if args.max_boxes is None else args.max_boxes
    else:
        args.imgsz = args.imgsz or 640
        args.batch = args.batch or 16
        args.episodes = args.episodes or 8
        args.policy_epochs = args.policy_epochs or 8
        args.eval_epochs = args.eval_epochs or 20
        args.eval_seeds = args.eval_seeds or "101 202 303"
        args.max_boxes = args.max_boxes


def main() -> None:
    ensure_packages()
    args = parse_args()
    apply_preset_defaults(args)
    if args.skip_val_loss_errors:
        patch_validator_to_skip_loss_shape_errors()
    args.out.mkdir(parents=True, exist_ok=True)
    args.data_yaml = prepare_globalwheat_splits(args.out, args.preset, args.max_boxes, seed=42)
    policy_path = train_policy(args)
    seeds = [int(s) for s in str(args.eval_seeds).split()]
    results = []
    for seed in seeds:
        for strategy in ("standard", "uniform", "manual", "random", "frozen_rl"):
            results.append(evaluate_strategy(args, strategy, seed, policy_path))
    summarize(results, args.out)
    if not args.no_zip:
        archive = shutil.make_archive(str(args.out), "zip", args.out)
        print(f"archive={archive}")


if __name__ == "__main__":
    main()
