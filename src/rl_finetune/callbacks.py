from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .controller import GROUP_NAMES, LRController, LRDecision

_LAYER_PATTERN = re.compile(r"(?:^|\.)model\.(\d+)\.")


def _detect_index(model: Any) -> int:
    modules = getattr(model, "model", None)
    if modules is None:
        raise ValueError("Expected an Ultralytics model with a top-level 'model' sequence.")
    for index, module in enumerate(modules):
        if module.__class__.__name__.lower() == "detect":
            return index
    return len(modules) - 1


def yolo_parameter_stages(
    model: Any,
    group_names: tuple[str, ...] = GROUP_NAMES,
    stage_mode: str = "semantic3",
) -> dict[int, str]:
    detect_index = _detect_index(model)
    backbone_end = max(1, int(round(detect_index * 0.45)))
    early_backbone_end = max(1, int(round(backbone_end * 0.5)))
    stages: dict[int, str] = {}
    for name, parameter in model.named_parameters():
        match = _LAYER_PATTERN.search(name)
        if match is None:
            stage = group_names[-1]
        else:
            layer_index = int(match.group(1))
            if stage_mode == "semantic4":
                if layer_index < early_backbone_end:
                    stage = "early_backbone"
                elif layer_index < backbone_end:
                    stage = "late_backbone"
                elif layer_index < detect_index:
                    stage = "neck"
                else:
                    stage = "head"
            elif layer_index < backbone_end:
                stage = "backbone"
            elif layer_index < detect_index:
                stage = "neck"
            else:
                stage = "head"
        stages[id(parameter)] = stage
    return stages


def split_optimizer_by_yolo_stage(
    trainer: Any,
    group_names: tuple[str, ...] = GROUP_NAMES,
    stage_mode: str = "semantic3",
) -> dict[str, int]:
    """Split existing decay/bias optimizer groups by YOLO semantic stage."""

    stage_by_parameter = yolo_parameter_stages(trainer.model, group_names, stage_mode)
    new_groups: list[dict[str, Any]] = []
    counts = {name: 0 for name in group_names}

    for old_group in trainer.optimizer.param_groups:
        partitions = {name: [] for name in group_names}
        for parameter in old_group["params"]:
            partitions[stage_by_parameter.get(id(parameter), group_names[-1])].append(parameter)

        for stage, parameters in partitions.items():
            if not parameters:
                continue
            group = {key: value for key, value in old_group.items() if key != "params"}
            group["params"] = parameters
            group["rl_stage"] = stage
            group["rl_base_lr"] = float(old_group.get("initial_lr", trainer.args.lr0))
            group["initial_lr"] = group["rl_base_lr"]
            new_groups.append(group)
            counts[stage] += sum(parameter.numel() for parameter in parameters)

    if not new_groups or any(count == 0 for count in counts.values()):
        raise ValueError(f"Could not create all YOLO parameter stages: {counts}")

    trainer.optimizer.param_groups[:] = new_groups
    trainer.scheduler.base_lrs = [group["initial_lr"] for group in new_groups]
    return counts


class LayerwiseLRCallback:
    def __init__(
        self,
        controller: LRController,
        output_dir: Path,
        reward_scale: float = 100.0,
        group_names: tuple[str, ...] = GROUP_NAMES,
        stage_mode: str = "semantic3",
    ) -> None:
        self.controller = controller
        self.output_dir = output_dir
        self.reward_scale = reward_scale
        self.group_names = group_names
        self.stage_mode = stage_mode
        self.history_path = output_dir / "lr_history.jsonl"
        self.current_decision = LRDecision(
            tuple(1.0 for _ in self.group_names),
            tuple(2 for _ in self.group_names),
        )
        self.current_map = 0.0
        self.previous_map: float | None = None
        self.map_delta = 0.0
        self.current_loss = 0.0
        self.group_parameter_counts: dict[str, int] = {}

    def register(self, yolo: Any) -> None:
        yolo.add_callback("on_train_start", self.on_train_start)
        yolo.add_callback("on_train_epoch_start", self.on_train_epoch_start)
        yolo.add_callback("on_train_batch_start", self.on_train_batch_start)
        yolo.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)

    def state(self, trainer: Any) -> np.ndarray:
        progress = float(trainer.epoch) / max(1, int(trainer.epochs) - 1)
        return np.asarray(
            [
                progress,
                self.current_map,
                self.map_delta,
                math.log1p(max(0.0, self.current_loss)),
            ],
            dtype=np.float32,
        )

    def on_train_start(self, trainer: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path.unlink(missing_ok=True)
        self.group_parameter_counts = split_optimizer_by_yolo_stage(
            trainer,
            self.group_names,
            self.stage_mode,
        )
        (self.output_dir / "group_parameter_counts.json").write_text(
            json.dumps(self.group_parameter_counts, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def on_train_epoch_start(self, trainer: Any) -> None:
        self.current_decision = self.controller.select(self.state(trainer))

    def on_train_batch_start(self, trainer: Any) -> None:
        multiplier_by_stage = dict(zip(self.group_names, self.current_decision.multipliers))
        schedule_factor = float(trainer.lf(trainer.epoch))
        for group in trainer.optimizer.param_groups:
            group["lr"] = (
                float(group["rl_base_lr"])
                * schedule_factor
                * multiplier_by_stage[group["rl_stage"]]
            )

    def on_fit_epoch_end(self, trainer: Any) -> None:
        if int(trainer.epoch) >= int(trainer.epochs):
            return
        metrics = trainer.metrics or {}
        current_map = float(metrics.get("metrics/mAP50-95(B)", 0.0))
        if trainer.tloss is not None:
            self.current_loss = float(trainer.tloss.detach().float().mean().cpu())

        if self.previous_map is None:
            reward = 0.0
            update_stats: dict[str, float] = {}
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
            "lr_multipliers": dict(zip(self.group_names, self.current_decision.multipliers)),
            "effective_lrs": {
                stage: next(
                    float(group["lr"])
                    for group in trainer.optimizer.param_groups
                    if group["rl_stage"] == stage
                )
                for stage in self.group_names
            },
            **update_stats,
        }
        with self.history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(record, sort_keys=True) + "\n")

        self.previous_map = current_map
        self.current_map = current_map
