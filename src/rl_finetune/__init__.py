"""RL-controlled layer-wise learning rates for Ultralytics detectors."""

from .callbacks import LayerwiseLRCallback, split_optimizer_by_yolo_stage
from .controller import (
    BASELINE_AWARE_LR_MULTIPLIERS,
    FOUR_GROUP_NAMES,
    GROUP_NAMES,
    LR_MULTIPLIERS,
    LR_SPACES,
    ManualController,
    RandomController,
    ReinforceController,
    UniformController,
    neutral_action_indices,
    resolve_lr_multipliers,
)

__all__ = [
    "GROUP_NAMES",
    "FOUR_GROUP_NAMES",
    "LR_MULTIPLIERS",
    "BASELINE_AWARE_LR_MULTIPLIERS",
    "LR_SPACES",
    "LayerwiseLRCallback",
    "ManualController",
    "RandomController",
    "ReinforceController",
    "UniformController",
    "neutral_action_indices",
    "resolve_lr_multipliers",
    "split_optimizer_by_yolo_stage",
]
