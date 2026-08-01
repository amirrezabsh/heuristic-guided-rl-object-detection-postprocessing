from __future__ import annotations

from typing import Dict, List

from .architecture import Architecture, LayerSpec
from .search_space import Action, YoloNeckSearchSpace


class YoloArchitectureAdapter:
    """Builds and exports constrained YOLOv8-style architectures.

    The exported model preserves YOLOv8's multi-scale FPN/PAN topology and
    Detect inputs. The sampled architecture configures the C2f blocks in that
    neck/head instead of replacing the detector with a fragile single-scale
    chain.
    """

    family = "yolo"

    def __init__(self, max_blocks: int = 4) -> None:
        self.search_space = YoloNeckSearchSpace(max_blocks=max_blocks)

    def empty_architecture(self) -> Architecture:
        return Architecture(family=self.family)

    def apply_action(self, architecture: Architecture, action: Action) -> None:
        if action.is_stop:
            architecture.stopped = True
            return
        architecture.add_layer(
            LayerSpec(
                block_type=action.block_type,
                channels=action.channels,
                repeats=action.repeats,
                kernel_size=action.kernel_size,
                shortcut=action.shortcut,
                metadata={"width_mult": action.width_mult},
            )
        )

    def to_yolo_dict(self, architecture: Architecture) -> Dict[str, object]:
        p4_top, p3, p4_bottom, p5 = self._neck_stage_configs(architecture)

        return {
            "nc": 80,
            "scale": "n",
            "scales": {"n": [0.33, 0.25, 1024]},
            "backbone": [
                [-1, 1, "Conv", [64, 3, 2]],
                [-1, 1, "Conv", [128, 3, 2]],
                [-1, 3, "C2f", [128, True]],
                [-1, 1, "Conv", [256, 3, 2]],
                [-1, 6, "C2f", [256, True]],
                [-1, 1, "Conv", [512, 3, 2]],
                [-1, 6, "C2f", [512, True]],
                [-1, 1, "Conv", [1024, 3, 2]],
                [-1, 3, "C2f", [1024, True]],
                [-1, 1, "SPPF", [1024, 5]],
            ],
            "head": [
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 6], 1, "Concat", [1]],
                [-1, p4_top["repeats"], "C2f", [p4_top["channels"], p4_top["shortcut"]]],
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 4], 1, "Concat", [1]],
                [-1, p3["repeats"], "C2f", [p3["channels"], p3["shortcut"]]],
                [-1, 1, "Conv", [p3["channels"], 3, 2]],
                [[-1, 12], 1, "Concat", [1]],
                [-1, p4_bottom["repeats"], "C2f", [p4_bottom["channels"], p4_bottom["shortcut"]]],
                [-1, 1, "Conv", [p4_bottom["channels"], 3, 2]],
                [[-1, 9], 1, "Concat", [1]],
                [-1, p5["repeats"], "C2f", [p5["channels"], p5["shortcut"]]],
                [[15, 18, 21], 1, "Detect", ["nc"]],
            ],
        }

    def to_yolo_yaml(self, architecture: Architecture) -> str:
        return _dump_simple_yaml(self.to_yolo_dict(architecture))

    def _neck_stage_configs(self, architecture: Architecture) -> tuple[dict, dict, dict, dict]:
        defaults = [
            {"channels": 512, "repeats": 3, "shortcut": False},
            {"channels": 256, "repeats": 3, "shortcut": False},
            {"channels": 512, "repeats": 3, "shortcut": False},
            {"channels": 1024, "repeats": 3, "shortcut": False},
        ]
        if not architecture.layers:
            return tuple(defaults)  # type: ignore[return-value]

        configs = []
        for index, default in enumerate(defaults):
            if index >= len(architecture.layers):
                configs.append(default)
                continue
            layer = architecture.layers[index]
            configs.append(
                {
                    "channels": self._stage_channels(layer, default["channels"]),
                    "repeats": layer.repeats * 3,
                    "shortcut": layer.shortcut,
                }
            )
        return tuple(configs)  # type: ignore[return-value]

    def _stage_channels(self, layer: LayerSpec, default_channels: int) -> int:
        width_mult = float(layer.metadata.get("width_mult", layer.channels / 100.0))
        return max(64, int(default_channels * width_mult))


def _dump_simple_yaml(value: object, indent: int = 0) -> str:
    """Small YAML writer for the simple dict/list/scalar structures used here."""

    pad = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_dump_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_format_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                rendered = _dump_simple_yaml(item, indent + 2)
                lines.append(f"{pad}- {rendered.lstrip()}")
            else:
                lines.append(f"{pad}- {_format_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_format_scalar(value)}"


def _format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)
