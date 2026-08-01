from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class LayerSpec:
    """A compact, model-agnostic layer/block description."""

    block_type: str
    channels: int
    repeats: int = 1
    kernel_size: int = 3
    shortcut: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "block_type": self.block_type,
            "channels": self.channels,
            "repeats": self.repeats,
            "kernel_size": self.kernel_size,
            "shortcut": self.shortcut,
            "metadata": dict(self.metadata),
        }


@dataclass
class Architecture:
    """Architecture candidate produced by a model-family adapter."""

    family: str
    layers: List[LayerSpec] = field(default_factory=list)
    stopped: bool = False

    def add_layer(self, layer: LayerSpec) -> None:
        if self.stopped:
            raise ValueError("Cannot add layers after the architecture is stopped")
        self.layers.append(layer)

    @property
    def depth(self) -> int:
        return sum(layer.repeats for layer in self.layers)

    def to_dict(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "stopped": self.stopped,
            "depth": self.depth,
            "layers": [layer.to_dict() for layer in self.layers],
        }

