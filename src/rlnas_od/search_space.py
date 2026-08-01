from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence


@dataclass(frozen=True)
class Action:
    """One valid architecture-construction decision."""

    name: str
    block_type: str = ""
    channels: int = 0
    width_mult: float = 1.0
    repeats: int = 1
    kernel_size: int = 3
    shortcut: bool = False

    @property
    def is_stop(self) -> bool:
        return self.name == "stop"


class SearchSpace:
    """Base class for model-family-specific search spaces."""

    def initial_state(self) -> Dict[str, int]:
        raise NotImplementedError

    def valid_actions(self, state: Dict[str, int]) -> Sequence[Action]:
        raise NotImplementedError

    def transition(self, state: Dict[str, int], action: Action) -> Dict[str, int]:
        raise NotImplementedError


class YoloNeckSearchSpace(SearchSpace):
    """Shortcut-only four-stage search space for a YOLOv8-style neck.

    Widths and effective repeat counts stay fixed at YOLOv8n values so
    pretrained tensors remain shape compatible. At every stage the controller
    only chooses whether the C2f shortcut is enabled.
    """

    def __init__(self, max_blocks: int = 4) -> None:
        if max_blocks != 4:
            raise ValueError("the direct YOLO neck search requires exactly 4 stages")
        self.max_blocks = max_blocks
        self._actions = [
            Action(
                name=f"s{int(shortcut)}",
                block_type="C2f",
                channels=100,
                width_mult=1.0,
                repeats=1,
                shortcut=shortcut,
            )
            for shortcut in (False, True)
        ]

    def initial_state(self) -> Dict[str, int]:
        return {"position": 0, "last_channels": 100}

    def valid_actions(self, state: Dict[str, int]) -> Sequence[Action]:
        if state["position"] >= self.max_blocks:
            return [Action("stop")]
        return list(self._actions)

    def transition(self, state: Dict[str, int], action: Action) -> Dict[str, int]:
        if action.is_stop:
            return dict(state)
        return {
            "position": state["position"] + 1,
            "last_channels": action.channels,
        }

    def action_names(self) -> Iterable[str]:
        yield from (action.name for action in self._actions)
