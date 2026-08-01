from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Sequence, Tuple

from .search_space import Action


@dataclass
class Decision:
    state_key: str
    action_name: str


class TabularPolicyController:
    """Small REINFORCE-style controller for discrete architecture actions."""

    def __init__(
        self,
        action_names: Sequence[str],
        learning_rate: float = 0.15,
        entropy_weight: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight
        self.rng = random.Random(seed)
        self.logits: DefaultDict[str, Dict[str, float]] = defaultdict(
            lambda: {name: 0.0 for name in action_names}
        )
        self.baseline = 0.0
        self.baseline_momentum = 0.9

    def select(self, state: Dict[str, int], actions: Sequence[Action]) -> Tuple[Action, Decision]:
        state_key = self._state_key(state)
        probs = self._probabilities(state_key, actions)
        sample = self.rng.random()
        cumulative = 0.0
        for action in actions:
            cumulative += probs[action.name]
            if sample <= cumulative:
                return action, Decision(state_key=state_key, action_name=action.name)
        action = actions[-1]
        return action, Decision(state_key=state_key, action_name=action.name)

    def update(self, decisions: Sequence[Decision], reward: float) -> None:
        advantage = reward - self.baseline
        self.baseline = (
            self.baseline_momentum * self.baseline
            + (1.0 - self.baseline_momentum) * reward
        )
        for decision in decisions:
            state_logits = self.logits[decision.state_key]
            action_names = list(state_logits)
            probs = self._softmax(state_logits)
            for name in action_names:
                grad = (1.0 if name == decision.action_name else 0.0) - probs[name]
                entropy_grad = -probs[name] * math.log(max(probs[name], 1e-12))
                state_logits[name] += self.learning_rate * (
                    advantage * grad + self.entropy_weight * entropy_grad
                )

    def _probabilities(self, state_key: str, actions: Sequence[Action]) -> Dict[str, float]:
        state_logits = self.logits[state_key]
        scoped_logits = {action.name: state_logits[action.name] for action in actions}
        return self._softmax(scoped_logits)

    def _softmax(self, logits: Dict[str, float]) -> Dict[str, float]:
        max_logit = max(logits.values())
        exp_values = {name: math.exp(value - max_logit) for name, value in logits.items()}
        total = sum(exp_values.values())
        return {name: value / total for name, value in exp_values.items()}

    def _state_key(self, state: Dict[str, int]) -> str:
        return f"pos={state['position']}|ch={state['last_channels']}"


class RandomController:
    """Uniform random baseline with the same controller interface."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def select(self, state: Dict[str, int], actions: Sequence[Action]) -> Tuple[Action, Decision]:
        action = self.rng.choice(list(actions))
        state_key = f"pos={state['position']}|ch={state['last_channels']}"
        return action, Decision(state_key=state_key, action_name=action.name)

    def update(self, decisions: Sequence[Decision], reward: float) -> None:
        return None


class ExhaustiveController:
    """Enumerates every four-stage shortcut configuration exactly once."""

    num_architectures = 16

    def __init__(self) -> None:
        self.architecture_index = 0

    def select(self, state: Dict[str, int], actions: Sequence[Action]) -> Tuple[Action, Decision]:
        if self.architecture_index >= self.num_architectures:
            raise ValueError("all shortcut architectures have already been evaluated")
        shortcut = (self.architecture_index >> state["position"]) & 1
        action_name = f"s{shortcut}"
        action = next(action for action in actions if action.name == action_name)
        state_key = f"pos={state['position']}|ch={state['last_channels']}"
        return action, Decision(state_key=state_key, action_name=action.name)

    def update(self, decisions: Sequence[Decision], reward: float) -> None:
        self.architecture_index += 1
