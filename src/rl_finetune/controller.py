from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
from torch import nn

GROUP_NAMES = ("backbone", "neck", "head")
FOUR_GROUP_NAMES = ("early_backbone", "late_backbone", "neck", "head")
LR_MULTIPLIERS = (0.25, 0.5, 1.0, 1.5, 2.0)
BASELINE_AWARE_LR_MULTIPLIERS = (0.75, 1.0, 1.25, 1.5)
LR_SPACES = {
    "original": LR_MULTIPLIERS,
    "baseline": BASELINE_AWARE_LR_MULTIPLIERS,
}


def resolve_lr_multipliers(lr_space: str | Sequence[float] = "original") -> tuple[float, ...]:
    if isinstance(lr_space, str):
        if lr_space not in LR_SPACES:
            raise ValueError(f"Unknown LR space {lr_space!r}. Expected one of {sorted(LR_SPACES)}.")
        return LR_SPACES[lr_space]
    multipliers = tuple(float(value) for value in lr_space)
    if not multipliers:
        raise ValueError("At least one LR multiplier is required.")
    return multipliers


def neutral_action_indices(group_count: int, lr_multipliers: Sequence[float]) -> tuple[int, ...]:
    neutral_index = min(
        range(len(lr_multipliers)),
        key=lambda index: abs(float(lr_multipliers[index]) - 1.0),
    )
    return tuple(neutral_index for _ in range(group_count))


@dataclass(frozen=True)
class LRDecision:
    multipliers: tuple[float, ...]
    action_indices: tuple[int, ...]


class LRController(Protocol):
    def select(self, state: np.ndarray) -> LRDecision: ...

    def update(self, reward: float) -> dict[str, float]: ...

    def save(self, path: Path) -> None: ...


class UniformController:
    def __init__(
        self,
        group_names: Sequence[str] = GROUP_NAMES,
        lr_multipliers: str | Sequence[float] = "original",
    ) -> None:
        self.group_names = tuple(group_names)
        self.lr_multipliers = resolve_lr_multipliers(lr_multipliers)
        self.action_indices = neutral_action_indices(len(self.group_names), self.lr_multipliers)

    def select(self, state: np.ndarray) -> LRDecision:
        del state
        return LRDecision(
            tuple(1.0 for _ in self.group_names),
            self.action_indices,
        )

    def update(self, reward: float) -> dict[str, float]:
        del reward
        return {}

    def save(self, path: Path) -> None:
        del path


class ManualController:
    """Conservative discriminative fine-tuning: lower LR in earlier layers."""

    def __init__(
        self,
        group_names: Sequence[str] = GROUP_NAMES,
        manual_action: Sequence[int] = (0, 1, 2),
        lr_multipliers: str | Sequence[float] = "original",
    ) -> None:
        self.group_names = tuple(group_names)
        self.manual_action = tuple(int(index) for index in manual_action)
        self.lr_multipliers = resolve_lr_multipliers(lr_multipliers)
        if len(self.group_names) != len(self.manual_action):
            raise ValueError("Manual action length must match group count.")
        if any(index < 0 or index >= len(self.lr_multipliers) for index in self.manual_action):
            raise ValueError("Manual action index is outside the LR multiplier space.")

    def select(self, state: np.ndarray) -> LRDecision:
        del state
        return LRDecision(
            tuple(self.lr_multipliers[index] for index in self.manual_action),
            self.manual_action,
        )

    def update(self, reward: float) -> dict[str, float]:
        del reward
        return {}

    def save(self, path: Path) -> None:
        del path


class RandomController:
    def __init__(
        self,
        seed: int,
        group_names: Sequence[str] = GROUP_NAMES,
        lr_multipliers: str | Sequence[float] = "original",
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.group_names = tuple(group_names)
        self.lr_multipliers = resolve_lr_multipliers(lr_multipliers)

    def select(self, state: np.ndarray) -> LRDecision:
        del state
        indices = tuple(
            int(v) for v in self.rng.integers(0, len(self.lr_multipliers), size=len(self.group_names))
        )
        return LRDecision(
            tuple(self.lr_multipliers[index] for index in indices),
            indices,
        )

    def update(self, reward: float) -> dict[str, float]:
        del reward
        return {}

    def save(self, path: Path) -> None:
        del path


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        group_count: int = len(GROUP_NAMES),
        action_count: int = len(LR_MULTIPLIERS),
        state_dim: int = 4,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.group_count = group_count
        self.action_count = action_count
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, group_count * action_count),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).reshape(-1, self.group_count, self.action_count)


class ReinforceController:
    """Small REINFORCE policy over per-stage learning-rate multipliers."""

    def __init__(
        self,
        seed: int,
        learning_rate: float = 3e-3,
        entropy_weight: float = 0.01,
        baseline_decay: float = 0.9,
        deterministic: bool = False,
        manual_prior_strength: float = 0.0,
        group_names: Sequence[str] = GROUP_NAMES,
        manual_action: Sequence[int] = (0, 1, 2),
        lr_multipliers: str | Sequence[float] = "original",
    ) -> None:
        torch.manual_seed(seed)
        self.group_names = tuple(group_names)
        self.manual_action = tuple(int(index) for index in manual_action)
        self.lr_multipliers = resolve_lr_multipliers(lr_multipliers)
        if len(self.group_names) != len(self.manual_action):
            raise ValueError("Manual action length must match group count.")
        if any(index < 0 or index >= len(self.lr_multipliers) for index in self.manual_action):
            raise ValueError("Manual action index is outside the LR multiplier space.")
        self.policy = PolicyNetwork(
            group_count=len(self.group_names),
            action_count=len(self.lr_multipliers),
        )
        self.manual_prior_strength = manual_prior_strength
        if manual_prior_strength > 0.0:
            self._initialize_manual_prior(manual_prior_strength)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.entropy_weight = entropy_weight
        self.baseline_decay = baseline_decay
        self.deterministic = deterministic
        self.reward_baseline = 0.0
        self._last_log_prob: torch.Tensor | None = None
        self._last_entropy: torch.Tensor | None = None

    def _initialize_manual_prior(self, strength: float) -> None:
        """Bias initial actions toward the configured conservative manual action."""

        output_layer = self.policy.network[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("Policy output layer must be linear.")
        with torch.no_grad():
            output_layer.weight.zero_()
            output_layer.bias.zero_()
            for group_index, action_index in enumerate(self.manual_action):
                output_layer.bias[group_index * len(self.lr_multipliers) + action_index] = strength

    def select(self, state: np.ndarray) -> LRDecision:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        logits = self.policy(state_tensor)[0]
        distributions = [torch.distributions.Categorical(logits=group_logits) for group_logits in logits]
        if self.deterministic:
            actions = [torch.argmax(group_logits) for group_logits in logits]
        else:
            actions = [distribution.sample() for distribution in distributions]

        self._last_log_prob = torch.stack(
            [distribution.log_prob(action) for distribution, action in zip(distributions, actions)]
        ).sum()
        self._last_entropy = torch.stack([distribution.entropy() for distribution in distributions]).mean()
        indices = tuple(int(action.item()) for action in actions)
        return LRDecision(
            tuple(self.lr_multipliers[index] for index in indices),
            indices,
        )

    def update(self, reward: float) -> dict[str, float]:
        if self.deterministic or self._last_log_prob is None or self._last_entropy is None:
            return {}

        advantage = float(reward) - self.reward_baseline
        self.reward_baseline = (
            self.baseline_decay * self.reward_baseline
            + (1.0 - self.baseline_decay) * float(reward)
        )
        loss = -(self._last_log_prob * advantage + self.entropy_weight * self._last_entropy)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._last_log_prob = None
        self._last_entropy = None
        return {
            "policy_loss": float(loss.detach()),
            "advantage": advantage,
            "reward_baseline": self.reward_baseline,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "reward_baseline": self.reward_baseline,
                "manual_prior_strength": self.manual_prior_strength,
                "group_names": self.group_names,
                "manual_action": self.manual_action,
                "lr_multipliers": self.lr_multipliers,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_group_names = tuple(checkpoint.get("group_names", GROUP_NAMES))
        if checkpoint_group_names != self.group_names:
            raise ValueError(
                f"Policy group names {checkpoint_group_names} do not match controller groups {self.group_names}."
            )
        checkpoint_lr_multipliers = tuple(checkpoint.get("lr_multipliers", LR_MULTIPLIERS))
        if checkpoint_lr_multipliers != self.lr_multipliers:
            raise ValueError(
                f"Policy LR multipliers {checkpoint_lr_multipliers} do not match controller multipliers "
                f"{self.lr_multipliers}."
            )
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "optimizer_state_dict" in checkpoint and not self.deterministic:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.reward_baseline = float(checkpoint.get("reward_baseline", 0.0))
