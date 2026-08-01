import unittest
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.rl_finetune.callbacks import split_optimizer_by_yolo_stage
from src.rl_finetune.controller import FOUR_GROUP_NAMES, LR_MULTIPLIERS, ReinforceController
from scripts.train_multiepisode_policy import read_episode_history


class Detect(nn.Linear):
    pass


class TinyYolo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Linear(4, 4),
                nn.Linear(4, 4),
                nn.Linear(4, 4),
                nn.Linear(4, 4),
                Detect(4, 2),
            ]
        )


class RLFineTuneTest(unittest.TestCase):
    def test_optimizer_is_split_into_all_semantic_stages(self):
        model = TinyYolo()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        trainer = SimpleNamespace(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=SimpleNamespace(lr0=0.01),
        )

        counts = split_optimizer_by_yolo_stage(trainer)

        self.assertTrue(all(counts[stage] > 0 for stage in ("backbone", "neck", "head")))
        self.assertEqual(
            {"backbone", "neck", "head"},
            {group["rl_stage"] for group in optimizer.param_groups},
        )
        self.assertEqual(len(optimizer.param_groups), len(scheduler.base_lrs))

    def test_optimizer_can_use_four_semantic_stages(self):
        model = TinyYolo()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        trainer = SimpleNamespace(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=SimpleNamespace(lr0=0.01),
        )

        counts = split_optimizer_by_yolo_stage(
            trainer,
            group_names=FOUR_GROUP_NAMES,
            stage_mode="semantic4",
        )

        self.assertTrue(all(counts[stage] > 0 for stage in FOUR_GROUP_NAMES))
        self.assertEqual(set(FOUR_GROUP_NAMES), {group["rl_stage"] for group in optimizer.param_groups})

    def test_reinforce_controller_returns_valid_action_and_updates(self):
        controller = ReinforceController(seed=7)
        decision = controller.select(np.zeros(4, dtype=np.float32))

        self.assertEqual(3, len(decision.multipliers))
        self.assertTrue(all(0.25 <= value <= 2.0 for value in decision.multipliers))
        stats = controller.update(0.1)
        self.assertIn("policy_loss", stats)

    def test_manual_prior_favors_discriminative_action(self):
        controller = ReinforceController(
            seed=7,
            deterministic=True,
            manual_prior_strength=2.0,
        )

        decision = controller.select(np.zeros(4, dtype=np.float32))

        self.assertEqual((0, 1, 2), decision.action_indices)
        self.assertEqual(
            (LR_MULTIPLIERS[0], LR_MULTIPLIERS[1], LR_MULTIPLIERS[2]),
            decision.multipliers,
        )

    def test_episode_history_can_be_read_for_resume(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            history_path = Path(tmp_dir) / "episodes.jsonl"
            history_path.write_text(
                json.dumps({"episode": 0}) + "\n" + json.dumps({"episode": 1}) + "\n",
                encoding="utf-8",
            )

            history = read_episode_history(history_path)

        self.assertEqual([0, 1], [record["episode"] for record in history])


if __name__ == "__main__":
    unittest.main()
