import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import ImageFile

from scripts.evaluate_frozen_postprocess_policies import (
    METRIC_NAMES,
    aggregate_rows,
    json_safe_protocol,
    load_policy_checkpoint,
    protocols_match,
    seed_for_policy,
)
from scripts.train_continuous_postprocess_rl import ContinuousPostprocessPolicy


class FinalEvaluationTest(unittest.TestCase):
    def test_frozen_policy_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "seed0" / "continuous_policy.pt"
            path.parent.mkdir()
            policy = ContinuousPostprocessPolicy(state_dim=15, action_dim=3)
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "conf_range": (0.005, 0.30),
                    "iou_range": (0.50, 0.90),
                    "max_det_range": (50, 400),
                    "min_area_range": (0.0, 0.02),
                    "top_k_range": (50, 1000),
                    "action_mode": "residual_combined",
                    "action_space": "base",
                    "residual_scale": 0.25,
                },
                path,
            )

            loaded, protocol = load_policy_checkpoint(path)

            self.assertEqual(3, loaded.action_dim)
            self.assertEqual("residual_combined", protocol["action_mode"])
            self.assertEqual("base", protocol["action_space"])
            self.assertEqual((50, 400), protocol["max_det_range"])
            self.assertEqual(
                [50, 400], json_safe_protocol(protocol)["max_det_range"]
            )

    def test_seed_prefers_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "not_named_after_seed" / "continuous_policy.pt"
            path.parent.mkdir()
            path.touch()
            (path.parent / "metadata.json").write_text(
                json.dumps({"seed": 2}), encoding="utf-8"
            )

            self.assertEqual(2, seed_for_policy(path))

    def test_aggregate_rows_reports_population_mean_and_std(self):
        def metrics(value):
            return {metric: value for metric in METRIC_NAMES}

        rows = [
            {"strategy": "standard", "seed": "", **metrics(0.1)},
            {"strategy": "heuristic_combined", "seed": "", **metrics(0.2)},
            {"strategy": "residual_heuristic_rl", "seed": 0, **metrics(0.2)},
            {"strategy": "residual_heuristic_rl", "seed": 1, **metrics(0.4)},
            {"strategy": "residual_heuristic_rl", "seed": 2, **metrics(0.6)},
        ]

        summary = aggregate_rows(rows)

        self.assertEqual([0, 1, 2], summary["rl_seeds"])
        self.assertAlmostEqual(0.4, summary["rl_map50_95_mean"])
        self.assertAlmostEqual((0.08 / 3) ** 0.5, summary["rl_map50_95_std"])
        self.assertEqual(0.1, summary["standard"]["map50_95"])

    def test_protocols_must_be_identical(self):
        self.assertTrue(protocols_match([{"a": 1}, {"a": 1}]))
        self.assertFalse(protocols_match([{"a": 1}, {"a": 2}]))
        self.assertFalse(protocols_match([]))

    def test_runner_keeps_training_off_test_and_final_test_full(self):
        runner = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_final_new_dataset_benchmarks.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--eval-split val", runner)
        self.assertIn("--eval-limit 0", runner)
        self.assertIn("YES_I_HAVE_FROZEN_THE_PROTOCOL", runner)
        self.assertIn('SKU_BATCH="${SKU_BATCH:-1}"', runner)
        self.assertIn("DETECTOR_TRAINING_COMPLETE.json", runner)
        self.assertIn('--resume "$last"', runner)
        self.assertIn('--epochs-per-process "$DETECTOR_EPOCHS_PER_PROCESS"', runner)
        self.assertIn('code" -eq 75', runner)
        self.assertIn("_clean", runner)
        self.assertIn("scripts/check_training_memory.py", runner)
        self.assertNotIn("detect val", runner)

    def test_prediction_cache_keeps_recoverable_truncated_images(self):
        self.assertTrue(ImageFile.LOAD_TRUNCATED_IMAGES)


if __name__ == "__main__":
    unittest.main()
