import json
import tempfile
import unittest
from pathlib import Path

from src.rlnas_od.architecture import Architecture, LayerSpec
from src.rlnas_od.controller import (
    ExhaustiveController,
    RandomController,
    TabularPolicyController,
)
from src.rlnas_od.evaluator import ProxyObjectDetectionEvaluator
from src.rlnas_od.search import ArchitectureSearchRunner
from src.rlnas_od.yolo_adapter import YoloArchitectureAdapter


class SearchRunnerTest(unittest.TestCase):
    def test_search_writes_best_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            adapter = YoloArchitectureAdapter(max_blocks=4)
            controller = TabularPolicyController(
                action_names=list(adapter.search_space.action_names()),
                seed=3,
            )
            runner = ArchitectureSearchRunner(
                adapter=adapter,
                controller=controller,
                evaluator=ProxyObjectDetectionEvaluator(),
                out_dir=out_dir,
            )

            best = runner.run(episodes=5)

            self.assertTrue((out_dir / "history.jsonl").exists())
            self.assertTrue((out_dir / "best_architecture.json").exists())
            self.assertTrue((out_dir / "best_yolo.yaml").exists())
            self.assertGreater(best.evaluation.reward, 0)

            payload = json.loads((out_dir / "best_architecture.json").read_text())
            self.assertEqual("yolo", payload["architecture"]["family"])

    def test_yolo_export_uses_multiscale_detect_inputs(self):
        adapter = YoloArchitectureAdapter(max_blocks=4)
        architecture = adapter.empty_architecture()
        state = adapter.search_space.initial_state()
        action = adapter.search_space.valid_actions(state)[0]
        adapter.apply_action(architecture, action)

        yolo_dict = adapter.to_yolo_dict(architecture)

        self.assertEqual([[15, 18, 21], 1, "Detect", ["nc"]], yolo_dict["head"][-1])

    def test_yolo_export_preserves_yolov8_fpn_head(self):
        adapter = YoloArchitectureAdapter(max_blocks=4)
        architecture = adapter.empty_architecture()
        state = adapter.search_space.initial_state()
        action = adapter.search_space.valid_actions(state)[0]
        adapter.apply_action(architecture, action)

        yolo_dict = adapter.to_yolo_dict(architecture)

        self.assertNotIn("neck", yolo_dict)
        self.assertEqual("nn.Upsample", yolo_dict["head"][0][2])
        self.assertEqual("Concat", yolo_dict["head"][1][2])
        self.assertEqual("C2f", yolo_dict["head"][2][2])
        self.assertEqual("Detect", yolo_dict["head"][-1][2])
        self.assertIsInstance(yolo_dict["head"][-1][0], list)

    def test_yolo_export_uses_four_sampled_neck_stages(self):
        adapter = YoloArchitectureAdapter(max_blocks=4)
        architecture = Architecture(
            family="yolo",
            layers=[
                LayerSpec(
                    "C2f",
                    channels=100,
                    repeats=1,
                    shortcut=True,
                    metadata={"width_mult": 1.0},
                ),
                LayerSpec(
                    "C2f",
                    channels=100,
                    repeats=1,
                    shortcut=True,
                    metadata={"width_mult": 1.0},
                ),
                LayerSpec(
                    "C2f",
                    channels=100,
                    repeats=1,
                    shortcut=False,
                    metadata={"width_mult": 1.0},
                ),
                LayerSpec(
                    "C2f",
                    channels=100,
                    repeats=1,
                    shortcut=False,
                    metadata={"width_mult": 1.0},
                ),
            ],
            stopped=True,
        )

        yolo_dict = adapter.to_yolo_dict(architecture)

        self.assertEqual([512, True], yolo_dict["head"][2][3])
        self.assertEqual(3, yolo_dict["head"][2][1])
        self.assertEqual([256, True], yolo_dict["head"][5][3])
        self.assertEqual(3, yolo_dict["head"][5][1])
        self.assertEqual([512, False], yolo_dict["head"][8][3])
        self.assertEqual(3, yolo_dict["head"][8][1])
        self.assertEqual([1024, False], yolo_dict["head"][11][3])
        self.assertEqual(3, yolo_dict["head"][11][1])

    def test_random_search_uses_same_runner_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = YoloArchitectureAdapter(max_blocks=4)
            runner = ArchitectureSearchRunner(
                adapter=adapter,
                controller=RandomController(seed=7),
                evaluator=ProxyObjectDetectionEvaluator(),
                out_dir=Path(tmp_dir),
            )

            runner.run(episodes=4)

            history = (Path(tmp_dir) / "history.jsonl").read_text(encoding="utf-8")
            self.assertEqual(4, len(history.splitlines()))

    def test_exhaustive_search_evaluates_all_shortcut_patterns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = YoloArchitectureAdapter(max_blocks=4)
            runner = ArchitectureSearchRunner(
                adapter=adapter,
                controller=ExhaustiveController(),
                evaluator=ProxyObjectDetectionEvaluator(),
                out_dir=Path(tmp_dir),
            )

            runner.run(episodes=16)

            rows = [
                json.loads(line)
                for line in (Path(tmp_dir) / "history.jsonl").read_text().splitlines()
            ]
            patterns = {
                tuple(layer["shortcut"] for layer in row["architecture"]["layers"])
                for row in rows
            }
            self.assertEqual(16, len(rows))
            self.assertEqual(16, len(patterns))


if __name__ == "__main__":
    unittest.main()
