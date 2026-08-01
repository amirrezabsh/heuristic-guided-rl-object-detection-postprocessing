import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.rlnas_od.evaluator import EvaluationResult, UltralyticsYoloEvaluator
from src.rlnas_od.yolo_adapter import YoloArchitectureAdapter


class UltralyticsYoloEvaluatorTest(unittest.TestCase):
    def test_cached_result_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = YoloArchitectureAdapter(max_blocks=4)
            architecture = adapter.empty_architecture()
            state = adapter.search_space.initial_state()
            action = adapter.search_space.valid_actions(state)[0]
            adapter.apply_action(architecture, action)

            evaluator = UltralyticsYoloEvaluator(
                adapter=adapter,
                data="unused.yaml",
                work_dir=Path(tmp_dir),
                epochs=1,
            )
            key = evaluator._architecture_key(architecture)
            cache_path = Path(tmp_dir) / "cache" / f"{key}.json"
            expected = EvaluationResult(
                reward=0.25,
                map50_95=0.30,
                params_m=1.0,
                flops_g=2.0,
                latency_ms=3.0,
            )
            cache_path.write_text(json.dumps(expected.to_dict()), encoding="utf-8")

            actual = evaluator.evaluate(architecture)

            self.assertEqual(expected, actual)

    def test_reward_penalizes_cost(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            evaluator = UltralyticsYoloEvaluator(
                adapter=YoloArchitectureAdapter(max_blocks=4),
                data="unused.yaml",
                work_dir=Path(tmp_dir),
                epochs=1,
            )

            low_cost = evaluator._reward(0.5, params_m=1.0, flops_g=2.0, latency_ms=3.0)
            high_cost = evaluator._reward(0.5, params_m=10.0, flops_g=20.0, latency_ms=30.0)

            self.assertGreater(low_cost, high_cost)

    def test_reward_uses_configured_penalty_weights(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            default_evaluator = UltralyticsYoloEvaluator(
                adapter=YoloArchitectureAdapter(max_blocks=4),
                data="unused.yaml",
                work_dir=Path(tmp_dir) / "default",
                epochs=1,
            )
            quality_first_evaluator = UltralyticsYoloEvaluator(
                adapter=YoloArchitectureAdapter(max_blocks=4),
                data="unused.yaml",
                work_dir=Path(tmp_dir) / "quality_first",
                epochs=1,
                flops_penalty=0.0002,
                params_penalty=0.0002,
                latency_penalty=0.00005,
            )

            default_reward = default_evaluator._reward(
                0.1,
                params_m=3.0,
                flops_g=4.0,
                latency_ms=20.0,
            )
            quality_first_reward = quality_first_evaluator._reward(
                0.1,
                params_m=3.0,
                flops_g=4.0,
                latency_ms=20.0,
            )

            self.assertGreater(quality_first_reward, default_reward)

    def test_validation_nms_settings_are_passed_to_ultralytics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = YoloArchitectureAdapter(max_blocks=4)
            architecture = adapter.empty_architecture()
            state = adapter.search_space.initial_state()
            action = adapter.search_space.valid_actions(state)[0]
            adapter.apply_action(architecture, action)

            calls = {}

            class FakeYOLO:
                def __init__(self, model_path, task):
                    calls["init"] = {"model_path": model_path, "task": task}

                def load(self, weights_path):
                    calls["load"] = weights_path
                    return self

                def train(self, **kwargs):
                    calls["train"] = kwargs
                    (Path(kwargs["project"]) / kwargs["name"] / "weights").mkdir(parents=True)
                    (Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt").write_text(
                        "fake",
                        encoding="utf-8",
                    )

                def val(self, **kwargs):
                    calls["val"] = kwargs
                    return SimpleNamespace(
                        box=SimpleNamespace(map=0.40),
                        speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0},
                    )

                def info(self, verbose=False):
                    return (0, 1_000_000, 0, 2.0)

            previous = sys.modules.get("ultralytics")
            sys.modules["ultralytics"] = SimpleNamespace(YOLO=FakeYOLO)
            try:
                evaluator = UltralyticsYoloEvaluator(
                    adapter=adapter,
                    data="unused.yaml",
                    work_dir=Path(tmp_dir),
                    epochs=1,
                    val_conf=0.35,
                    val_iou=0.6,
                    max_det=50,
                    pretrained_weights="yolov8n.pt",
                    freeze=10,
                )

                evaluator.evaluate(architecture)
            finally:
                if previous is None:
                    sys.modules.pop("ultralytics", None)
                else:
                    sys.modules["ultralytics"] = previous

            self.assertEqual(0.35, calls["train"]["conf"])
            self.assertEqual(0.6, calls["train"]["iou"])
            self.assertEqual(50, calls["train"]["max_det"])
            self.assertFalse(calls["train"]["val"])
            self.assertEqual("yolov8n.pt", calls["load"])
            self.assertEqual(10, calls["train"]["freeze"])
            self.assertEqual(0.35, calls["val"]["conf"])
            self.assertEqual(0.6, calls["val"]["iou"])
            self.assertEqual(50, calls["val"]["max_det"])
            self.assertFalse((Path(calls["train"]["project"]) / calls["train"]["name"]).exists())


if __name__ == "__main__":
    unittest.main()
