from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .architecture import Architecture
from .yolo_adapter import YoloArchitectureAdapter


@dataclass(frozen=True)
class EvaluationResult:
    reward: float
    map50_95: float
    params_m: float
    flops_g: float
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "reward": self.reward,
            "map50_95": self.map50_95,
            "params_m": self.params_m,
            "flops_g": self.flops_g,
            "latency_ms": self.latency_ms,
        }


class Evaluator:
    def evaluate(self, architecture: Architecture) -> EvaluationResult:
        raise NotImplementedError


class ProxyObjectDetectionEvaluator(Evaluator):
    """Deterministic stand-in for expensive object-detection training.

    This is intentionally not a scientific result. It lets us develop the
    search machinery before wiring in YOLO training.
    """

    def __init__(
        self,
        flops_penalty: float = 0.003,
        params_penalty: float = 0.004,
        latency_penalty: float = 0.001,
    ) -> None:
        self.flops_penalty = flops_penalty
        self.params_penalty = params_penalty
        self.latency_penalty = latency_penalty

    def evaluate(self, architecture: Architecture) -> EvaluationResult:
        params_m = 2.2
        flops_g = 7.5
        latency_ms = 4.0
        quality = 0.23

        for layer in architecture.layers:
            channel_factor = layer.channels / 128.0
            params_m += 0.18 * channel_factor * layer.repeats
            flops_g += 0.55 * channel_factor * layer.repeats
            latency_ms += 0.30 * channel_factor * layer.repeats
            if layer.block_type == "C2f":
                quality += 0.030 + 0.006 * channel_factor
            elif layer.block_type == "SPPF":
                quality += 0.020
                latency_ms += 0.15
            elif layer.block_type == "Conv":
                quality += 0.015 + 0.003 * channel_factor

        quality -= max(0, len(architecture.layers) - 4) * 0.018
        quality += self._stable_noise(architecture) * 0.012
        map50_95 = max(0.05, min(0.62, quality))
        reward = (
            map50_95
            - self.flops_penalty * flops_g
            - self.params_penalty * params_m
            - self.latency_penalty * latency_ms
        )
        return EvaluationResult(
            reward=round(reward, 6),
            map50_95=round(map50_95, 6),
            params_m=round(params_m, 4),
            flops_g=round(flops_g, 4),
            latency_ms=round(latency_ms, 4),
        )

    def _stable_noise(self, architecture: Architecture) -> float:
        payload = json.dumps(architecture.to_dict(), sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:12], 16))
        return rng.uniform(-1.0, 1.0)


class UltralyticsYoloEvaluator(Evaluator):
    """Evaluator that trains and validates sampled YOLO architectures.

    This evaluator is intentionally budgeted by CLI options. For architecture
    search, start with a tiny dataset subset and low epoch count, then retrain
    the best architecture with a larger budget.
    """

    def __init__(
        self,
        adapter: YoloArchitectureAdapter,
        data: str,
        work_dir: Path,
        epochs: int = 3,
        imgsz: int = 320,
        batch: int = 8,
        device: str = "cpu",
        workers: int = 0,
        val_conf: float = 0.25,
        val_iou: float = 0.7,
        max_det: int = 100,
        keep_training_runs: bool = False,
        pretrained_weights: Optional[str] = None,
        freeze: int = 0,
        flops_penalty: float = 0.003,
        params_penalty: float = 0.004,
        latency_penalty: float = 0.001,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if not 0.0 <= val_conf <= 1.0:
            raise ValueError("val_conf must be between 0 and 1")
        if not 0.0 <= val_iou <= 1.0:
            raise ValueError("val_iou must be between 0 and 1")
        if max_det < 1:
            raise ValueError("max_det must be positive")
        if freeze < 0:
            raise ValueError("freeze cannot be negative")
        self.adapter = adapter
        self.data = data
        self.work_dir = work_dir.resolve()
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.workers = workers
        self.val_conf = val_conf
        self.val_iou = val_iou
        self.max_det = max_det
        self.keep_training_runs = keep_training_runs
        self.pretrained_weights = pretrained_weights
        self.freeze = freeze
        self.flops_penalty = flops_penalty
        self.params_penalty = params_penalty
        self.latency_penalty = latency_penalty

        self.cache_dir = self.work_dir / "cache"
        self.candidate_dir = self.work_dir / "candidates"
        self.train_dir = self.work_dir / "ultralytics"
        self.mpl_config_dir = self.work_dir / "matplotlib"
        for path in (self.cache_dir, self.candidate_dir, self.train_dir, self.mpl_config_dir):
            path.mkdir(parents=True, exist_ok=True)

    def evaluate(self, architecture: Architecture) -> EvaluationResult:
        key = self._architecture_key(architecture)
        cache_path = self.cache_dir / f"{key}.json"
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        yaml_path = self.candidate_dir / f"{key}.yaml"
        yaml_path.write_text(self.adapter.to_yolo_yaml(architecture), encoding="utf-8")

        # Ultralytics imports matplotlib at import time, so keep its cache local.
        os.environ.setdefault("MPLCONFIGDIR", str(self.mpl_config_dir))
        from ultralytics import YOLO

        model = YOLO(str(yaml_path), task="detect")
        if self.pretrained_weights:
            model.load(self.pretrained_weights)
        model.train(
            data=self.data,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            workers=self.workers,
            project=str(self.train_dir),
            name=key,
            exist_ok=True,
            verbose=False,
            plots=False,
            save=False,
            val=False,
            freeze=self.freeze,
            conf=self.val_conf,
            iou=self.val_iou,
            max_det=self.max_det,
        )
        metrics = model.val(
            data=self.data,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            workers=self.workers,
            conf=self.val_conf,
            iou=self.val_iou,
            max_det=self.max_det,
            verbose=False,
            plots=False,
        )

        map50_95 = self._extract_map50_95(metrics)
        params_m, flops_g = self._extract_model_cost(model)
        latency_ms = self._extract_latency_ms(metrics)
        reward = self._reward(map50_95, params_m, flops_g, latency_ms)
        result = EvaluationResult(
            reward=round(reward, 6),
            map50_95=round(map50_95, 6),
            params_m=round(params_m, 4),
            flops_g=round(flops_g, 4),
            latency_ms=round(latency_ms, 4),
        )
        cache_path.write_text(json.dumps(result.to_dict(), sort_keys=True), encoding="utf-8")
        if not self.keep_training_runs:
            shutil.rmtree(self.train_dir / key, ignore_errors=True)
        return result

    def _architecture_key(self, architecture: Architecture) -> str:
        payload = json.dumps(
            {
                "cache_version": 2,
                "architecture": architecture.to_dict(),
                "yolo_model": self.adapter.to_yolo_dict(architecture),
                "data": self.data,
                "epochs": self.epochs,
                "imgsz": self.imgsz,
                "batch": self.batch,
                "val_conf": self.val_conf,
                "val_iou": self.val_iou,
                "max_det": self.max_det,
                "pretrained_weights": self.pretrained_weights,
                "freeze": self.freeze,
                "flops_penalty": self.flops_penalty,
                "params_penalty": self.params_penalty,
                "latency_penalty": self.latency_penalty,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _read_cache(self, cache_path: Path) -> Optional[EvaluationResult]:
        if not cache_path.exists():
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return EvaluationResult(
            reward=float(payload["reward"]),
            map50_95=float(payload["map50_95"]),
            params_m=float(payload["params_m"]),
            flops_g=float(payload["flops_g"]),
            latency_ms=float(payload["latency_ms"]),
        )

    def _reward(
        self,
        map50_95: float,
        params_m: float,
        flops_g: float,
        latency_ms: float,
    ) -> float:
        return (
            map50_95
            - self.flops_penalty * flops_g
            - self.params_penalty * params_m
            - self.latency_penalty * latency_ms
        )

    def _extract_map50_95(self, metrics: object) -> float:
        box = getattr(metrics, "box", None)
        if box is not None:
            value = getattr(box, "map", None)
            if value is not None:
                return float(value)
        results_dict = getattr(metrics, "results_dict", None)
        if isinstance(results_dict, dict):
            for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95"):
                if key in results_dict:
                    return float(results_dict[key])
        raise ValueError("Could not extract mAP50-95 from Ultralytics metrics")

    def _extract_model_cost(self, model: object) -> tuple[float, float]:
        info = model.info(verbose=False)
        if isinstance(info, tuple):
            if len(info) >= 4:
                params = float(info[1])
                flops = float(info[3])
                return params / 1_000_000.0, flops
            if len(info) >= 2:
                params = float(info[0])
                flops = float(info[-1])
                return params / 1_000_000.0, flops

        torch_model = getattr(model, "model", None)
        if torch_model is not None:
            params = sum(parameter.numel() for parameter in torch_model.parameters())
            flops = 0.0
            try:
                from ultralytics.utils.torch_utils import get_flops

                flops = float(get_flops(torch_model, imgsz=self.imgsz))
            except Exception:
                flops = 0.0
            return params / 1_000_000.0, flops
        return 0.0, 0.0

    def _extract_latency_ms(self, metrics: object) -> float:
        speed = getattr(metrics, "speed", None)
        if isinstance(speed, dict):
            return float(sum(speed.values()))
        return 0.0
