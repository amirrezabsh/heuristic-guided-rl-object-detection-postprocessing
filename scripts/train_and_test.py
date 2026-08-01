#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one selected model and evaluate it on the untouched test split."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--pretrained-weights")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/final_models"))
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str((args.project / "matplotlib").resolve()))
    from ultralytics import YOLO

    model = YOLO(args.model, task="detect")
    if args.pretrained_weights:
        model.load(args.pretrained_weights)

    project = args.project.resolve()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(project),
        name=args.name,
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
        conf=0.05,
        iou=0.7,
        max_det=100,
    )

    best_weights = project / args.name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"best weights not found: {best_weights}")

    final_model = YOLO(str(best_weights), task="detect")
    metrics = final_model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=0.001,
        iou=0.7,
        max_det=300,
        project=str(project),
        name=f"{args.name}_final_test",
        exist_ok=True,
        plots=True,
    )
    result = {
        "model": args.model,
        "pretrained_weights": args.pretrained_weights,
        "seed": args.seed,
        "best_weights": str(best_weights),
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "speed": dict(metrics.speed),
    }
    result_path = project / args.name / "final_metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"map50_95={result['map50_95']}")
    print(f"map50={result['map50']}")
    print(f"final_metrics={result_path}")


if __name__ == "__main__":
    main()
