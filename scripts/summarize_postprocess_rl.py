#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize continuous post-processing RL runs.")
    parser.add_argument("--root", type=Path, default=Path("runs/torchvision_continuous_postprocess_rl"))
    parser.add_argument("--min-eval-images", type=int, default=0)
    args = parser.parse_args()

    rows: list[dict] = []
    for run_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        metadata_path = run_dir / "metadata.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        if int(metadata.get("eval_images", 0) or 0) < args.min_eval_images:
            continue
        for metrics_path in sorted(run_dir.glob("*_metrics.json")):
            result = load_json(metrics_path)
            rows.append(
                {
                    "model": metadata.get("model", ""),
                    "run": run_dir.name,
                    "strategy": result.get("strategy", metrics_path.stem.removesuffix("_metrics")),
                    "seed": result.get("seed", metadata.get("seed", "")),
                    "map50_95": result["map50_95"],
                    "map50": result["map50"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "avg_predictions_per_image": result.get("avg_predictions_per_image", ""),
                    "train_images": metadata.get("train_images", ""),
                    "eval_images": metadata.get("eval_images", ""),
                    "path": str(metrics_path),
                }
            )

    if not rows:
        raise SystemExit(f"No *_metrics.json files found under {args.root}")

    output_path = args.root / "summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["strategy"]))].append(float(row["map50_95"]))

    for (model, strategy), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        print(
            f"{model or 'unknown'} {strategy}: "
            f"n={len(values)} mean_map50_95={mean:.6f} std={math.sqrt(variance):.6f}"
        )
    print(f"summary={output_path}")


if __name__ == "__main__":
    main()
