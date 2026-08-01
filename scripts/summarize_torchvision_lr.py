#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TorchVision detector LR-control runs.")
    parser.add_argument("--root", type=Path, default=Path("runs/torchvision_lr/evaluation"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("*/final_metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "strategy": result["strategy"],
                "model": result["model"],
                "seed": result["seed"],
                "map50_95": result["map50_95"],
                "map50": result["map50"],
                "precision": result["precision"],
                "recall": result["recall"],
                "path": str(path),
            }
        )
    if not rows:
        raise SystemExit(f"No final_metrics.json files found under {args.root}")

    output_path = args.root / "comparison.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(float(row["map50_95"]))
    for strategy, values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        print(f"{strategy}: n={len(values)} mean_map50_95={mean:.6f} std={variance ** 0.5:.6f}")
    print(f"comparison={output_path}")


if __name__ == "__main__":
    main()
