#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.train_continuous_postprocess_rl import heuristic_actions, parse_range, summarize_actions
from scripts.train_postprocess_policy import cache_name, evaluate_records, load_cached_records, write_result


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_saved_range(value: object, cast=float) -> tuple:
    if isinstance(value, str):
        return parse_range(value, cast)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (cast(value[0]), cast(value[1]))
    raise ValueError(f"Cannot parse range value: {value!r}")


def find_eval_cache(run_dir: Path, metadata: dict) -> Path:
    cache_dir = run_dir / "cache"
    eval_split = str(metadata.get("eval_split", metadata.get("eval-split", "val")))
    eval_limit = metadata.get("eval_limit", metadata.get("eval-limit", metadata.get("eval_images")))
    if eval_limit is not None:
        try:
            eval_limit = int(eval_limit)
        except (TypeError, ValueError):
            pass
    expected = cache_dir / cache_name(eval_split, eval_limit)
    if expected.exists():
        return expected
    candidates = sorted(cache_dir.glob("*_predictions.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No cached prediction files found in {cache_dir}")
    eval_images = int(metadata.get("eval_images", 0) or 0)
    for candidate in candidates:
        if f"limit{eval_images}" in candidate.name:
            return candidate
    return candidates[-1]


def update_comparison_csv(run_dir: Path, heuristic_rows: list[dict]) -> None:
    comparison_path = run_dir / "comparison.csv"
    rows: list[dict] = []
    if comparison_path.exists():
        with comparison_path.open(newline="", encoding="utf-8") as source:
            rows = [row for row in csv.DictReader(source)]
    rows = [row for row in rows if not str(row.get("strategy", "")).startswith("heuristic_")]
    for row in heuristic_rows:
        rows.append(
            {
                "strategy": row["strategy"],
                "seed": row.get("seed", ""),
                "map50_95": row["map50_95"],
                "map50": row["map50"],
                "precision": row["precision"],
                "recall": row["recall"],
                "avg_predictions_per_image": row["avg_predictions_per_image"],
                "fixed_action": row.get("fixed_action"),
                "action_summary": row.get("action_summary"),
            }
        )
    if not rows:
        return
    fieldnames = [
        "strategy",
        "seed",
        "map50_95",
        "map50",
        "precision",
        "recall",
        "avg_predictions_per_image",
        "fixed_action",
        "action_summary",
    ]
    with comparison_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill heuristic post-processing metrics from cached predictions.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-eval-images", type=int, default=0)
    args = parser.parse_args()

    for run_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = load_json(metadata_path)
        eval_images = int(metadata.get("eval_images", 0) or 0)
        if eval_images < args.min_eval_images:
            continue
        cache_path = find_eval_cache(run_dir, metadata)
        records = load_cached_records(cache_path)
        conf_range = parse_saved_range(metadata.get("conf_range", "0.005,0.30"), float)
        iou_range = parse_saved_range(metadata.get("iou_range", "0.50,0.90"), float)
        max_det_range = parse_saved_range(metadata.get("max_det_range", "50,300"), int)
        seed = metadata.get("seed", "")
        heuristic_rows: list[dict] = []
        for rule in ("density", "confidence", "combined"):
            strategy = f"heuristic_{rule}"
            actions = heuristic_actions(records, conf_range, iou_range, max_det_range, rule=rule)
            metrics = evaluate_records(records, actions)
            row = {
                "strategy": strategy,
                "seed": seed,
                **metrics,
                "fixed_action": None,
                "action_summary": summarize_actions(actions),
            }
            heuristic_rows.append(row)
            write_result(run_dir / f"{strategy}_metrics.json", row)
            print(f"{run_dir.name} {strategy}: map50_95={metrics['map50_95']:.6f}")
        update_comparison_csv(run_dir, heuristic_rows)


if __name__ == "__main__":
    main()
