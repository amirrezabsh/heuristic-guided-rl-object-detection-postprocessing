#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=RESULT_DIRECTORY")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=RESULT_DIRECTORY")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine guarded final-test results.")
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    summaries: dict[str, dict] = {}
    for dataset, root in args.dataset:
        marker = root / "FINAL_TEST_COMPLETE.json"
        comparison = root / "comparison.csv"
        summary_path = root / "summary.json"
        for required in (marker, comparison, summary_path):
            if not required.is_file():
                raise SystemExit(f"Incomplete final result for {dataset}: missing {required}")
        with comparison.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                rows.append({"dataset": dataset, **row})
        summaries[dataset] = json.loads(summary_path.read_text(encoding="utf-8"))

    args.output.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"combined_comparison={comparison_path}")
    print(f"combined_summary={summary_path}")


if __name__ == "__main__":
    main()
