#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert extracted SKU-110K fixed annotations to YOLO format."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Extracted SKU110K_fixed directory containing images/ and annotations/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for dataset YAML and split lists. Defaults to <root>/subsets.",
    )
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def yolo_line(row: dict[str, str]) -> str | None:
    width = float(row["image_width"])
    height = float(row["image_height"])
    x1 = max(0.0, min(width, float(row["x1"])))
    y1 = max(0.0, min(height, float(row["y1"])))
    x2 = max(0.0, min(width, float(row["x2"])))
    y2 = max(0.0, min(height, float(row["y2"])))
    if x2 <= x1 or y2 <= y1 or width <= 0 or height <= 0:
        return None

    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"0 {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"


def convert_split(root: Path, split: str) -> list[Path]:
    annotations_path = root / "annotations" / f"annotations_{split}.csv"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {annotations_path}")

    labels_by_image: dict[str, list[str]] = defaultdict(list)
    with annotations_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(
            handle,
            fieldnames=[
                "image_name",
                "x1",
                "y1",
                "x2",
                "y2",
                "class",
                "image_width",
                "image_height",
            ],
        )
        for row in reader:
            line = yolo_line(row)
            if line is not None:
                labels_by_image[row["image_name"]].append(line)

    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for image_name, lines in sorted(labels_by_image.items()):
        image_path = root / "images" / image_name
        if not image_path.is_file():
            continue
        (labels_dir / f"{Path(image_name).stem}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        image_paths.append(image_path.resolve())
    return image_paths


def write_list(path: Path, images: list[Path]) -> None:
    path.write_text("".join(f"{image}\n" for image in images), encoding="utf-8")


def sample(images: list[Path], limit: int, rng: random.Random) -> list[Path]:
    if limit <= 0 or limit >= len(images):
        return sorted(images)
    return sorted(rng.sample(images, limit))


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out = (args.out.expanduser().resolve() if args.out else root / "subsets")
    out.mkdir(parents=True, exist_ok=True)

    if not (root / "images").is_dir() or not (root / "annotations").is_dir():
        raise FileNotFoundError(
            f"{root} must contain both images/ and annotations/ directories."
        )

    converted = {
        split: convert_split(root, split)
        for split in ("train", "val", "test")
    }

    rng = random.Random(args.seed)
    pilot = {
        "train": sample(converted["train"], args.train_limit, rng),
        "val": sample(converted["val"], args.val_limit, rng),
        "test": sample(converted["test"], args.test_limit, rng),
    }

    full_paths = {}
    pilot_paths = {}
    for split in ("train", "val", "test"):
        full_paths[split] = out / f"full_{split}.txt"
        pilot_paths[split] = out / f"pilot_{split}.txt"
        write_list(full_paths[split], sorted(converted[split]))
        write_list(pilot_paths[split], pilot[split])

    def write_yaml(path: Path, paths: dict[str, Path]) -> None:
        path.write_text(
            "\n".join(
                [
                    f"path: {root}",
                    f"train: {paths['train']}",
                    f"val: {paths['val']}",
                    f"test: {paths['test']}",
                    "names:",
                    "  0: object",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    full_yaml = out / "full.yaml"
    pilot_yaml = out / "pilot.yaml"
    write_yaml(full_yaml, full_paths)
    write_yaml(pilot_yaml, pilot_paths)

    metadata = {
        "seed": args.seed,
        "root": str(root),
        "full_counts": {split: len(converted[split]) for split in converted},
        "pilot_counts": {split: len(pilot[split]) for split in pilot},
        "full_yaml": str(full_yaml),
        "pilot_yaml": str(pilot_yaml),
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"full_yaml={full_yaml}")
    print(f"pilot_yaml={pilot_yaml}")
    print(f"labels_dir={root / 'labels'}")
    print(json.dumps(metadata["pilot_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
