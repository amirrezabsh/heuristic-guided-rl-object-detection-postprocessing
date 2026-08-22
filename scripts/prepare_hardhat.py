#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_TO_ID = {
    "helmet": 0,
    "head": 1,
    "person": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert extracted Hard Hat Workers Pascal VOC XML annotations to YOLO format."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Extracted dataset root containing images/ and annotations/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for split lists and YAML files. Defaults to <root>/subsets.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def text(node: ET.Element | None, default: str = "") -> str:
    return node.text.strip() if node is not None and node.text is not None else default


def to_yolo_line(obj: ET.Element, width: float, height: float) -> str | None:
    class_name = text(obj.find("name")).lower()
    if class_name not in CLASS_TO_ID:
        return None

    box = obj.find("bndbox")
    if box is None or width <= 0 or height <= 0:
        return None

    x1 = max(0.0, min(width, float(text(box.find("xmin"), "0"))))
    y1 = max(0.0, min(height, float(text(box.find("ymin"), "0"))))
    x2 = max(0.0, min(width, float(text(box.find("xmax"), "0"))))
    y2 = max(0.0, min(height, float(text(box.find("ymax"), "0"))))
    if x2 <= x1 or y2 <= y1:
        return None

    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return (
        f"{CLASS_TO_ID[class_name]} "
        f"{x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
    )


def convert_annotations(root: Path) -> list[Path]:
    annotations_dir = root / "annotations"
    images_dir = root / "images"
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        annotation = tree.getroot()
        filename = text(annotation.find("filename"))
        image_path = images_dir / filename
        if not image_path.is_file():
            continue

        size = annotation.find("size")
        if size is None:
            continue
        width = float(text(size.find("width"), "0"))
        height = float(text(size.find("height"), "0"))

        lines = [
            line
            for line in (
                to_yolo_line(obj, width, height)
                for obj in annotation.findall("object")
            )
            if line is not None
        ]
        (labels_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        image_paths.append(image_path.resolve())

    return image_paths


def split_images(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[Path]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("split ratios must sum to a positive value")
    shuffled = list(images)
    rng.shuffle(shuffled)

    train_count = int(len(shuffled) * train_ratio / ratio_sum)
    val_count = int(len(shuffled) * val_ratio / ratio_sum)
    return {
        "train": sorted(shuffled[:train_count]),
        "val": sorted(shuffled[train_count : train_count + val_count]),
        "test": sorted(shuffled[train_count + val_count :]),
    }


def sample(images: list[Path], limit: int, rng: random.Random) -> list[Path]:
    if limit <= 0 or limit >= len(images):
        return sorted(images)
    return sorted(rng.sample(images, limit))


def write_list(path: Path, images: list[Path]) -> None:
    path.write_text("".join(f"{image}\n" for image in images), encoding="utf-8")


def write_yaml(path: Path, root: Path, paths: dict[str, Path]) -> None:
    path.write_text(
        "\n".join(
            [
                f"path: {root}",
                f"train: {paths['train']}",
                f"val: {paths['val']}",
                f"test: {paths['test']}",
                "names:",
                "  0: helmet",
                "  1: head",
                "  2: person",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve() if args.out else root / "subsets"
    out.mkdir(parents=True, exist_ok=True)

    if not (root / "images").is_dir() or not (root / "annotations").is_dir():
        raise FileNotFoundError(
            f"{root} must contain both images/ and annotations/ directories."
        )

    rng = random.Random(args.seed)
    images = convert_annotations(root)
    full = split_images(
        images,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        rng,
    )
    pilot = {
        "train": sample(full["train"], args.train_limit, rng),
        "val": sample(full["val"], args.val_limit, rng),
        "test": sample(full["test"], args.test_limit, rng),
    }

    full_paths = {}
    pilot_paths = {}
    for split in ("train", "val", "test"):
        full_paths[split] = out / f"full_{split}.txt"
        pilot_paths[split] = out / f"pilot_{split}.txt"
        write_list(full_paths[split], full[split])
        write_list(pilot_paths[split], pilot[split])

    full_yaml = out / "full.yaml"
    pilot_yaml = out / "pilot.yaml"
    write_yaml(full_yaml, root, full_paths)
    write_yaml(pilot_yaml, root, pilot_paths)

    metadata = {
        "seed": args.seed,
        "root": str(root),
        "classes": CLASS_TO_ID,
        "full_counts": {split: len(full[split]) for split in full},
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
