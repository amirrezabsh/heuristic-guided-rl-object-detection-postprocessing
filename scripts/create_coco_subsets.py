#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic, non-overlapping COCO search splits."
    )
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/coco_subsets"),
    )
    return parser.parse_args()


def sample_images(directory: Path, count: int, rng: random.Random) -> list[Path]:
    images = sorted(directory.glob("*.jpg"))
    if count < 1:
        raise ValueError("split sizes must be positive")
    if count > len(images):
        raise ValueError(f"requested {count} images from {directory}, found {len(images)}")
    return rng.sample(images, count)


def write_paths(path: Path, images: list[Path]) -> None:
    path.write_text(
        "".join(f"{image.resolve()}\n" for image in sorted(images)),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    train_dir = repo_root / "datasets/coco/images/train2017"
    val_dir = repo_root / "datasets/coco/images/val2017"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            "COCO images are missing. Expected datasets/coco/images/train2017 "
            "and datasets/coco/images/val2017."
        )

    rng = random.Random(args.seed)
    search_train = sample_images(train_dir, args.train_size, rng)
    held_out = sample_images(val_dir, args.val_size + args.test_size, rng)
    search_val = held_out[: args.val_size]
    final_test = held_out[args.val_size :]

    out_dir = (repo_root / args.out).resolve() if not args.out.is_absolute() else args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    train_list = out_dir / "search_train.txt"
    val_list = out_dir / "search_val.txt"
    test_list = out_dir / "final_test.txt"
    write_paths(train_list, search_train)
    write_paths(val_list, search_val)
    write_paths(test_list, final_test)

    yaml_path = out_dir / "search_data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {repo_root}",
                f"train: {train_list}",
                f"val: {val_list}",
                f"test: {test_list}",
                "nc: 80",
                "",
            ]
        ),
        encoding="utf-8",
    )
    metadata = {
        "seed": args.seed,
        "train_size": len(search_train),
        "val_size": len(search_val),
        "test_size": len(final_test),
        "train_source": str(train_dir),
        "val_test_source": str(val_dir),
        "data_yaml": str(yaml_path),
    }
    (out_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"data_yaml={yaml_path}")
    print(f"train_images={len(search_train)}")
    print(f"val_images={len(search_val)}")
    print(f"test_images={len(final_test)}")


if __name__ == "__main__":
    main()
