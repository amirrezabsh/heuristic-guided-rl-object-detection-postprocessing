#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from scripts.train_continuous_postprocess_rl import (
    ContinuousPostprocessPolicy,
    deterministic_actions,
    extended_base_actions,
    heuristic_actions,
    summarize_actions,
)
from scripts.train_postprocess_policy import (
    Action,
    cache_name,
    cache_split_predictions,
    evaluate_records,
    fixed_actions,
    image_paths_from_split,
    load_data_yaml,
    split_file,
    write_result,
)

METRIC_NAMES = (
    "map50_95",
    "map50",
    "precision",
    "recall",
    "avg_predictions_per_image",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate already-frozen post-processing policies once on an untouched test split."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--policy", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=0,
        help="Maximum images; 0 evaluates the complete test split.",
    )
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--candidate-conf", type=float, default=0.001)
    parser.add_argument("--candidate-iou", type=float, default=0.95)
    parser.add_argument("--candidate-max-det", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-final-test",
        required=True,
        help="Must be exactly YES_I_HAVE_FROZEN_THE_PROTOCOL.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_for_policy(path: Path) -> int:
    metadata_path = path.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "seed" in metadata:
            return int(metadata["seed"])
    match = re.search(r"seed(\d+)", str(path))
    if not match:
        raise ValueError(f"Cannot determine seed for policy: {path}")
    return int(match.group(1))


def load_policy_checkpoint(path: Path) -> tuple[ContinuousPostprocessPolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["policy_state_dict"]
    state_dim = int(state_dict["net.0.weight"].shape[1])
    action_dim = int(state_dict["mean.weight"].shape[0])
    policy = ContinuousPostprocessPolicy(state_dim=state_dim, action_dim=action_dim)
    policy.load_state_dict(state_dict)
    policy.eval()
    protocol = {
        "conf_range": tuple(checkpoint["conf_range"]),
        "iou_range": tuple(checkpoint["iou_range"]),
        "max_det_range": tuple(checkpoint["max_det_range"]),
        "min_area_range": tuple(checkpoint.get("min_area_range", (0.0, 0.0))),
        "top_k_range": tuple(checkpoint.get("top_k_range") or checkpoint["max_det_range"]),
        "action_mode": str(checkpoint.get("action_mode", "direct")),
        "action_space": str(checkpoint.get("action_space", "base")),
        "residual_scale": float(checkpoint.get("residual_scale", 0.25)),
    }
    return policy, protocol


def json_safe_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return the exact JSON-compatible representation used in test markers."""
    return json.loads(json.dumps(protocol, sort_keys=True))


def validate_frozen_inputs(args: argparse.Namespace) -> list[tuple[int, Path]]:
    if args.confirm_final_test != "YES_I_HAVE_FROZEN_THE_PROTOCOL":
        raise SystemExit(
            "Refusing to open the test split. Pass "
            "--confirm-final-test YES_I_HAVE_FROZEN_THE_PROTOCOL after freezing all settings."
        )
    if args.eval_limit < 0:
        raise SystemExit("--eval-limit must be 0 (full split) or a positive integer.")
    required = [args.model, args.data, *args.policy]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing frozen input(s):\n- " + "\n- ".join(missing))
    seeded = sorted((seed_for_policy(path), path.resolve()) for path in args.policy)
    seeds = [seed for seed, _path in seeded]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"Duplicate policy seeds are not allowed: {seeds}")
    if seeds != [0, 1, 2]:
        raise SystemExit(f"Final protocol requires policy seeds [0, 1, 2], got {seeds}")
    for seed, path in seeded:
        metadata_path = path.parent / "metadata.json"
        if not metadata_path.is_file():
            raise SystemExit(f"Missing policy metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("seed", -1)) != seed:
            raise SystemExit(f"Policy seed disagrees with metadata: {path}")
        if str(metadata.get("eval_split")) != "val":
            raise SystemExit(f"Policy was not validated on the val split: {path}")
        metadata_model = Path(str(metadata.get("model", ""))).resolve()
        metadata_data = Path(str(metadata.get("data", ""))).resolve()
        if metadata_model != args.model.resolve():
            raise SystemExit(f"Policy was trained with a different detector: {path}")
        if metadata_data != args.data.resolve():
            raise SystemExit(f"Policy was trained with a different dataset YAML: {path}")
    return seeded


def protocols_match(protocols: list[dict[str, Any]]) -> bool:
    if not protocols:
        return False
    first = protocols[0]
    return all(protocol == first for protocol in protocols[1:])


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rl_rows = [row for row in rows if row["strategy"] == "residual_heuristic_rl"]
    summary: dict[str, Any] = {"rl_seeds": [int(row["seed"]) for row in rl_rows]}
    for metric in METRIC_NAMES:
        values = [float(row[metric]) for row in rl_rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summary[f"rl_{metric}_mean"] = mean
        summary[f"rl_{metric}_std"] = variance**0.5
    for strategy in ("standard", "heuristic_combined"):
        row = next(item for item in rows if item["strategy"] == strategy)
        summary[strategy] = {metric: row[metric] for metric in METRIC_NAMES}
    return summary


def write_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "strategy",
        "seed",
        *METRIC_NAMES,
        "policy",
        "action_summary",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["action_summary"] = json.dumps(payload.get("action_summary"), sort_keys=True)
            writer.writerow({key: payload.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    seeded_policies = validate_frozen_inputs(args)
    output = args.output.resolve()
    completion_marker = output / "FINAL_TEST_COMPLETE.json"
    if completion_marker.exists():
        raise SystemExit(
            f"Final test is already complete at {output}. Refusing to run it again."
        )
    output.mkdir(parents=True, exist_ok=True)

    loaded = []
    for seed, path in seeded_policies:
        policy, protocol = load_policy_checkpoint(path)
        loaded.append((seed, path, policy, protocol))
    protocols = [protocol for _seed, _path, _policy, protocol in loaded]
    if not protocols_match(protocols):
        raise SystemExit("Frozen policy checkpoints use different action protocols.")
    protocol = protocols[0]
    if protocol["action_mode"] != "residual_combined" or protocol["action_space"] != "base":
        raise SystemExit(
            "Final protocol requires action_mode=residual_combined and action_space=base."
        )

    data = load_data_yaml(args.data)
    split_path = split_file(data, args.split)
    image_paths = image_paths_from_split(split_path, args.eval_limit)
    if not image_paths:
        raise SystemExit(f"No images found in {args.split!r} split: {split_path}")

    manifest = {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "split": args.split,
        "split_path": str(split_path.resolve()),
        "split_sha256": sha256_file(split_path) if split_path.is_file() else None,
        "eval_limit": args.eval_limit,
        "eval_images": len(image_paths),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "candidate_conf": args.candidate_conf,
        "candidate_iou": args.candidate_iou,
        "candidate_max_det": args.candidate_max_det,
        "policies": [
            {"seed": seed, "path": str(path), "sha256": sha256_file(path)}
            for seed, path in seeded_policies
        ],
        "protocol": json_safe_protocol(protocol),
    }
    started_marker = output / "FINAL_TEST_STARTED.json"
    if started_marker.exists():
        previous = json.loads(started_marker.read_text(encoding="utf-8"))
        immutable_keys = (
            "model_sha256",
            "data_sha256",
            "split",
            "split_sha256",
            "eval_limit",
            "eval_images",
            "imgsz",
            "batch",
            "device",
            "candidate_conf",
            "candidate_iou",
            "candidate_max_det",
            "policies",
            "protocol",
        )
        changed = [key for key in immutable_keys if previous.get(key) != manifest.get(key)]
        if changed:
            raise SystemExit(
                "A partial final test exists with a different frozen protocol. "
                f"Refusing to continue; changed fields: {changed}"
            )
        manifest["started_at_utc"] = previous["started_at_utc"]
    else:
        write_result(started_marker, manifest)

    records = cache_split_predictions(
        model_path=str(args.model),
        image_paths=image_paths,
        cache_path=output / "cache" / cache_name(args.split, args.eval_limit),
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        candidate_conf=args.candidate_conf,
        candidate_iou=args.candidate_iou,
        candidate_max_det=args.candidate_max_det,
    )

    conf_range = protocol["conf_range"]
    iou_range = protocol["iou_range"]
    max_det_range = protocol["max_det_range"]
    min_area_range = protocol["min_area_range"]
    top_k_range = protocol["top_k_range"]
    base_actions = heuristic_actions(
        records, conf_range, iou_range, max_det_range, rule="combined"
    )
    if protocol["action_space"] == "extended":
        base_actions = extended_base_actions(base_actions, top_k_range)

    rows: list[dict[str, Any]] = []
    baselines = [
        (
            "standard",
            fixed_actions(records, Action(conf=0.25, iou=0.70, max_det=300, agnostic=False)),
        ),
        ("heuristic_combined", base_actions),
    ]
    for strategy, actions in baselines:
        metrics = evaluate_records(records, actions)
        row = {
            "strategy": strategy,
            "seed": "",
            **metrics,
            "policy": "",
            "action_summary": summarize_actions(actions),
        }
        rows.append(row)
        write_result(output / f"{strategy}_metrics.json", row)

    for seed, path, policy, _policy_protocol in loaded:
        actions = deterministic_actions(
            records,
            policy,
            conf_range,
            iou_range,
            max_det_range,
            min_area_range,
            top_k_range,
            base_actions=base_actions,
            residual_scale=protocol["residual_scale"],
        )
        metrics = evaluate_records(records, actions)
        row = {
            "strategy": "residual_heuristic_rl",
            "seed": seed,
            **metrics,
            "policy": str(path),
            "action_summary": summarize_actions(actions),
        }
        rows.append(row)
        write_result(output / f"residual_heuristic_rl_seed{seed}_metrics.json", row)

    write_comparison(output / "comparison.csv", rows)
    summary = aggregate_rows(rows)
    write_result(output / "summary.json", summary)
    manifest.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "comparison": str(output / "comparison.csv"),
            "summary": str(output / "summary.json"),
        }
    )
    write_result(completion_marker, manifest)
    print(f"final_test_complete={completion_marker}")
    print(f"comparison={output / 'comparison.csv'}")
    print(f"summary={output / 'summary.json'}")


if __name__ == "__main__":
    main()
