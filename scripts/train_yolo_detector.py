#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rl_finetune.ultralytics_patches import patch_validator_to_skip_loss_shape_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector while tolerating dense-target validation-loss shape errors."
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.75)
    parser.add_argument("--empty-cache-interval", type=int, default=50)
    parser.add_argument(
        "--epochs-per-process",
        type=int,
        default=0,
        help="Stop cleanly after this many newly completed epochs; 0 runs continuously.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str((args.project / "matplotlib").resolve()))

    # Ultralytics 8.4.x can fail only while computing validation loss for very
    # dense targets (notably SKU-110K) on MPS. Detection metrics remain valid.
    patch_validator_to_skip_loss_shape_errors()

    import torch
    from ultralytics import YOLO

    if args.resume and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    if not 0.0 < args.mps_memory_fraction <= 1.0:
        raise ValueError("--mps-memory-fraction must be in (0, 1].")
    if args.empty_cache_interval < 0:
        raise ValueError("--empty-cache-interval must be non-negative.")
    if args.epochs_per_process < 0:
        raise ValueError("--epochs-per-process must be non-negative.")

    using_mps = str(args.device).lower() == "mps" and torch.backends.mps.is_available()
    if using_mps:
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)

    model_path = str(args.resume) if args.resume else args.model
    model = YOLO(model_path, task="detect")
    if args.epochs_per_process:
        completed_this_process = 0

        def disable_intermediate_final_eval(trainer) -> None:
            # Each epoch already performs validation before saving. Avoid a
            # second best.pt validation when this short-lived process exits.
            trainer.final_eval = lambda: None

        def stop_after_saved_epoch(trainer) -> None:
            nonlocal completed_this_process
            completed_this_process += 1
            if completed_this_process >= args.epochs_per_process:
                trainer.stop = True

        model.add_callback("on_pretrain_routine_end", disable_intermediate_final_eval)
        model.add_callback("on_model_save", stop_after_saved_epoch)
    if using_mps and args.empty_cache_interval:
        batches_seen = 0

        def release_mps_cache(_trainer) -> None:
            nonlocal batches_seen
            batches_seen += 1
            if batches_seen % args.empty_cache_interval == 0:
                torch.mps.empty_cache()

        model.add_callback("on_train_batch_end", release_mps_cache)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        mosaic=args.mosaic,
        seed=args.seed,
        plots=False,
        save=True,
        val=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        resume=str(args.resume) if args.resume else False,
    )

    run_dir = args.project.resolve() / args.name
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Training produced no results file: {results_path}")
    with results_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    completed_epochs = int(float(rows[-1]["epoch"])) if rows else 0
    if completed_epochs < args.epochs:
        partial_marker = run_dir / "DETECTOR_TRAINING_PARTIAL.json"
        partial_marker.write_text(
            json.dumps(
                {
                    "status": "partial",
                    "completed_epochs": completed_epochs,
                    "target_epochs": args.epochs,
                    "last_weights": str(run_dir / "weights" / "last.pt"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"detector_training_partial={partial_marker}")
        print(f"completed_epochs={completed_epochs}/{args.epochs}")
        raise SystemExit(75)

    marker = run_dir / "DETECTOR_TRAINING_COMPLETE.json"
    (run_dir / "DETECTOR_TRAINING_PARTIAL.json").unlink(missing_ok=True)
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "epochs": args.epochs,
                "completed_epochs": completed_epochs,
                "batch": args.batch,
                "device": args.device,
                "resumed_from": str(args.resume.resolve()) if args.resume else None,
                "best_weights": str(run_dir / "weights" / "best.pt"),
                "last_weights": str(run_dir / "weights" / "last.pt"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"detector_training_complete={marker}")


if __name__ == "__main__":
    main()
