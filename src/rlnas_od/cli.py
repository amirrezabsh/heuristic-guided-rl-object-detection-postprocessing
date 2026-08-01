from __future__ import annotations

import argparse
from pathlib import Path

from .controller import ExhaustiveController, RandomController, TabularPolicyController
from .evaluator import ProxyObjectDetectionEvaluator, UltralyticsYoloEvaluator
from .search import ArchitectureSearchRunner
from .yolo_adapter import YoloArchitectureAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-NAS for object detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="run architecture search")
    search_parser.add_argument("--episodes", type=int, default=25)
    search_parser.add_argument("--max-blocks", type=int, default=4)
    search_parser.add_argument("--seed", type=int, default=0)
    search_parser.add_argument("--out", type=Path, default=Path("runs/search"))
    search_parser.add_argument(
        "--strategy",
        choices=("rl", "random", "exhaustive"),
        default="rl",
        help="candidate sampling strategy",
    )
    search_parser.add_argument(
        "--evaluator",
        choices=("proxy", "ultralytics"),
        default="proxy",
        help="evaluation backend for sampled architectures",
    )
    search_parser.add_argument(
        "--data",
        default="datasets/coco8_local.yaml",
        help="Ultralytics dataset YAML",
    )
    search_parser.add_argument("--epochs", type=int, default=3)
    search_parser.add_argument("--imgsz", type=int, default=320)
    search_parser.add_argument("--batch", type=int, default=8)
    search_parser.add_argument("--device", default="cpu")
    search_parser.add_argument("--workers", type=int, default=0)
    search_parser.add_argument(
        "--val-conf",
        type=float,
        default=0.25,
        help="validation confidence threshold; raise it if NMS is too slow",
    )
    search_parser.add_argument(
        "--val-iou",
        type=float,
        default=0.7,
        help="validation NMS IoU threshold",
    )
    search_parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="maximum detections per image during validation",
    )
    search_parser.add_argument(
        "--keep-training-runs",
        action="store_true",
        help="keep per-candidate Ultralytics training folders instead of cleaning them",
    )
    search_parser.add_argument(
        "--pretrained-weights",
        default=None,
        help="optional YOLO weights to transfer into each sampled architecture",
    )
    search_parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="number of initial layers to freeze during Ultralytics training",
    )
    search_parser.add_argument(
        "--flops-penalty",
        type=float,
        default=0.003,
        help="reward penalty per GFLOP",
    )
    search_parser.add_argument(
        "--params-penalty",
        type=float,
        default=0.004,
        help="reward penalty per million parameters",
    )
    search_parser.add_argument(
        "--latency-penalty",
        type=float,
        default=0.001,
        help="reward penalty per millisecond of validation latency",
    )

    args = parser.parse_args()
    if args.command == "search":
        adapter = YoloArchitectureAdapter(max_blocks=args.max_blocks)
        if args.strategy == "rl":
            controller = TabularPolicyController(
                action_names=list(adapter.search_space.action_names()),
                seed=args.seed,
            )
        elif args.strategy == "random":
            controller = RandomController(seed=args.seed)
        else:
            controller = ExhaustiveController()
        if args.evaluator == "proxy":
            evaluator = ProxyObjectDetectionEvaluator(
                flops_penalty=args.flops_penalty,
                params_penalty=args.params_penalty,
                latency_penalty=args.latency_penalty,
            )
        else:
            evaluator = UltralyticsYoloEvaluator(
                adapter=adapter,
                data=args.data,
                work_dir=args.out / "eval",
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                val_conf=args.val_conf,
                val_iou=args.val_iou,
                max_det=args.max_det,
                keep_training_runs=args.keep_training_runs,
                pretrained_weights=args.pretrained_weights,
                freeze=args.freeze,
                flops_penalty=args.flops_penalty,
                params_penalty=args.params_penalty,
                latency_penalty=args.latency_penalty,
            )
        runner = ArchitectureSearchRunner(
            adapter=adapter,
            controller=controller,
            evaluator=evaluator,
            out_dir=args.out,
        )
        episodes = ExhaustiveController.num_architectures if args.strategy == "exhaustive" else args.episodes
        best = runner.run(episodes)
        print(f"best_episode={best.episode}")
        print(f"strategy={args.strategy}")
        print(f"evaluated_architectures={episodes}")
        print(f"best_reward={best.evaluation.reward}")
        print(f"best_map50_95={best.evaluation.map50_95}")
        print(f"best_architecture={args.out / 'best_architecture.json'}")
        print(f"best_yolo_yaml={args.out / 'best_yolo.yaml'}")


if __name__ == "__main__":
    main()
