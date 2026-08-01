#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
DATA="${DATA:-$ROOT_DIR/datasets/coco_subsets/search_data.yaml}"
EPISODES="${EPISODES:-12}"
EPOCHS="${EPOCHS:-12}"
IMGSZ="${IMGSZ:-416}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-0 7 42}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/thesis_search_shortcuts}"

for seed in $SEEDS; do
  for strategy in rl random; do
    out_dir="$OUT_ROOT/${strategy}_seed${seed}"
    echo "[run] strategy=$strategy seed=$seed out=$out_dir"
    "$PYTHON" -m src.rlnas_od.cli search \
      --strategy "$strategy" \
      --evaluator ultralytics \
      --data "$DATA" \
      --episodes "$EPISODES" \
      --max-blocks 4 \
      --epochs "$EPOCHS" \
      --imgsz "$IMGSZ" \
      --batch "$BATCH" \
      --device "$DEVICE" \
      --workers 0 \
      --seed "$seed" \
      --pretrained-weights "$ROOT_DIR/yolov8n.pt" \
      --freeze 0 \
      --val-conf 0.05 \
      --val-iou 0.7 \
      --max-det 100 \
      --flops-penalty 0 \
      --params-penalty 0 \
      --latency-penalty 0 \
      --out "$out_dir"
  done
done

echo "[run] strategy=exhaustive out=$OUT_ROOT/exhaustive_seed0"
"$PYTHON" -m src.rlnas_od.cli search \
  --strategy exhaustive \
  --evaluator ultralytics \
  --data "$DATA" \
  --max-blocks 4 \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers 0 \
  --seed 0 \
  --pretrained-weights "$ROOT_DIR/yolov8n.pt" \
  --freeze 0 \
  --val-conf 0.05 \
  --val-iou 0.7 \
  --max-det 100 \
  --flops-penalty 0 \
  --params-penalty 0 \
  --latency-penalty 0 \
  --out "$OUT_ROOT/exhaustive_seed0"

"$PYTHON" "$ROOT_DIR/scripts/summarize_searches.py" "$OUT_ROOT"
