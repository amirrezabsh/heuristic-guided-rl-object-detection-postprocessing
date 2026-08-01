#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATA="${DATA:-datasets/coco_subsets/search_data.yaml}"
MODEL="${MODEL:-yolov8n.pt}"
DEVICE="${DEVICE:-mps}"
EPOCHS="${EPOCHS:-12}"
IMGSZ="${IMGSZ:-416}"
BATCH="${BATCH:-8}"
SEEDS="${SEEDS:-0 7 42}"
OUT_ROOT="${OUT_ROOT:-runs/lr_finetuning}"
LR_SPACE="${LR_SPACE:-original}"

for seed in $SEEDS; do
  for strategy in uniform manual random rl; do
    "$PYTHON" scripts/train_layerwise_rl.py \
      --strategy "$strategy" \
      --model "$MODEL" \
      --data "$DATA" \
      --epochs "$EPOCHS" \
      --imgsz "$IMGSZ" \
      --batch "$BATCH" \
      --device "$DEVICE" \
      --lr-space "$LR_SPACE" \
      --seed "$seed" \
      --test-split val \
      --project "$OUT_ROOT" \
      --name "${strategy}_seed${seed}"
  done
done

"$PYTHON" scripts/summarize_lr_finetuning.py --root "$OUT_ROOT"
