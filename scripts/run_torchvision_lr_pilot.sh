#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATA="${DATA:-datasets/coco_subsets/search_data.yaml}"
MODEL="${MODEL:-fasterrcnn_mobilenet}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-2}"
TRAIN_LIMIT="${TRAIN_LIMIT:-200}"
VAL_LIMIT="${VAL_LIMIT:-100}"
SEEDS="${SEEDS:-101}"
OUT_ROOT="${OUT_ROOT:-runs/torchvision_lr_pilot}"

for seed in $SEEDS; do
  for strategy in uniform manual random; do
    "$PYTHON" scripts/train_torchvision_layerwise_rl.py \
      --strategy "$strategy" \
      --model "$MODEL" \
      --data "$DATA" \
      --epochs "$EPOCHS" \
      --batch "$BATCH" \
      --device "$DEVICE" \
      --train-limit "$TRAIN_LIMIT" \
      --val-limit "$VAL_LIMIT" \
      --seed "$seed" \
      --project "$OUT_ROOT/evaluation" \
      --name "${strategy}_seed${seed}"
  done

  "$PYTHON" scripts/train_torchvision_layerwise_rl.py \
    --strategy rl \
    --model "$MODEL" \
    --data "$DATA" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --train-limit "$TRAIN_LIMIT" \
    --val-limit "$VAL_LIMIT" \
    --seed "$seed" \
    --project "$OUT_ROOT/evaluation" \
    --name "rl_seed${seed}" \
    --policy-out "$OUT_ROOT/evaluation/rl_seed${seed}/policy.pt"
done

"$PYTHON" scripts/summarize_torchvision_lr.py --root "$OUT_ROOT/evaluation"
