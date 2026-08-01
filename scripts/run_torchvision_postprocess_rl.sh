#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATA="${DATA:-datasets/coco_subsets/search_data.yaml}"
MODELS="${MODELS:-fasterrcnn_mobilenet}"
DEVICE="${DEVICE:-cpu}"
BATCH="${BATCH:-2}"
TRAIN_LIMIT="${TRAIN_LIMIT:-100}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"
POLICY_EPOCHS="${POLICY_EPOCHS:-20}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-8}"
POLICY_LR="${POLICY_LR:-0.0005}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.02}"
LATENCY_WEIGHT="${LATENCY_WEIGHT:-0.0}"
ACTION_MODE="${ACTION_MODE:-direct}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.25}"
ACTION_SPACE="${ACTION_SPACE:-base}"
MIN_AREA_RANGE="${MIN_AREA_RANGE:-0.0,0.02}"
TOP_K_RANGE="${TOP_K_RANGE:-50,300}"
FREEZE_EXTENDED_EPOCHS="${FREEZE_EXTENDED_EPOCHS:-0}"
SEEDS="${SEEDS:-0 1 2}"
OUT_ROOT="${OUT_ROOT:-runs/torchvision_continuous_postprocess_rl}"

for model in $MODELS; do
  for seed in $SEEDS; do
    "$PYTHON" scripts/train_torchvision_continuous_postprocess_rl.py \
      --model "$model" \
      --data "$DATA" \
      --train-split train \
      --eval-split val \
      --train-limit "$TRAIN_LIMIT" \
      --eval-limit "$EVAL_LIMIT" \
      --batch "$BATCH" \
      --device "$DEVICE" \
      --policy-epochs "$POLICY_EPOCHS" \
      --rl-batch-size "$RL_BATCH_SIZE" \
      --policy-lr "$POLICY_LR" \
      --entropy-weight "$ENTROPY_WEIGHT" \
      --latency-weight "$LATENCY_WEIGHT" \
      --action-mode "$ACTION_MODE" \
      --residual-scale "$RESIDUAL_SCALE" \
      --action-space "$ACTION_SPACE" \
      --min-area-range "$MIN_AREA_RANGE" \
      --top-k-range "$TOP_K_RANGE" \
      --freeze-extended-epochs "$FREEZE_EXTENDED_EPOCHS" \
      --seed "$seed" \
      --project "$OUT_ROOT" \
      --name "${model}_seed${seed}_e${POLICY_EPOCHS}_n${EVAL_LIMIT}"
  done
done

"$PYTHON" scripts/summarize_postprocess_rl.py --root "$OUT_ROOT" --min-eval-images "$EVAL_LIMIT"
