#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATA="${DATA:-datasets/coco_subsets/search_data.yaml}"
MODEL="${MODEL:-yolov8n.pt}"
DEVICE="${DEVICE:-mps}"
EPISODES="${EPISODES:-8}"
POLICY_EPOCHS="${POLICY_EPOCHS:-6}"
EVAL_EPOCHS="${EVAL_EPOCHS:-6}"
IMGSZ="${IMGSZ:-416}"
BATCH="${BATCH:-8}"
TRAIN_SEED="${TRAIN_SEED:-0}"
EVAL_SEEDS="${EVAL_SEEDS:-101}"
OUT_ROOT="${OUT_ROOT:-runs/lr_policy_development}"
RESUME="${RESUME:-0}"
STAGE_MODE="${STAGE_MODE:-semantic3}"
LR_SPACE="${LR_SPACE:-original}"

resume_args=()
if [ "$RESUME" = "1" ]; then
  resume_args+=(--resume)
fi

policy_train_args=(
  --model "$MODEL"
  --data "$DATA"
  --episodes "$EPISODES"
  --epochs-per-episode "$POLICY_EPOCHS"
  --imgsz "$IMGSZ"
  --batch "$BATCH"
  --device "$DEVICE"
  --stage-mode "$STAGE_MODE"
  --lr-space "$LR_SPACE"
  --seed "$TRAIN_SEED"
  --out "$OUT_ROOT/policy_training"
)
if [ "$RESUME" = "1" ]; then
  policy_train_args+=(--resume)
fi

"$PYTHON" scripts/train_multiepisode_policy.py "${policy_train_args[@]}"

for seed in $EVAL_SEEDS; do
  for strategy in uniform manual random; do
    "$PYTHON" scripts/train_layerwise_rl.py \
      --strategy "$strategy" \
      --model "$MODEL" \
      --data "$DATA" \
      --epochs "$EVAL_EPOCHS" \
      --imgsz "$IMGSZ" \
      --batch "$BATCH" \
      --device "$DEVICE" \
      --stage-mode "$STAGE_MODE" \
      --lr-space "$LR_SPACE" \
      --seed "$seed" \
      --test-split val \
      --project "$OUT_ROOT/evaluation" \
      --name "${strategy}_seed${seed}"
  done

  "$PYTHON" scripts/train_layerwise_rl.py \
    --strategy rl \
    --model "$MODEL" \
    --data "$DATA" \
    --epochs "$EVAL_EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --stage-mode "$STAGE_MODE" \
    --lr-space "$LR_SPACE" \
    --seed "$seed" \
    --test-split val \
    --policy-in "$OUT_ROOT/policy_training/policy.pt" \
    --freeze-policy \
    --project "$OUT_ROOT/evaluation" \
    --name "frozen_rl_seed${seed}"
done

"$PYTHON" scripts/summarize_lr_finetuning.py --root "$OUT_ROOT/evaluation"
