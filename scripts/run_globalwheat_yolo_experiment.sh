#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATA="${DATA:-GlobalWheat2020.yaml}"
MODEL="${MODEL:-yolov8n.pt}"
DEVICE="${DEVICE:-mps}"
EPISODES="${EPISODES:-6}"
POLICY_EPOCHS="${POLICY_EPOCHS:-4}"
EVAL_EPOCHS="${EVAL_EPOCHS:-6}"
IMGSZ="${IMGSZ:-512}"
BATCH="${BATCH:-8}"
MOSAIC="${MOSAIC:-1.0}"
TRAIN_SEED="${TRAIN_SEED:-0}"
EVAL_SEEDS="${EVAL_SEEDS:-101}"
OUT_ROOT="${OUT_ROOT:-runs/globalwheat_yolo_lr_policy}"
RESUME="${RESUME:-0}"
STAGE_MODE="${STAGE_MODE:-semantic3}"
LR_SPACE="${LR_SPACE:-original}"
TEST_SPLIT="${TEST_SPLIT:-val}"
SKIP_VAL_LOSS_ERRORS="${SKIP_VAL_LOSS_ERRORS:-1}"

skip_val_loss_args=()
if [ "$SKIP_VAL_LOSS_ERRORS" = "1" ]; then
  skip_val_loss_args+=(--skip-val-loss-errors)
fi

policy_train_args=(
  --model "$MODEL"
  --data "$DATA"
  --episodes "$EPISODES"
  --epochs-per-episode "$POLICY_EPOCHS"
  --imgsz "$IMGSZ"
  --batch "$BATCH"
  --mosaic "$MOSAIC"
  --device "$DEVICE"
  --stage-mode "$STAGE_MODE"
  --lr-space "$LR_SPACE"
  --seed "$TRAIN_SEED"
  --out "$OUT_ROOT/policy_training"
  "${skip_val_loss_args[@]}"
)
if [ "$RESUME" = "1" ]; then
  policy_train_args+=(--resume)
fi

"$PYTHON" scripts/train_multiepisode_policy.py "${policy_train_args[@]}"

for seed in $EVAL_SEEDS; do
  for strategy in standard uniform manual random; do
    "$PYTHON" scripts/train_layerwise_rl.py \
      --strategy "$strategy" \
      --model "$MODEL" \
      --data "$DATA" \
      --epochs "$EVAL_EPOCHS" \
      --imgsz "$IMGSZ" \
      --batch "$BATCH" \
      --mosaic "$MOSAIC" \
      --device "$DEVICE" \
      --stage-mode "$STAGE_MODE" \
      --lr-space "$LR_SPACE" \
      --seed "$seed" \
      --test-split "$TEST_SPLIT" \
      --project "$OUT_ROOT/evaluation" \
      --name "${strategy}_seed${seed}" \
      "${skip_val_loss_args[@]}"
  done

  "$PYTHON" scripts/train_layerwise_rl.py \
    --strategy rl \
    --model "$MODEL" \
    --data "$DATA" \
    --epochs "$EVAL_EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --mosaic "$MOSAIC" \
    --device "$DEVICE" \
    --stage-mode "$STAGE_MODE" \
    --lr-space "$LR_SPACE" \
    --seed "$seed" \
    --test-split "$TEST_SPLIT" \
    --policy-in "$OUT_ROOT/policy_training/policy.pt" \
    --freeze-policy \
    --project "$OUT_ROOT/evaluation" \
    --name "frozen_rl_seed${seed}" \
    "${skip_val_loss_args[@]}"
done

"$PYTHON" scripts/summarize_lr_finetuning.py --root "$OUT_ROOT/evaluation"
