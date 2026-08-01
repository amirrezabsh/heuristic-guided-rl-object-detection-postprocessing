#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-runs/globalwheat_yolo_baselineaware/evaluation/standard_seed101/weights/best.pt}"
DATA="${DATA:-datasets/globalwheat_subsets/pilot.yaml}"
DEVICE="${DEVICE:-mps}"
IMGSZ="${IMGSZ:-416}"
BATCH="${BATCH:-4}"
TRAIN_LIMIT="${TRAIN_LIMIT:-200}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
IMITATION_EPOCHS="${IMITATION_EPOCHS:-25}"
POLICY_EPOCHS="${POLICY_EPOCHS:-0}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-runs/globalwheat_postprocess_rl}"
NAME="${NAME:-pilot_seed${SEED}}"

"$PYTHON" scripts/train_postprocess_policy.py \
  --model "$MODEL" \
  --data "$DATA" \
  --train-split train \
  --eval-split test \
  --train-limit "$TRAIN_LIMIT" \
  --eval-limit "$EVAL_LIMIT" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --candidate-conf 0.001 \
  --candidate-iou 0.95 \
  --candidate-max-det 1000 \
  --conf-values "${CONF_VALUES:-0.03,0.05,0.10,0.20}" \
  --iou-values "${IOU_VALUES:-0.60,0.70,0.80}" \
  --max-det-values "${MAX_DET_VALUES:-100,200,300}" \
  --imitation-epochs "$IMITATION_EPOCHS" \
  --oracle-latency-weight 0.0 \
  --policy-epochs "$POLICY_EPOCHS" \
  --policy-lr "${POLICY_LR:-0.003}" \
  --entropy-weight "${ENTROPY_WEIGHT:-0.01}" \
  --latency-weight "${LATENCY_WEIGHT:-0.0}" \
  --rl-reward-mode "${RL_REWARD_MODE:-image}" \
  --rl-batch-size "${RL_BATCH_SIZE:-16}" \
  --seed "$SEED" \
  --project "$OUT_ROOT" \
  --name "$NAME"
