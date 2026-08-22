#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
IMGSZ="${IMGSZ:-416}"
SKU_BATCH="${SKU_BATCH:-1}"
HARDHAT_BATCH="${HARDHAT_BATCH:-4}"
WORKERS="${WORKERS:-0}"
SEEDS="${SEEDS:-0 1 2}"

TRAIN_LIMIT="${TRAIN_LIMIT:-1000}"
VAL_LIMIT="${VAL_LIMIT:-500}"
POLICY_EPOCHS="${POLICY_EPOCHS:-40}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-16}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.25}"
DETECTOR_EPOCHS_PER_PROCESS="${DETECTOR_EPOCHS_PER_PROCESS:-1}"
MAX_DETECTOR_RETRIES="${MAX_DETECTOR_RETRIES:-3}"

SKU_DATA="${SKU_DATA:-/Users/amir/Documents/MSc/Thesis/datasets/SKU110K_fixed/subsets/full.yaml}"
HARDHAT_DATA="${HARDHAT_DATA:-/Users/amir/Documents/MSc/Thesis/datasets/hardhat_workers/subsets/full.yaml}"

SKU_FT_EPOCHS="${SKU_FT_EPOCHS:-20}"
HARDHAT_FT_EPOCHS="${HARDHAT_FT_EPOCHS:-20}"

usage() {
  echo "Usage: $0 train | final-test | status"
  echo "  train       Train detectors and policies using train/val only."
  echo "  final-test  Evaluate frozen artifacts once on the complete test splits."
  echo "  status      Show which frozen artifacts and completion markers exist."
}

detector_batch() {
  local dataset="$1"
  if [ "$dataset" = "sku110k" ]; then
    echo "$SKU_BATCH"
  else
    echo "$HARDHAT_BATCH"
  fi
}

detector_weights() {
  local dataset="$1"
  local epochs="$2"
  echo "$(detector_run_dir "$dataset" "$epochs")/weights/best.pt"
}

detector_run_dir() {
  local dataset="$1"
  local epochs="$2"
  local batch
  batch="$(detector_batch "$dataset")"
  echo "runs/final_${dataset}_yolo_supervised/yolov8n_e${epochs}_b${batch}_chunk${DETECTOR_EPOCHS_PER_PROCESS}_clean"
}

policy_root() {
  local dataset="$1"
  local batch
  batch="$(detector_batch "$dataset")"
  echo "runs/final_${dataset}_postprocess_rl_b${batch}_clean"
}

policy_dir() {
  local dataset="$1"
  local seed="$2"
  echo "$(policy_root "$dataset")/seed${seed}_e${POLICY_EPOCHS}_n${TRAIN_LIMIT}_${VAL_LIMIT}"
}

train_detector() {
  local dataset="$1"
  local data="$2"
  local epochs="$3"
  local batch="$4"
  local project="runs/final_${dataset}_yolo_supervised"
  local run_dir
  run_dir="$(detector_run_dir "$dataset" "$epochs")"
  local name
  name="$(basename "$run_dir")"
  local weights
  weights="$(detector_weights "$dataset" "$epochs")"
  local last="$run_dir/weights/last.pt"
  local marker="$run_dir/DETECTOR_TRAINING_COMPLETE.json"

  if [ -f "$weights" ] && [ -f "$marker" ]; then
    echo "[skip] detector training is complete: $weights"
    return
  fi

  local failures=0
  while [ ! -f "$marker" ]; do
    local resume_args=()
    if [ -f "$last" ]; then
      echo "[resume] next detector epoch from: $last"
      resume_args+=(--resume "$last")
    fi

    local code=0
    if "$PYTHON" scripts/train_yolo_detector.py \
      --model yolov8n.pt \
      --data "$data" \
      --epochs "$epochs" \
      --imgsz "$IMGSZ" \
      --batch "$batch" \
      --device "$DEVICE" \
      --workers "$WORKERS" \
      --optimizer AdamW \
      --lr0 0.001 \
      --mosaic 0.0 \
      --seed 0 \
      --project "$project" \
      --name "$name" \
      --mps-memory-fraction 0.75 \
      --empty-cache-interval 50 \
      --epochs-per-process "$DETECTOR_EPOCHS_PER_PROCESS" \
      ${resume_args[@]+"${resume_args[@]}"}; then
      code=0
    else
      code=$?
    fi

    if [ "$code" -eq 0 ]; then
      failures=0
    elif [ "$code" -eq 75 ]; then
      failures=0
      echo "[restart] saved epoch complete; releasing process memory before resuming"
    else
      failures=$((failures + 1))
      if [ "$failures" -ge "$MAX_DETECTOR_RETRIES" ]; then
        echo "Detector failed $failures consecutive times (last exit=$code)." >&2
        exit "$code"
      fi
      echo "[retry $failures/$MAX_DETECTOR_RETRIES] detector exited $code; resuming last saved epoch"
    fi
  done

  if [ ! -f "$weights" ] || [ ! -f "$marker" ]; then
    echo "Detector training did not produce all completion artifacts: $run_dir" >&2
    exit 1
  fi
}

train_policies() {
  local dataset="$1"
  local data="$2"
  local model="$3"
  local batch="$4"
  local root
  root="$(policy_root "$dataset")"

  for seed in $SEEDS; do
    local run_dir
    run_dir="$(policy_dir "$dataset" "$seed")"
    if [ -f "$run_dir/continuous_policy.pt" ] && [ -f "$run_dir/metadata.json" ]; then
      echo "[skip] frozen policy exists: $run_dir/continuous_policy.pt"
      continue
    fi

    "$PYTHON" scripts/train_continuous_postprocess_rl.py \
      --model "$model" \
      --data "$data" \
      --train-split train \
      --eval-split val \
      --train-limit "$TRAIN_LIMIT" \
      --eval-limit "$VAL_LIMIT" \
      --imgsz "$IMGSZ" \
      --batch "$batch" \
      --device "$DEVICE" \
      --policy-epochs "$POLICY_EPOCHS" \
      --rl-batch-size "$RL_BATCH_SIZE" \
      --action-mode residual_combined \
      --action-space base \
      --residual-scale "$RESIDUAL_SCALE" \
      --seed "$seed" \
      --project "$root" \
      --name "$(basename "$run_dir")"
  done

  "$PYTHON" scripts/summarize_postprocess_rl.py \
    --root "$root" \
    --min-eval-images "$VAL_LIMIT"
}

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "Missing required frozen artifact: $path" >&2
    return 1
  fi
}

preflight_final_test() {
  local missing=0
  require_file "$SKU_DATA" || missing=1
  require_file "$HARDHAT_DATA" || missing=1
  require_file "$(detector_weights sku110k "$SKU_FT_EPOCHS")" || missing=1
  require_file "$(detector_weights hardhat "$HARDHAT_FT_EPOCHS")" || missing=1
  require_file "$(detector_run_dir sku110k "$SKU_FT_EPOCHS")/DETECTOR_TRAINING_COMPLETE.json" || missing=1
  require_file "$(detector_run_dir hardhat "$HARDHAT_FT_EPOCHS")/DETECTOR_TRAINING_COMPLETE.json" || missing=1
  for dataset in sku110k hardhat; do
    for seed in 0 1 2; do
      require_file "$(policy_dir "$dataset" "$seed")/continuous_policy.pt" || missing=1
      require_file "$(policy_dir "$dataset" "$seed")/metadata.json" || missing=1
    done
  done
  if [ "$missing" -ne 0 ]; then
    echo "Final-test preflight failed. Run '$0 train' to completion first." >&2
    exit 1
  fi
  if [ "$SEEDS" != "0 1 2" ]; then
    echo "Final test requires SEEDS='0 1 2'; current value is '$SEEDS'." >&2
    exit 1
  fi
  if [ "${CONFIRM_FINAL_TEST:-}" != "YES_I_HAVE_FROZEN_THE_PROTOCOL" ]; then
    echo "Refusing to open test splits without explicit confirmation." >&2
    echo "Run with CONFIRM_FINAL_TEST=YES_I_HAVE_FROZEN_THE_PROTOCOL." >&2
    exit 1
  fi
}

evaluate_dataset_once() {
  local dataset="$1"
  local data="$2"
  local epochs="$3"
  local batch="$4"
  local output="runs/final_${dataset}_test"
  local marker="$output/FINAL_TEST_COMPLETE.json"
  if [ -f "$marker" ]; then
    echo "[skip] final test already complete: $marker"
    return
  fi

  local policy_args=()
  for seed in 0 1 2; do
    policy_args+=(--policy "$(policy_dir "$dataset" "$seed")/continuous_policy.pt")
  done

  "$PYTHON" scripts/evaluate_frozen_postprocess_policies.py \
    --model "$(detector_weights "$dataset" "$epochs")" \
    --data "$data" \
    "${policy_args[@]}" \
    --split test \
    --eval-limit 0 \
    --imgsz "$IMGSZ" \
    --batch "$batch" \
    --device "$DEVICE" \
    --candidate-conf 0.001 \
    --candidate-iou 0.95 \
    --candidate-max-det 1000 \
    --output "$output" \
    --confirm-final-test YES_I_HAVE_FROZEN_THE_PROTOCOL
}

show_status() {
  local legacy_sku="runs/final_sku110k_yolo_supervised/yolov8n_e20/weights/last.pt"
  if [ -f "$legacy_sku" ]; then
    echo "[preserved-pilot] $legacy_sku"
  fi
  for dataset in sku110k hardhat; do
    local epochs="$SKU_FT_EPOCHS"
    if [ "$dataset" = "hardhat" ]; then
      epochs="$HARDHAT_FT_EPOCHS"
    fi
    local weights
    weights="$(detector_weights "$dataset" "$epochs")"
    local detector_marker
    detector_marker="$(detector_run_dir "$dataset" "$epochs")/DETECTOR_TRAINING_COMPLETE.json"
    if [ -f "$weights" ] && [ -f "$detector_marker" ]; then
      echo "[ready] $weights"
    elif [ -f "$(detector_run_dir "$dataset" "$epochs")/weights/last.pt" ]; then
      echo "[partial] $(detector_run_dir "$dataset" "$epochs")/weights/last.pt"
    else
      echo "[missing] $weights"
    fi
    for seed in 0 1 2; do
      local policy
      policy="$(policy_dir "$dataset" "$seed")/continuous_policy.pt"
      if [ -f "$policy" ]; then echo "[ready] $policy"; else echo "[missing] $policy"; fi
    done
    local marker="runs/final_${dataset}_test/FINAL_TEST_COMPLETE.json"
    if [ -f "$marker" ]; then echo "[complete] $marker"; else echo "[not-run] $marker"; fi
  done
}

mode="${1:-}"
case "$mode" in
  train)
    if [ "${ALLOW_MEMORY_PRESSURE:-0}" != "1" ]; then
      "$PYTHON" scripts/check_training_memory.py
    else
      echo "[warning] memory preflight was explicitly bypassed"
    fi
    train_detector sku110k "$SKU_DATA" "$SKU_FT_EPOCHS" "$SKU_BATCH"
    train_policies sku110k "$SKU_DATA" "$(detector_weights sku110k "$SKU_FT_EPOCHS")" "$SKU_BATCH"
    train_detector hardhat "$HARDHAT_DATA" "$HARDHAT_FT_EPOCHS" "$HARDHAT_BATCH"
    train_policies hardhat "$HARDHAT_DATA" "$(detector_weights hardhat "$HARDHAT_FT_EPOCHS")" "$HARDHAT_BATCH"
    echo "Training/validation complete. Inspect validation summaries, then freeze the protocol."
    echo "sku_validation=$(policy_root sku110k)/summary.csv"
    echo "hardhat_validation=$(policy_root hardhat)/summary.csv"
    ;;
  final-test)
    preflight_final_test
    evaluate_dataset_once sku110k "$SKU_DATA" "$SKU_FT_EPOCHS" "$SKU_BATCH"
    evaluate_dataset_once hardhat "$HARDHAT_DATA" "$HARDHAT_FT_EPOCHS" "$HARDHAT_BATCH"
    "$PYTHON" scripts/summarize_final_new_dataset_tests.py \
      --dataset sku110k=runs/final_sku110k_test \
      --dataset hardhat=runs/final_hardhat_test \
      --output runs/final_new_dataset_tests
    ;;
  status)
    show_status
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
