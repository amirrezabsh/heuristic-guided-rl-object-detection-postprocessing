#!/usr/bin/env bash
set -euo pipefail

STAGE_MODE=semantic4 \
OUT_ROOT="${OUT_ROOT:-runs/lr_policy_yolo_fourgroup_ablation}" \
EPISODES="${EPISODES:-8}" \
POLICY_EPOCHS="${POLICY_EPOCHS:-6}" \
EVAL_EPOCHS="${EVAL_EPOCHS:-6}" \
EVAL_SEEDS="${EVAL_SEEDS:-101 202 303}" \
scripts/run_lr_policy_development.sh
