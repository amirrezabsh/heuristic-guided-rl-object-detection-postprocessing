#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPLY=0
DROP_COCO2017=0
DROP_RUNS=0

usage() {
  cat <<USAGE
Usage: scripts/cleanup_storage.sh [--apply] [--drop-coco2017] [--drop-runs]

Default mode is a dry run. It reports reclaimable storage without deleting.

Options:
  --apply          actually delete selected generated artifacts
  --drop-coco2017 delete extracted full COCO 2017 data under datasets/coco
  --drop-runs     delete experiment outputs under runs

Safe-by-default cleanup removes only temporary download archives:
  datasets/.tmp_coco2017
  datasets/.tmp_coco_labels
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --apply)
      APPLY=1
      ;;
    --drop-coco2017)
      DROP_COCO2017=1
      ;;
    --drop-runs)
      DROP_RUNS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

targets=(
  "$ROOT_DIR/datasets/.tmp_coco2017"
  "$ROOT_DIR/datasets/.tmp_coco_labels"
)

if [ "$DROP_COCO2017" = "1" ]; then
  targets+=("$ROOT_DIR/datasets/coco")
fi

if [ "$DROP_RUNS" = "1" ]; then
  targets+=("$ROOT_DIR/runs")
fi

echo "[summary] current storage"
du -sh "$ROOT_DIR/datasets" "$ROOT_DIR/runs" "$ROOT_DIR/.venv" 2>/dev/null || true

echo
echo "[targets]"
for target in "${targets[@]}"; do
  if [ -e "$target" ]; then
    du -sh "$target"
  else
    echo "0B	$target (missing)"
  fi
done

if [ "$APPLY" != "1" ]; then
  echo
  echo "[dry-run] no files deleted. Re-run with --apply to delete listed targets."
  exit 0
fi

echo
echo "[delete]"
for target in "${targets[@]}"; do
  if [ -e "$target" ]; then
    echo "$target"
    rm -rf "$target"
  fi
done

echo
echo "[done] storage cleanup complete"
