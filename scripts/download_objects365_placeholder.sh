#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="$ROOT_DIR/datasets/Objects365"

mkdir -p "$DATASETS_DIR"

cat <<'MSG'
Objects365 is very large (hundreds of GB), so this project keeps setup reproducible
without forcing an immediate download.

Use Ultralytics built-in downloader when you are ready:

  .venv/bin/yolo detect train model=yolov8n.pt data=Objects365.yaml epochs=1 imgsz=640 batch=16 device=cpu

Or use your own mirror/pipeline and then create a local YAML under datasets/
pointing to your final train/val image+label paths.
MSG

echo "[done] Placeholder prepared at $DATASETS_DIR"

