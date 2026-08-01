#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="$ROOT_DIR/datasets"

mkdir -p "$DATASETS_DIR"

download_and_extract() {
  local name="$1"
  local url="$2"
  local zip_path="$DATASETS_DIR/${name}.zip"

  if [ -d "$DATASETS_DIR/$name" ]; then
    echo "[skip] $name already exists at $DATASETS_DIR/$name"
    return
  fi

  echo "[download] $name"
  curl -L "$url" -o "$zip_path"
  echo "[extract] $name"
  unzip -o "$zip_path" -d "$DATASETS_DIR"
  rm -f "$zip_path"
}

download_and_extract "coco8" "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
download_and_extract "coco128" "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

echo "[done] datasets are ready under $DATASETS_DIR"

