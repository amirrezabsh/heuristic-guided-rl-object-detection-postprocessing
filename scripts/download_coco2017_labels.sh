#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="$ROOT_DIR/datasets"
COCO_DIR="$DATASETS_DIR/coco"
TMP_DIR="$DATASETS_DIR/.tmp_coco_labels"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

mkdir -p "$COCO_DIR" "$TMP_DIR"

LABELS_ZIP="$TMP_DIR/coco2017labels.zip"
URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"

if [ ! -f "$LABELS_ZIP" ]; then
  echo "[download] coco2017labels.zip"
  curl -L "$URL" -o "$LABELS_ZIP"
else
  echo "[skip] coco2017labels.zip already downloaded"
fi

if [ -d "$COCO_DIR/labels/train2017" ] && [ -d "$COCO_DIR/labels/val2017" ]; then
  echo "[skip] COCO labels already extracted"
else
  echo "[extract] labels to $COCO_DIR"
  unzip -o "$LABELS_ZIP" -d "$COCO_DIR"
fi

if [ -d "$COCO_DIR/coco/labels" ] && [ ! -d "$COCO_DIR/labels" ]; then
  mv "$COCO_DIR/coco/labels" "$COCO_DIR/labels"
fi
if [ -f "$COCO_DIR/coco/train2017.txt" ] && [ ! -f "$COCO_DIR/train2017.txt" ]; then
  mv "$COCO_DIR/coco/train2017.txt" "$COCO_DIR/train2017.txt"
fi
if [ -f "$COCO_DIR/coco/val2017.txt" ] && [ ! -f "$COCO_DIR/val2017.txt" ]; then
  mv "$COCO_DIR/coco/val2017.txt" "$COCO_DIR/val2017.txt"
fi
if [ -f "$COCO_DIR/coco/test-dev2017.txt" ] && [ ! -f "$COCO_DIR/test-dev2017.txt" ]; then
  mv "$COCO_DIR/coco/test-dev2017.txt" "$COCO_DIR/test-dev2017.txt"
fi
if [ -d "$COCO_DIR/coco/annotations" ] && [ ! -d "$COCO_DIR/annotations" ]; then
  mv "$COCO_DIR/coco/annotations" "$COCO_DIR/annotations"
fi
if [ -d "$COCO_DIR/coco/images/train2017" ] && [ ! -d "$COCO_DIR/images/train2017" ]; then
  mkdir -p "$COCO_DIR/images"
  mv "$COCO_DIR/coco/images/train2017" "$COCO_DIR/images/train2017"
fi
if [ -d "$COCO_DIR/coco/images/val2017" ] && [ ! -d "$COCO_DIR/images/val2017" ]; then
  mkdir -p "$COCO_DIR/images"
  mv "$COCO_DIR/coco/images/val2017" "$COCO_DIR/images/val2017"
fi
rmdir "$COCO_DIR/coco/images" 2>/dev/null || true

if [ "$KEEP_ARCHIVES" = "1" ]; then
  echo "[keep] downloaded label archive remains under $TMP_DIR"
else
  echo "[cleanup] removing downloaded label archive under $TMP_DIR"
  rm -rf "$TMP_DIR"
fi

echo "[done] COCO labels are ready under $COCO_DIR/labels"
