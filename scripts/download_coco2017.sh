#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="$ROOT_DIR/datasets/coco"
IMAGES_DIR="$DATASETS_DIR/images"
TMP_DIR="$ROOT_DIR/datasets/.tmp_coco2017"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

mkdir -p "$DATASETS_DIR" "$IMAGES_DIR" "$TMP_DIR"

download_if_needed_or_resume() {
  local filename="$1"
  local url="$2"
  local completed_path="$3"
  local destination="$TMP_DIR/$filename"
  local code=0
  if [ -e "$completed_path" ]; then
    echo "[skip] $filename not needed (already extracted: $completed_path)"
    return
  fi
  if [ -f "$destination" ]; then
    echo "[resume] $filename"
  else
    echo "[download] $filename"
  fi
  set +e
  curl -L -C - "$url" -o "$destination"
  code=$?
  set -e
  if [ "$code" -ne 0 ]; then
    if [ "$code" -ne 33 ]; then
      return "$code"
    fi
    echo "[info] $filename already complete on disk (curl exit 33)"
  fi
}

extract_if_missing_dir() {
  local archive="$1"
  local expected_dir="$2"
  local target_dir="${3:-$DATASETS_DIR}"
  if [ -d "$target_dir/$expected_dir" ]; then
    echo "[skip] $expected_dir already extracted"
    return
  fi
  if [ ! -f "$TMP_DIR/$archive" ]; then
    echo "[error] archive missing: $TMP_DIR/$archive"
    return 1
  fi
  echo "[extract] $archive"
  unzip -o "$TMP_DIR/$archive" -d "$target_dir"
}

extract_if_missing_file() {
  local archive="$1"
  local expected_file="$2"
  if [ -f "$DATASETS_DIR/$expected_file" ]; then
    echo "[skip] $expected_file already extracted"
    return
  fi
  if [ ! -f "$TMP_DIR/$archive" ]; then
    echo "[error] archive missing: $TMP_DIR/$archive"
    return 1
  fi
  echo "[extract] $archive"
  unzip -o "$TMP_DIR/$archive" -d "$DATASETS_DIR"
}

download_if_needed_or_resume "train2017.zip" "http://images.cocodataset.org/zips/train2017.zip" "$IMAGES_DIR/train2017"
download_if_needed_or_resume "val2017.zip" "http://images.cocodataset.org/zips/val2017.zip" "$IMAGES_DIR/val2017"
download_if_needed_or_resume "test2017.zip" "http://images.cocodataset.org/zips/test2017.zip" "$IMAGES_DIR/test2017"
download_if_needed_or_resume "annotations_trainval2017.zip" "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$DATASETS_DIR/annotations/instances_train2017.json"
download_if_needed_or_resume "image_info_test2017.zip" "http://images.cocodataset.org/annotations/image_info_test2017.zip" "$DATASETS_DIR/annotations/image_info_test2017.json"

extract_if_missing_dir "train2017.zip" "train2017" "$IMAGES_DIR"
extract_if_missing_dir "val2017.zip" "val2017" "$IMAGES_DIR"
extract_if_missing_dir "test2017.zip" "test2017" "$IMAGES_DIR"
extract_if_missing_file "annotations_trainval2017.zip" "annotations/instances_train2017.json"
extract_if_missing_file "image_info_test2017.zip" "annotations/image_info_test2017.json"

if [ "$KEEP_ARCHIVES" = "1" ]; then
  echo "[keep] downloaded archives remain under $TMP_DIR"
else
  echo "[cleanup] removing downloaded archives under $TMP_DIR"
  rm -rf "$TMP_DIR"
fi

echo "[done] COCO 2017 is ready under $DATASETS_DIR"
