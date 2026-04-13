#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <image_dir> <work_dir>"
  echo "Example: $0 data/raw/shelf_scene data/processed/reconstruction/scene01"
  exit 1
fi

IMAGE_DIR="$1"
WORK_DIR="$2"
DB_PATH="$WORK_DIR/database.db"
SPARSE_DIR="$WORK_DIR/sparse"
DENSE_DIR="$WORK_DIR/dense"

mkdir -p "$WORK_DIR" "$SPARSE_DIR" "$DENSE_DIR"

colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$IMAGE_DIR" \
  --ImageReader.single_camera 1

colmap exhaustive_matcher \
  --database_path "$DB_PATH"

mkdir -p "$SPARSE_DIR/0"
colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$IMAGE_DIR" \
  --output_path "$SPARSE_DIR"

colmap image_undistorter \
  --image_path "$IMAGE_DIR" \
  --input_path "$SPARSE_DIR/0" \
  --output_path "$DENSE_DIR" \
  --output_type COLMAP

echo "[colmap] Finished. Outputs in: $WORK_DIR"
