#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <gs_repo_dir> <source_path> <model_path>"
  echo "Example: $0 third_party/gaussian-splatting data/processed/reconstruction/scene01/dense data/processed/reconstruction/scene01/gs_model"
  exit 1
fi

GS_REPO_DIR="$1"
SOURCE_PATH="$2"
MODEL_PATH="$3"

if [[ ! -f "$GS_REPO_DIR/train.py" ]]; then
  echo "Error: train.py not found in $GS_REPO_DIR"
  exit 1
fi

mkdir -p "$MODEL_PATH"

python "$GS_REPO_DIR/train.py" \
  -s "$SOURCE_PATH" \
  -m "$MODEL_PATH"

echo "[3dgs] Training started/completed. Model dir: $MODEL_PATH"
