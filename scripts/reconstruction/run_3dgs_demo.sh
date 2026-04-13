#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <source_path> <model_output_dir> [iterations]"
  echo "Example: $0 data/raw/demo_colmap_scene data/processed/reconstruction/demo_model 300"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
ENV_NAME="gaussian_splatting"
GS_DIR="$ROOT_DIR/third_party/gaussian-splatting"
SOURCE_PATH="$1"
MODEL_PATH="$2"
ITERATIONS="${3:-300}"

if [[ ! -f "$GS_DIR/train.py" ]]; then
  echo "Error: gaussian-splatting repo not found at $GS_DIR"
  echo "Run: bash scripts/reconstruction/setup_3dgs.sh"
  exit 1
fi

if [[ ! -d "$SOURCE_PATH" ]]; then
  echo "Error: source_path does not exist: $SOURCE_PATH"
  exit 1
fi

mkdir -p "$MODEL_PATH"

# Activate conda environment
source "$CONDA_HOME/bin/activate" "$ENV_NAME"

python "$GS_DIR/train.py" \
  -s "$SOURCE_PATH" \
  -m "$MODEL_PATH" \
  --iterations "$ITERATIONS"

echo "[done] Demo training finished. Model path: $MODEL_PATH"
