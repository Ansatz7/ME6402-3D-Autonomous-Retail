#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
ENV_NAME="gaussian_splatting"

echo "[check] Project Root: $ROOT_DIR"
echo "[check] Using Conda environment: $ENV_NAME"
echo ""

check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    echo "[✓] System command: $name"
  else
    echo "[✗] Missing system command: $name"
  fi
}

check_py_pkg() {
  local pkg="$1"
  if python -c "import importlib.util;import sys;sys.exit(0 if importlib.util.find_spec('$pkg') else 1)" 2>/dev/null; then
    echo "[✓] Python package: $pkg"
  else
    echo "[✗] Missing Python package: $pkg"
  fi
}

# Activate conda environment
echo "[setup] Activating Conda environment..."
source "$CONDA_HOME/bin/activate" "$ENV_NAME"

echo ""
echo "=== System Dependencies ==="
check_cmd git
check_cmd nvidia-smi
check_cmd nvcc
check_cmd colmap

echo ""
echo "=== Python Environment ==="
echo "[✓] Python: $CONDA_PREFIX"
python --version

echo ""
echo "=== Python Packages ==="
check_py_pkg torch
check_py_pkg torchvision
check_py_pkg numpy
check_py_pkg cv2
check_py_pkg diff_gaussian_rasterization
check_py_pkg fused_ssim
check_py_pkg plyfile
check_py_pkg tqdm

echo ""
echo "=== Troubleshooting ==="
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[hint] CUDA toolkit missing. To install on Ubuntu:"
  echo "  sudo apt update"
  echo "  sudo apt install -y nvidia-cuda-toolkit"
fi

if ! command -v colmap >/dev/null 2>&1; then
  echo "[hint] COLMAP missing. To install on Ubuntu:"
  echo "  sudo apt install -y colmap"
fi
