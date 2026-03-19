#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

echo "[check] Root: $ROOT_DIR"

check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    echo "[ok] $name found: $(command -v "$name")"
  else
    echo "[missing] $name"
  fi
}

check_py_pkg() {
  local pkg="$1"
  if "$PYTHON_BIN" -c "import importlib.util;import sys;sys.exit(0 if importlib.util.find_spec('$pkg') else 1)"; then
    echo "[ok] python package: $pkg"
  else
    echo "[missing] python package: $pkg"
  fi
}

check_cmd git
check_cmd nvidia-smi
check_cmd nvcc
check_cmd colmap

if [[ -x "$PYTHON_BIN" ]]; then
  echo "[ok] python: $PYTHON_BIN"
  check_py_pkg torch
  check_py_pkg torchvision
  check_py_pkg numpy
  check_py_pkg cv2
else
  echo "[missing] python venv executable: $PYTHON_BIN"
fi

echo ""
echo "[hint] If nvcc or colmap is missing, run these manually on Ubuntu:"
echo "  sudo apt update"
echo "  sudo apt install -y nvidia-cuda-toolkit colmap"
echo ""
echo "[hint] Install Python dependencies:"
echo "  $PYTHON_BIN -m pip install --upgrade pip"
echo "  $PYTHON_BIN -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
echo "  cd $ROOT_DIR/third_party/gaussian-splatting"
echo "  $PYTHON_BIN -m pip install opencv-python tqdm plyfile joblib"
echo "  $PYTHON_BIN -m pip install ./submodules/diff-gaussian-rasterization ./submodules/simple-knn ./submodules/fused-ssim"
