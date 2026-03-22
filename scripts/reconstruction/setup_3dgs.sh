#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"
GS_DIR="$THIRD_PARTY_DIR/gaussian-splatting"

mkdir -p "$THIRD_PARTY_DIR"

if [[ ! -d "$GS_DIR/.git" ]]; then
  echo "[setup] Cloning gaussian-splatting into $GS_DIR"
  git clone https://github.com/graphdeco-inria/gaussian-splatting.git "$GS_DIR"
else
  echo "[setup] gaussian-splatting already exists: $GS_DIR"
fi

echo "[setup] Done. Next steps:"
echo "  1) Ensure Conda environment 'gaussian_splatting' is created and has PyTorch installed"
echo "  2) Activate the environment: conda activate gaussian_splatting"
echo "  3) Install requirements: pip install -r $GS_DIR/requirements.txt"
echo "  4) Build extensions as documented by upstream repo:"
echo "     cd $GS_DIR/submodules/diff-gaussian-rasterization"
echo "     pip install -e ."
echo "     cd ../simple-knn"
echo "     pip install -e ."
echo "     cd ../fused-ssim"
echo "     pip install -e ."
