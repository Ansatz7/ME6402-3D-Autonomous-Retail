#!/usr/bin/env bash
# =============================================================================
# 3D Gaussian Splatting 上游仓库安装脚本
# 锁定 commit: 54c035f（验证环境：Python 3.10 + CUDA 11.8 + Ubuntu 20.04）
# =============================================================================
set -euo pipefail

# 锁定版本（此 commit 为本项目验证通过的版本）
GS_COMMIT="54c035f"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"
GS_DIR="$THIRD_PARTY_DIR/gaussian-splatting"

mkdir -p "$THIRD_PARTY_DIR"

if [[ ! -d "$GS_DIR/.git" ]]; then
  echo "[setup] Cloning gaussian-splatting into $GS_DIR"
  git clone https://github.com/graphdeco-inria/gaussian-splatting.git "$GS_DIR"
  git -C "$GS_DIR" checkout "$GS_COMMIT"
  echo "[setup] Checked out commit: $GS_COMMIT"
else
  echo "[setup] gaussian-splatting already exists: $GS_DIR"
  CURRENT=$(git -C "$GS_DIR" rev-parse --short HEAD)
  if [[ "$CURRENT" != "$GS_COMMIT" ]]; then
    echo "[setup] WARNING: current commit ($CURRENT) differs from pinned ($GS_COMMIT)"
    echo "[setup] Run: git -C $GS_DIR checkout $GS_COMMIT  to pin to verified version"
  else
    echo "[setup] Commit matches pinned version: $GS_COMMIT"
  fi
fi

echo ""
echo "[setup] Done. Next steps:"
echo "  1) Ensure Conda environment 'gaussian_splatting' is created:"
echo "     conda create -n gaussian_splatting python=3.10 -y"
echo "     conda activate gaussian_splatting"
echo "     conda install pytorch==2.1.2 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"
echo "  2) Install requirements:"
echo "     pip install -r $GS_DIR/requirements.txt"
echo "  3) Build CUDA extensions (requires CUDA 11.8 nvcc):"
echo "     pip install --no-build-isolation -e $GS_DIR/submodules/diff-gaussian-rasterization"
echo "     pip install --no-build-isolation -e $GS_DIR/submodules/simple-knn"
echo "     pip install --no-build-isolation -e $GS_DIR/submodules/fused-ssim"
echo "  4) (Optional) Build SIBR viewer — or use Docker instead:"
echo "     bash scripts/reconstruction/build_sibr_docker_image.sh"
