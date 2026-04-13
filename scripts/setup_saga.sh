#!/usr/bin/env bash
# =============================================================================
# SAGA 环境安装脚本
# 在已有的 gaussian_splatting conda 环境中安装 SAGA 所需额外依赖
# 锁定 commit: 4acdaa6（验证环境：Python 3.10 + CUDA 11.8 + Ubuntu 20.04）
#
# 使用方法：
#   source $HOME/miniconda3/bin/activate gaussian_splatting
#   bash scripts/setup_saga.sh
# =============================================================================
set -euo pipefail

# 锁定版本（此 commit 为本项目验证通过的版本）
SAGA_COMMIT="4acdaa6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo " SAGA 安装脚本"
echo " 项目根目录：$PROJECT_ROOT"
echo "=============================================="

# ── 1. 验证 conda 环境 ──────────────────────────────────────────
EXPECTED_ENV="gaussian_splatting"
CURRENT_ENV="${CONDA_DEFAULT_ENV:-}"
if [[ "$CURRENT_ENV" != "$EXPECTED_ENV" ]]; then
    echo "⚠️  请先激活 $EXPECTED_ENV 环境："
    echo "    source \$HOME/miniconda3/bin/activate $EXPECTED_ENV"
    exit 1
fi
echo "✓ Conda 环境：$CURRENT_ENV"

# ── 2. 克隆 SAGA 仓库（含子模块）──────────────────────────────────
SAGA_DIR="$PROJECT_ROOT/third_party/SAGA"
if [[ ! -d "$SAGA_DIR/.git" ]]; then
    echo ""
    echo ">>> 克隆 SAGA 仓库..."
    git clone --recursive https://github.com/Jumpat/SegAnyGAussians.git "$SAGA_DIR"
    git -C "$SAGA_DIR" checkout "$SAGA_COMMIT"
    git -C "$SAGA_DIR" submodule update --init --recursive
    echo "✓ 已切换到锁定 commit: $SAGA_COMMIT"
else
    echo "✓ SAGA 仓库已存在，更新子模块..."
    CURRENT=$(git -C "$SAGA_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
    if [[ "$CURRENT" != "$SAGA_COMMIT" ]]; then
        echo "⚠️  当前 commit ($CURRENT) 与锁定版本 ($SAGA_COMMIT) 不同"
        echo "   如需切换：git -C $SAGA_DIR checkout $SAGA_COMMIT"
    else
        echo "✓ Commit 匹配锁定版本：$SAGA_COMMIT"
    fi
    git -C "$SAGA_DIR" submodule update --init --recursive
fi

# ── 3. 安装 SAM（Segment Anything）──────────────────────────────
echo ""
echo ">>> 安装 segment-anything..."
SAM_PKG="$SAGA_DIR/third_party/segment-anything"
if [[ -d "$SAM_PKG" ]]; then
    pip install -e "$SAM_PKG" --quiet
    echo "✓ segment-anything 安装完成（本地源）"
else
    pip install segment-anything --quiet
    echo "✓ segment-anything 安装完成（PyPI）"
fi

# ── 4. 安装 kmeans_pytorch ───────────────────────────────────────
echo ""
echo ">>> 安装 kmeans_pytorch..."
KM_PKG="$SAGA_DIR/third_party/kmeans_pytorch"
if [[ -d "$KM_PKG" ]]; then
    pip install -e "$KM_PKG" --quiet
    echo "✓ kmeans_pytorch 安装完成（本地源）"
else
    pip install kmeans-pytorch --quiet
    echo "✓ kmeans_pytorch 安装完成（PyPI）"
fi

# ── 5. 安装其他 Python 依赖 ──────────────────────────────────────
echo ""
echo ">>> 安装 open_clip_torch, hdbscan..."
pip install open-clip-torch hdbscan --quiet
echo "✓ open_clip_torch, hdbscan 安装完成"

# ── 6. 编译 SAGA 定制 CUDA 光栅化模块 ───────────────────────────
# 确保使用 Conda 的 nvcc（11.8），而不是系统的
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"

echo ""
echo ">>> 编译 diff-gaussian-rasterization_contrastive_f..."
CONTRASTIVE_PKG="$SAGA_DIR/submodules/diff-gaussian-rasterization_contrastive_f"
if [[ -d "$CONTRASTIVE_PKG" ]]; then
    pip install --no-build-isolation -e "$CONTRASTIVE_PKG" --quiet
    echo "✓ diff-gaussian-rasterization_contrastive_f 编译完成"
else
    echo "⚠️  未找到 $CONTRASTIVE_PKG，请检查 SAGA 子模块是否已初始化："
    echo "    git -C $SAGA_DIR submodule update --init --recursive"
fi

echo ""
echo ">>> 编译 diff-gaussian-rasterization-depth..."
DEPTH_PKG="$SAGA_DIR/submodules/diff-gaussian-rasterization-depth"
if [[ -d "$DEPTH_PKG" ]]; then
    pip install --no-build-isolation -e "$DEPTH_PKG" --quiet
    echo "✓ diff-gaussian-rasterization-depth 编译完成"
else
    echo "⚠️  未找到 $DEPTH_PKG，请检查 SAGA 子模块是否已初始化："
    echo "    git -C $SAGA_DIR submodule update --init --recursive"
fi

# ── 7. 创建 SAM checkpoint 目录 ──────────────────────────────────
SAM_CKPT_DIR="$PROJECT_ROOT/dependencies/sam_ckpt"
mkdir -p "$SAM_CKPT_DIR"
SAM_CKPT="$SAM_CKPT_DIR/sam_vit_h_4b8939.pth"

echo ""
echo "=============================================="
echo " SAGA 依赖安装完成"
echo "=============================================="
echo ""

if [[ -f "$SAM_CKPT" ]]; then
    echo "✓ SAM checkpoint 已存在：$SAM_CKPT"
else
    echo "⚠️  还需下载 SAM ViT-H checkpoint（~2.5 GB）："
    echo ""
    echo "    wget -P $SAM_CKPT_DIR \\"
    echo "      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    echo ""
    echo "    或者在 Notebook 的 SAGA Section 中运行下载单元格。"
fi

echo ""
echo "下一步：在 Notebook 中打开 SAGA Section 并按顺序运行。"
