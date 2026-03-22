#!/bin/bash

# 3DGS 快速训练演示脚本
# 用途：运行 3DGS 300 iteration 快速演示，验证环境和模型功能

set -e

# 获取脚本所在目录并推导项目根目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
export ENV_NAME="gaussian_splatting"
export GS_DIR="$PROJECT_ROOT/third_party/gaussian-splatting"
export DATA_DIR="$PROJECT_ROOT/data/minimal_dataset"
export OUTPUT_DIR="$PROJECT_ROOT/outputs/3dgs_demo_300iter"

echo "========================================"
echo "3DGS 快速演示 - 300 Iteration"
echo "========================================"
echo ""

# 1. 激活环境
echo "[1/4] 激活 Conda 环境..."
source $CONDA_HOME/bin/activate $ENV_NAME
echo "✓ 环境已激活: $CONDA_PREFIX"
echo ""

# 2. 验证环境
echo "[2/4] 快速环境检查..."
python -c "
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA: {torch.version.cuda}')
print(f'✓ GPU: {torch.cuda.device_count()} 个')
from diff_gaussian_rasterization import GaussianRasterizer
print(f'✓ GaussianRasterizer 可用')
" || exit 1
echo ""

# 3. 创建输出目录
echo "[3/4] 准备输出目录..."
mkdir -p "$OUTPUT_DIR"
echo "✓ 输出目录: $OUTPUT_DIR"
echo ""

# 4. 运行训练
echo "[4/4] 启动 3DGS 训练（300 iterations）..."
echo "  数据路径: $DATA_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo ""
echo "⏳ 训练进行中（预计 2-5 分钟）..."
echo "==========================================="
echo ""

cd "$GS_DIR"

python train.py \
  -s "$DATA_DIR" \
  -m "$OUTPUT_DIR" \
  --iterations 300 \
  --resolution 1 \
  --sh_degree 3 \
  --save_iterations 300 \
  --test_iterations 300 \
  --quiet

echo ""
echo "==========================================="
echo ""
echo "✅ 训练演示完成！"
echo ""
echo "📁 结果位置："
echo "  - 模型文件: $OUTPUT_DIR/point_cloud/iteration_300/point_cloud.ply"
echo "  - 完整输出: $OUTPUT_DIR/"
echo ""
echo "📊 后续步骤："
echo "  1. 检查输出 PLY 文件是否生成"
echo "  2. 用官方 SIBR viewer 打开查看 3D 模型"
echo "  3. 在 Jupyter notebook 中查看详细日志"
echo ""
echo "💡 提示："
echo "  - 这是快速演示（300 iter），完整训练用 30000 iter"
echo "  - 修改 --iterations 参数改变训练轮数"
echo "  - 修改 --resolution 参数改变图像分辨率（1=原分,2=1/2,4=1/4）"
echo ""
