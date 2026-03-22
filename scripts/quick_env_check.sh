#!/bin/bash

# 快速环境检查脚本
# 用途：验证 Conda 环境和 3DGS 核心依赖是否完整可用

set -e

CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
ENV_NAME="gaussian_splatting"

echo "========================================"
echo "3DGS 环境快速检查"
echo "========================================"
echo ""

# 1. 激活环境
echo "[1/7] 激活 Conda 环境..."
source $CONDA_HOME/bin/activate $ENV_NAME
echo "✓ 环境已激活: $CONDA_PREFIX"
echo ""

# 2. 检查 Python
echo "[2/7] 检查 Python..."
python_version=$(python --version)
echo "✓ $python_version"
echo ""

# 3. 检查 PyTorch 和 CUDA
echo "[3/7] 检查 PyTorch 和 CUDA..."
python << 'EOF'
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.version.cuda}")
print(f"✓ CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
EOF
echo ""

# 4. 检查核心依赖
echo "[4/7] 检查核心 Python 依赖..."
python << 'EOF'
import sys
packages = [
    ('numpy', 'numpy'),
    ('opencv', 'cv2'),
    ('plyfile', 'plyfile'),
    ('tqdm', 'tqdm'),
    ('scipy', 'scipy'),
    ('scikit-image', 'skimage')
]

for name, module in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
    except ImportError:
        print(f"✗ {name}: 未安装")
        sys.exit(1)
EOF
echo ""

# 5. 检查 CUDA 模块
echo "[5/7] 检查 CUDA 编译模块..."
python << 'EOF'
try:
    from diff_gaussian_rasterization import GaussianRasterizer
    print("✓ diff_gaussian_rasterization: 已编译")
except ImportError as e:
    print(f"✗ diff_gaussian_rasterization: {e}")
    exit(1)

try:
    import fused_ssim
    print("✓ fused_ssim: 已编译")
except ImportError as e:
    print(f"⚠ fused_ssim: {e}")
EOF
echo ""

# 6. 检查 3DGS 仓库
echo "[6/7] 检查 3DGS 仓库结构..."
gs_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../third_party/gaussian-splatting" && pwd)"
if [ -f "$gs_path/train.py" ]; then
    echo "✓ train.py 存在"
else
    echo "✗ train.py 不存在"
    exit 1
fi

if [ -d "$gs_path/submodules" ]; then
    echo "✓ submodules 目录存在"
else
    echo "✗ submodules 目录不存在"
    exit 1
fi
echo ""

# 7. 检查 COLMAP
echo "[7/7] 检查 COLMAP..."
if command -v colmap &> /dev/null; then
    colmap_version=$(colmap -h | head -1)
    echo "✓ COLMAP 已安装: $colmap_version"
else
    echo "⚠ COLMAP 未在 PATH 中（可选）"
fi
echo ""

echo "========================================"
echo "✓ 环境检查完成！所有关键组件可用。"
echo "========================================"
echo ""
echo "快速启动 3DGS 训练："
echo "  cd \$PWD/third_party/gaussian-splatting"
echo "  source $CONDA_HOME/bin/activate $ENV_NAME"
echo "  python train.py -s <dataset_path> --iterations 300 -m output_demo"
echo ""
