# 团队开发指南 - ME6402 3D 自主零售

## 概述

本项目已优化为支持团队协作。所有脚本和配置均使用**相对路径**，确保团队成员在任何安装路径上都能顺利使用该项目。

## 关键特性

✅ **相对路径设计** - 所有脚本独立于绝对路径工作  
✅ **Conda 环境** - 统一使用 `gaussian_splatting` 环境  
✅ **一体化 Notebook** - Jupyter 管道控制中心  
✅ **快速启动脚本** - 300 iteration 快速演示  
✅ **完整文档** - PIPELINE_GUIDE.md 中的详细说明

---

## 快速开始

### 方式 1：使用 Jupyter Notebook（推荐）

```bash
# 1. 激活 Conda 环境
conda activate gaussian_splatting

# 2. 启动 Jupyter Notebook
cd /path/to/ME6402-3D-Autonomous-Retail
jupyter notebook notebooks/pipeline_control_center.ipynb
```

在 notebook 中，你可以：
- 配置管道参数（一处修改，全局生效）
- 运行完整的 3DGS 训练
- 进行 COLMAP 预处理
- 分析训练结果

### 方式 2：命令行脚本

#### 快速演示（300 iteration）

```bash
bash scripts/run_3dgs_demo_quick.sh
```

#### 完整训练（自定义参数）

```bash
bash scripts/reconstruction/run_3dgs_demo.sh <source_path> <model_output> [iterations]

# 例子
bash scripts/reconstruction/run_3dgs_demo.sh \
  data/raw/my_scene \
  data/processed/my_model \
  1000
```

#### 环境验证

```bash
bash scripts/quick_env_check.sh
bash scripts/reconstruction/check_3dgs_env.sh
```

---

## 项目结构

```
ME6402-3D-Autonomous-Retail/
├── notebooks/
│   └── pipeline_control_center.ipynb    # Jupyter 管道控制中心
├── scripts/
│   ├── quick_env_check.sh              # 快速环境检查 (7 步)
│   ├── run_3dgs_demo_quick.sh          # 300 iteration 快速演示
│   └── reconstruction/                  # 重建脚本集合
│       ├── check_3dgs_env.sh           # Conda 环境验证
│       ├── run_3dgs_demo.sh            # 3DGS 训练（参数化）
│       ├── run_3dgs_train.sh           # 完整训练脚本
│       ├── run_colmap.sh               # COLMAP 预处理
│       └── setup_3dgs.sh               # 3DGS 克隆和设置
├── data/
│   ├── raw/                            # 原始数据存储（需创建）
│   ├── processed/                      # 处理后的数据（自动创建）
│   └── minimal_dataset/                # 最小演示数据集
├── third_party/
│   └── gaussian-splatting/             # 3DGS 源代码（Git 子模块）
├── outputs/                            # 训练输出（自动创建）
├── .gitignore                          # 排除大文件和依赖项
├── PIPELINE_GUIDE.md                   # 详细使用文档
└── TEAM_SETUP_GUIDE.md                 # 本文件
```

---

## 相对路径设计说明

### Bash 脚本

所有 Bash 脚本使用以下模式自动检测项目根目录：

```bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
GS_DIR="$PROJECT_ROOT/third_party/gaussian-splatting"
```

**优势**：
- 无论从何处调用脚本，都能正确找到项目文件
- 支持 symlink 和任意安装路径
- 团队成员可使用不同的本地路径

### Python/Jupyter

Notebook 使用以下逻辑检测项目根目录：

```python
from pathlib import Path
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / '.git').exists():
    PROJECT_ROOT = Path.cwd().parent
```

---

## Conda 环境验证

运行以下命令验证环境已正确配置：

```bash
# 激活环境
conda activate gaussian_splatting

# 验证关键包
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('✓ GaussianRasterizer')"
```

**预期输出**：
```
PyTorch: 2.1.2
CUDA: 11.8
✓ GaussianRasterizer
```

---

## .gitignore 重要说明

以下项已添加到 `.gitignore` 并**不会被 Git 上传**：

```
# 第三方依赖（仅供项目使用，不上传）
third_party/gaussian-splatting/

# 大文件
*.pt, *.ply, *.pth, *.ckpt, *.onnx
data/raw/*, data/processed/*
outputs/, logs/

# Python 缓存
__pycache__/, .ipynb_checkpoints/
```

⚠️ **重要**：`third_party/gaussian-splatting` 是 3DGS 的本地副本，用于项目开发和编译。它**不是**用于上游合并的 Fork。

---

## 常见问题解决

### 问题：脚本说找不到 gaussian-splatting

**解决**：
```bash
cd /path/to/ME6402-3D-Autonomous-Retail
bash scripts/reconstruction/setup_3dgs.sh
```

### 问题：Conda 环境激活出错

**解决**：
```bash
# 列出可用环境
conda env list

# 确认环境存在
conda activate gaussian_splatting

# 如果不存在，从 env.yml 重新创建（联系团队维护者）
```

### 问题：CUDA/GPU 不可用

**解决**：
```bash
# 验证 CUDA 设备
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 检查系统依赖
nvidia-smi
nvcc --version
```

### 问题：相对路径仍然不工作

**原因**：通常是脚本的 shebang 或执行方式有问题。

**验证脚本的执行方式**：
```bash
# ✓ 正确方式
bash scripts/run_3dgs_demo_quick.sh

# ✓ 也可以
./scripts/run_3dgs_demo_quick.sh

# ✗ 不要这样（直接用 Python 执行 Bash 脚本）
python scripts/run_3dgs_demo_quick.sh
```

---

## 下一步

1. **运行快速演示**
   ```bash
   bash scripts/run_3dgs_demo_quick.sh
   ```

2. **查看输出**
   ```bash
   ls outputs/3dgs_demo_300iter/point_cloud/iteration_300/
   ```

3. **导入你的数据** - 修改 notebook 中的数据路径，运行完整管道

4. **贡献代码** - 提交 PR 时请确保所有路径使用相对路径

---

## 联系和支持

- 📖 详细文档：见 [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
- 🐛 问题报告：在项目上提交 GitHub Issues
- 💬 团队讨论：联系项目维护者

---

**最后更新**：2024年  
**版本**：1.0 - 团队协作就绪
