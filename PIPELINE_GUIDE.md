# 3DGS Pipeline 演示和控制中枢

## 📖 文档导航

本项目提供两种方式运行 3DGS 演示和管理整个 pipeline：

### 方式 1️⃣：**Jupyter Notebook（推荐）**

位置：`notebooks/pipeline_control_center.ipynb`

**优点**：
- 可视化界面
- 支持逐步执行（cell-by-cell）
- 实时日志和结果查看
- 参数配置在一个地方，全局控制

**快速启动**：
```bash
# 激活环境
source $HOME/miniconda3/bin/activate gaussian_splatting

# 启动 Jupyter
cd /home/ansatz/github/ME6402-3D-Autonomous-Retail
jupyter notebook notebooks/pipeline_control_center.ipynb

# 或用 VS Code 打开（需要 Jupyter 扩展）
```

**Notebook 结构**：
1. **导入和配置** - 环境设置和路径
2. **环境检查** - PyTorch、CUDA、GPU 验证
3. **Pipeline 参数** - 一处修改全局配置（数据源、迭代数、输出路径）
4. **3DGS 快速训练** - 300 iter 演示（~2-5 分钟）
5. **COLMAP 处理** - 相机标定（可选）
6. **训练结果评估** - 分析输出和指标
7. **完整 Pipeline** - 一键运行（COLMAP → 3DGS → 评估）
8. **日志和参考** - 查看执行日志、常用命令
9. **快速导航** - 使用指南

### 方式 2️⃣：**命令行脚本（快速验证）**

位置：`scripts/run_3dgs_demo_quick.sh`

**优点**：
- 快速运行，无需 Jupyter
- 适合自动化流程
- 输出清晰

**快速启动**：
```bash
# 直接运行演示
bash scripts/run_3dgs_demo_quick.sh

# 或手动执行
source $HOME/miniconda3/bin/activate gaussian_splatting
cd third_party/gaussian-splatting
python train.py -s ../../data/minimal_dataset -m ../../outputs/3dgs_demo_300iter --iterations 300
```

---

## 🚀 快速开始（首次用户）

### Step 1: 验证环境
```bash
source $HOME/miniconda3/bin/activate gaussian_splatting
bash scripts/quick_env_check.sh
```

### Step 2: 选择运行方式

**选项 A：用 Notebook（推荐）**
```bash
jupyter notebook notebooks/pipeline_control_center.ipynb
# 然后按顺序运行 cell，从 Section 2️⃣ 开始
```

**选项 B：用脚本（快速验证）**
```bash
bash scripts/run_3dgs_demo_quick.sh
```

### Step 3: 检查结果
```bash
# 查看输出
ls -la outputs/3dgs_demo_300iter/point_cloud/

# 应该看到 point_cloud.ply 文件
ls -la outputs/3dgs_demo_300iter/point_cloud/iteration_300/
```

---

## 📊 配置参数说明

在 Notebook 第 3️⃣ 部分修改 `config` 字典来控制整个 pipeline：

```python
config = {
    "dataset": {
        "source": "minimal",           # "minimal" | "custom" | "nerf_synthetic"
        "path": "data/minimal_dataset"
    },
    "3dgs": {
        "iterations": 300,             # 修改此处改变训练轮数（300=演示, 30000=完整）
        "resolution": 1,               # 1=原分辨率, 2=1/2, 4=1/4
        "white_background": False,
        "output_dir": "outputs/3dgs_demo"
    }
}
```

**常用配置**：
- 快速演示：`iterations=300` + `resolution=4` （~1 分钟）
- 标准验证：`iterations=300` + `resolution=2` （~2-3 分钟）
- 完整训练：`iterations=30000` + `resolution=1` （~1 小时）

---

## 📁 文件结构

```
project/
├── notebooks/
│   └── pipeline_control_center.ipynb      ← 推荐：Pipeline 控制中枢
├── scripts/
│   ├── quick_env_check.sh                 ← 环境快速检查
│   ├── run_3dgs_demo_quick.sh             ← 命令行快速演示
│   └── reconstruction/
│       ├── run_colmap.sh
│       ├── run_3dgs_train.sh
│       └── ...
├── data/
│   ├── minimal_dataset/                   ← 最小演示数据
│   └── nerf_synthetic/                    ← NeRF Synthetic 数据（可选）
├── outputs/                               ← 训练输出
│   ├── 3dgs_demo/
│   ├── 3dgs_demo_300iter/
│   └── ...
└── logs/                                  ← 执行日志
    └── pipeline_*.log
```

---

## 🔧 环境信息

**已验证的环境配置**：
- Python: 3.10.20
- PyTorch: 2.1.2
- CUDA: 11.8（via Conda）
- GPU: RTX 4060 (8GB) + RTX 2080 Ti (22GB)
- Framework: 3D Gaussian Splatting (官方)

**激活命令**：
```bash
source $HOME/miniconda3/bin/activate gaussian_splatting
```

---

## ❓ 常见问题

**Q: 为什么我的训练很慢？**
A: 检查 GPU 是否被占用
```bash
nvidia-smi  # 查看 GPU 使用情况
```

**Q: PLY 文件没有生成？**
A: 检查 `outputs/` 目录是否存在，训练日志中是否有错误

**Q: 如何修改训练数据？**
A: 修改 Notebook 第 3️⃣ 部分的 `config["dataset"]["path"]`

**Q: 如何跳过 COLMAP？**
A: 确保数据中有 `transforms.json`，然后设置 `config["dataset"]["use_colmap"] = False`

---

## 📝 更新日期

- **2026-03-22** - 创建 Notebook 控制中枢和快速演示脚本
- **2026-03-22** - 完成 Conda 环境配置和 3DGS 编译

更多信息见 `目标和进度.md`

