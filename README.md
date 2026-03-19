# 3D-Vision Enabled Autonomous Retail System | 3D视觉赋能无人零售系统

ME6402 Course Project: Planogram Compliance and Auto-Checkout  
ME6402课程项目：货架陈列合规检测与自动结算

---

## English Guide

[English](#english-guide) | [中文](#中文指南)

### Project Overview
This project develops a smart retail checkout and planogram compliance system by elevating 2D object detection into a 3D interactive digital twin. It targets heavy occlusion and dense shelf scenes, and tracks product states (compliant, misplaced, out-of-bound, or taken) in reconstructed 3D space.

### System Pipeline
1. Multi-view image capture from shelf cameras.
2. 3D reconstruction with SfM (COLMAP) and 3D Gaussian Splatting (3DGS).
3. 2D detection with YOLOv10 / RT-DETR and 2D-to-3D mapping.
4. Planogram compliance scoring using geometric metrics.

### Evaluation Metrics
- Position Deviation: $\Delta D = ||c_{det} - c_{plan}||_2$
- Pose Deviation: 3D IoU between detected and planogram boxes.
- PCR (Planogram Compliance Rate): binary pass/fail with thresholds.
- WCI (Weighted Compliance Index): continuous weighted score:

$$
S_i = \alpha \cdot \max\left(0, 1 - \frac{\Delta D}{D_{max}}\right) + \beta \cdot IoU
$$

### Tech Stack
- Reconstruction: COLMAP, 3D Gaussian Splatting
- Detection: YOLOv10 / RT-DETR, OpenCV, PyTorch
- Evaluation: NumPy, SciPy, custom 3D IoU and compliance scoring

### Repository Tree
```text
.
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- data/
|   |-- raw/
|   |   `-- .gitkeep
|   `-- processed/
|       `-- .gitkeep
|-- notebooks/
|   `-- .gitkeep
|-- weights/
|   `-- .gitkeep
`-- src/
	|-- __init__.py
	|-- reconstruction/
	|   |-- __init__.py
	|   |-- .gitkeep
	|   `-- runner.py
	|-- detection/
	|   |-- __init__.py
	|   |-- .gitkeep
	|   `-- runner.py
	`-- evaluation/
		|-- __init__.py
		|-- .gitkeep
		|-- planogram_evaluator.py
		`-- metrics_demo.py
```

### File and Folder Responsibilities
- `data/raw/`: input multi-view images and raw annotations (ignored by git).
- `data/processed/`: intermediate outputs such as reconstruction artifacts (ignored by git).
- `notebooks/`: experiment notebooks for quick analysis and visualization.
- `weights/`: model checkpoints and exported weights (ignored by git except `.gitkeep`).
- `src/reconstruction/runner.py`: reconstruction stage entry script (currently scaffold).
- `src/detection/runner.py`: detection and mapping stage entry script (currently scaffold).
- `src/evaluation/planogram_evaluator.py`: core metric implementation for Delta D, 3D IoU, PCR, WCI.
- `src/evaluation/metrics_demo.py`: minimal runnable example for evaluator output.
- `scripts/reconstruction/setup_3dgs.sh`: clone/check `graphdeco-inria/gaussian-splatting` into `third_party/`.
- `scripts/reconstruction/check_3dgs_env.sh`: check GPU/CUDA/COLMAP/Python dependencies before running training.
- `scripts/reconstruction/run_colmap.sh`: one-click COLMAP sparse+dense preparation pipeline.
- `scripts/reconstruction/run_3dgs_train.sh`: launch upstream `train.py` for 3DGS training.
- `scripts/reconstruction/run_3dgs_demo.sh`: run a short training on a prepared COLMAP-format demo dataset.
- `configs/reconstruction/scene_example.env`: example path configuration for quick trial.
- `requirements.txt`: Python dependency baseline for all members.

### Quick Start: Reproduce 3DGS Pipeline (Feature Branch)
0. Integration strategy (recommended):
	- Do NOT copy all 3DGS files into project root.
	- Keep upstream repo under `third_party/gaussian-splatting` and call it via scripts.
	- This keeps your repo clean and makes upstream updates easier.
1. Switch to feature branch:
	- `git checkout feat/pipeline-skeleton`
2. Clone upstream 3DGS repo:
	- `bash scripts/reconstruction/setup_3dgs.sh`
3. Run environment check:
	- `bash scripts/reconstruction/check_3dgs_env.sh`
4. Fast demo path (if you already have a COLMAP-format demo dataset):
	- `bash scripts/reconstruction/run_3dgs_demo.sh <source_path> <model_output_dir> 300`
5. Run COLMAP preprocessing for your own photos:
	- `bash scripts/reconstruction/run_colmap.sh data/raw/scene01 data/processed/reconstruction/scene01`
6. Launch 3DGS training:
	- `bash scripts/reconstruction/run_3dgs_train.sh third_party/gaussian-splatting data/processed/reconstruction/scene01/dense data/processed/reconstruction/scene01/gs_model`
7. Optional dry-run command orchestration test:
	- `python3 src/reconstruction/runner.py`

Notes:
- Ensure `colmap` is installed and available in PATH.
- Upstream gaussian-splatting has GPU/CUDA build requirements; follow its official docs for full environment setup.
- This repo provides integration scripts and orchestration, while heavy training code stays in upstream checkout.

---

## 中文指南

[English](#english-guide) | [中文](#中文指南)

### 项目概述
本项目面向无人零售场景，目标是将传统2D检测升级为3D交互数字孪生系统，用于解决货架遮挡严重、视角盲区和密集摆放带来的识别困难，并在3D空间中判断商品状态（合规、错放、越界、被取走）。

### 系统流程
1. 多视角图像采集：从不同角度获取货架图像。
2. 三维重建：基于SfM（COLMAP）和3DGS重建货架场景。
3. 二维检测与三维映射：使用YOLOv10/RT-DETR检测并映射到3D空间。
4. 合规评估：通过几何指标计算货架陈列合规程度。

### 评价指标
- 位置偏差（Delta D）：$\Delta D = ||c_{det} - c_{plan}||_2$
- 姿态偏差：检测框与标准框的3D IoU。
- PCR（Planogram Compliance Rate）：基于阈值的二值合规率。
- WCI（Weighted Compliance Index）：连续加权评分。

### 技术栈
- 三维重建：COLMAP、3D Gaussian Splatting
- 目标检测：YOLOv10 / RT-DETR、OpenCV、PyTorch
- 评估计算：NumPy、SciPy、自定义3D IoU与合规评分

### 仓库结构与协作说明
- `src/reconstruction/`: 三维重建模块。
- `src/detection/`: 2D检测与2D到3D映射模块。
- `src/evaluation/`: 指标计算与评估模块。
- 各目录中的 `runner.py` 为阶段入口，便于组员并行开发。
- `planogram_evaluator.py` 为评价算法核心实现，建议统一维护接口，避免组内调用不一致。
- `scripts/reconstruction/check_3dgs_env.sh` 会在训练前检查依赖，建议先执行。
- `scripts/reconstruction/run_3dgs_demo.sh` 用于快速验证示例数据可训练。

### 3DGS 快速复现（功能分支）
0. 集成策略（推荐）：
	- 不要把3DGS仓库内容直接铺到项目根目录。
	- 建议放在 `third_party/gaussian-splatting`，由本项目脚本调用。
	- 这样更容易维护、升级上游版本，也不会污染主仓库结构。
1. 切换到功能分支：
	- `git checkout feat/pipeline-skeleton`
2. 拉取上游3DGS仓库：
	- `bash scripts/reconstruction/setup_3dgs.sh`
3. 先做环境检查：
	- `bash scripts/reconstruction/check_3dgs_env.sh`
4. 快速示例路径（你已有 COLMAP 格式示例数据时）：
	- `bash scripts/reconstruction/run_3dgs_demo.sh <source_path> <model_output_dir> 300`
5. 对你自己的图片执行COLMAP预处理：
	- `bash scripts/reconstruction/run_colmap.sh data/raw/scene01 data/processed/reconstruction/scene01`
6. 启动3DGS训练：
	- `bash scripts/reconstruction/run_3dgs_train.sh third_party/gaussian-splatting data/processed/reconstruction/scene01/dense data/processed/reconstruction/scene01/gs_model`
7. 可选：先做 dry-run 验证命令链：
	- `python3 src/reconstruction/runner.py`

说明：
- 需提前安装并配置 `colmap`。
- gaussian-splatting 对 CUDA/GPU 环境有要求，请按其官方文档安装依赖。
- 本仓库负责“接入与编排”，重训练核心代码保持在上游仓库，便于后续同步更新。
