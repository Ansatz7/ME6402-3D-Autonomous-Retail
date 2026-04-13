# 3D-Vision Enabled Autonomous Retail System | 3D 视觉赋能无人零售系统

ME6402 Course Project | ME6402 课程项目

[English](#english-guide) | [中文](#中文指南)

---

## English Guide

### Project Overview

The original goal of this project was to build a full planogram compliance system for retail shelves — detecting whether products are in their correct 3D positions by comparing detected bounding boxes against a reference planogram.

**What we actually built and got working:**

A complete pipeline from raw video to 3D semantic segmentation of shelf products, consisting of two integrated parts:

- **3D Reconstruction**: multi-view video → FFmpeg frame extraction → COLMAP Structure-from-Motion → 3D Gaussian Splatting (3DGS) reconstruction → SIBR / Open3D visualization
- **3D Semantic Segmentation (SAGA)**: on top of the trained 3DGS model, apply SAM to auto-segment all objects → extract CLIP features → train a 3D contrastive feature field → text-query to locate products with 3D bounding boxes

**What was not reached:**

The planogram compliance scoring module (`src/evaluation/planogram_evaluator.py`) — covering ΔD position deviation, 3D IoU, PCR and WCI metrics — was designed and implemented as a standalone evaluator but was **never integrated into the pipeline**. It exists as a theoretical framework. Reaching it requires clean, reliable 3D bounding boxes from the segmentation step, which proved difficult given the limitations of SAGA (described below).

The entire pipeline is controlled through a single Jupyter Notebook: `notebooks/pipeline_control_center.ipynb`. All parameters live in `configs/pipeline.yaml`.

---

### System Pipeline

```
Raw Video
    │
    ▼  [Section 2] FFmpeg — frame extraction
Multi-view Frames (JPEG)
    │
    ▼  [Section 3] COLMAP SfM — camera pose estimation
Sparse Point Cloud + Camera Poses
    │
    ▼  [Section 4] 3D Gaussian Splatting — scene reconstruction
3DGS Model (.ply)
    │
    ├──▶  [Section 5–6] SIBR Viewer (Docker) / Open3D — interactive visualization
    │
    └──▶  [Section 9] SAGA — 3D semantic segmentation
                │
                ├── 9.6.1  Create downsampled images (×downsample)
                ├── 9.6.2  SAM — auto-segment every object per frame → sam_masks/
                ├── 9.6.2  CLIP — extract semantic features per mask → clip_features/
                ├── 9.6.3  Compute 3D physical scale per mask → mask_scales/
                ├── 9.6.4  Train 3D contrastive feature field (~10–40 min)
                └── 9.7    Text query / click → 3D bounding box
                                │
                                ▼
                    ⚠ Results poor: scattered point clouds,
                      imprecise bounding boxes (see Limitations)
                                │
                                ▼
                    [ Planogram compliance evaluation ]
                    [ ΔD / 3D IoU / PCR / WCI          ]
                    [ ← NOT REACHED; framework only    ]
```

---

### Honest Assessment of Results

**3D Reconstruction**: works well. Shelf products are reconstructed at good quality. However, taking many photos inevitably introduces exposure variation between frames. The 3DGS algorithm bakes these lighting inconsistencies into the scene as floating artifact "clouds", which later interfere with segmentation.

**SAGA — 2D CLIP query**: impressive. Typing a text query like `"coca cola can"` or `"tissue box"` into the 2D interface highlights the correct region with no labeling required.

**SAGA — 3D segmentation**: poor. The 3D contrastive feature training uses only SAM masks as supervision (learning geometric consistency), not CLIP features directly. CLIP only comes in at query time. This separation means semantic boundaries in 3D are blurry, and the artifact clouds from reconstruction make the feature distribution noisy. The resulting point clouds are scattered rather than tightly clustered around the target object, making the 3D bounding boxes unreliable.

---

### Repository Structure

```
ME6402-3D-Autonomous-Retail/
│
├── notebooks/
│   └── pipeline_control_center.ipynb   ← main entry point for all operations
│
├── configs/
│   ├── pipeline.yaml                   ← all parameters (edit here, re-run Cell 1)
│   ├── pinned_packages.txt             ← numpy/scipy version pins for SAGA
│   └── reconstruction/
│       └── scene_example.env
│
├── src/
│   ├── pipeline/                       ← core pipeline modules (imported by notebook)
│   │   ├── utils.py                    ← project root detection, config loading, logging
│   │   ├── video.py                    ← FFmpeg frame extraction
│   │   ├── colmap.py                   ← COLMAP SfM + intrinsics parsing + path fixup
│   │   ├── training.py                 ← 3DGS training launcher + OOM auto-downscale
│   │   ├── viewer.py                   ← Open3D viewer + SIBR Docker launcher
│   │   └── saga.py                     ← full SAGA integration (6-step pipeline)
│   │
│   └── evaluation/                     ← compliance scoring (designed, not integrated)
│       ├── planogram_evaluator.py      ← ΔD, 3D IoU, PCR, WCI — standalone only
│       └── metrics_demo.py             ← minimal runnable example
│
├── scripts/
│   ├── setup_saga.sh                   ← clone SAGA + compile CUDA extensions
│   ├── quick_env_check.sh
│   ├── run_3dgs_demo_quick.sh
│   └── reconstruction/
│       ├── setup_3dgs.sh               ← clone gaussian-splatting (pinned commit)
│       ├── check_3dgs_env.sh
│       ├── run_colmap.sh
│       ├── run_3dgs_train.sh
│       ├── run_3dgs_demo.sh
│       ├── build_sibr_docker_image.sh
│       └── run_sibr_in_docker.sh
│
└── requirements.txt
```

---

### Environment Setup

**Requirements**
- Ubuntu 20.04 / 22.04, x86\_64
- NVIDIA GPU, VRAM ≥ 8 GB (≥ 16 GB recommended for SAGA SAM step)
- CUDA 11.8 (managed via Conda)
- `colmap` installed system-wide (`sudo apt install colmap`)
- Docker (optional, for SIBR Viewer)

**Step 1 — Clone this repo**
```bash
git clone <this-repo-url>
cd ME6402-3D-Autonomous-Retail
```

**Step 2 — Extract the data archive**

Place `ME6402_data.tar.gz` in the project root, then extract:

```bash
cd ME6402-3D-Autonomous-Retail
tar xzf ME6402_data.tar.gz
```

The archive unpacks directly into the project root and provides:

```
dependencies/sam_ckpt/sam_vit_h_4b8939.pth       ← SAM weights (~2.5 GB)
data/colmap_workspace/custom_scene_01/             ← scene images + COLMAP output + CLIP features
outputs/3dgs_custom_scene_01_5000iter/             ← trained 3DGS model (ready for SAGA)
outputs/3dgs_team_data2_30000iter/                 ← high-quality model for SAGA demo
```

> Third-party source repos (`gaussian-splatting`, `SAGA`) are **not** in this archive — they are cloned by the setup scripts in Step 3–4.

**Step 3 — Set up the 3DGS Conda environment**
```bash
conda create -n gaussian_splatting python=3.10 -y
conda activate gaussian_splatting
conda install pytorch==2.1.2 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Clone upstream repo and pin to verified commit (54c035f)
bash scripts/reconstruction/setup_3dgs.sh

# Install CUDA extensions
source $HOME/miniconda3/bin/activate gaussian_splatting
pip install --no-build-isolation -e third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install --no-build-isolation -e third_party/gaussian-splatting/submodules/simple-knn
pip install --no-build-isolation -e third_party/gaussian-splatting/submodules/fused-ssim

bash scripts/reconstruction/check_3dgs_env.sh     # verify
```

**Step 4 — Install SAGA dependencies**
```bash
source $HOME/miniconda3/bin/activate gaussian_splatting
bash scripts/setup_saga.sh   # clones SAGA (pinned commit 4acdaa6) + compiles CUDA submodules
```

Compilation takes ~5–15 minutes and requires `nvcc` (CUDA 11.8).

**Step 5 — Install Python dependencies**
```bash
pip install -r requirements.txt
pip install -r configs/pinned_packages.txt   # version pins for SAGA compatibility
```

**Step 6 — (Optional) Build SIBR Docker image**
```bash
bash scripts/reconstruction/build_sibr_docker_image.sh
```

---

### Quick Start

**Launch the notebook (recommended)**
```bash
source $HOME/miniconda3/bin/activate gaussian_splatting
jupyter notebook notebooks/pipeline_control_center.ipynb
```

| Section | Content | Status |
|---------|---------|--------|
| Cell 1 | Initialize — load all modules and config | required first |
| Section 1 | Environment check (PyTorch / CUDA / COLMAP) | |
| Section 2 | Video frame extraction (FFmpeg) | optional |
| Section 3 | COLMAP camera calibration | needed for custom data |
| Section 4 | 3DGS training (300 iter preview → 30 000 iter full) | core |
| Section 5–6 | SIBR / Open3D visualization | |
| Section 7 | One-click full pipeline | |
| Section 9 | SAGA 3D semantic segmentation + text query | |

**To reproduce results on the provided scene (`custom_scene_01`)**:

The archive already contains the COLMAP-processed images and a 5 000-iteration 3DGS model. Start directly from **Section 5** (view reconstruction) or **Section 9** (SAGA segmentation) — no need to re-run COLMAP or training.

---

### Configuration

Edit `configs/pipeline.yaml`. Changes take effect after re-running Cell 1 in the notebook.

```yaml
video:
  enabled: false             # set true to extract frames from a video first
  input_video: ~/Videos/recording.mp4
  scene_name: custom_scene_01
  fps: 2

colmap:
  workspace: data/colmap_workspace
  camera_model: PINHOLE

training:
  iterations: 30000          # 300–5000 for quick preview; 30000 for quality
  resolution: 2              # 1 = full; 2 = half; auto-downscales on OOM
  output_dir: outputs

saga:
  sam_checkpoint: dependencies/sam_ckpt/sam_vit_h_4b8939.pth
  downsample: 4              # ×4 for 8 GB VRAM; ×2 for ≥16 GB
  image_root: "data/colmap_workspace/custom_scene_01/dense"
  model_path:  "outputs/3dgs_custom_scene_01_5000iter"
```

---

### External Components

**Third-party repos (not archived — cloned by scripts)**

| Repo | Upstream | Pinned commit | How to get |
|------|----------|---------------|------------|
| 3D Gaussian Splatting | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | `54c035f` | `bash scripts/reconstruction/setup_3dgs.sh` |
| SAGA | [Jumpat/SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians) | `4acdaa6` | `bash scripts/setup_saga.sh` |

Verified on: Python 3.10.20 · PyTorch 2.1.2 · CUDA 11.8 · Ubuntu 20.04 · x86\_64

> The SIBR Viewer (1.6 GB compiled C++ binary) is handled entirely via Docker; local compilation is not required.

**Files in the accompanying archive**

| Component | Path | Size |
|-----------|------|------|
| SAM ViT-H weights | `dependencies/sam_ckpt/sam_vit_h_4b8939.pth` | ~2.5 GB |
| Custom scene data | `data/colmap_workspace/custom_scene_01/` | ~200 MB |
| 5k-iter 3DGS model | `outputs/3dgs_custom_scene_01_5000iter/` | ~300 MB |
| 30k-iter model | `outputs/3dgs_team_data2_30000iter/` | ~800 MB |

Download SAM weights manually if needed:
```bash
mkdir -p dependencies/sam_ckpt
wget -P dependencies/sam_ckpt \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

### Limitations and Future Work

**Root cause of poor 3D segmentation:**

1. **Artifact clouds in reconstruction.** Shooting many photos inevitably produces exposure differences between frames. 3DGS bakes these as floating artifact clouds in 3D space, making the semantic feature distribution noisy before SAGA even begins.
2. **CLIP is disconnected from 3D training.** SAGA's contrastive feature training uses only SAM mask consistency as supervision — it learns geometric coherence, not semantics. CLIP features are injected only at query time, too late to fix incorrect clustering boundaries.

**Future directions:**

- Apply **Bilateral Grid** exposure normalization across all input frames before reconstruction to suppress artifact clouds.
- Replace SAGA with **LERF** (Language Embedded Radiance Fields), which distills CLIP features directly into the 3D training loop rather than keeping the two tracks separate.
- Once reliable 3D bounding boxes are available, wire them into `src/evaluation/planogram_evaluator.py` to complete the original compliance scoring goal.

---

### References

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al., SIGGRAPH 2023
- [Segment Anything](https://segment-anything.com/) — Kirillov et al., ICCV 2023
- [SAGA: Segment Any 3D Gaussians](https://jumpat.github.io/SAGA/) — Cen et al., AAAI 2025
- [CLIP](https://openai.com/research/clip) — Radford et al., ICML 2021
- [COLMAP](https://colmap.github.io/) — Structure-from-Motion and Multi-View Stereo

---

## 中文指南

### 项目概述

本项目的最初目标是构建一套完整的货架陈列合规检测系统——通过将检测到的商品三维位置与标准货架图（planogram）对比，判断摆放是否合规。

**实际完成并可运行的部分：**

从原始视频到货架商品 3D 语义分割的完整 pipeline，分为两大模块：

- **三维重建**：多视角视频 → FFmpeg 抽帧 → COLMAP SfM 相机位姿估计 → 3D 高斯泼溅（3DGS）重建 → SIBR / Open3D 可视化
- **3D 语义分割（SAGA）**：在已训练的 3DGS 模型上，用 SAM 自动分割各帧物体 → 提取 CLIP 特征 → 训练 3D 对比特征场 → 文字查询定位商品并输出 3D 包围盒

**未能完成的部分：**

货架合规评分模块（`src/evaluation/planogram_evaluator.py`）——包含 ΔD 位置偏差、3D IoU、PCR 合规率、WCI 加权合规指数——已独立设计并实现，但**从未被接入到 pipeline 中**。它目前仅作为理论框架存在。要真正使用它，前提是 3D 语义分割能输出可靠的包围盒，而 SAGA 在本项目中未能做到这一点（见局限性部分）。

整个流程由一个 Jupyter Notebook 统一控制：`notebooks/pipeline_control_center.ipynb`，参数集中在 `configs/pipeline.yaml` 中管理。

---

### 系统流程

```
原始视频
    │
    ▼  [Section 2] FFmpeg 抽帧
多视角 JPEG 图像
    │
    ▼  [Section 3] COLMAP SfM 相机位姿估计
稀疏点云 + 相机位姿
    │
    ▼  [Section 4] 3D Gaussian Splatting 三维重建
3DGS 高斯点云模型 (.ply)
    │
    ├──▶  [Section 5–6] SIBR Viewer (Docker) / Open3D 交互可视化
    │
    └──▶  [Section 9] SAGA 3D 语义分割
                │
                ├── 9.6.1  创建缩小版图像（×downsample）
                ├── 9.6.2  SAM 自动分割每帧物体 → sam_masks/
                ├── 9.6.2  CLIP 提取每个 mask 的语义特征 → clip_features/
                ├── 9.6.3  估算 mask 3D 物理尺度 → mask_scales/
                ├── 9.6.4  训练 3D 对比特征场（10–40 分钟）
                └── 9.7    文字/点击查询 → 3D 包围盒
                                │
                                ▼
                    ⚠ 效果不理想：点云散布广泛，
                      包围盒精度不足（见局限性）
                                │
                                ▼
                    [ 货架合规评分 ]
                    [ ΔD / 3D IoU / PCR / WCI ]
                    [ ← 未完成，仅有框架代码  ]
```

---

### 实际效果的诚实说明

**三维重建**：效果良好。货架上的商品重建质量不错。但大量多角度拍摄时各帧曝光差异不可避免，3DGS 会把光度残差烘焙为空间中的浮云杂影，干扰后续分割。

**SAGA — 2D CLIP 文字查询**：效果出色。输入"coca cola can"或"tissue box"等词语，可在 2D 图像上无需任何标注地高亮出正确区域。

**SAGA — 3D 分割**：效果差。3D 对比特征训练只以 SAM mask 为监督（学几何一致性），CLIP 语义特征只在查询时介入，两条线路完全分离。这导致 3D 语义边界模糊，叠加上重建产生的杂云，特征分布本就嘈杂。最终选出的点云散布广泛，并非紧密聚集在目标物体周围，包围盒不可靠。

---

### 仓库结构

```
ME6402-3D-Autonomous-Retail/
│
├── notebooks/
│   └── pipeline_control_center.ipynb   ← 所有操作的主入口
│
├── configs/
│   ├── pipeline.yaml                   ← 所有参数（改这里，重跑 Cell 1 生效）
│   ├── pinned_packages.txt             ← SAGA 兼容性版本锁定
│   └── reconstruction/
│       └── scene_example.env
│
├── src/
│   ├── pipeline/                       ← 核心 pipeline 模块（Notebook import）
│   │   ├── utils.py                    ← 项目根目录检测、配置加载、日志
│   │   ├── video.py                    ← FFmpeg 视频抽帧
│   │   ├── colmap.py                   ← COLMAP SfM + 内参解析 + 路径自动修正
│   │   ├── training.py                 ← 3DGS 训练启动 + OOM 自动降档重试
│   │   ├── viewer.py                   ← Open3D 查看 + SIBR Docker 启动
│   │   └── saga.py                     ← SAGA 全流程集成（六步封装）
│   │
│   └── evaluation/                     ← 合规评分（已设计，未接入 pipeline）
│       ├── planogram_evaluator.py      ← ΔD、3D IoU、PCR、WCI，仅独立运行
│       └── metrics_demo.py             ← 最小可运行示例
│
├── scripts/
│   ├── setup_saga.sh                   ← 克隆 SAGA + 编译 CUDA 扩展
│   ├── quick_env_check.sh
│   ├── run_3dgs_demo_quick.sh
│   └── reconstruction/
│       ├── setup_3dgs.sh               ← 克隆 gaussian-splatting（锁定 commit）
│       ├── check_3dgs_env.sh
│       ├── run_colmap.sh
│       ├── run_3dgs_train.sh
│       ├── run_3dgs_demo.sh
│       ├── build_sibr_docker_image.sh
│       └── run_sibr_in_docker.sh
│
└── requirements.txt
```

---

### 环境配置

**系统要求**
- Ubuntu 20.04 / 22.04，x86\_64
- NVIDIA GPU，显存 ≥ 8 GB（SAGA SAM 步骤建议 ≥ 16 GB）
- CUDA 11.8（通过 Conda 管理，无需系统级全局安装）
- 系统安装 `colmap`（`sudo apt install colmap`）
- Docker（可选，用于 SIBR Viewer）

**Step 1 — 克隆本仓库**
```bash
git clone <this-repo-url>
cd ME6402-3D-Autonomous-Retail
```

**Step 2 — 解压实验数据压缩包**

将 `ME6402_data.tar.gz` 放到项目根目录，然后执行：

```bash
cd ME6402-3D-Autonomous-Retail
tar xzf ME6402_data.tar.gz
```

压缩包会直接解压到项目根目录，包含以下内容：

```
dependencies/sam_ckpt/sam_vit_h_4b8939.pth       ← SAM 权重（~2.5 GB）
data/colmap_workspace/custom_scene_01/             ← 场景图像 + COLMAP 输出 + CLIP 特征
outputs/3dgs_custom_scene_01_5000iter/             ← 已训练的 3DGS 模型（可直接用于 SAGA）
outputs/3dgs_team_data2_30000iter/                 ← 高质量模型，用于 SAGA 效果演示
```

> 第三方源码仓库（`gaussian-splatting`、`SAGA`）**不在**压缩包中，由 Step 3–4 的脚本自动克隆。

**Step 3 — 创建 3DGS Conda 环境**
```bash
conda create -n gaussian_splatting python=3.10 -y
conda activate gaussian_splatting
conda install pytorch==2.1.2 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 克隆上游仓库并切换到锁定版本（commit 54c035f）
bash scripts/reconstruction/setup_3dgs.sh

# 编译 CUDA 扩展
source $HOME/miniconda3/bin/activate gaussian_splatting
pip install --no-build-isolation -e third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install --no-build-isolation -e third_party/gaussian-splatting/submodules/simple-knn
pip install --no-build-isolation -e third_party/gaussian-splatting/submodules/fused-ssim

bash scripts/reconstruction/check_3dgs_env.sh     # 验证
```

**Step 4 — 安装 SAGA 依赖**
```bash
source $HOME/miniconda3/bin/activate gaussian_splatting
bash scripts/setup_saga.sh   # 克隆 SAGA（锁定 commit 4acdaa6）并编译 CUDA 子模块
```

编译约需 5–15 分钟，依赖 CUDA 11.8 的 `nvcc`。

**Step 5 — 安装 Python 基础依赖**
```bash
pip install -r requirements.txt
pip install -r configs/pinned_packages.txt   # SAGA 与 numpy 版本锁定
```

**Step 6 — （可选）构建 SIBR Docker 镜像**
```bash
bash scripts/reconstruction/build_sibr_docker_image.sh
```

---

### 快速开始

**启动 Notebook（推荐）**
```bash
source $HOME/miniconda3/bin/activate gaussian_splatting
jupyter notebook notebooks/pipeline_control_center.ipynb
```

| Section | 内容 | 说明 |
|---------|------|------|
| Cell 1 | 初始化——加载所有模块和配置 | 每次打开必须先运行 |
| Section 1 | 环境检查（PyTorch / CUDA / COLMAP） | |
| Section 2 | 视频抽帧（FFmpeg） | 有自有视频时使用 |
| Section 3 | COLMAP 相机标定 | 自有数据必须运行 |
| Section 4 | 3DGS 训练（300 iter 快速预览 → 30 000 iter 高质量） | 核心 |
| Section 5–6 | SIBR / Open3D 交互查看 | |
| Section 7 | 一键完整流程 | |
| Section 9 | SAGA 3D 语义分割 + 文字查询 | |

**直接复现提供的场景（`custom_scene_01`）**：

压缩包已包含 COLMAP 处理后图像和 5 000 iter 的 3DGS 模型，可直接从 **Section 5**（查看重建）或 **Section 9**（SAGA 分割）开始，无需重新运行 COLMAP 和训练。

---

### 配置参数

编辑 `configs/pipeline.yaml`，在 Notebook 中重新运行 Cell 1 即可生效，无需重启 kernel。

```yaml
video:
  enabled: false             # 改为 true 可先从视频抽帧
  input_video: ~/Videos/recording.mp4
  scene_name: custom_scene_01
  fps: 2                     # 建议 1–5

colmap:
  workspace: data/colmap_workspace
  camera_model: PINHOLE

training:
  iterations: 30000          # 300–5000 快速预览；30000 高质量
  resolution: 2              # 1=原始；OOM 时自动降至 4
  output_dir: outputs

saga:
  sam_checkpoint: dependencies/sam_ckpt/sam_vit_h_4b8939.pth
  downsample: 4              # RTX 4060 8GB 建议 4；≥16GB 可用 2
  image_root: "data/colmap_workspace/custom_scene_01/dense"
  model_path:  "outputs/3dgs_custom_scene_01_5000iter"
```

---

### 需外部获取的组件

**第三方源码仓库（不打包，由脚本自动克隆）**

| 仓库 | 上游地址 | 锁定 commit | 获取方式 |
|------|----------|-------------|----------|
| 3D Gaussian Splatting | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | `54c035f` | `bash scripts/reconstruction/setup_3dgs.sh` |
| SAGA | [Jumpat/SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians) | `4acdaa6` | `bash scripts/setup_saga.sh` |

验证环境：Python 3.10.20 · PyTorch 2.1.2 · CUDA 11.8 · Ubuntu 20.04 · x86\_64

> SIBR Viewer（1.6 GB C++ 二进制）完全通过 Docker 运行，无需本地编译。

**压缩包中的文件（实验数据和训练产出）**

| 组件 | 放置路径 | 大小 |
|------|----------|------|
| SAM ViT-H 权重 | `dependencies/sam_ckpt/sam_vit_h_4b8939.pth` | ~2.5 GB |
| 自定义场景数据 | `data/colmap_workspace/custom_scene_01/` | ~200 MB |
| 5k iter 训练模型 | `outputs/3dgs_custom_scene_01_5000iter/` | ~300 MB |
| 30k iter 高质量模型 | `outputs/3dgs_team_data2_30000iter/` | ~800 MB |

如需自行下载 SAM 权重：
```bash
mkdir -p dependencies/sam_ckpt
wget -P dependencies/sam_ckpt \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

### 局限性与未来方向

**3D 语义分割效果差的根本原因：**

1. **重建杂云**：大量拍摄时各帧曝光差异不可避免，3DGS 将光度残差烘焙为空间浮云，导致语义特征分布本就嘈杂。
2. **CLIP 与 3D 训练完全脱节**：SAGA 的对比特征训练只以 SAM mask 几何一致性为监督，CLIP 语义特征仅在查询时才介入，过晚介入无法修正错误的聚类边界。

**未来方向：**

- 引入 **Bilateral Grid** 曝光归一化，从源头减少杂云。
- 考虑 **LERF**（Language Embedded Radiance Fields）替代 SAGA——在 3D 训练阶段直接蒸馏 CLIP 特征，而非后期叠加。
- 一旦 3D 包围盒足够可靠，将其接入 `src/evaluation/planogram_evaluator.py`，完成最初的合规检测目标。

---

### 参考文献

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al., SIGGRAPH 2023
- [Segment Anything](https://segment-anything.com/) — Kirillov et al., ICCV 2023
- [SAGA: Segment Any 3D Gaussians](https://jumpat.github.io/SAGA/) — Cen et al., AAAI 2025
- [CLIP](https://openai.com/research/clip) — Radford et al., ICML 2021
- [COLMAP](https://colmap.github.io/) — Structure-from-Motion and Multi-View Stereo
