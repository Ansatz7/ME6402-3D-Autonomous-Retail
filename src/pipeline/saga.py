"""
SAGA（Segment Any 3D Gaussians）集成模块
AAAI 2025 — https://github.com/Jumpat/SegAnyGAussians

工作流程（在已训练好的 3DGS 模型上直接运行，无需重新训练）：

  Step 1: create_downsampled_images — 创建缩小图像目录（downsample≠1 时需要）
  Step 2: extract_sam_masks         — SAM 自动分割，生成每帧 mask（sam_masks/*.pt）
  Step 3: extract_sam_features      — 从图像+mask 提取 CLIP 特征（clip_features/*.pt）
  Step 4: train_saga_features       — 在冻结的 3DGS 上训练对比特征（~10-40 分钟）
  Step 5: compute_scales            — 估算每个 mask 的 3D 物理尺度（可选）
  Step 6: open_saga_notebook        — 打开 SAGA 交互 Notebook（文字/点击分割）

使用方式：
    from src.pipeline.saga import *
    saga_cfg = load_saga_config()
    extract_sam_features(saga_cfg)
    ...
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .utils import PROJECT_ROOT, OUTPUT_DIR, logger

SAGA_DIR = PROJECT_ROOT / "third_party" / "SAGA"
SAM_DEFAULT_CKPT = PROJECT_ROOT / "dependencies" / "sam_ckpt" / "sam_vit_h_4b8939.pth"
SAM_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


# ──────────────────────────────────────────────────────────────
# 安装检查
# ──────────────────────────────────────────────────────────────

def check_saga_ready() -> bool:
    """检查 SAGA 依赖是否全部就绪，打印状态报告。"""
    print("=" * 55)
    print("SAGA 环境检查")
    print("=" * 55)
    ok = True

    # SAGA 仓库
    saga_ok = SAGA_DIR.exists() and (SAGA_DIR / "train_contrastive_feature.py").exists()
    _status(saga_ok, f"SAGA 仓库  ({SAGA_DIR})")
    if not saga_ok:
        print("   → 运行：bash scripts/setup_saga.sh")
        ok = False

    # 关键子模块
    contrastive = SAGA_DIR / "submodules" / "diff-gaussian-rasterization_contrastive_f"
    _status(contrastive.exists(), "子模块 diff-gaussian-rasterization_contrastive_f")
    if not contrastive.exists():
        ok = False

    # Python 包
    for pkg, label in [
        ("segment_anything",  "segment-anything (SAM)"),
        ("kmeans_pytorch",    "kmeans_pytorch"),
        ("open_clip",         "open_clip_torch"),
        ("hdbscan",           "hdbscan"),
    ]:
        installed = _try_import(pkg)
        _status(installed, label)
        if not installed:
            ok = False

    # SAM checkpoint
    ckpt_ok = SAM_DEFAULT_CKPT.exists()
    _status(ckpt_ok, f"SAM ViT-H checkpoint  ({SAM_DEFAULT_CKPT.name})")
    if not ckpt_ok:
        print(f"   → 下载命令（见下方 download_sam_checkpoint()）")
        ok = False

    print()
    if ok:
        print("✅ SAGA 环境就绪")
    else:
        print("⚠️  部分依赖缺失，请先运行：bash scripts/setup_saga.sh")
    return ok


def download_sam_checkpoint(dest: Optional[Path] = None) -> bool:
    """下载 SAM ViT-H checkpoint（~2.5 GB）。"""
    ckpt = Path(dest) if dest else SAM_DEFAULT_CKPT
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    if ckpt.exists():
        print(f"✓ SAM checkpoint 已存在：{ckpt}")
        return True

    print(f"⬇️  正在下载 SAM ViT-H checkpoint → {ckpt}")
    print(f"   URL：{SAM_DOWNLOAD_URL}")
    print(f"   大小：约 2.5 GB，请耐心等待...\n")

    cmd = ["wget", "-q", "--show-progress", "-O", str(ckpt), SAM_DOWNLOAD_URL]
    if not shutil.which("wget"):
        cmd = ["curl", "-L", "-o", str(ckpt), SAM_DOWNLOAD_URL]

    result = subprocess.run(cmd)
    if result.returncode == 0 and ckpt.exists():
        mb = ckpt.stat().st_size / 1024 / 1024
        print(f"\n✓ 下载完成（{mb:.0f} MB）：{ckpt}")
        return True
    else:
        print("\n✗ 下载失败，请手动下载：")
        print(f"   wget -O {ckpt} {SAM_DOWNLOAD_URL}")
        return False


# ──────────────────────────────────────────────────────────────
# Step 1: 提取 SAM 图像特征
# ──────────────────────────────────────────────────────────────

def extract_sam_features(saga_cfg: dict) -> bool:
    """
    Step 1（mask 提取后）：从图像 + SAM mask 中提取 CLIP 特征。

    依赖：sam_masks/ 目录须已存在（先运行 extract_sam_masks()）
    输出：<image_root>/clip_features/*.pt
    VRAM：~4 GB
    """
    image_root, _ = _get_paths(saga_cfg)

    script = SAGA_DIR / "get_clip_features.py"
    if not _check_script(script):
        return False
    if not _check_images_dir(image_root):
        return False

    sam_masks_dir = image_root / "sam_masks"
    if not sam_masks_dir.exists():
        print(f"✗ sam_masks/ 目录不存在：{sam_masks_dir}")
        print("  请先运行 extract_sam_masks(saga_cfg)")
        return False

    clip_features_dir = image_root / "clip_features"
    if clip_features_dir.exists() and len(list(clip_features_dir.glob("*.pt"))) > 0:
        n = len(list(clip_features_dir.glob("*.pt")))
        print(f"✓ CLIP 特征已存在（{n} 个），跳过。如需重新提取，删除 {clip_features_dir}/")
        return True

    cmd = [
        sys.executable,
        str(script),
        "--image_root", str(image_root),
    ]
    print(f"\n[SAGA 9.6.2] 提取 CLIP 语义特征（从 SAM mask）...")
    print(f"   图像目录：{image_root / 'images'}")
    print(f"   输出目录：{clip_features_dir}")
    return _run_saga_script(cmd, SAGA_DIR)


# ──────────────────────────────────────────────────────────────
# CLIP 特征质量验证（可选）
# ──────────────────────────────────────────────────────────────

def validate_clip_features(saga_cfg: dict, queries: list[str] | None = None) -> None:
    """
    验证 CLIP 特征提取结果是否有效。

    做三件事：
      1. 统计 clip_features/*.pt 文件数量和每文件特征维度
      2. 对一批查询词，计算与所有 mask 特征的余弦相似度，打印 top-5 分数
      3. 打印分数分布直方图（用 ASCII 显示）

    使用示例（Notebook 中）：
        validate_clip_features(saga_cfg, queries=["mouse", "bottle", "shelf"])
    """
    import os
    import torch
    import numpy as np

    image_root, _ = _get_paths(saga_cfg)
    clip_feat_dir = image_root / "clip_features"

    if not clip_feat_dir.exists():
        print("✗ clip_features/ 不存在，请先运行 extract_sam_features(saga_cfg)")
        return

    pt_files = sorted(clip_feat_dir.glob("*.pt"))
    if not pt_files:
        print("✗ clip_features/ 为空，请先运行 extract_sam_features(saga_cfg)")
        return

    # ── 1. 文件统计 ───────────────────────────────────────────────
    print("=" * 55)
    print("CLIP 特征文件统计")
    print("=" * 55)
    print(f"  文件数：{len(pt_files)}")

    sample = torch.load(pt_files[0], map_location="cpu")
    print(f"  特征维度：{sample.shape}  (N_mask × 512)")
    print(f"  示例文件：{pt_files[0].name}")

    total_masks = 0
    all_features = []
    for f in pt_files:
        feat = torch.load(f, map_location="cpu")
        total_masks += feat.shape[0]
        all_features.append(feat)
    all_features = torch.cat(all_features, dim=0).float()  # (总mask数, 512)
    print(f"  总 mask 数：{total_masks}")

    # 检查特征是否已归一化
    norms = all_features.norm(dim=1)
    print(f"  特征范数均值：{norms.mean():.4f}（≈1.0 表示已归一化）")

    # ── 2. 文字查询相似度 ─────────────────────────────────────────
    if queries is None:
        queries = ["mouse", "bottle", "book", "cup", "shelf"]

    # 尝试在 SAGA 目录下导入 open_clip
    try:
        os.chdir(str(SAGA_DIR))
        import sys as _sys
        if str(SAGA_DIR) not in _sys.path:
            _sys.path.insert(0, str(SAGA_DIR))
        from clip_utils.clip_utils import load_clip
        clip_model = load_clip()
        clip_model.eval()

        # 归一化图像特征
        norm_features = torch.nn.functional.normalize(all_features, dim=-1).cuda()

        print()
        print("=" * 55)
        print("文字查询相似度验证（top-5 分数）")
        print("=" * 55)
        for query in queries:
            with torch.no_grad():
                clip_model.set_positives([query])
                # get_relevancy 返回 (N, 2)，[:,0] 是正类概率
                scores = clip_model.get_relevancy(norm_features, 0)[:, 0]
            scores_cpu = scores.cpu().numpy()
            top5 = np.sort(scores_cpu)[::-1][:5]
            mean_s = scores_cpu.mean()
            print(f"  '{query}': top5={[f'{s:.3f}' for s in top5]}  mean={mean_s:.3f}")

            # ASCII 直方图（10 个 bin，范围 0~1）
            bins = np.zeros(10, dtype=int)
            for s in scores_cpu:
                idx = min(int(s * 10), 9)
                bins[idx] += 1
            bar = "  分布: |"
            for b in bins:
                bar += "#" * min(b // max(1, total_masks // 50), 8) + " "
            print(bar + f"|  (0→1)")
        norm_features = norm_features.cpu()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n  (跳过文字查询验证：{e})")

    print()
    print("✓ 验证完成。分数 top5 越高（越接近1.0）代表 CLIP 特征识别能力越强。")


# ──────────────────────────────────────────────────────────────
# Step 2: 创建缩小版图像目录（mask 提取用）
# ──────────────────────────────────────────────────────────────

def create_downsampled_images(saga_cfg: dict) -> bool:
    """
    Step 2（前置）：生成 images_<N>/ 目录供 mask 提取使用。

    extract_segment_everything_masks.py 在 downsample≠1 时
    读取 images_4/ 而不是 images/，此函数自动创建。
    """
    image_root, _ = _get_paths(saga_cfg)
    downsample = int(saga_cfg.get("downsample", 4))

    if downsample == 1:
        return True  # 直接用 images/，无需创建

    src_dir  = image_root / "images"
    dest_dir = image_root / f"images_{downsample}"

    if dest_dir.exists() and len(list(dest_dir.glob("*"))) > 0:
        print(f"✓ {dest_dir.name}/ 已存在，跳过创建。")
        return True

    if not src_dir.exists():
        print(f"✗ 图像目录不存在：{src_dir}")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in src_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not imgs:
        print(f"✗ {src_dir} 下没有图像文件")
        return False

    try:
        from PIL import Image as PilImage
    except ImportError:
        print("✗ Pillow 未安装（pip install Pillow）")
        return False

    print(f"\n[SAGA 9.6.1 前置] 创建 ×{downsample} 缩小图像目录...")
    for p in imgs:
        img = PilImage.open(p)
        w, h = img.width // downsample, img.height // downsample
        img = img.resize((w, h), PilImage.LANCZOS)
        img.save(dest_dir / p.name)

    print(f"✓ 已创建 {len(imgs)} 张缩小图像 → {dest_dir}")
    return True


# ──────────────────────────────────────────────────────────────
# Step 3: 提取 SAM 自动分割 mask
# ──────────────────────────────────────────────────────────────

def extract_sam_masks(saga_cfg: dict) -> bool:
    """
    Step 3：SAM 自动分割，生成每帧 mask。

    输出：<image_root>/sam_masks/*.pt
    VRAM：~7-8 GB（推荐 RTX 2080 Ti）
    """
    image_root, sam_ckpt = _get_paths(saga_cfg)
    downsample = int(saga_cfg.get("downsample", 4))

    script = SAGA_DIR / "extract_segment_everything_masks.py"
    if not _check_script(script):
        return False
    if not _check_sam_ckpt(sam_ckpt):
        return False

    masks_dir = image_root / "sam_masks"
    if masks_dir.exists() and len(list(masks_dir.glob("*.pt"))) > 0:
        n = len(list(masks_dir.glob("*.pt")))
        print(f"✓ SAM mask 已存在（{n} 个），跳过。如需重新生成，删除 {masks_dir}/")
        return True

    # 确保缩小版图像目录存在
    if not create_downsampled_images(saga_cfg):
        return False

    cmd = [
        sys.executable,
        str(script),
        "--image_root", str(image_root),
        "--sam_checkpoint_path", str(sam_ckpt),
        "--sam_arch", saga_cfg.get("sam_arch", "vit_h"),
        "--downsample", str(downsample),
    ]
    print(f"\n[SAGA 9.6.1] 提取 SAM 自动分割 mask...")
    print(f"   图像目录：{image_root / f'images_{downsample}'}")
    print(f"   输出目录：{masks_dir}")
    return _run_saga_script(cmd, SAGA_DIR)


# ──────────────────────────────────────────────────────────────
# Step 4: 训练对比特征
# ──────────────────────────────────────────────────────────────

def train_saga_features(saga_cfg: dict) -> bool:
    """
    Step 4：在冻结的 3DGS 模型上训练对比特征（~10-40 分钟）。

    需要：<image_root>/features/ 和 <image_root>/sam_masks/ 已存在。
    输出：<model_path>/point_cloud/iteration_<N>/contrastive_feature_point_cloud.ply
          <model_path>/sam_proj.pt
    """
    image_root, _ = _get_paths(saga_cfg)
    model_path = _get_model_path(saga_cfg)
    if model_path is None:
        return False

    script = SAGA_DIR / "train_contrastive_feature.py"
    if not _check_script(script):
        return False

    # 检查前置步骤输出
    if not (image_root / "clip_features").exists():
        print("✗ 缺少 clip_features/，请先运行 Step 1：extract_sam_features()")
        return False
    if not (image_root / "sam_masks").exists():
        print("✗ 缺少 sam_masks/，请先运行 Step 3：extract_sam_masks()")
        return False

    cmd = [
        sys.executable,
        str(script),
        "-m", str(model_path),
        "-s", str(image_root),   # 覆盖 cfg_args 中的 source_path（支持本地路径）
        "--iteration", "-1",
        "--num_sampled_rays", "1000",    # 必须 >0，官方示例值（属性名含 s）
        "--iterations", "10000",
        "--feature_lr", "0.0025",
        "--feature_dim", "32",           # cfg_args 无此字段，必须显式传入
        "--allow_principle_point_shift", # cfg_args 无此字段，SAGA Scene 初始化需要
    ]
    print(f"\n[SAGA 9.6.4] 训练对比特征...")
    print(f"   模型路径：{model_path}")
    print(f"   场景数据：{image_root}")
    print(f"   预计时间：10~40 分钟")
    return _run_saga_script(cmd, SAGA_DIR)


# ──────────────────────────────────────────────────────────────
# Step 5: 估算 mask 3D 尺度（可选，提升分割精度）
# ──────────────────────────────────────────────────────────────

def compute_scales(saga_cfg: dict) -> bool:
    """
    Step 5（可选）：估算每个 mask 在 3D 空间中的物理尺度。

    用于 prompt_segmenting.ipynb 中的 scale-aware 分割。
    输出：<image_root>/mask_scales/*.pt
    """
    image_root, _ = _get_paths(saga_cfg)
    model_path = _get_model_path(saga_cfg)
    if model_path is None:
        return False

    script = SAGA_DIR / "get_scale.py"
    if not _check_script(script):
        return False

    cmd = [
        sys.executable,
        str(script),
        "-m", str(model_path),
        "-s", str(image_root),   # 覆盖 cfg_args 中的 source_path（支持本地路径）
        "--image_root", str(image_root),
        "--iteration", "-1",
    ]
    print(f"\n[SAGA 9.6.3] 估算 mask 3D 物理尺度...")
    return _run_saga_script(cmd, SAGA_DIR)


# ──────────────────────────────────────────────────────────────
# Step 6: 启动 SAGA 交互 Notebook
# ──────────────────────────────────────────────────────────────

def open_saga_gui(saga_cfg: dict, gpu: int = 0) -> None:
    """
    Step 6A：启动 SAGA 交互式 GUI（saga_gui.py）。

    操作方式：
      - 左键拖动：旋转视角
      - 右键点击物体：放置分割点（需先勾选 clickmode）
      - segment3d：执行 3D 分割
      - save as：保存分割结果到 segmentation_res/<name>.pt
    """
    gui_script = SAGA_DIR / "saga_gui.py"
    if not gui_script.exists():
        print(f"✗ 未找到 saga_gui.py：{gui_script}")
        return

    model_path = _get_model_path(saga_cfg)
    if not model_path:
        return

    # 自动检测 contrastive feature 的 iteration
    feat_iters = sorted(
        (p.parent for p in model_path.glob("point_cloud/*/contrastive_feature_point_cloud.ply")),
        key=lambda p: int(p.name.replace("iteration_", ""))
    )
    scene_iters = sorted(
        (p.parent for p in model_path.glob("point_cloud/*/point_cloud.ply")),
        key=lambda p: int(p.name.replace("iteration_", ""))
    )
    f_iter = int(feat_iters[-1].name.replace("iteration_", "")) if feat_iters else 10000
    s_iter = int(scene_iters[-1].name.replace("iteration_", "")) if scene_iters else 5000

    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"\n[SAGA GUI] 启动交互式分割界面...")
    print(f"   模型：{model_path}")
    print(f"   场景 iter：{s_iter}，特征 iter：{f_iter}，GPU：{gpu}")
    print(f"   操作：勾选 clickmode → 右键点击物体 → segment3d → save as")

    subprocess.Popen(
        [sys.executable, str(gui_script),
         "--model_path", str(model_path),
         "-f", str(f_iter),
         "-s", str(s_iter)],
        cwd=str(SAGA_DIR),
        env=env,
    )
    print("✓ GUI 已启动，等待窗口弹出（约 10-20 秒）")


def open_bbox_viewer(
    bboxes: "dict | list",
    saga_cfg: dict,
    gpu: int = 1,
) -> None:
    """
    启动 3D BBox 可交互查看器（bbox_viewer.py）。

    在高斯泼溅真实感渲染上叠加 3D 识别框 + 标签，支持旋转/平移/缩放。
    saga_gui.py 独立运行，互不影响。

    Args:
        bboxes:    query_by_text() 返回的 dict，或多个 bbox 的 list
        saga_cfg:  SAGA 配置字典
        gpu:       GPU 编号（默认 1 = RTX 2080 Ti）
    """
    import json, tempfile

    viewer_script = SAGA_DIR / "bbox_viewer.py"
    if not viewer_script.exists():
        print(f"✗ 未找到 bbox_viewer.py：{viewer_script}")
        return

    model_path = _get_model_path(saga_cfg)
    if not model_path:
        return

    # 支持单个 dict 或 list
    bbox_list = bboxes if isinstance(bboxes, list) else [bboxes]

    # 写临时 JSON
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(bbox_list, tmp)
    tmp.close()

    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    labels = [b["label"] for b in bbox_list]
    print(f"\n[BBox Viewer] 启动 3D 识别框查看器...")
    print(f"   模型：{model_path}")
    print(f"   识别物体：{labels}，GPU：{gpu}")
    print(f"   操作：左键旋转 / 右键平移 / 滚轮缩放")

    subprocess.Popen(
        [sys.executable, str(viewer_script),
         "-m", str(model_path),
         "--bbox_json", tmp.name],
        cwd=str(SAGA_DIR),
        env=env,
    )
    print("✓ 查看器已启动，等待窗口弹出（约 10-20 秒）")


def open_saga_notebook(saga_cfg: dict) -> None:
    """
    Step 6B：在 SAGA 目录下启动 prompt_segmenting.ipynb（文字查询）。

    支持：文字 CLIP 查询（输入 "mouse" 自动识别）→ 输出 3D Bounding Box
    """
    nb = SAGA_DIR / "prompt_segmenting.ipynb"
    if not nb.exists():
        print(f"✗ 未找到 SAGA Notebook：{nb}")
        print(f"  请先运行：bash scripts/setup_saga.sh")
        return

    model_path = _get_model_path(saga_cfg)
    image_root, _ = _get_paths(saga_cfg)

    print("\n[SAGA Step 6B] 启动文字查询 Notebook")
    print(f"   Notebook：{nb}")
    print(f"   打开后修改 notebook 顶部变量：")
    print(f"     model_path = \"{model_path}\"")
    print(f"     image_root = \"{image_root}\"")
    print()

    subprocess.Popen(
        ["jupyter", "lab", "--no-browser", "--port=8888"],
        cwd=str(SAGA_DIR),
    )
    print("✓ JupyterLab 已启动，复制终端链接到浏览器，打开 prompt_segmenting.ipynb")


# ──────────────────────────────────────────────────────────────
# Step 7: 文字查询 → 自动 3D Bounding Box（无需手动开 Notebook）
# ──────────────────────────────────────────────────────────────

def query_by_text(
    text: str,
    saga_cfg: dict,
    save_name: str | None = None,
    gpu: int = 0,
    score_threshold: float = 0.0,
) -> "dict | None":
    """
    用文字查询在 3D 高斯场景中定位物体，输出 3D bounding box。

    原理：加载已训练的 SAGA 对比特征模型，对每张训练相机渲染特征图并
    收集 CLIP 图像特征（clip_features），再用 CLIP 文字编码计算相似度，
    找到最相关的 Gaussian 集合，最终输出 3D AABB bbox。

    Args:
        text:             查询文字，如 "mouse", "bottle", "yogurt"
        saga_cfg:         SAGA 配置字典（含 model_path, image_root 等）
        save_name:        mask 保存文件名（不含 .pt），默认用 text
        gpu:              使用的 GPU 编号
        score_threshold:  相似度阈值，0.0-1.0（0 = 自动用最优 cluster）

    Returns:
        bbox dict: {"label": text, "center": [...], "size": [...],
                    "bbox_min": [...], "bbox_max": [...], "n_gaussians": N}
        None if failed
    """
    import os

    # ── 0. CUDA device 必须在 import torch 之前设置 ──────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # ── 1. 解析路径 ───────────────────────────────────────────────
    image_root, _ = _get_paths(saga_cfg)
    model_path = _get_model_path(saga_cfg)
    if model_path is None:
        return None

    # 检查必要的前置输出目录
    clip_feat_dir = image_root / "clip_features"
    if not clip_feat_dir.exists():
        print(f"✗ clip_features/ 目录不存在：{clip_feat_dir}")
        print("  请先运行 extract_sam_features(saga_cfg)")
        return None

    # ── 2. 切换工作目录并注入 SAGA 到 sys.path ───────────────────
    import sys
    original_cwd = os.getcwd()
    original_path = sys.path[:]
    try:
        os.chdir(str(SAGA_DIR))
        if str(SAGA_DIR) not in sys.path:
            sys.path.insert(0, str(SAGA_DIR))

        return _query_by_text_inner(
            text=text,
            model_path=model_path,
            image_root=image_root,
            save_name=save_name or text.replace(" ", "_"),
            score_threshold=score_threshold,
        )
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path


def _query_by_text_inner(
    text: str,
    model_path: "Path",
    image_root: "Path",
    save_name: str,
    score_threshold: float,
) -> "dict | None":
    """
    实际执行文字查询的内部函数（已在 SAGA_DIR 下运行，sys.path 已注入）。
    对应 prompt_segmenting.ipynb Cell 1~54 的完整逻辑。
    """
    import os
    import torch
    import numpy as np
    from copy import deepcopy
    from argparse import ArgumentParser, Namespace

    # SAGA 模块（需要在 SAGA_DIR 下才能正确导入）
    from arguments import ModelParams, PipelineParams
    from scene import Scene, GaussianModel, FeatureGaussianModel
    from gaussian_renderer import render_contrastive_feature
    from sklearn.preprocessing import QuantileTransformer
    import hdbscan as hdbscan_module

    FEATURE_DIM = 32

    # ── A. 找到 contrastive feature 的 iteration ─────────────────
    feat_plys = sorted(
        model_path.glob("point_cloud/iteration_*/contrastive_feature_point_cloud.ply"),
        key=lambda p: int(p.parent.name.replace("iteration_", ""))
    )
    if not feat_plys:
        print(f"✗ 未找到 contrastive_feature_point_cloud.ply，请先运行 train_saga_features()")
        return None
    feat_iter = int(feat_plys[-1].parent.name.replace("iteration_", ""))
    print(f"   使用 contrastive feature iter={feat_iter}")

    scale_gate_path = str(model_path / f"point_cloud/iteration_{feat_iter}/scale_gate.pt")

    # ── B. 加载 scale_gate ────────────────────────────────────────
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, 32, bias=True),
        torch.nn.Sigmoid()
    )
    scale_gate.load_state_dict(torch.load(scale_gate_path, map_location="cpu"))
    scale_gate = scale_gate.cuda()

    # ── C. 解析场景配置，加载 GaussianModel ──────────────────────
    def get_combined_args(parser: ArgumentParser, mp: str, sp: str = None) -> Namespace:
        cmdlne_string = ["--model_path", mp]
        if sp:
            cmdlne_string += ["--source_path", sp]
        args_cmdline = parser.parse_args(cmdlne_string)
        cfgfilepath = os.path.join(mp, "cfg_args")
        cfgfile_string = "Namespace()"
        try:
            with open(cfgfilepath) as f:
                cfgfile_string = f.read()
        except (TypeError, FileNotFoundError):
            pass
        args_cfgfile = eval(cfgfile_string)
        merged = vars(args_cfgfile).copy()
        for k, v in vars(args_cmdline).items():
            if v is not None:
                merged[k] = v
        return Namespace(**merged)

    parser = ArgumentParser(description="query_by_text")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--target", default="scene", type=str)

    args = get_combined_args(parser, str(model_path), str(image_root))

    dataset = model_params.extract(args)
    dataset.need_features = True   # 加载 clip_features/
    dataset.need_masks = True
    dataset.allow_principle_point_shift = False

    scene_gaussians = GaussianModel(dataset.sh_degree)
    feature_gaussians = FeatureGaussianModel(FEATURE_DIM)

    scene = Scene(
        dataset, scene_gaussians, feature_gaussians,
        load_iteration=-1,
        feature_load_iteration=feat_iter,
        shuffle=False,
        mode="eval",
        target="contrastive_feature",
    )

    # ── D. 构建 quantile transformer（基于全部 mask scale）────────
    all_scales = []
    for cam in scene.getTrainCameras():
        all_scales.append(cam.mask_scales)
    all_scales = torch.cat(all_scales)

    qt = QuantileTransformer(output_distribution="uniform")
    qt.fit(all_scales.detach().cpu().numpy().reshape(-1, 1))

    def q_trans(s: torch.Tensor) -> torch.Tensor:
        shape = s.shape
        return torch.tensor(
            qt.transform(s.detach().cpu().numpy().reshape(-1, 1)),
            dtype=torch.float32,
        ).to(s.device).reshape(shape)

    # ── E. 采集 anchor point features（稀疏采样）────────────────
    all_point_features = feature_gaussians.get_point_features  # (N, FEATURE_DIM)
    anchor_mask = torch.rand(all_point_features.shape[0]) > 0.99
    anchor_point_features = all_point_features[anchor_mask]
    print(f"   anchor points: {len(anchor_point_features)}")

    # ── F. 遍历所有训练相机，收集 clip_features 和 seg_features ──
    bg_color = [0.0] * FEATURE_DIM
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    seg_features = []
    clip_features = []
    scales = []
    mask_identifiers = []
    camera_id_mask_id = []

    cameras = scene.getTrainCameras()
    print(f"   训练相机数：{len(cameras)}，开始收集 CLIP 特征...")

    for cam_i, view in enumerate(cameras):
        torch.cuda.empty_cache()
        clip_features.append(view.original_features)

        tmp_view = deepcopy(view)
        tmp_view.feature_height = view.original_image.shape[-2]
        tmp_view.feature_width = view.original_image.shape[-1]

        rendered_feature = render_contrastive_feature(
            tmp_view, feature_gaussians,
            pipeline_params.extract(args), background,
            norm_point_features=True,
        )["render"]
        feature_h, feature_w = rendered_feature.shape[-2:]

        with torch.no_grad():
            # 降采样到 1/4 分辨率（与 notebook 保持一致）
            rf_small = torch.nn.functional.interpolate(
                rendered_feature.unsqueeze(0),
                (feature_h // 4, feature_w // 4),
                mode="bilinear",
            ).squeeze()

            sam_masks = view.original_masks.cuda().unsqueeze(1)
            sam_masks = torch.nn.functional.interpolate(
                sam_masks.float(), (feature_h // 4, feature_w // 4), mode="bilinear"
            )
            # 腐蚀：convolution-based erosion，kernel 3×3，threshold=2
            sam_masks = torch.conv2d(
                sam_masks.float().cpu(),
                torch.full((3, 3), 1.0).view(1, 1, 3, 3).cpu(),
                padding=1,
            )
            sam_masks = (sam_masks >= 2).cuda()

            mask_scales = view.mask_scales.cuda().unsqueeze(-1)
            mask_scales_q = q_trans(mask_scales)
            scale_gates = scale_gate(mask_scales_q)

            # scale-conditioned anchor features: (N_scale, N_anchor, C)
            sca_feat = scale_gates.unsqueeze(1) * anchor_point_features.unsqueeze(0)
            sca_feat = torch.nn.functional.normalize(sca_feat, dim=-1, p=2)

            # scale-conditioned rendered features: (N_scale, C, H, W)
            sc_render = rf_small.unsqueeze(0) * scale_gates.unsqueeze(-1).unsqueeze(-1)
            sc_render = torch.nn.functional.normalize(sc_render, dim=1, p=2)

            # mask-pooled features: (N_mask, C)
            mask_feat = (
                (sam_masks * sc_render).sum(dim=-1).sum(dim=-1)
                / (sam_masks.sum(dim=-1).sum(dim=-1) + 1e-9)
            )
            mask_feat = torch.nn.functional.normalize(mask_feat, dim=-1, p=2)

            mask_id = torch.einsum("nmc,nc->nm", sca_feat, mask_feat) > 0.5

            mask_identifiers.append(mask_id.cpu())
            seg_features.append(mask_feat)
            scales.append(view.mask_scales.cuda().unsqueeze(-1))

            for j in range(len(mask_feat)):
                camera_id_mask_id.append((cam_i, j))

    torch.cuda.empty_cache()

    flattened_mask_features = torch.cat(seg_features, dim=0)
    flattened_clip_features = torch.cat(clip_features, dim=0)
    flattened_clip_features = torch.nn.functional.normalize(
        flattened_clip_features.float(), dim=-1, p=2
    )
    flattened_scales = torch.cat(scales, dim=0)
    flattened_mask_identifiers = torch.cat(mask_identifiers, dim=0).to(torch.float16).cuda()

    print(f"   累计 mask 数：{flattened_mask_features.shape[0]}")

    # ── G. 用 mask_identifiers 构建 Jaccard 距离矩阵，做聚类 ─────
    with torch.no_grad():
        intersection = torch.einsum(
            "mc,nc->mn", flattened_mask_identifiers, flattened_mask_identifiers
        )
        union = (
            flattened_mask_identifiers.sum(dim=-1).unsqueeze(-1)
            + flattened_mask_identifiers.sum(dim=-1).unsqueeze(0)
            - intersection
            + 1e-6
        )
        distance_map = (1 - intersection / union).detach().cpu().numpy().astype(np.float64)

    clusterer = hdbscan_module.HDBSCAN(
        min_cluster_size=30, cluster_selection_epsilon=0.25, metric="precomputed"
    )
    cluster_labels = clusterer.fit_predict(distance_map)
    cluster_labels = torch.from_numpy(cluster_labels).to(
        device=flattened_clip_features.device, dtype=torch.long
    )

    # ── H. 用 CLIP 文字编码对每个 cluster 打分 ───────────────────
    from clip_utils import get_scores_with_template
    from clip_utils.clip_utils import load_clip

    clip_model = load_clip()
    clip_model.eval()

    # 零售货架专用 prompt 模板，比通用 ImageNet 模板对商品识别更准
    retail_template = [
        'a photo of a {}.',
        'a product photo of a {}.',
        'a retail shelf item: {}.',
        'a close-up photo of a {}.',
        'a {} on a store shelf.',
        'a {} product.',
        'a packaged {}.',
        'a {} for sale.',
        'a photo of the {}.',
        'a good photo of a {}.',
    ]
    scores = get_scores_with_template(
        clip_model, flattened_clip_features.cuda(), text, template=retail_template
    ).squeeze()

    # 每个 cluster 取均值分数（cluster_labels 从 -1 开始，-1 = noise）
    unique_clusters = cluster_labels.unique()
    cluster_scores = torch.zeros(len(unique_clusters), device=cluster_labels.device)
    for idx, cid in enumerate(unique_clusters):
        cluster_scores[idx] = scores[cluster_labels == cid].mean()

    # ── I. 选取好 cluster，得到对应的 mask features + scale ──────
    SCORE_THRESH = score_threshold if score_threshold > 0.0 else 0.45
    good_mask = cluster_scores > SCORE_THRESH
    good_indices = torch.where(good_mask)[0]
    if len(good_indices) == 0:
        # 退化：选最高分 cluster
        good_indices = torch.tensor([cluster_scores.argmax()])

    good_clusters = [unique_clusters[i] for i in good_indices]

    clip_query_features = []
    corresponding_scales = []
    for g in good_clusters:
        in_cluster = cluster_labels == g
        cluster_mask_scores = scores[in_cluster]
        best_idx = cluster_mask_scores.argmax()
        feat = torch.nn.functional.normalize(
            flattened_mask_features[in_cluster][best_idx], dim=-1, p=2
        )
        clip_query_features.append(feat)
        corresponding_scales.append(flattened_scales[in_cluster][best_idx].item())

    # ── J. 计算每个 Gaussian 与 query feature 的相似度 ───────────
    point_features = feature_gaussians.get_point_features  # (N, FEATURE_DIM)

    final_similarities = torch.zeros(point_features.shape[0], device="cuda")
    for feat, scale_val in zip(clip_query_features, corresponding_scales):
        scale_t = torch.full((1,), scale_val).cuda()
        scale_t = q_trans(scale_t)
        gates = scale_gate(scale_t).detach().squeeze()

        sc_pt = point_features * gates.unsqueeze(0)
        sc_pt_normed = torch.nn.functional.normalize(sc_pt, dim=-1, p=2)
        sims = torch.einsum("C,NC->N", feat.cuda(), sc_pt_normed)
        # 取多个 cluster 中的最大相似度
        final_similarities = torch.maximum(final_similarities, sims)

    # ── K. 生成 mask 并保存 ───────────────────────────────────────
    # 自动选阈值：若用户未指定有效阈值，用 0.85（notebook 默认值）
    threshold = score_threshold if score_threshold > 0.0 else 0.85
    final_mask = final_similarities > threshold

    seg_dir = model_path / "segmentation_res"
    seg_dir.mkdir(exist_ok=True)
    mask_save_path = seg_dir / f"{save_name}.pt"
    torch.save(final_mask, str(mask_save_path))
    print(f"✓ mask 已保存 → {mask_save_path}  (选中 Gaussian 数：{final_mask.sum().item()})")

    # ── L. 调用 get_3d_bbox_from_mask 返回 bbox ───────────────────
    # 找最新 scene point_cloud.ply
    scene_plys = sorted(
        model_path.glob("point_cloud/iteration_*/point_cloud.ply"),
        key=lambda p: int(p.parent.name.replace("iteration_", ""))
    )
    if not scene_plys:
        print(f"✗ 未找到 point_cloud.ply：{model_path}")
        return None
    ply_path = scene_plys[-1]

    try:
        from plyfile import PlyData
    except ImportError:
        print("✗ 缺少 plyfile（pip install plyfile）")
        return None

    ply = PlyData.read(str(ply_path))
    verts = ply["vertex"]
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1)  # (N, 3)

    mask_np = final_mask.detach().cpu().numpy().flatten().astype(bool)
    if len(mask_np) != len(xyz):
        print(f"✗ mask 长度 {len(mask_np)} ≠ 点云 Gaussian 数 {len(xyz)}")
        return None

    selected = xyz[mask_np]
    if len(selected) == 0:
        print(f"✗ 阈值 {threshold:.2f} 下没有选中任何 Gaussian，请尝试降低 score_threshold")
        return None

    padding = 0.05
    bbox_min = selected.min(axis=0)
    bbox_max = selected.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    size = bbox_max - bbox_min
    p = 1 + padding

    result = {
        "label":      text,
        "center":     center.tolist(),
        "size":       (size * p).tolist(),
        "bbox_min":   (center - size / 2 * p).tolist(),
        "bbox_max":   (center + size / 2 * p).tolist(),
        "n_gaussians": int(mask_np.sum()),
        "mask_path":  str(mask_save_path),
    }

    print(f"\n✅ 3D Bounding Box — {text}")
    print(f"   中心坐标 : [{', '.join(f'{v:.4f}' for v in result['center'])}]  （单位：米）")
    print(f"   尺寸 XYZ : [{', '.join(f'{v:.4f}' for v in result['size'])}]")
    print(f"   包含 Gaussian 数 : {result['n_gaussians']:,}")
    logger.info(f"query_by_text [{text}]: center={result['center']}, size={result['size']}")

    return result


# ──────────────────────────────────────────────────────────────
# 结果解析：从分割结果中提取 3D Bounding Box
# ──────────────────────────────────────────────────────────────

def get_3d_bbox_from_mask(mask_pt: str | Path, model_path: str | Path,
                           label: str = "object", padding: float = 0.05) -> dict | None:
    """
    从 SAGA 输出的 final_mask.pt 中读取 3D Bounding Box。

    参数：
        mask_pt     — SAGA segmentation_res/final_mask.pt 路径
        model_path  — 3DGS 模型目录（含 point_cloud.ply）
        label       — 物体标签
        padding     — bbox 向外扩展比例（默认 5%）

    返回：
        {
            "label": "yogurt",
            "center": [x, y, z],       # 世界坐标系，单位：米
            "size":   [dx, dy, dz],    # 三轴边长
            "bbox_min": [x, y, z],
            "bbox_max": [x, y, z],
        }
    """
    try:
        import torch
        import numpy as np
        from plyfile import PlyData
    except ImportError as e:
        print(f"✗ 缺少依赖：{e}")
        return None

    mask_path = Path(mask_pt)
    if not mask_path.exists():
        print(f"✗ mask 文件不存在：{mask_path}")
        print("  请在 SAGA Notebook 中运行分割，生成 segmentation_res/final_mask.pt")
        return None

    # 找到最新的 point_cloud.ply
    model_dir = Path(model_path)
    plys = sorted(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not plys:
        print(f"✗ 未找到 point_cloud.ply：{model_dir}")
        return None
    ply_path = plys[-1]
    print(f"   使用点云：{ply_path.relative_to(PROJECT_ROOT)}")

    # 读取 Gaussian 位置
    ply = PlyData.read(str(ply_path))
    verts = ply["vertex"]
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1)  # (N, 3)

    # 读取 mask（True = 属于该物体）
    mask = torch.load(str(mask_path), map_location="cpu")
    if hasattr(mask, "numpy"):
        mask = mask.numpy().astype(bool)
    else:
        mask = np.array(mask, dtype=bool)

    mask = mask.flatten()
    if len(mask) != len(xyz):
        print(f"✗ mask 长度 {len(mask)} ≠ 点云 Gaussian 数 {len(xyz)}")
        print("  请确认 mask 与模型来自同一次训练")
        return None

    selected = xyz[mask]
    if len(selected) == 0:
        print("✗ mask 内没有选中任何 Gaussian")
        return None

    # 计算 AABB
    p = 1 + padding
    bbox_min = selected.min(axis=0)
    bbox_max = selected.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    size = bbox_max - bbox_min

    # 加 padding
    bbox_min_padded = center - size / 2 * p
    bbox_max_padded = center + size / 2 * p

    result = {
        "label":    label,
        "center":   center.tolist(),
        "size":     (size * p).tolist(),
        "bbox_min": bbox_min_padded.tolist(),
        "bbox_max": bbox_max_padded.tolist(),
        "n_gaussians": int(mask.sum()),
    }

    print(f"\n✅ 3D Bounding Box — {label}")
    print(f"   中心坐标 : [{', '.join(f'{v:.4f}' for v in result['center'])}]  （单位：米）")
    print(f"   尺寸 XYZ : [{', '.join(f'{v:.4f}' for v in result['size'])}]")
    print(f"   包含 Gaussian 数 : {result['n_gaussians']:,}")
    logger.info(f"SAGA bbox [{label}]: center={result['center']}, size={result['size']}")

    return result


def export_bboxes(bboxes: list[dict], output_path: str | Path | None = None) -> Path:
    """将多个 3D Bounding Box 导出为 JSON，供合规评估模块使用。"""
    if output_path is None:
        output_path = PROJECT_ROOT / "outputs" / "saga_bboxes.json"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bboxes, indent=2, ensure_ascii=False))
    print(f"✓ 已导出 {len(bboxes)} 个 bbox → {out}")
    return out


# ──────────────────────────────────────────────────────────────
# 一键完整 SAGA 流程（Step 1~5）
# ──────────────────────────────────────────────────────────────

def run_saga_pipeline(saga_cfg: dict) -> bool:
    """
    一键运行 Step 1~5（特征提取 + mask 提取 + 训练），
    完成后提示打开 SAGA Notebook 进行交互分割。
    """
    print("\n" + "=" * 55)
    print("SAGA Pipeline 开始")
    print("=" * 55)

    steps = [
        ("[1/4] 创建缩小图像目录",     lambda: create_downsampled_images(saga_cfg)),
        ("[2/4] 提取 SAM 自动 mask",  lambda: extract_sam_masks(saga_cfg)),
        ("[3/4] 提取 CLIP 特征",       lambda: extract_sam_features(saga_cfg)),
        ("[4/4] 训练对比特征",         lambda: train_saga_features(saga_cfg)),
    ]

    for label, fn in steps:
        print(f"\n{label}...")
        if not fn():
            print(f"\n✗ SAGA Pipeline 中止于：{label}")
            return False

    print("\n" + "=" * 55)
    print("✅ SAGA 特征训练完成！")
    print("=" * 55)
    print("\n下一步：运行 open_saga_notebook(saga_cfg) 打开交互分割界面")
    print("  - 输入文字（如 'yogurt bottle'）→ CLIP 匹配 → 自动输出 3D bbox")
    print("  - 点击渲染图像中的物体 → 精确选定")
    print("  - 完成后调用 get_3d_bbox_from_mask() 获取坐标")
    return True


# ──────────────────────────────────────────────────────────────
# 内部辅助函数
# ──────────────────────────────────────────────────────────────

def load_saga_config(cfg: dict | None = None) -> dict:
    """
    从主 pipeline cfg 中提取 saga 子配置，
    并补充默认值。
    """
    from .utils import load_config
    base = cfg or load_config()
    saga = base.get("saga", {})

    # 默认：使用主配置的数据路径和模型路径
    saga.setdefault("sam_checkpoint", str(SAM_DEFAULT_CKPT))
    saga.setdefault("sam_arch", "vit_h")
    saga.setdefault("downsample", 4)

    # image_root：SAGA 要求有 images/ 目录的场景根目录
    if not saga.get("image_root"):
        saga["image_root"] = base["dataset"]["path"]

    # model_path：使用训练输出目录
    if not saga.get("model_path"):
        saga["model_path"] = base["training"]["output_dir"]

    return saga


def _get_paths(saga_cfg: dict) -> tuple[Path, Path]:
    image_root = Path(saga_cfg["image_root"]).expanduser()
    sam_ckpt   = Path(saga_cfg.get("sam_checkpoint", str(SAM_DEFAULT_CKPT))).expanduser()
    if not image_root.is_absolute():
        image_root = (PROJECT_ROOT / image_root).resolve()
    if not sam_ckpt.is_absolute():
        sam_ckpt = (PROJECT_ROOT / sam_ckpt).resolve()
    return image_root, sam_ckpt


def _get_model_path(saga_cfg: dict) -> Path | None:
    raw = saga_cfg.get("model_path", "")
    if not raw:
        print("✗ saga_cfg['model_path'] 未设置，请指定已训练的 3DGS 模型目录")
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    if not p.exists():
        print(f"✗ 模型目录不存在：{p}")
        return None
    return p


def _check_script(script: Path) -> bool:
    if not script.exists():
        print(f"✗ SAGA 脚本不存在：{script}")
        print("  请先运行：bash scripts/setup_saga.sh")
        return False
    return True


def _check_sam_ckpt(ckpt: Path) -> bool:
    if not ckpt.exists():
        print(f"✗ SAM checkpoint 不存在：{ckpt}")
        print("  运行：download_sam_checkpoint() 或 bash scripts/setup_saga.sh")
        return False
    return True


def _check_images_dir(image_root: Path) -> bool:
    images = image_root / "images"
    if not images.exists():
        print(f"✗ 图像目录不存在：{images}")
        print("  SAGA 要求场景目录下有 images/ 子目录")
        return False
    imgs = [p for p in images.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not imgs:
        print(f"✗ {images} 下没有找到图像文件")
        return False
    print(f"   图像目录：{images}（{len(imgs)} 张）")
    return True


def _run_saga_script(cmd: list[str], cwd: Path) -> bool:
    """在 SAGA 目录下运行脚本，实时打印输出。"""
    import os
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    print(f"   命令：{' '.join(cmd[:3])} ...")
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    tail: list[str] = []
    for line in proc.stdout:
        print(line, end="")
        tail.append(line.rstrip())
        if len(tail) > 100:
            tail = tail[-100:]
    rc = proc.wait()
    if rc != 0:
        print(f"\n✗ 脚本退出码 {rc}")
        return False
    print("✓ 完成")
    return True


def _status(ok: bool, label: str) -> None:
    print(f"  {'✓' if ok else '✗'} {label}")


def _try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def visualize_query_results(*results) -> None:
    """
    弹窗显示 query_by_text 返回的 3D 可视化（每个结果一个独立窗口）。
    用法：visualize_query_results(result1, result2, ...)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.switch_backend("TkAgg")   # 弹出独立窗口；如报错可换 "Qt5Agg"

    try:
        from plyfile import PlyData
    except ImportError:
        print("✗ 请先安装 plyfile：pip install plyfile")
        return

    # 所有结果共用同一份点云，只读一次
    if not results:
        print("没有传入任何查询结果")
        return

    model_path = Path(results[0]["mask_path"]).parent.parent
    ply_path = model_path / "point_cloud/iteration_30000/point_cloud.ply"
    ply = PlyData.read(ply_path)
    xyz = np.stack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]], axis=1)
    step = max(1, len(xyz) // 8000)
    bg_xyz = xyz[::step]

    colors = ["red", "limegreen", "dodgerblue", "orange"]

    for idx, result in enumerate(results):
        import torch
        mask = torch.load(result["mask_path"]).cpu().numpy().astype(bool)
        selected = xyz[mask]

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(bg_xyz[:, 0], bg_xyz[:, 1], bg_xyz[:, 2],
                   c="lightgray", s=0.3, alpha=0.2)
        ax.scatter(selected[:, 0], selected[:, 1], selected[:, 2],
                   c=colors[idx % len(colors)], s=8, alpha=0.9,
                   label=f"{result['label']} ({result['n_gaussians']})")

        mn, mx = result["bbox_min"], result["bbox_max"]
        corners = [
            [mn[0],mn[1],mn[2]], [mx[0],mn[1],mn[2]], [mx[0],mx[1],mn[2]], [mn[0],mx[1],mn[2]],
            [mn[0],mn[1],mx[2]], [mx[0],mn[1],mx[2]], [mx[0],mx[1],mx[2]], [mn[0],mx[1],mx[2]],
        ]
        for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            ax.plot([corners[i][0], corners[j][0]],
                    [corners[i][1], corners[j][1]],
                    [corners[i][2], corners[j][2]], "b-", linewidth=1.2, alpha=0.7)

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("Query: " + result["label"])
        ax.legend()
        plt.tight_layout()

    plt.show()  # 一次性显示所有窗口，可同时拖动旋转


def validate_3d_fusion(
    mask_label: str,
    saga_cfg: dict,
    n_views: int = 6,
) -> None:
    """
    验证 3D 语义融合质量：把 query_by_text 选中的 3D Gaussians
    投影回原始图像，目视检查落点是否与真实物体位置吻合。

    用法：
        validate_3d_fusion("coca_cola_can", saga_cfg)
        validate_3d_fusion("facial_tissue_box", saga_cfg, n_views=4)

    参数：
        mask_label : segmentation_res/ 下的文件名（不含 .pt）
        saga_cfg   : pipeline.yaml 中的 saga 节
        n_views    : 随机抽取几个相机视角显示（默认 6）
    """
    import json
    import random
    import cv2
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from plyfile import PlyData

    plt.switch_backend("TkAgg")

    model_path = _get_model_path(saga_cfg)
    if model_path is None:
        return
    image_root, _ = _get_paths(saga_cfg)

    mask_pt = model_path / "segmentation_res" / (mask_label + ".pt")
    if not mask_pt.exists():
        print(f"✗ 找不到 {mask_pt}，先运行 query_by_text()")
        return

    # 读取 Gaussian 位置
    ply_path = model_path / "point_cloud/iteration_30000/point_cloud.ply"
    ply = PlyData.read(ply_path)
    xyz = np.stack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]], axis=1)

    # 读取 3D mask
    mask = torch.load(mask_pt, map_location="cpu").numpy().astype(bool)
    selected_xyz = xyz[mask]           # (N_selected, 3)
    print(f"   '{mask_label}'：选中 {mask.sum()} 个 Gaussians，投影到 {n_views} 个视角")

    # 读取相机参数（cameras.json 是 3DGS 导出的，rotation = camera→world）
    with open(model_path / "cameras.json") as f:
        cameras = json.load(f)

    # 随机采样 n_views 个相机
    sampled = random.sample(cameras, min(n_views, len(cameras)))

    ncols = 3
    nrows = (len(sampled) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax_i, cam in enumerate(sampled):
        img_name = cam["img_name"]
        img_path = image_root / "images" / img_name
        if not img_path.exists():
            axes[ax_i].axis("off")
            continue

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        # 相机内参
        fx, fy = cam["fx"], cam["fy"]
        cx, cy = W / 2.0, H / 2.0

        # 外参：cameras.json 中 rotation 是 camera→world，position 是相机中心（world）
        R = np.array(cam["rotation"])   # (3, 3)  camera→world
        t = np.array(cam["position"])   # (3,)    world position

        # world → camera
        R_cw = R.T                           # (3, 3)
        p_cam = (selected_xyz - t) @ R_cw.T  # (N, 3)

        # 过滤掉在相机后方的点
        in_front = p_cam[:, 2] > 0.01
        p_cam = p_cam[in_front]

        # 投影到像素
        u = fx * p_cam[:, 0] / p_cam[:, 2] + cx
        v = fy * p_cam[:, 1] / p_cam[:, 2] + cy

        # 过滤出图像范围外的点
        in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v = u[in_frame], v[in_frame]

        # 显示：降采样大图避免卡顿
        scale = min(1.0, 800 / W)
        disp = cv2.resize(img, (int(W * scale), int(H * scale)))
        axes[ax_i].imshow(disp)

        if len(u) > 0:
            axes[ax_i].scatter(u * scale, v * scale,
                               c="red", s=2, alpha=0.5, linewidths=0)
        n_proj = int(in_frame.sum())
        axes[ax_i].set_title(f"{img_name}\n{n_proj} pts projected", fontsize=8)
        axes[ax_i].axis("off")

    # 关闭多余的子图
    for ax_i in range(len(sampled), len(axes)):
        axes[ax_i].axis("off")

    fig.suptitle(f'3D fusion 验证 — "{mask_label}"', fontsize=12)
    plt.tight_layout()
    plt.show()


def query_2d_image(image_name: str, query: str, saga_cfg: dict, top_k: int = 3) -> None:
    """
    在单张 2D 图像上做文字查询，高亮显示 top-k 个最匹配的 SAM mask。

    原理：OpenCLIP 本身没有定位能力，我们用 SAM mask 作为 region proposal，
    对每个 mask 的预计算 CLIP 特征做 text-image cosine 相似度打分。

    用法：
        query_2d_image("IMG_20260402_120500.jpg", "coca cola can", saga_cfg)

    参数：
        image_name : 图像文件名（带扩展名），需在 image_root/images/ 下存在
        query      : 自然语言查询词
        saga_cfg   : pipeline.yaml 中的 saga 节
        top_k      : 叠加显示前 k 个 mask（默认 3）
    """
    import sys
    import cv2
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.switch_backend("TkAgg")

    image_root, _ = _get_paths(saga_cfg)
    stem = Path(image_name).stem

    img_path  = image_root / "images" / image_name
    feat_path = image_root / "clip_features" / (stem + ".pt")
    mask_path = image_root / "sam_masks"     / (stem + ".pt")

    if not img_path.exists():
        print(f"✗ 图像不存在：{img_path}")
        return
    if not feat_path.exists():
        print(f"✗ CLIP 特征不存在：{feat_path}（先跑 extract_clip_features）")
        return
    if not mask_path.exists():
        print(f"✗ SAM mask 不存在：{mask_path}（先跑 extract_sam_masks）")
        return

    # 读取图像
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # 读取预计算特征 [N_mask, 512]
    features = torch.load(feat_path, map_location="cpu")   # [N, 512]

    # 读取 SAM masks [N_mask, h, w]（低分辨率，需 resize 到原图）
    masks_raw = torch.load(mask_path, map_location="cpu")  # [N, h, w]
    masks = torch.nn.functional.interpolate(
        masks_raw.unsqueeze(0).float(), size=(H, W), mode="bilinear"
    ).squeeze(0)
    masks = (masks > 0.5).float()   # [N, H, W]

    # 用 CLIP 模型打分（复用 clip_utils）
    saga_dir = PROJECT_ROOT / "third_party" / "SAGA"
    if str(saga_dir) not in sys.path:
        sys.path.insert(0, str(saga_dir))

    from clip_utils.clip_utils import load_clip
    from clip_utils import get_scores

    clip_model = load_clip()
    clip_model.eval()

    features_cuda = features.cuda()
    scores = get_scores(clip_model, features_cuda, query).cpu().float()  # [N]

    # 取 top-k mask，叠加到原图
    topk_idx = scores.topk(min(top_k, len(scores))).indices.tolist()

    overlay = img.copy().astype(np.float32)
    colors  = [(255, 80, 80), (80, 200, 80), (80, 130, 255)]   # R G B per rank

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：原图
    axes[0].imshow(img)
    axes[0].set_title("原图")
    axes[0].axis("off")

    # 右图：叠加 top-k mask
    result_img = img.copy().astype(np.float32)
    legend_patches = []
    for rank, idx in enumerate(topk_idx):
        color = colors[rank % len(colors)]
        m = masks[idx].numpy()                      # [H, W]  0/1
        for c in range(3):
            result_img[:, :, c] = np.where(
                m > 0.5,
                result_img[:, :, c] * 0.4 + color[c] * 0.6,
                result_img[:, :, c]
            )
        score_val = scores[idx].item()
        label = f"#{rank+1}  score={score_val:.3f}"
        legend_patches.append(mpatches.Patch(
            facecolor=[c/255 for c in color], label=label))

    axes[1].imshow(result_img.astype(np.uint8))
    axes[1].set_title(f'Query: "{query}"  top-{top_k} masks')
    axes[1].axis("off")
    axes[1].legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.suptitle(image_name, fontsize=10, color="gray")
    plt.tight_layout()
    plt.show()
