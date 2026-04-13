"""环境检查：PyTorch、CUDA、COLMAP、3DGS 模块。"""

from __future__ import annotations

import shutil
import subprocess

from .utils import GS_DIR, DATA_DIR, OUTPUT_DIR, PROJECT_ROOT, logger


def check_environment(cfg: dict | None = None) -> bool:
    """打印完整的环境检查报告，返回是否全部通过。"""
    print("=" * 60)
    print("3DGS 环境检查")
    print("=" * 60)

    ok = True

    # PyTorch + CUDA
    try:
        import torch
        print(f"\nPyTorch:  {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA 可用: {cuda_ok}")
        if cuda_ok:
            print(f"CUDA 版本: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.1f} GB)")
        else:
            print("  ⚠️  CUDA 不可用，训练将在 CPU 上运行（极慢）")
            ok = False
    except ImportError:
        print("✗ PyTorch 未安装")
        ok = False

    # 核心依赖
    print("\n核心依赖:")
    _check_import("cv2",     "OpenCV")
    _check_import("numpy",   "NumPy")
    _check_import("plyfile", "plyfile")
    _check_import("scipy",   "SciPy")
    _check_import("open3d",  "Open3D")
    gs_ok = _check_import("diff_gaussian_rasterization", "diff_gaussian_rasterization (CUDA 模块)")
    if not gs_ok:
        ok = False

    # COLMAP CLI
    print("\nCOLMAP:")
    colmap = shutil.which("colmap")
    if colmap:
        print(f"  ✓ {colmap}")
    else:
        print("  ✗ 未找到 colmap 命令（自有数据流程需要）")

    # Docker（可选，SIBR 用）
    print("\nDocker（SIBR 查看器）:")
    docker = shutil.which("docker")
    if docker:
        probe = subprocess.run(["docker", "info"], capture_output=True)
        if probe.returncode == 0:
            print("  ✓ Docker daemon 可访问")
        else:
            print("  ⚠️  docker 命令存在但 daemon 不可访问（可能需要加入 docker 组）")
    else:
        print("  ⚠️  未找到 docker（仅影响 SIBR 查看器，Open3D 不受影响）")

    # 项目目录
    print("\n项目目录:")
    print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"  GS_DIR       : {GS_DIR}  {'✓' if GS_DIR.exists() else '✗ 不存在'}")
    print(f"  DATA_DIR     : {DATA_DIR}  {'✓' if DATA_DIR.exists() else '✗ 不存在'}")
    print(f"  OUTPUT_DIR   : {OUTPUT_DIR}  ✓")

    print("\n" + ("✅ 环境检查通过" if ok else "⚠️  部分检查未通过，见上方提示"))
    logger.info(f"环境检查完成，结果: {'通过' if ok else '部分未通过'}")
    return ok


def _check_import(module: str, label: str) -> bool:
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "")
        print(f"  ✓ {label}{f'  {ver}' if ver else ''}")
        return True
    except ImportError as e:
        print(f"  ✗ {label}  ({e})")
        return False
