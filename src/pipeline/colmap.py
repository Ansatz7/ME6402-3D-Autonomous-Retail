"""COLMAP 相机标定：SfM 五步流程 + 内参解析 + 路径自动修正。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .utils import PROJECT_ROOT, logger


# ──────────────────────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────────────────────

def _run_with_live_output(cmd: list[str], timeout: int = 7200) -> tuple[int, str]:
    """实时打印子进程输出，返回 (returncode, 最近 200 行)。"""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    tail: list[str] = []
    try:
        for line in proc.stdout:
            print(line, end="")
            tail.append(line.rstrip("\n"))
            if len(tail) > 200:
                tail = tail[-200:]
        return proc.wait(timeout=timeout), "\n".join(tail)
    except subprocess.TimeoutExpired:
        proc.kill()
        return 124, "\n".join(tail)


def _ensure_text_model(model_dir: Path) -> bool:
    """确保 sparse/0 下有 cameras.txt（若只有 .bin 则自动转换）。"""
    cameras_txt = model_dir / "cameras.txt"
    if cameras_txt.exists():
        return True
    cameras_bin = model_dir / "cameras.bin"
    if not cameras_bin.exists() or not shutil.which("colmap"):
        return False
    cmd = [
        "colmap", "model_converter",
        "--input_path", str(model_dir),
        "--output_path", str(model_dir),
        "--output_type", "TXT",
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and cameras_txt.exists()


def _parse_intrinsics(cameras_txt: Path) -> dict | None:
    """从 cameras.txt 读取第一个相机的内参。"""
    if not cameras_txt.exists():
        return None
    for raw in cameras_txt.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        model, w, h = parts[1], int(parts[2]), int(parts[3])
        params = [float(x) for x in parts[4:]]
        intr = {"model": model, "width": w, "height": h, "raw_params": params}
        if model in ("SIMPLE_PINHOLE",) and len(params) >= 3:
            intr.update({"fx": params[0], "fy": params[0], "cx": params[1], "cy": params[2]})
        elif model == "PINHOLE" and len(params) >= 4:
            intr.update({"fx": params[0], "fy": params[1], "cx": params[2], "cy": params[3]})
        elif model in ("SIMPLE_RADIAL", "RADIAL") and len(params) >= 3:
            intr.update({"fx": params[0], "fy": params[0], "cx": params[1], "cy": params[2]})
        return intr
    return None


def _ensure_sparse_zero(scene: Path) -> bool:
    """确保 COLMAP 输出满足 3DGS 约定：sparse/0/cameras.*。"""
    sparse = scene / "sparse"
    zero   = sparse / "0"
    if not sparse.exists():
        return False
    if zero.exists():
        return True
    prefixes = ("cameras", "images", "points3D")
    files = [p for p in sparse.iterdir()
             if p.is_file() and any(p.name.startswith(x) for x in prefixes)]
    if not files:
        return False
    zero.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = zero / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"🧩 已整理 sparse 目录：{sparse} → {zero}")
    return True


def _is_recognizable(path: Path) -> bool:
    """判断路径是否是 3DGS 可识别的数据结构。"""
    if (path / "sparse" / "0").exists() and (path / "images").exists():
        return True
    if (path / "transforms_train.json").exists() or (path / "transforms.json").exists():
        return True
    return False


# ──────────────────────────────────────────────────────────────
# 公开 API
# ──────────────────────────────────────────────────────────────

def run_colmap(cfg: dict) -> bool:
    """
    运行 COLMAP SfM 流程，完成后自动更新 cfg['dataset']['path']
    指向 undistorted dense 输出（3DGS 可直接读取的格式）。
    """
    if not cfg["dataset"].get("use_colmap", False):
        print("ℹ️  use_colmap=false，跳过 COLMAP。")
        return True

    colmap_script = PROJECT_ROOT / "scripts" / "reconstruction" / "run_colmap.sh"
    if not colmap_script.exists():
        print(f"✗ COLMAP 脚本不存在: {colmap_script}")
        return False

    dataset_path = Path(cfg["dataset"]["path"])
    images_dir   = dataset_path / cfg["dataset"].get("images_dir", "images")
    image_dir    = images_dir if images_dir.exists() else dataset_path
    if not image_dir.exists():
        print(f"✗ 图像目录不存在: {image_dir}")
        return False

    workspace = Path(cfg["colmap"]["workspace"])
    workspace.mkdir(parents=True, exist_ok=True)

    cmd = ["bash", str(colmap_script), str(image_dir), str(workspace)]
    print(f"\n🧭 运行 COLMAP  →  {workspace}")
    print(f"   {' '.join(cmd)}")

    rc, tail = _run_with_live_output(cmd, timeout=7200)
    if rc != 0:
        print(f"\n✗ COLMAP 失败（返回码 {rc}）")
        if rc == 124:
            print("  超时（>2h）")
        if tail:
            print("--- 最近输出 ---")
            print(tail[-2000:])
        return False

    # 切换训练路径到 undistorted dense 输出
    dense = workspace / "dense"
    use_undistorted = cfg.get("colmap", {}).get("use_undistorted_for_training", True)
    if use_undistorted and (dense / "sparse").exists() and (dense / "images").exists():
        cfg["dataset"]["path"] = str(dense)
        cfg["dataset"]["images_dir"] = "images"
        print(f"\n🧭 训练数据已切换 →  {dense}")
    else:
        _ensure_sparse_zero(workspace)

    # 回填内参
    if cfg.get("colmap", {}).get("auto_update_intrinsics", True):
        model_dir = workspace / "sparse" / "0"
        _ensure_text_model(model_dir)
        intr = _parse_intrinsics(model_dir / "cameras.txt")
        if intr:
            cfg["colmap"]["estimated_intrinsics"] = intr
            print(f"\n📷 内参已写入 cfg['colmap']['estimated_intrinsics']：")
            print(json.dumps(intr, indent=2, ensure_ascii=False))

    logger.info(f"COLMAP 完成，workspace={workspace}")
    print("\n✅ COLMAP 完成")
    return True


def fix_dataset_layout(cfg: dict) -> bool:
    """
    训练前对数据目录做两项自动修复：
    1. 若 sparse/ 下没有 0/ 子目录，自动创建
    2. 若路径不被 3DGS 识别，尝试切换到 COLMAP dense 输出
    返回是否做了修正。
    """
    dataset_path = Path(cfg["dataset"]["path"])

    # 先在当前路径尝试修复
    _ensure_sparse_zero(dataset_path)
    if _is_recognizable(dataset_path):
        return False

    # 尝试切换到 dense
    workspace = cfg.get("colmap", {}).get("workspace", "")
    if not workspace:
        return False
    dense = Path(workspace) / "dense"
    _ensure_sparse_zero(dense)
    if _is_recognizable(dense):
        cfg["dataset"]["path"] = str(dense)
        cfg["dataset"]["images_dir"] = "images"
        print(f"🧭 训练前自动修正数据路径 →  {dense}")
        logger.info(f"训练前自动修正 dataset.path: {dense}")
        return True

    return False
