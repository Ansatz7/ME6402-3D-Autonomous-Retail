"""结果查看：Open3D 交互查看 + SIBR Docker 查看。"""

from __future__ import annotations

import getpass
import shutil
import subprocess
from pathlib import Path

from .utils import OUTPUT_DIR, PROJECT_ROOT, logger


def find_latest_model(search_root: Path | None = None) -> Path | None:
    """在 outputs/ 下搜索最新迭代的 point_cloud.ply，返回其所在模型目录。"""
    root = search_root or OUTPUT_DIR
    best_iter = -1
    best_dir: Path | None = None

    for cfg_file in root.glob("**/cfg_args"):
        model_dir = cfg_file.parent
        for ply in model_dir.glob("point_cloud/iteration_*/point_cloud.ply"):
            try:
                it = int(ply.parent.name.split("_")[-1])
                if it > best_iter:
                    best_iter, best_dir = it, model_dir
            except ValueError:
                pass

    return best_dir


def find_latest_ply(cfg: dict | None = None) -> Path | None:
    """返回最新训练输出中的 point_cloud.ply 路径。"""
    if cfg:
        search = Path(cfg["training"]["output_dir"])
        plys = sorted(search.glob("**/point_cloud/iteration_*/point_cloud.ply"))
        if plys:
            return plys[-1]
    # 全局搜索
    plys = sorted(OUTPUT_DIR.glob("**/point_cloud/iteration_*/point_cloud.ply"))
    return plys[-1] if plys else None


# ──────────────────────────────────────────────────────────────
# Open3D 查看
# ──────────────────────────────────────────────────────────────

def open_viewer(cfg: dict | None = None, ply_path: Path | str | None = None) -> bool:
    """
    用 Open3D 打开点云（推荐）。

    优先级：
    1. 显式传入的 ply_path
    2. cfg['training']['output_dir'] 下最新的 ply
    3. outputs/ 下全局搜索最新的 ply
    """
    if ply_path:
        target = Path(ply_path)
    else:
        target = find_latest_ply(cfg)

    if target is None or not target.exists():
        print("✗ 未找到 point_cloud.ply，请先完成训练（Section 4）")
        if cfg:
            print(f"   搜索目录：{cfg['training']['output_dir']}")
        return False

    mb = target.stat().st_size / 1024 / 1024
    print(f"✅ 找到点云：{target}")
    print(f"   大小：{mb:.1f} MB")

    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(target))
        print(f"   点数：{len(pcd.points):,}")
        print("   打开 Open3D 交互窗口（关闭窗口后继续）...")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=f"3DGS — {target.parent.parent.name}",
            width=1280,
            height=720,
        )
        logger.info(f"Open3D 查看完成: {target}")
        return True
    except ImportError:
        print("⚠️  Open3D 未安装，尝试 CloudCompare ...")
        if shutil.which("CloudCompare"):
            subprocess.Popen(["CloudCompare", str(target)])
            print("   CloudCompare 已启动（后台运行）")
            return True
        print(f"   请手动打开文件：{target}")
        return False


# ──────────────────────────────────────────────────────────────
# SIBR 查看
# ──────────────────────────────────────────────────────────────

def _has_docker() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def launch_sibr(cfg: dict, model_dir: Path | str | None = None) -> bool:
    """
    在 Docker 中启动 SIBR Viewer。

    若不传 model_dir，自动搜索 outputs/ 下的可用模型目录并提示选择。
    """
    sibr_cfg   = cfg.get("viewer", {}).get("sibr", {})
    script     = Path(sibr_cfg.get("script", PROJECT_ROOT / "scripts/reconstruction/run_sibr_in_docker.sh"))
    docker_img = sibr_cfg.get("docker_image", "sibr-builder:ubuntu22.04-cuda11.8")

    if not script.exists():
        print(f"✗ SIBR 脚本不存在: {script}")
        return False

    # 确定模型目录
    if model_dir:
        selected = Path(model_dir)
    else:
        selected = _pick_model_interactively(cfg)
        if selected is None:
            return False

    if not selected.exists():
        print(f"✗ 模型目录不存在: {selected}")
        return False
    if not (selected / "cfg_args").exists():
        print(f"✗ 模型目录下缺少 cfg_args: {selected}")
        return False

    cmd = ["bash", str(script), docker_img, str(selected)]
    print(f"\n🖼️  启动 SIBR Viewer ...")
    print(f"   {' '.join(cmd)}")
    print("   提示：关闭 SIBR 窗口后，该单元继续运行。")

    try:
        if _has_docker():
            r = subprocess.run(cmd, cwd=PROJECT_ROOT)
            ok = r.returncode == 0
        elif shutil.which("sudo"):
            print("ℹ️  无直接 Docker 权限，尝试 sudo ...")
            pwd = getpass.getpass("sudo 密码（不可见）: ")
            if not pwd:
                print("✗ 已取消")
                return False
            r = subprocess.run(
                ["sudo", "-S", "bash", str(script), docker_img, str(selected)],
                cwd=PROJECT_ROOT, input=pwd + "\n", text=True,
            )
            ok = r.returncode == 0
        else:
            print("✗ 无 Docker 权限且无 sudo，请将用户加入 docker 组后重启 Jupyter")
            return False

        if ok:
            print("✓ SIBR 正常退出")
        else:
            print(f"✗ SIBR 退出码非 0")
        logger.info(f"SIBR 查看完成: {selected}，ok={ok}")
        return ok
    except Exception as e:
        print(f"✗ 启动 SIBR 异常: {e}")
        return False


def _pick_model_interactively(cfg: dict) -> Path | None:
    """列出可用模型，让用户选择。"""
    candidates: list[tuple[int, Path]] = []
    for cfg_file in OUTPUT_DIR.glob("**/cfg_args"):
        model_dir = cfg_file.parent
        best = -1
        for ply in model_dir.glob("point_cloud/iteration_*/point_cloud.ply"):
            try:
                best = max(best, int(ply.parent.name.split("_")[-1]))
            except ValueError:
                pass
        candidates.append((best, model_dir))

    if not candidates:
        print(f"✗ 未找到可用的 3DGS 模型（outputs/ 下无 cfg_args 文件）")
        return None

    candidates.sort(reverse=True)
    print("\n可用模型（按最新迭代降序）:")
    for i, (it, d) in enumerate(candidates, 1):
        tag = f"iter={it}" if it >= 0 else "iter=?"
        print(f"  [{i}] {d.name}  ({tag})")

    raw = input("\n输入序号或路径（回车=第 1 个）: ").strip()
    if not raw:
        return candidates[0][1]
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx][1]
        print(f"✗ 序号超出范围")
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p
