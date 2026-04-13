"""3DGS 训练：启动 train.py，支持 OOM 自动降档。"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

from .colmap import fix_dataset_layout, _is_recognizable, _ensure_sparse_zero
from .utils import GS_DIR, PROJECT_ROOT, logger


def _build_cmd(cfg: dict) -> list[str]:
    tcfg = cfg["training"]
    cmd = [
        "python", "train.py",
        "-s", cfg["dataset"]["path"],
        "-m", tcfg["output_dir"],
        "--iterations", str(tcfg["iterations"]),
        "--resolution", str(tcfg["resolution"]),
        "--sh_degree", str(tcfg["sh_degree"]),
    ]
    if tconfig := tcfg.get("save_iterations"):
        cmd += ["--save_iterations"] + [str(x) for x in tconfig]
    if tconfig := tcfg.get("test_iterations"):
        cmd += ["--test_iterations"] + [str(x) for x in tconfig]
    if tcfg.get("white_background"):
        cmd.append("-w")
    return cmd


def _run_once(cfg: dict, timeout: int = 3600) -> tuple[bool, int, bool]:
    """单次训练，返回 (success, returncode, had_oom)。"""
    cmd = _build_cmd(cfg)
    logger.info(f"训练命令: {' '.join(cmd)}")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

    tail: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            print(line, end="")
            tail.append(line.rstrip("\n"))
            if len(tail) > 400:
                tail = tail[-400:]
        rc = proc.wait(timeout=timeout)
        oom = any("out of memory" in l.lower() for l in tail)
        return rc == 0, rc, oom
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.error("训练超时（>1h）")
        return False, 124, False
    except Exception as e:
        logger.error(f"训练异常: {e}")
        return False, 1, False


def run_training(cfg: dict) -> bool:
    """
    启动 3DGS 训练。

    - 训练前自动检测并修复数据目录结构
    - OOM 时自动将 resolution 降档（×1 → ×2 → ×4）重试
    """
    logger.info("=" * 50)
    logger.info("启动 3DGS 训练")
    logger.info("=" * 50)

    # 定位 GS_DIR
    gs_dir = GS_DIR
    if not gs_dir.exists():
        for p in [Path.cwd(), *Path.cwd().parents]:
            c = p / "third_party" / "gaussian-splatting"
            if c.exists():
                gs_dir = c
                break

    if not gs_dir.exists():
        print(f"✗ 3DGS 目录不存在: {gs_dir}")
        return False

    # 数据路径修复
    fix_dataset_layout(cfg)
    dataset_path = Path(cfg["dataset"]["path"])
    _ensure_sparse_zero(dataset_path)

    if not _is_recognizable(dataset_path):
        print("✗ 数据路径不被 3DGS 识别，请检查：")
        print(f"   {dataset_path}")
        print("   期望：<path>/sparse/0/ + <path>/images/ 同时存在")
        print("       或 <path>/transforms.json 存在（NeRF Blender 格式）")
        return False

    # 创建输出目录
    Path(cfg["training"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    os.chdir(gs_dir)

    # OOM 自动降档
    base_res = int(cfg["training"].get("resolution", 1))
    retry_res = [base_res] + [r for r in [2, 4] if r > base_res]
    original_res = base_res

    start = datetime.now()
    print(f"\n⏳ 训练开始")
    print(f"   数据   : {cfg['dataset']['path']}")
    print(f"   迭代   : {cfg['training']['iterations']}")
    print(f"   输出   : {cfg['training']['output_dir']}")
    print(f"   分辨率 : ×{base_res}")

    for idx, res in enumerate(retry_res):
        cfg["training"]["resolution"] = res
        if idx > 0:
            print(f"\n⚠️  OOM，自动降档重试：resolution=×{res}")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        success, rc, had_oom = _run_once(cfg)

        if success:
            elapsed = (datetime.now() - start).total_seconds() / 60
            print(f"\n✅ 训练完成  （{elapsed:.1f} 分钟）")
            print(f"   模型输出: {cfg['training']['output_dir']}")
            logger.info(f"训练完成，耗时 {elapsed:.1f}min，输出={cfg['training']['output_dir']}")
            cfg["training"]["resolution"] = original_res
            return True

        if had_oom and idx < len(retry_res) - 1:
            continue

        print(f"\n✗ 训练失败（返回码 {rc}）")
        if had_oom:
            print("  显存不足且已降到最低档（×4），建议减少 iterations 后重试")
        break

    cfg["training"]["resolution"] = original_res
    return False


def analyze_results(cfg: dict) -> None:
    """分析训练输出目录：列出 PLY 文件及大小，打印 results.json。"""
    import json as _json

    output = Path(cfg["training"]["output_dir"])
    if not output.exists():
        print(f"✗ 输出目录不存在: {output}")
        return

    print("=" * 60)
    print("训练结果分析")
    print("=" * 60)

    plys = sorted(output.glob("**/*.ply"))
    if plys:
        print(f"\n📁 PLY 点云文件（共 {len(plys)} 个）:")
        for ply in plys:
            mb = ply.stat().st_size / 1024 / 1024
            print(f"  {ply.relative_to(output)}  ({mb:.1f} MB)")
    else:
        print("\n⚠️  未找到 PLY 文件")

    results_file = output / "results.json"
    if results_file.exists():
        results = _json.loads(results_file.read_text())
        print("\n📊 训练指标 (results.json):")
        for k, v in results.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")

    print(f"\n输出目录: {output}")
    logger.info(f"结果分析完成: {output}")
