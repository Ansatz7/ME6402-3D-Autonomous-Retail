"""工具函数：项目根目录检测、配置加载、日志初始化。"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import yaml

# ──────────────────────────────────────────────────────────────
# 项目根目录检测
# ──────────────────────────────────────────────────────────────

def find_project_root(start: Path | None = None) -> Path:
    """从任意子目录向上查找项目根目录。"""
    if start is None:
        start = Path.cwd()

    markers = [
        lambda p: (p / "third_party" / "gaussian-splatting").exists(),
        lambda p: (p / "configs" / "pipeline.yaml").exists(),
        lambda p: (p / "src" / "pipeline").exists(),
    ]

    for p in [start, *start.parents]:
        if all(m(p) for m in markers):
            return p

    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p

    return start


PROJECT_ROOT: Path = find_project_root()

# ──────────────────────────────────────────────────────────────
# 标准路径（只在这里定义一次）
# ──────────────────────────────────────────────────────────────

GS_DIR    = PROJECT_ROOT / "third_party" / "gaussian-splatting"
DATA_DIR  = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR   = PROJECT_ROOT / "logs"
CONFIG_FILE = PROJECT_ROOT / "configs" / "pipeline.yaml"

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 日志
# ──────────────────────────────────────────────────────────────

def setup_logger(name: str = "pipeline") -> logging.Logger:
    log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


logger = setup_logger()

# ──────────────────────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────────────────────

_OFFICIAL_PATHS = {
    # 常见的 T&T + DB 场景路径（项目内已下载）
    "playroom": DATA_DIR / "official" / "tandt_db" / "db" / "playroom",
    "truck":    DATA_DIR / "official" / "tandt_db" / "tandt" / "truck",
    "train":    DATA_DIR / "official" / "tandt_db" / "tandt" / "train",
    "drjohnson":DATA_DIR / "official" / "tandt_db" / "db" / "drjohnson",
}


def _resolve_dataset_path(cfg: dict) -> Path:
    """根据 dataset.source 和 scene 解析出实际数据路径。"""
    source = cfg["dataset"]["source"]
    scene  = cfg["dataset"].get("scene", "playroom")

    if source == "official_tandt":
        if scene in _OFFICIAL_PATHS:
            return _OFFICIAL_PATHS[scene]
        return DATA_DIR / "official" / "tandt_db" / "tandt" / scene

    if source == "minimal":
        return DATA_DIR / "minimal_dataset"

    if source == "nerf_synthetic":
        return DATA_DIR / "nerf_synthetic" / scene

    if source == "custom":
        raw = cfg["dataset"].get("custom_path", f"data/raw/{scene}")
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()

    raise ValueError(f"未知 dataset.source: {source!r}")


def load_config(path: Path | str | None = None) -> dict:
    """
    加载 configs/pipeline.yaml 并解析为运行时 config 字典。

    每次调用都会重新读取文件，因此修改 YAML 后只需重跑 Cell 1。
    返回的 config 中所有路径均为绝对路径字符串。
    """
    cfg_path = Path(path) if path else CONFIG_FILE
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 数据集路径
    raw["dataset"]["path"] = str(_resolve_dataset_path(raw))

    # COLMAP workspace → <root>/<scene_name>/
    scene_name = raw["dataset"].get("scene") or raw["video"].get("scene_name", "scene")
    colmap_ws = PROJECT_ROOT / raw["colmap"]["workspace"] / scene_name
    raw["colmap"]["workspace"] = str(colmap_ws)

    # 训练输出目录 → <output_dir>/<scene_name>_<iters>iter/
    iters = raw["training"]["iterations"]
    out_root = PROJECT_ROOT / raw["training"]["output_dir"]
    raw["training"]["output_dir"] = str(out_root / f"{scene_name}_{iters}iter")

    # Viewer script 路径
    raw["viewer"]["sibr"]["script"] = str(PROJECT_ROOT / raw["viewer"]["sibr"]["script"])

    return raw


def resolve_path(p: str | Path) -> Path:
    """将相对路径解析为基于 PROJECT_ROOT 的绝对路径。"""
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
