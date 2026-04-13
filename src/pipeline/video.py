"""视频抽帧：从 mp4/mov 提取 JPEG 帧到 data/raw/<scene>/images/。"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

from .utils import DATA_DIR, PROJECT_ROOT, logger


def extract_frames(cfg: dict) -> bool:
    """
    按 cfg['video'] 配置从视频中抽帧。

    抽帧完成后自动更新 cfg['dataset'] 指向该场景，
    并将 use_colmap 置为 True，方便直接运行下一步 COLMAP。
    """
    vcfg = cfg.get("video", {})
    if not vcfg.get("enabled", False):
        print("ℹ️  video.enabled=false，跳过抽帧。")
        print("   若要抽帧，请在 configs/pipeline.yaml 中将 video.enabled 改为 true。")
        return False

    # ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("✗ 未找到 ffmpeg，请安装：sudo apt install ffmpeg")
        return False

    # 输入视频
    video_path = Path(vcfg["input_video"]).expanduser()
    if not video_path.is_absolute():
        video_path = (PROJECT_ROOT / video_path).resolve()
    if not video_path.exists():
        print(f"✗ 视频文件不存在: {video_path}")
        return False

    # 输出目录
    scene_name = str(vcfg.get("scene_name", "custom_scene_01")).strip() or "custom_scene_01"
    images_dir = DATA_DIR / "raw" / scene_name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧帧（避免帧号冲突）
    for old in images_dir.glob("*.jpg"):
        old.unlink()

    # 构建 ffmpeg 命令
    fps           = int(vcfg.get("fps", 2))
    jpeg_quality  = int(vcfg.get("jpeg_quality", 2))
    start_sec     = int(vcfg.get("start_seconds", 0))
    duration_sec  = int(vcfg.get("duration_seconds", 0))

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", str(video_path)]
    if duration_sec > 0:
        cmd += ["-t", str(duration_sec)]
    cmd += ["-vf", f"fps={fps}", "-q:v", str(jpeg_quality),
            "-start_number", "1", str(images_dir / "%06d.jpg"), "-y"]

    print("🎬 开始视频抽帧 ...")
    print(f"   输入  : {video_path}")
    print(f"   输出  : {images_dir}")
    print(f"   命令  : {' '.join(shlex.quote(x) for x in cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("✗ ffmpeg 失败")
        print(result.stderr[-2000:])
        return False

    n = len(list(images_dir.glob("*.jpg")))
    print(f"✓ 抽帧完成，共 {n} 张")
    if n < 20:
        print("⚠️  帧数偏少（建议 ≥20 张），COLMAP 成功率可能降低")

    # 自动更新配置，指向新场景
    cfg["dataset"]["source"]     = "custom"
    cfg["dataset"]["path"]       = str(images_dir.parent)
    cfg["dataset"]["images_dir"] = "images"
    cfg["dataset"]["use_colmap"] = True
    cfg["colmap"]["workspace"]   = str(DATA_DIR / "colmap_workspace" / scene_name)
    cfg["training"]["output_dir"] = str(
        PROJECT_ROOT / "outputs" / f"{scene_name}_{cfg['training']['iterations']}iter"
    )
    print(f"\n🧭 已自动更新配置：")
    print(f"   dataset.path       = {cfg['dataset']['path']}")
    print(f"   colmap.workspace   = {cfg['colmap']['workspace']}")
    print(f"   training.output_dir= {cfg['training']['output_dir']}")
    logger.info(f"抽帧完成：{n} 张，场景={scene_name}")
    return True
