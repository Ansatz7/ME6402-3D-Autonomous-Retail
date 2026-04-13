"""
src.pipeline — 3DGS Pipeline 功能模块

在 Notebook Cell 1 中执行：
    from src.pipeline import *
即可导入所有函数，无需任何其他初始化。
"""

from .utils import (
    load_config,
    find_project_root,
    resolve_path,
    setup_logger,
    PROJECT_ROOT,
    GS_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    LOG_DIR,
)

from .env import check_environment

from .video import extract_frames

from .colmap import run_colmap, fix_dataset_layout

from .training import run_training, analyze_results

from .viewer import open_viewer, launch_sibr, find_latest_ply, find_latest_model

from .saga import (
    check_saga_ready,
    download_sam_checkpoint,
    create_downsampled_images,
    extract_sam_masks,
    extract_sam_features,
    train_saga_features,
    compute_scales,
    open_saga_gui,
    open_bbox_viewer,
    open_saga_notebook,
    query_by_text,
    get_3d_bbox_from_mask,
    export_bboxes,
    run_saga_pipeline,
    load_saga_config,
)


def run_pipeline(cfg: dict) -> bool:
    """
    一键运行完整流程：COLMAP（可选）→ 训练 → 查看结果。

    使用 cfg['dataset']['use_colmap'] 决定是否执行 COLMAP。
    使用 cfg['viewer']['backend'] 决定查看器类型。
    """
    from datetime import datetime
    from .utils import logger

    logger.info("=" * 50)
    logger.info("Pipeline 开始")
    logger.info("=" * 50)
    start = datetime.now()

    # Step 1: COLMAP
    if cfg["dataset"].get("use_colmap", False):
        print("\n[1/2] COLMAP 相机标定 ...")
        if not run_colmap(cfg):
            print("\n✗ Pipeline 中止：COLMAP 失败")
            return False
    else:
        print("\n[1/2] 跳过 COLMAP（use_colmap=false）")

    # Step 2: 训练
    print("\n[2/2] 3DGS 训练 ...")
    if not run_training(cfg):
        print("\n✗ Pipeline 中止：训练失败")
        return False

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n{'='*50}")
    print(f"✅ Pipeline 完成！（{elapsed:.1f} 分钟）")
    print(f"{'='*50}")
    logger.info(f"Pipeline 完成，耗时 {elapsed:.1f}min")

    analyze_results(cfg)

    # 查看
    backend = cfg.get("viewer", {}).get("backend", "open3d")
    if backend == "open3d":
        open_viewer(cfg)
    elif backend == "sibr":
        launch_sibr(cfg)

    return True


__all__ = [
    "load_config",
    "find_project_root",
    "resolve_path",
    "setup_logger",
    "PROJECT_ROOT",
    "GS_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "LOG_DIR",
    "check_environment",
    "extract_frames",
    "run_colmap",
    "fix_dataset_layout",
    "run_training",
    "analyze_results",
    "open_viewer",
    "launch_sibr",
    "find_latest_ply",
    "find_latest_model",
    "run_pipeline",
    # SAGA
    "check_saga_ready",
    "download_sam_checkpoint",
    "create_downsampled_images",
    "extract_sam_masks",
    "extract_sam_features",
    "train_saga_features",
    "compute_scales",
    "open_saga_gui",
    "open_bbox_viewer",
    "open_saga_notebook",
    "query_by_text",
    "get_3d_bbox_from_mask",
    "export_bboxes",
    "run_saga_pipeline",
    "load_saga_config",
]
