# 3D-Vision Enabled Autonomous Retail System

ME6402 Course Project: Planogram Compliance and Auto-Checkout

## Project Overview
This project develops a smart retail checkout and planogram compliance system by elevating 2D object detection into a 3D interactive digital twin. The system targets occlusion-heavy and dense shelf environments, and tracks product states in reconstructed 3D space.

## System Pipeline
1. 2D Multi-View Capture: Collect shelf images from multiple viewpoints.
2. 3D Reconstruction: Use SfM (COLMAP) and 3D Gaussian Splatting (3DGS) to build a 3D scene.
3. 2D-to-3D Recognition: Detect 4 SKUs with YOLOv10 or RT-DETR and project detections into 3D.
4. Planogram Compliance Evaluation: Quantify spatial consistency using Delta D and 3D IoU.

## Evaluation Metrics
- PCR (Planogram Compliance Rate): Binary pass/fail check based on threshold rules.
- WCI (Weighted Compliance Index): Continuous weighted score per item.

## Tech Stack
- Reconstruction: COLMAP, 3D Gaussian Splatting
- Vision: YOLOv10 / RT-DETR, OpenCV, PyTorch
- Algorithms: Euclidean distance, 3D IoU, sequence alignment
