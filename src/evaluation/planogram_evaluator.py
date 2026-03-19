from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


Box3D = Tuple[float, float, float, float, float, float]


@dataclass
class PlanogramEvaluator:
    """Evaluate planogram compliance using geometric and threshold metrics."""

    d_max: float = 0.15
    iou_min: float = 0.5
    alpha: float = 0.5
    beta: float = 0.5

    def __post_init__(self) -> None:
        if self.d_max <= 0:
            raise ValueError("d_max must be > 0")
        if not 0.0 <= self.iou_min <= 1.0:
            raise ValueError("iou_min must be within [0, 1]")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if self.alpha + self.beta == 0:
            raise ValueError("alpha + beta must be > 0")

    @staticmethod
    def position_deviation(
        detected_center: Sequence[float], planogram_center: Sequence[float]
    ) -> float:
        """Euclidean distance between detected center and planogram center."""
        detected = np.asarray(detected_center, dtype=float)
        reference = np.asarray(planogram_center, dtype=float)

        if detected.shape != (3,) or reference.shape != (3,):
            raise ValueError("centers must be 3D vectors of shape (3,)")

        return float(np.linalg.norm(detected - reference))

    @staticmethod
    def iou_3d(box_a: Box3D, box_b: Box3D) -> float:
        """Compute 3D IoU for axis-aligned boxes in (xmin, ymin, zmin, xmax, ymax, zmax)."""
        a = np.asarray(box_a, dtype=float)
        b = np.asarray(box_b, dtype=float)

        if a.shape != (6,) or b.shape != (6,):
            raise ValueError("boxes must be shape (6,)")
        if np.any(a[:3] > a[3:]) or np.any(b[:3] > b[3:]):
            raise ValueError("invalid box: min coordinates must be <= max coordinates")

        inter_min = np.maximum(a[:3], b[:3])
        inter_max = np.minimum(a[3:], b[3:])
        inter_dim = np.maximum(0.0, inter_max - inter_min)
        inter_vol = float(np.prod(inter_dim))

        vol_a = float(np.prod(np.maximum(0.0, a[3:] - a[:3])))
        vol_b = float(np.prod(np.maximum(0.0, b[3:] - b[:3])))
        union = vol_a + vol_b - inter_vol

        if union <= 0.0:
            return 0.0
        return inter_vol / union

    def pcr(self, delta_d: float, iou: float) -> int:
        """Binary compliance decision based on threshold rule."""
        return int(delta_d <= self.d_max and iou >= self.iou_min)

    def wci(self, delta_d: float, iou: float) -> float:
        """Continuous weighted compliance index.

        Formula:
            score = alpha * max(0, 1 - delta_d / d_max) + beta * iou
        """
        distance_score = max(0.0, 1.0 - (delta_d / self.d_max))
        raw = self.alpha * distance_score + self.beta * float(iou)
        normalizer = self.alpha + self.beta
        return float(raw / normalizer)

    def evaluate(
        self,
        detected_center: Sequence[float],
        planogram_center: Sequence[float],
        detected_box: Box3D,
        planogram_box: Box3D,
    ) -> dict:
        """Convenience wrapper to compute all metrics for one instance."""
        delta_d = self.position_deviation(detected_center, planogram_center)
        iou = self.iou_3d(detected_box, planogram_box)
        return {
            "delta_d": delta_d,
            "iou_3d": iou,
            "pcr": self.pcr(delta_d, iou),
            "wci": self.wci(delta_d, iou),
        }
