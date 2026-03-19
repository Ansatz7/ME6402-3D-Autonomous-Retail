from __future__ import annotations

from planogram_evaluator import PlanogramEvaluator


def main() -> None:
    evaluator = PlanogramEvaluator(d_max=0.2, iou_min=0.5, alpha=0.6, beta=0.4)

    detected_center = (0.2, 0.3, 0.4)
    planogram_center = (0.18, 0.25, 0.45)
    detected_box = (0.1, 0.2, 0.3, 0.3, 0.4, 0.5)
    planogram_box = (0.12, 0.18, 0.28, 0.32, 0.42, 0.52)

    result = evaluator.evaluate(
        detected_center=detected_center,
        planogram_center=planogram_center,
        detected_box=detected_box,
        planogram_box=planogram_box,
    )
    print(result)


if __name__ == "__main__":
    main()
