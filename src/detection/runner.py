from __future__ import annotations

from pathlib import Path


class DetectionRunner:
    """Placeholder runner for 2D detection and 2D-to-3D projection stage."""

    def __init__(self, image_dir: str, weights_path: str) -> None:
        self.image_dir = Path(image_dir)
        self.weights_path = Path(weights_path)

    def validate_inputs(self) -> None:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

    def run(self) -> dict:
        self.validate_inputs()
        return {
            "status": "stub",
            "message": "Detection pipeline placeholder created.",
            "image_dir": str(self.image_dir),
            "weights": str(self.weights_path),
        }


if __name__ == "__main__":
    runner = DetectionRunner("data/raw", "weights/model.pt")
    print(runner.run())
