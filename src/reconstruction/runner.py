from __future__ import annotations

from pathlib import Path


class ReconstructionRunner:
    """Placeholder runner for COLMAP + 3DGS reconstruction workflow."""

    def __init__(self, image_dir: str, output_dir: str) -> None:
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)

    def validate_inputs(self) -> None:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        self.validate_inputs()
        return {
            "status": "stub",
            "message": "Reconstruction pipeline placeholder created.",
            "image_dir": str(self.image_dir),
            "output_dir": str(self.output_dir),
        }


if __name__ == "__main__":
    runner = ReconstructionRunner("data/raw", "data/processed/reconstruction")
    print(runner.run())
