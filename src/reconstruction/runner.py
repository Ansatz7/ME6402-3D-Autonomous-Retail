from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List


class ReconstructionRunner:
    """Command orchestrator for COLMAP + 3DGS reconstruction workflow."""

    def __init__(
        self,
        image_dir: str,
        output_dir: str,
        gs_repo_dir: str = "third_party/gaussian-splatting",
        model_dir: str | None = None,
        dry_run: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.gs_repo_dir = Path(gs_repo_dir)
        self.model_dir = Path(model_dir) if model_dir else self.output_dir / "gs_model"
        self.dry_run = dry_run
        self.repo_root = Path(__file__).resolve().parents[2]

    def validate_inputs(self) -> None:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _run(self, cmd: List[str]) -> None:
        if self.dry_run:
            return
        subprocess.run(cmd, check=True)

    def build_commands(self) -> List[List[str]]:
        setup_script = self.repo_root / "scripts" / "reconstruction" / "setup_3dgs.sh"
        colmap_script = self.repo_root / "scripts" / "reconstruction" / "run_colmap.sh"
        train_script = self.repo_root / "scripts" / "reconstruction" / "run_3dgs_train.sh"
        dense_dir = self.output_dir / "dense"

        return [
            ["bash", str(setup_script)],
            ["bash", str(colmap_script), str(self.image_dir), str(self.output_dir)],
            [
                "bash",
                str(train_script),
                str(self.gs_repo_dir),
                str(dense_dir),
                str(self.model_dir),
            ],
        ]

    def run(self) -> dict:
        self.validate_inputs()
        commands = self.build_commands()
        for cmd in commands:
            self._run(cmd)
        return {
            "status": "dry-run" if self.dry_run else "executed",
            "message": "Reconstruction pipeline commands prepared.",
            "image_dir": str(self.image_dir),
            "output_dir": str(self.output_dir),
            "gs_repo_dir": str(self.gs_repo_dir),
            "model_dir": str(self.model_dir),
            "commands": [" ".join(cmd) for cmd in commands],
        }


if __name__ == "__main__":
    runner = ReconstructionRunner(
        image_dir="data/raw",
        output_dir="data/processed/reconstruction",
        dry_run=True,
    )
    print(runner.run())
