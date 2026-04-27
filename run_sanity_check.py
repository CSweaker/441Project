"""Quick smoke test for the project.

This runs a small training job and verifies that reports/artifacts are created.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    reports_dir = BASE_DIR / "reports" / "sanity_check"
    artifact_path = BASE_DIR / "artifacts" / "sanity_ids_artifacts.joblib"
    cmd = [
        sys.executable,
        str(BASE_DIR / "train_pipeline.py"),
        "--quick",
        "--sample-size",
        "3000",
        "--test-sample-size",
        "1000",
        "--out",
        str(reports_dir),
        "--artifacts",
        str(artifact_path),
    ]
    subprocess.run(cmd, check=True)
    required = [
        reports_dir / "run_summary.json",
        reports_dir / "binary_model_comparison.csv",
        artifact_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Smoke test did not create: {missing}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
