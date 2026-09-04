from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        "filament-segmentation-2026",
        "-p",
        str(output_dir),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
