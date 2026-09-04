from __future__ import annotations

import zipfile
from pathlib import Path


def main() -> None:
    raw_dir = Path("data/raw")
    zip_files = sorted(raw_dir.glob("*.zip"))
    if not zip_files:
        raise SystemExit("No zip files found in data/raw. Run the Kaggle download command first.")

    for zip_path in zip_files:
        target_dir = raw_dir / zip_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target_dir)
        print(f"extracted {zip_path} -> {target_dir}")


if __name__ == "__main__":
    main()

