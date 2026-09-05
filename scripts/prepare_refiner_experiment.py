"""Stage the shared inference script for a short, separate Kaggle refiner experiment."""
from __future__ import annotations

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/coco_refiner_kernel"
    output.mkdir(parents=True, exist_ok=True)
    source = (root / "kaggle_inference/optimize_inference.py").read_text()
    replacements = {"DEFAULT_REFINER_EPOCHS = 0": "DEFAULT_REFINER_EPOCHS = 4",
                    "RESOLUTIONS = (1024, 1536, 2048)": "RESOLUTIONS = (1024,)"}
    for original, replacement in replacements.items():
        if source.count(original) != 1:
            raise ValueError(f"Cannot locate experiment setting: {original}")
        source = source.replace(original, replacement)
    (output / "optimize_inference.py").write_text(source)
    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    metadata.update(id="abhimanyu122/solar-filament-coco-refiner",
                    title="Solar Filament COCO Refiner")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
