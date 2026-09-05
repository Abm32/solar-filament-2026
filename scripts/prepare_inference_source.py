"""Package the accepted V6 checkpoints with verifiable split and file provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=root / "outputs/kaggle_kernel_v6")
    parser.add_argument("--output", type=Path, default=root / "outputs/v6_inference_source")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = next((root / "data/raw").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    coco = json.loads(annotations.read_text())
    files = sorted({image["file_name"] for image in coco["images"]})
    random.Random(42).shuffle(files)
    n_valid = round(len(files) * 0.2)
    for name in ("yolo_best.pt", "best_refiner.pt"):
        shutil.copy2(args.checkpoints / name, args.output / name)
    source = (root / "kaggle_kernel/train_and_submit.py").read_text()
    # Checkpoint metadata records this exact Ultralytics version.
    source = source.replace('"-q", "ultralytics"', '"-q", "ultralytics==8.4.138"')
    (args.output / "solar_base.py").write_text(source)
    provenance = {
        "experiment": "V6 all annotators single class",
        "kernel": "abhimanyu122/solar-filament-yolo-refiner/6",
        "submission_ref": 56017396,
        "public_score": 0.35,
        "ultralytics": "8.4.138",
        "seed": 42,
        "validation_files": sorted(files[:n_valid]),
        "training_files": sorted(files[n_valid:]),
        "annotation_sha256": sha256(annotations),
        "checkpoint_selection": "Detector mAP and refiner Dice on the same validation files",
        "sha256": {name: sha256(args.output / name) for name in
                   ("yolo_best.pt", "best_refiner.pt", "solar_base.py")},
    }
    (args.output / "v6_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"directory": str(args.output), "training_files": len(files) - n_valid,
                      "validation_files": n_valid, "sha256": provenance["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
