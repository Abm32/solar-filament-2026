"""V29 full training: COCO-pretrained YOLOv8l-seg at native 2048.

This reproduces the only model-generated public 0.55 regime. It trains on the
canonical 566-file / 937-annotator-record split and writes checkpoints only.
It never reads audit/test images and never exports or submits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

MODEL_SHA256 = "ca3ec94c445aeaf79c2b57982fb451060f18a7561e2c20b4d26bb9dc1fcabf93"
EPOCHS = 20
IMAGE_SIZE = 2048
MASK_RATIO = 4
GLOBAL_BATCH = 4
SEED = 20260913


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.model) != MODEL_SHA256:
        raise RuntimeError("official YOLOv8l-seg checkpoint identity changed")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    train_args = dict(
        data=str(args.data), epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=GLOBAL_BATCH,
        device=[0, 1, 2, 3], workers=2, optimizer="AdamW", lr0=1e-4, lrf=0.1,
        momentum=0.937, weight_decay=5e-4, warmup_epochs=1.0,
        patience=0, seed=SEED, deterministic=True, single_cls=True,
        mask_ratio=MASK_RATIO, overlap_mask=True, amp=True, rect=False,
        multi_scale=0.0, cache=False, mosaic=0.3, close_mosaic=5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=10.0, translate=0.1,
        scale=0.3, fliplr=0.5, flipud=0.5, plots=False, val=False,
        save=True, save_period=5, project=str(args.output / "runs"),
        name="yolov8l_native2048", exist_ok=False, verbose=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_contract.json").write_text(json.dumps({
        "model_sha256": MODEL_SHA256, "arguments": train_args,
        "train_population": {"physical_files": 566, "records": 937, "instances": 6606},
        "reads_audit": False, "reads_test": False, "submits": False,
    }, indent=2) + "\n")
    model.train(**train_args)


if __name__ == "__main__":
    main()
