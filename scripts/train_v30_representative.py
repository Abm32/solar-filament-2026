"""V30 public-regime training: native YOLOv8l-2048 on representative labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

MODEL_SHA256 = "ca3ec94c445aeaf79c2b57982fb451060f18a7561e2c20b4d26bb9dc1fcabf93"
EPOCHS = 40


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.model) != MODEL_SHA256:
        raise RuntimeError("official YOLOv8l-seg checkpoint changed")
    manifest = json.loads((args.data.parent / "manifest.json").read_text())
    if manifest["train_files"] != 566:
        raise RuntimeError("representative training population changed")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    model.train(
        data=str(args.data), epochs=EPOCHS, imgsz=2048, batch=4,
        device=[0, 1, 2, 3], workers=2, optimizer="AdamW", lr0=1e-4, lrf=0.1,
        momentum=0.937, weight_decay=5e-4, warmup_epochs=1.0, patience=0,
        seed=20260913, deterministic=True, single_cls=True, mask_ratio=4,
        overlap_mask=True, amp=True, rect=False, multi_scale=0.0, cache=False,
        mosaic=0.3, close_mosaic=5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        degrees=10.0, translate=0.1, scale=0.3, fliplr=0.5, flipud=0.5,
        plots=False, val=False, save=True, save_period=5,
        project=str(args.output / "runs"), name="representative_yolov8l_native2048",
        exist_ok=False, verbose=True,
    )


if __name__ == "__main__":
    main()
