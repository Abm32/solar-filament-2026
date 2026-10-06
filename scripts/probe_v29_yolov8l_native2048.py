"""V29 feasibility probe for the public 0.55 upstream model regime.

Runs official COCO-pretrained YOLOv8l-seg at native 2048 with Ultralytics'
default mask_ratio=4. This is deliberately different from V18, which used a
small YOLO11s warm start at 1536 with mask_ratio=1 and failed in mask loss.
The probe trains exactly one complete epoch on the canonical 937-record train
split so sustained allocator behavior and high-instance records are exercised.
It never reads audit/test images and never exports a submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

MODEL_NAME = "yolov8l-seg.pt"
IMAGE_SIZE = 2048
MASK_RATIO = 4
BATCH_SIZE = 1
EPOCHS = 1
EXPECTED_TRAIN_RECORDS = 937
EXPECTED_TRAIN_INSTANCES = 6606
EXPECTED_VAL_RECORDS = 141
SEED = 20260913


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_dataset(root: Path) -> dict:
    result = {}
    for split, expected_records in (("train", EXPECTED_TRAIN_RECORDS), ("val", EXPECTED_VAL_RECORDS)):
        images = sorted(p for p in (root / "images" / split).iterdir() if p.suffix.lower() == ".jpeg")
        labels = sorted((root / "labels" / split).glob("*.txt"))
        if len(images) != expected_records or len(labels) != expected_records:
            raise RuntimeError(
                f"{split} population changed: images={len(images)} labels={len(labels)} "
                f"expected={expected_records}"
            )
        if {p.stem for p in images} != {p.stem for p in labels}:
            raise RuntimeError(f"{split} image/label stems differ")
        result[f"{split}_records"] = len(images)
        result[f"{split}_instances"] = sum(
            sum(bool(line.strip()) for line in path.read_text().splitlines()) for path in labels
        )
    if result["train_instances"] != EXPECTED_TRAIN_INSTANCES:
        raise RuntimeError(f"train instances changed: {result['train_instances']}")
    return result


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def probe(data_yaml: Path, model_path: Path, output: Path) -> dict:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("V29 feasibility requires CUDA")
    dataset = count_dataset(data_yaml.parent)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = YOLO(str(model_path))
    arguments = dict(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
        device=0, workers=2, optimizer="AdamW", lr0=1e-4, lrf=0.1,
        warmup_epochs=1.0, patience=0, seed=SEED, deterministic=True,
        single_cls=True, mask_ratio=MASK_RATIO, overlap_mask=True, amp=True,
        rect=False, multi_scale=0.0, cache=False, mosaic=0.3, close_mosaic=0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=10.0, translate=0.1,
        scale=0.3, fliplr=0.5, flipud=0.5, plots=False, val=False,
        save=True, save_period=-1, project=str(output / "runs"), name="full_epoch_probe",
        exist_ok=False, verbose=True,
    )
    status = "complete"
    error = None
    try:
        model.train(**arguments)
    except torch.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc)[:1000]
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(0))
    result = {
        "status": status,
        "model": MODEL_NAME,
        "model_sha256": sha256(model_path),
        "dataset": dataset,
        "arguments": arguments,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": peak,
        "peak_allocated_mib": peak / 2**20,
        "cuda_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "error": error,
        "reads_audit": False,
        "reads_test": False,
        "exports_submission": False,
    }
    write_json(output / "feasibility.json", result)
    print(json.dumps(result, indent=2))
    if status != "complete":
        raise RuntimeError("V29 native-2048 YOLOv8l feasibility failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.data, args.model, args.output)


if __name__ == "__main__":
    main()
