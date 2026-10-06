"""V37 G2: bounded RF-DETR-Seg-Nano memorization gate on 16 fit records only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.train_v37_rfdetr_gate import (
    ANNOTATION_SHA256,
    BATCH_SIZE,
    CHECKPOINT_MD5,
    CHECKPOINT_SHA256,
    MASK_DOWNSAMPLE_RATIO,
    MEMORY_GATE_MIB,
    RFDETR_VERSION,
    RESOLUTION,
    SEED,
    SELECTION_ID,
    MemoryMonitor,
    build_dataset,
    digest,
    finite_losses,
    package_versions,
    write_json,
)

EPOCHS = 125
MAX_STEPS = 2000
CHECKPOINT_INTERVAL = 25
CONFIDENCE_GRID = (0.05, 0.10, 0.20, 0.30, 0.50)
PQ_GATE = 0.85
RECALL_GATE = 0.90
MEDIAN_IOU_GATE = 0.80


def decoded_ground_truth(dataset: Path) -> list[dict]:
    from pycocotools import mask as mask_utils

    document = json.loads((dataset / "train/_annotations.coco.json").read_text())
    by_image: dict[int, list[dict]] = {}
    for annotation in document["annotations"]:
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    records = []
    for info in sorted(document["images"], key=lambda row: int(row["id"])):
        masks = []
        for annotation in by_image.get(int(info["id"]), []):
            segmentation = annotation["segmentation"]
            rles = mask_utils.frPyObjects(segmentation, int(info["height"]), int(info["width"]))
            masks.append(mask_utils.decode(mask_utils.merge(rles)).astype(bool))
        records.append({"alias": info["file_name"], "masks": masks})
    if len(records) != 16 or sum(len(row["masks"]) for row in records) != 57:
        raise RuntimeError("V37 G2 decoded population changed")
    return records


def strict_match(predictions: list[np.ndarray], targets: list[np.ndarray]) -> tuple[list[float], int, int, int]:
    if not predictions or not targets:
        return [], 0, len(predictions), len(targets)
    pred_areas = np.asarray([int(mask.sum()) for mask in predictions], dtype=np.float64)
    target_areas = np.asarray([int(mask.sum()) for mask in targets], dtype=np.float64)
    ious = np.zeros((len(predictions), len(targets)), dtype=np.float64)
    for pred_index, prediction in enumerate(predictions):
        for target_index, target in enumerate(targets):
            intersection = int(np.logical_and(prediction, target).sum())
            union = pred_areas[pred_index] + target_areas[target_index] - intersection
            if union > 0:
                ious[pred_index, target_index] = intersection / union
    rows, columns = linear_sum_assignment(-ious)
    matched = [float(ious[row, column]) for row, column in zip(rows, columns) if ious[row, column] > 0.5]
    tp = len(matched)
    return matched, tp, len(predictions) - tp, len(targets) - tp


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)", path.stem)
    if not match:
        raise ValueError(f"cannot parse checkpoint epoch: {path}")
    # Lightning names checkpoints with the zero-based completed epoch.
    return int(match.group(1)) + 1


def evaluate_checkpoint(checkpoint: Path, dataset: Path) -> dict:
    from rfdetr import RFDETRSegNano

    records = decoded_ground_truth(dataset)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = RFDETRSegNano(
            num_classes=1,
            resolution=RESOLUTION,
            pretrain_weights=str(checkpoint),
            mask_downsample_ratio=MASK_DOWNSAMPLE_RATIO,
            gradient_checkpointing=True,
            device="cuda",
        )
        load_warnings = [str(item.message) for item in caught]
    partial = [message for message in load_warnings if "loaded only partially" in message.lower()]
    if partial:
        raise RuntimeError(f"partial G2 checkpoint load: {partial}")

    totals = {
        threshold: {"ious": [], "tp": 0, "fp": 0, "fn": 0, "per_record": []}
        for threshold in CONFIDENCE_GRID
    }
    for record in records:
        image = dataset / "train" / record["alias"]
        detections = model.predict(str(image), threshold=min(CONFIDENCE_GRID), include_source_image=False)
        masks = detections.mask
        confidence = detections.confidence
        if masks is None or confidence is None:
            masks = np.zeros((0, 2048, 2048), dtype=bool)
            confidence = np.zeros((0,), dtype=np.float32)
        if masks.shape[1:] != (2048, 2048):
            raise RuntimeError(f"source-resolution prediction changed: {masks.shape}")
        for threshold in CONFIDENCE_GRID:
            selected = [mask.astype(bool, copy=False) for mask, score in zip(masks, confidence) if float(score) >= threshold and bool(mask.any())]
            matched, tp, fp, fn = strict_match(selected, record["masks"])
            cell = totals[threshold]
            cell["ious"].extend(matched)
            cell["tp"] += tp; cell["fp"] += fp; cell["fn"] += fn
            cell["per_record"].append({"alias": record["alias"], "tp": tp, "fp": fp, "fn": fn, "iou_sum": sum(matched)})

    cells = []
    for threshold in CONFIDENCE_GRID:
        cell = totals[threshold]
        denominator = cell["tp"] + 0.5 * cell["fp"] + 0.5 * cell["fn"]
        pq = sum(cell["ious"]) / denominator if denominator else 0.0
        recall = cell["tp"] / 57
        median_iou = float(np.median(cell["ious"])) if cell["ious"] else 0.0
        cells.append({
            "threshold": threshold,
            "pq": pq,
            "recall": recall,
            "median_matched_iou": median_iou,
            "iou_sum": sum(cell["ious"]),
            "tp": cell["tp"], "fp": cell["fp"], "fn": cell["fn"],
            "per_record": cell["per_record"],
        })
    del model
    import torch
    torch.cuda.empty_cache()
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint), "epoch": checkpoint_epoch(checkpoint), "load_warnings": load_warnings, "cells": cells}


def run(args) -> dict:
    started = time.monotonic()
    result = {
        "status": "started", "experiment": "v37_rfdetr_seg_nano_azure_g2",
        "selection_id": SELECTION_ID, "resolution": RESOLUTION,
        "mask_downsample_ratio": MASK_DOWNSAMPLE_RATIO, "epochs": EPOCHS,
        "max_steps": MAX_STEPS, "batch_size": BATCH_SIZE,
        "confidence_grid": list(CONFIDENCE_GRID), "memory_gate_mib": MEMORY_GATE_MIB,
        "package_versions": package_versions(),
        "reads_tune": False, "reads_audit": False, "reads_test": False,
        "exports": False, "submits": False,
    }
    monitor = MemoryMonitor()
    try:
        import torch
        from rfdetr import RFDETRSegNano

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        gpu_capability = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        result["gpu"] = {"name": gpu_name, "capability": gpu_capability}
        if torch.cuda.device_count() != 1 or gpu_name != "Tesla T4" or gpu_capability != [7, 5]:
            raise RuntimeError(f"V37 G2 requires one visible Tesla T4/7.5, got {result['gpu']}")
        if result["package_versions"]["rfdetr"] != RFDETR_VERSION:
            raise RuntimeError(f"expected rfdetr {RFDETR_VERSION}")
        if digest(args.annotations) != ANNOTATION_SHA256:
            raise RuntimeError("canonical annotation identity changed")
        if digest(args.checkpoint) != CHECKPOINT_SHA256 or digest(args.checkpoint, "md5") != CHECKPOINT_MD5:
            raise RuntimeError("official checkpoint identity changed")

        manifest = build_dataset(args.annotations, args.images, args.work / "dataset")
        result["dataset"] = manifest
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()
        model = RFDETRSegNano(
            num_classes=1, resolution=RESOLUTION, pretrain_weights=str(args.checkpoint),
            mask_downsample_ratio=MASK_DOWNSAMPLE_RATIO, gradient_checkpointing=True, device="cuda",
        )
        train_output = args.work / "training"
        with monitor:
            model.train(
                dataset_dir=str(args.work / "dataset"), output_dir=str(train_output),
                epochs=EPOCHS, batch_size=BATCH_SIZE, eval_batch_size=1,
                grad_accum_steps=1, lr=1e-4, num_workers=0, seed=SEED,
                early_stopping=False, eval_interval=CHECKPOINT_INTERVAL, use_ema=False,
                run_test=False, tensorboard=False, wandb=False,
                multi_scale=False, scale_jitter=False, aug_config={},
                augmentation_backend="torchvision", checkpoint_interval=CHECKPOINT_INTERVAL,
                device="cuda", save_dataset_grids=False, progress_bar="tqdm",
                compute_val_loss=True,
                notes={"experiment": "v37_azure_g2", "selection_id": SELECTION_ID},
            )
        result["losses"] = finite_losses(train_output)
        del model; torch.cuda.empty_cache()
        checkpoints = sorted(train_output.glob("checkpoint_*.ckpt"), key=checkpoint_epoch)
        expected_epochs = [25, 50, 75, 100, 125]
        observed_epochs = [checkpoint_epoch(path) for path in checkpoints]
        if observed_epochs != expected_epochs:
            raise RuntimeError(f"periodic checkpoints changed: {observed_epochs}")
        result["evaluations"] = [evaluate_checkpoint(path, args.work / "dataset") for path in checkpoints]
        candidates = [
            {**cell, "epoch": evaluation["epoch"], "checkpoint": evaluation["checkpoint"], "checkpoint_sha256": evaluation["checkpoint_sha256"]}
            for evaluation in result["evaluations"] for cell in evaluation["cells"]
        ]
        selected = max(candidates, key=lambda row: (row["pq"], row["recall"], row["median_matched_iou"], -row["threshold"], -row["epoch"]))
        result["selected"] = selected
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "error"; result["error_type"] = type(error).__name__
        result["error"] = str(error); result["traceback"] = traceback.format_exc()
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["peak_external_mib"] = monitor.peak_mib; result["memory_samples"] = monitor.samples
        try:
            import torch
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
        except Exception:
            result["peak_allocated_mib"] = None
        selected = result.get("selected", {})
        completed_steps = EPOCHS * 16 if result["status"] == "complete" else None
        result["completed_steps"] = completed_steps
        result["gates"] = {
            "status_complete": result["status"] == "complete",
            "finite_losses": result.get("losses", {}).get("all_finite", False),
            "pq": selected.get("pq", 0.0) >= PQ_GATE,
            "recall": selected.get("recall", 0.0) >= RECALL_GATE,
            "median_matched_iou": selected.get("median_matched_iou", 0.0) >= MEDIAN_IOU_GATE,
            "memory": 0 < result["peak_external_mib"] < MEMORY_GATE_MIB,
            "step_budget": completed_steps is not None and completed_steps <= MAX_STEPS,
            "data_firewall": not any((result["reads_tune"], result["reads_audit"], result["reads_test"], result["exports"], result["submits"])),
        }
        result["g2_passed"] = all(result["gates"].values())
        result["full_design_authorized"] = result["g2_passed"]
        result["action"] = "freeze V37 full-population design" if result["g2_passed"] else "close V37 before full training"
        write_json(args.output, result)
        print(json.dumps({key: result.get(key) for key in ("status", "elapsed_seconds", "peak_external_mib", "selected", "gates", "g2_passed", "action")}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
