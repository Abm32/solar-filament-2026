"""V25: missing-label-aware uncertain-background classification loss.

Modes prepare/verify build the unchanged all-annotator dataset plus 2,659 omitted-
rater uncertainty boxes. Smoke/train modes use the same custom runtime for both
arms; only uncertain-negative classification weight differs (1.0 control, 0.1 arm).
No mode evaluates audit/test data, exports, or submits.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from scripts import train_v24_consensus as v24

BASE = Path(os.environ.get("V25_BASE", "/mnt/solar-v25-missing-label-loss"))
DATA_ROOT = Path(os.environ.get(
    "V25_DATA_ROOT",
    "/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/"
    "MAGFiLO_1.0_Kaggle_2026",
))
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
RAW_IMAGES = DATA_ROOT / "train/train_images"
INPUTS = BASE / "inputs"
SOURCE_DETECTOR = INPUTS / "v17_detector.pt"
PROVENANCE = INPUTS / "v6_source/v6_provenance.json"
DATASET = BASE / "dataset/all_annotators"
UNCERTAINTY_JSON = BASE / "uncertainty_boxes.json"
OUTPUT = BASE / "outputs"

ANNOTATION_SHA256 = v24.ANNOTATION_SHA256
DETECTOR_SHA256 = v24.DETECTOR_SHA256
PROVENANCE_SHA256 = v24.PROVENANCE_SHA256
PRIMARY_SEED = 42
REPLICATION_SEED = 20260912
TRAIN_RESOLUTION = 896
BATCH = 1
EPOCHS = 20
CONTROL_WEIGHT = 1.0
ARM_WEIGHT = 0.1
MEMORY_GATE_MIB = 15872
EXPECTED_UNCERTAINTY = 2659
EXPECTED_RECORDS_WITH_UNCERTAINTY = 541


def configure_v24_paths() -> None:
    v24.ANNOTATIONS = ANNOTATIONS
    v24.RAW_IMAGES = RAW_IMAGES
    v24.PROVENANCE = PROVENANCE


def omitted_regions(images: dict, annotations: dict, partitions: dict) -> tuple[dict, dict]:
    mapping = {}
    support_histogram: dict[int, int] = {}
    records_with_uncertainty = 0
    for name in partitions["train_files"]:
        image_ids = partitions["by_file"][name]
        clusters = v24.consensus_clusters(image_ids, images, annotations)
        for cluster in clusters:
            support_histogram[cluster["support"]] = support_histogram.get(cluster["support"], 0) + 1
        for image_id in image_ids:
            own = {str(annotation["id"]) for annotation in annotations.get(image_id, [])}
            missing = [
                cluster for cluster in clusters
                if not own.intersection(cluster["member_annotation_ids"])
            ]
            stem = f"{Path(name).stem}_ann{image_id}"
            entries = []
            info = images[image_id]
            for cluster in missing:
                annotation = cluster["annotation"]
                x, y, width, height = map(float, annotation["bbox"])
                if width <= 0 or height <= 0:
                    raise ValueError(f"invalid uncertainty bbox in {stem}")
                entries.append({
                    "annotation_id": str(annotation["id"]),
                    "support": int(cluster["support"]),
                    "xywhn": [
                        (x + width / 2) / info["width"],
                        (y + height / 2) / info["height"],
                        width / info["width"],
                        height / info["height"],
                    ],
                })
            if entries:
                mapping[stem] = sorted(entries, key=lambda entry: entry["annotation_id"])
                records_with_uncertainty += 1
    summary = {
        "uncertainty_instances": sum(map(len, mapping.values())),
        "records_with_uncertainty": records_with_uncertainty,
        "max_per_record": max(map(len, mapping.values())),
        "support_histogram": dict(sorted(support_histogram.items())),
    }
    return mapping, summary


def prepare() -> None:
    configure_v24_paths()
    if v24.sha256(ANNOTATIONS) != ANNOTATION_SHA256:
        raise RuntimeError("canonical annotation identity changed")
    if v24.sha256(PROVENANCE) != PROVENANCE_SHA256:
        raise RuntimeError("V6 provenance identity changed")
    _, images, annotations = v24.load_population()
    provenance = json.loads(PROVENANCE.read_text())
    partitions = v24.dataset_partitions(images, annotations, provenance)
    mapping, summary = omitted_regions(images, annotations, partitions)
    if summary["uncertainty_instances"] != EXPECTED_UNCERTAINTY:
        raise RuntimeError(f"uncertainty count changed: {summary}")
    if summary["records_with_uncertainty"] != EXPECTED_RECORDS_WITH_UNCERTAINTY:
        raise RuntimeError(f"uncertainty record count changed: {summary}")
    if DATASET.exists() or UNCERTAINTY_JSON.exists():
        raise ValueError("V25 prepared data already exists; use verify")
    empty_consensus = {name: [] for name in partitions["train_files"]}
    dataset_manifest = v24.build_dataset(
        DATASET, "all_annotators", images, annotations, partitions, empty_consensus, "control"
    )
    v24.write_json(UNCERTAINTY_JSON, mapping)
    uncertainty_id = v24.digest_json(mapping)
    payload = {
        "status": "complete",
        "annotation_sha256": ANNOTATION_SHA256,
        "source_detector_sha256": DETECTOR_SHA256,
        "dataset_manifest": dataset_manifest,
        "train_files": len(partitions["train_files"]),
        "train_records": len(partitions["train_ids"]),
        "positive_instances": sum(len(annotations[image_id]) for image_id in partitions["train_ids"]),
        "uncertainty_id": uncertainty_id,
        "uncertainty_summary": summary,
        "weights": {"control": CONTROL_WEIGHT, "uncertain": ARM_WEIGHT},
    }
    v24.write_json(BASE / "experiment_manifest.json", payload)
    print(json.dumps(payload, indent=2))


def training_arguments(epochs: int = EPOCHS, seed: int = PRIMARY_SEED) -> dict:
    return dict(
        epochs=epochs, patience=0, batch=BATCH, imgsz=TRAIN_RESOLUTION, workers=4,
        resume=False, optimizer="AdamW", lr0=1e-4, lrf=0.1, momentum=0.9,
        weight_decay=5e-4, cos_lr=False, warmup_epochs=1.0, warmup_bias_lr=0.0,
        nbs=64, seed=seed, deterministic=True, single_cls=True, mask_ratio=1,
        overlap_mask=False, amp=True, rect=False, multi_scale=0.0, fraction=1.0,
        freeze=None, hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=10, translate=0.1,
        scale=0.3, shear=0.0, perspective=0.0, fliplr=0.0, flipud=0.0, bgr=0.0,
        mosaic=0.3, close_mosaic=5 if epochs > 1 else 0, mixup=0.0, cutmix=0.0,
        copy_paste=0.0, save_period=5, val=False, plots=False,
    )


def verify() -> dict:
    if v24.sha256(SOURCE_DETECTOR) != DETECTOR_SHA256:
        raise RuntimeError("V17 detector identity changed")
    if v24.sha256(PROVENANCE) != PROVENANCE_SHA256:
        raise RuntimeError("V6 provenance identity changed")
    manifest = json.loads((BASE / "experiment_manifest.json").read_text())
    mapping = json.loads(UNCERTAINTY_JSON.read_text())
    if v24.digest_json(mapping) != manifest["uncertainty_id"]:
        raise RuntimeError("uncertainty mapping identity changed")
    if sum(map(len, mapping.values())) != EXPECTED_UNCERTAINTY:
        raise RuntimeError("uncertainty population changed")
    for split, expected in (("train", 937), ("val", 141)):
        images = list((DATASET / "images" / split).glob("*.jpeg"))
        labels = list((DATASET / "labels" / split).glob("*.txt"))
        if len(images) != expected or len(labels) != expected:
            raise RuntimeError(f"{split} population mismatch")
    payload = {
        "status": "verified",
        "source_detector_sha256": v24.sha256(SOURCE_DETECTOR),
        "annotation_sha256": v24.sha256(ANNOTATIONS),
        "uncertainty_id": manifest["uncertainty_id"],
        "uncertainty_instances": EXPECTED_UNCERTAINTY,
        "training_arguments": training_arguments(),
        "memory_gate_mib": MEMORY_GATE_MIB,
        "submits": False,
    }
    v24.write_json(BASE / "verification.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def train(policy: str, smoke: bool, seed: int) -> None:
    import pandas as pd
    import torch
    from ultralytics import YOLO
    from scripts import v25_loss_runtime

    verification = verify()
    if policy not in ("control", "uncertain"):
        raise ValueError(f"unknown policy {policy}")
    weight = CONTROL_WEIGHT if policy == "control" else ARM_WEIGHT
    trainer = v25_loss_runtime.configure(UNCERTAINTY_JSON, weight)
    seed_label = "seed1" if seed == PRIMARY_SEED else "seed2"
    stage = "smoke" if smoke else "train"
    output = OUTPUT / stage / policy / seed_label
    marker, final = output / "result.json", output / "final.pt"
    arguments = training_arguments(epochs=1 if smoke else EPOCHS, seed=seed)
    signature = {
        "policy": policy,
        "uncertainty_weight": weight,
        "seed": seed,
        "smoke": smoke,
        "source_sha256": v24.sha256(SOURCE_DETECTOR),
        "uncertainty_id": verification["uncertainty_id"],
        "dataset_manifest": json.loads((DATASET / "manifest.json").read_text()),
        "arguments": arguments,
    }
    if marker.is_file() and final.is_file():
        saved = json.loads(marker.read_text())
        if saved["signature"] != signature or saved["checkpoint_sha256"] != v24.sha256(final):
            raise RuntimeError("saved V25 stage identity changed")
        print(json.dumps(saved, indent=2)); return
    if output.exists():
        raise RuntimeError(f"refusing to overwrite partial stage {output}")
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = YOLO(str(SOURCE_DETECTOR))
    model.train(
        trainer=trainer,
        data=str(DATASET / "data.yaml"),
        device="0",
        project=str(output),
        name="run",
        exist_ok=False,
        **arguments,
    )
    elapsed = time.monotonic() - started
    shutil.copy2(Path(model.trainer.last), final)
    results_path = Path(model.trainer.save_dir) / "results.csv"
    results = pd.read_csv(results_path)
    criterion = getattr(model.trainer.model, "criterion", None)
    record = {
        "signature": signature,
        "elapsed_seconds": elapsed,
        "epochs_completed": len(results),
        "checkpoint_sha256": v24.sha256(final),
        "results_sha256": v24.sha256(results_path),
        "peak_allocated_mib": int(torch.cuda.max_memory_allocated(0)) / 2**20,
        "projected_hours": elapsed * (EPOCHS if smoke else 1) / 3600,
        "finite_results": bool(np.isfinite(results.select_dtypes(include=[np.number]).to_numpy()).all()),
        "last_uncertain_negative_anchors": getattr(criterion, "last_uncertain_negative_anchors", None),
        "last_total_negative_anchors": getattr(criterion, "last_total_negative_anchors", None),
        "total_uncertain_negative_anchors": getattr(criterion, "total_uncertain_negative_anchors", None),
        "total_negative_anchors": getattr(criterion, "total_negative_anchors", None),
        "loss_batches": getattr(criterion, "loss_batches", None),
    }
    if not record["finite_results"]:
        raise RuntimeError("non-finite V25 training result")
    v24.write_json(marker, record)
    print(json.dumps(record, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(
        "prepare", "verify", "smoke-control", "smoke-uncertain",
        "train-control", "train-uncertain",
    ))
    parser.add_argument("--seed", type=int, choices=(PRIMARY_SEED, REPLICATION_SEED),
                        default=PRIMARY_SEED)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); cv2.setNumThreads(2)
    if args.mode == "prepare": prepare()
    elif args.mode == "verify": verify()
    elif args.mode == "smoke-control": train("control", True, args.seed)
    elif args.mode == "smoke-uncertain": train("uncertain", True, args.seed)
    elif args.mode == "train-control": train("control", False, args.seed)
    else: train("uncertain", False, args.seed)


if __name__ == "__main__":
    main()
