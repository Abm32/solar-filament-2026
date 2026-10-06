"""V23: deterministic solar radial-flattening/CLAHE preprocessing and matched training.

Modes:
  prepare       Build transformed competition-only labeled images and both YOLO datasets.
  verify        Verify source identities, populations, transformed geometry and frozen config.
  smoke-control One-epoch two-GPU raw-pixel smoke.
  smoke-arm     One-epoch two-GPU transformed-pixel smoke.
  train-control 20-epoch two-GPU raw-pixel continuation from V17.
  train-arm     20-epoch two-GPU transformed-pixel continuation from V17.

Evaluation is separate and has no submission path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("V23_BASE", "/mnt/solar-v23-radial-clahe"))
DATA_ROOT = Path(os.environ.get(
    "V23_DATA_ROOT",
    "/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/"
    "MAGFiLO_1.0_Kaggle_2026",
))
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
RAW_IMAGES = DATA_ROOT / "train/train_images"
TRANSFORMED = BASE / "transformed/train_images"
RAW_DATASET = BASE / "datasets/raw"
ARM_DATASET = BASE / "datasets/transformed"
OUTPUT = BASE / "outputs"
SOURCE_DETECTOR = BASE / "inputs/v17_detector.pt"
PROVENANCE = BASE / "inputs/v6_source/v6_provenance.json"

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
DETECTOR_SHA256 = "ae476e663039fbbbfda77ec519635ca15e2d50d1fe9f1af54735f84dbdc251ef"
PROVENANCE_SHA256 = "ceb09f980d5be195db35f2954c26fd19ea7ffc0f318b2bae5b7c803504b17122"
EXPECTED = {"files": 707, "records": 1154, "annotations": 8199}
TRAIN_EXPECTED = {"files": 566, "records": 937, "annotations": 6606}
SEED = 42
REPLICATION_SEED = 20260912
GLOBAL_BATCH = 1
TRAIN_RESOLUTION = 896
PROFILE_SIZE = 512
RADIAL_BINS = 256
PROFILE_SIGMA_BINS = 15.0
DISK_LIMIT = 0.98
CENTER_TOLERANCE_FRACTION = 0.06
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fit_disk(gray: np.ndarray) -> dict:
    if gray.shape != (2048, 2048):
        raise ValueError(f"expected 2048x2048 image, found {gray.shape}")
    work = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("solar disk contour not found")
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    filled = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(filled, [largest], -1, 1, cv2.FILLED)
    disk_fraction = float(filled.mean())
    if abs(cx - 1023.5) > CENTER_TOLERANCE_FRACTION * 2048 or abs(cy - 1023.5) > CENTER_TOLERANCE_FRACTION * 2048:
        raise ValueError(f"disk center outside frozen tolerance: {(cx, cy)}")
    if not 0.35 * 2048 <= radius <= 0.55 * 2048:
        raise ValueError(f"disk radius outside frozen tolerance: {radius}")
    if not 0.45 <= disk_fraction <= 0.80:
        raise ValueError(f"disk fraction outside frozen tolerance: {disk_fraction}")
    return {"cx": float(cx), "cy": float(cy), "radius": float(radius),
            "disk_fraction": disk_fraction}


def radial_profile(gray: np.ndarray, disk: dict) -> np.ndarray:
    small = cv2.resize(gray, (PROFILE_SIZE, PROFILE_SIZE), interpolation=cv2.INTER_AREA)
    scale = PROFILE_SIZE / 2048.0
    cx, cy, radius = disk["cx"] * scale, disk["cy"] * scale, disk["radius"] * scale
    yy, xx = np.indices(small.shape, dtype=np.float32)
    radial = np.hypot(xx - cx, yy - cy) / radius
    valid = radial <= DISK_LIMIT
    indices = np.minimum((radial * RADIAL_BINS).astype(np.int32), RADIAL_BINS - 1)
    profile = np.full(RADIAL_BINS, np.nan, dtype=np.float32)
    for index in range(RADIAL_BINS):
        values = small[valid & (indices == index)]
        if values.size:
            profile[index] = float(np.median(values))
    finite = np.flatnonzero(np.isfinite(profile))
    if len(finite) < RADIAL_BINS * 0.9:
        raise ValueError(f"radial profile has only {len(finite)} finite bins")
    missing = np.flatnonzero(~np.isfinite(profile))
    profile[missing] = np.interp(missing, finite, profile[finite])
    profile = cv2.GaussianBlur(
        profile[None, :], (0, 0), sigmaX=PROFILE_SIGMA_BINS,
        borderType=cv2.BORDER_REPLICATE,
    )[0]
    if float(profile.min()) <= 1.0:
        raise ValueError("radial profile contains invalid background")
    return profile


def transform_image(gray: np.ndarray) -> tuple[np.ndarray, dict]:
    disk = fit_disk(gray)
    profile = radial_profile(gray, disk)
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    radial = np.hypot(xx - disk["cx"], yy - disk["cy"]) / disk["radius"]
    mask = radial <= DISK_LIMIT
    coordinates = np.clip(radial * (RADIAL_BINS - 1), 0, RADIAL_BINS - 1)
    low = np.floor(coordinates).astype(np.int32)
    high = np.minimum(low + 1, RADIAL_BINS - 1)
    fraction = coordinates - low
    background = profile[low] * (1.0 - fraction) + profile[high] * fraction
    reference = float(np.median(profile))
    flattened = gray.astype(np.float32) / np.maximum(background, 1.0) * reference
    values = flattened[mask]
    lo, hi = np.percentile(values, (2, 98))
    scaled = np.clip((flattened - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
    scaled = cv2.GaussianBlur(scaled, (3, 3), 0.7)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    enhanced = clahe.apply(scaled)
    enhanced[~mask] = 0
    before = gray[mask].astype(np.float32)
    after = enhanced[mask].astype(np.float32)
    diagnostics = {
        **disk,
        "profile_min": float(profile.min()), "profile_median": reference,
        "profile_max": float(profile.max()),
        "raw_p02": float(np.percentile(before, 2)),
        "raw_p50": float(np.percentile(before, 50)),
        "raw_p98": float(np.percentile(before, 98)),
        "transformed_p02": float(np.percentile(after, 2)),
        "transformed_p50": float(np.percentile(after, 50)),
        "transformed_p98": float(np.percentile(after, 98)),
        "raw_std": float(before.std()), "transformed_std": float(after.std()),
        "changed_fraction": float(np.mean(gray != enhanced)),
    }
    return enhanced, diagnostics


def load_population() -> tuple[dict, dict, dict]:
    if sha256(ANNOTATIONS) != ANNOTATION_SHA256:
        raise RuntimeError("canonical annotation identity changed")
    coco = json.loads(ANNOTATIONS.read_text())
    images = {image["id"]: image for image in coco["images"]}
    annotations: dict = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    actual = {
        "files": len({image["file_name"] for image in images.values()}),
        "records": len(images), "annotations": len(coco["annotations"]),
    }
    if actual != EXPECTED:
        raise RuntimeError(f"population mismatch: {actual} != {EXPECTED}")
    return coco, images, annotations


def best_annotator_ids(images: dict, annotations: dict) -> dict[str, object]:
    by_file: dict[str, list] = defaultdict(list)
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)

    def completeness(image_id):
        entries = annotations.get(image_id, [])
        return len(entries), sum(entry.get("area", 0) for entry in entries)

    return {name: max(ids, key=completeness) for name, ids in by_file.items()}


def dataset_partitions(images: dict, annotations: dict, provenance: dict) -> dict:
    by_file: dict[str, list] = defaultdict(list)
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    best = best_annotator_ids(images, annotations)
    files = sorted(best)
    random.Random(SEED).shuffle(files)
    held_count = round(len(files) * 0.2)
    held = files[:held_count]
    train_files = sorted(files[held_count:])
    if train_files != sorted(provenance["training_files"]):
        raise ValueError("reconstructed train split differs from V6 provenance")
    if sorted(held) != sorted(provenance["validation_files"]):
        raise ValueError("reconstructed held-out split differs from V6 provenance")
    train_ids = [image_id for name in train_files for image_id in sorted(by_file[name])]
    valid_ids = [best[name] for name in sorted(held)]
    instances = sum(len(annotations.get(image_id, [])) for image_id in train_ids)
    actual = {"files": len(train_files), "records": len(train_ids), "annotations": instances}
    if actual != TRAIN_EXPECTED:
        raise ValueError(f"train population mismatch: {actual}")
    return {"train_files": train_files, "valid_files": sorted(held),
            "train_ids": train_ids, "valid_ids": valid_ids}


def polygon_line(polygon, width: int, height: int) -> str:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError("polygon has fewer than three points")
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in points.ravel())


def build_dataset(root: Path, image_root: Path, images: dict, annotations: dict,
                  partitions: dict, transform_id: str) -> dict:
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text())
        verify_dataset(root, partitions)
        return saved
    if root.exists():
        raise ValueError(f"refusing to overwrite unverified dataset {root}")
    counts = {}
    for split, image_ids, unique in (
        ("train", partitions["train_ids"], True),
        ("val", partitions["valid_ids"], False),
    ):
        image_out, label_out = root / "images" / split, root / "labels" / split
        image_out.mkdir(parents=True); label_out.mkdir(parents=True)
        labels = 0
        for image_id in image_ids:
            info = images[image_id]
            source = image_root / info["file_name"]
            if not source.is_file():
                raise ValueError(f"missing dataset source {source}")
            original = Path(info["file_name"])
            target_name = f"{original.stem}_ann{image_id}{original.suffix}" if unique else original.name
            (image_out / target_name).symlink_to(source)
            lines = [
                polygon_line(polygon, info["width"], info["height"])
                for annotation in annotations.get(image_id, [])
                for polygon in annotation["segmentation"] if len(polygon) >= 6
            ]
            labels += len(lines)
            (label_out / f"{Path(target_name).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
        counts[split] = {"records": len(image_ids), "labels": labels}
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n"
    )
    manifest = {"transform_id": transform_id, "counts": counts,
                "annotation_sha256": ANNOTATION_SHA256}
    write_json(manifest_path, manifest)
    verify_dataset(root, partitions)
    return manifest


def verify_dataset(root: Path, partitions: dict) -> dict:
    result = {}
    for split, expected in (("train", len(partitions["train_ids"])),
                            ("val", len(partitions["valid_ids"]))):
        image_paths = sorted((root / "images" / split).glob("*.jpeg"))
        label_paths = sorted((root / "labels" / split).glob("*.txt"))
        if len(image_paths) != expected or len(label_paths) != expected:
            raise ValueError(f"{root}/{split} count mismatch")
        if {p.stem for p in image_paths} != {p.stem for p in label_paths}:
            raise ValueError(f"{root}/{split} stems differ")
        result[split] = expected
    labels = sum(len([line for line in path.read_text().splitlines() if line.strip()])
                 for path in (root / "labels/train").glob("*.txt"))
    if labels != TRAIN_EXPECTED["annotations"]:
        raise ValueError(f"training label mismatch {labels}")
    result["train_labels"] = labels
    return result


def prepare() -> None:
    _, images, annotations = load_population()
    if sha256(PROVENANCE) != PROVENANCE_SHA256:
        raise RuntimeError("V6 provenance identity changed")
    provenance = json.loads(PROVENANCE.read_text())
    partitions = dataset_partitions(images, annotations, provenance)
    TRANSFORMED.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    source_hashes, output_hashes = {}, {}
    names = sorted({info["file_name"] for info in images.values()})
    started = time.monotonic()
    for index, name in enumerate(names, 1):
        source, destination = RAW_IMAGES / name, TRANSFORMED / name
        source_hashes[name] = sha256(source)
        if not destination.is_file():
            gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"could not read {source}")
            transformed, row = transform_image(gray)
            bgr = cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)
            if not cv2.imwrite(str(destination), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"could not write {destination}")
            row["file_name"] = name
            diagnostics.append(row)
        output_hashes[name] = sha256(destination)
        if index % 50 == 0 or index == len(names):
            print(f"transformed {index}/{len(names)}", flush=True)
    # Diagnostics must be complete on first construction. On resume, recompute
    # them from source deterministically rather than trusting a partial table.
    if len(diagnostics) != len(names):
        diagnostics = []
        for name in names:
            gray = cv2.imread(str(RAW_IMAGES / name), cv2.IMREAD_GRAYSCALE)
            _, row = transform_image(gray)
            row["file_name"] = name
            diagnostics.append(row)
    if len(set(output_hashes.values())) != len(names):
        raise RuntimeError("transformed images unexpectedly collide")
    transform_contract = {
        "profile_size": PROFILE_SIZE, "radial_bins": RADIAL_BINS,
        "profile_sigma_bins": PROFILE_SIGMA_BINS, "disk_limit": DISK_LIMIT,
        "center_tolerance_fraction": CENTER_TOLERANCE_FRACTION,
        "clahe_clip": CLAHE_CLIP, "clahe_grid": list(CLAHE_GRID),
        "jpeg_quality": 95,
    }
    transform_id = digest_json({"contract": transform_contract, "outputs": output_hashes})
    build_dataset(RAW_DATASET, RAW_IMAGES, images, annotations, partitions, "raw")
    build_dataset(ARM_DATASET, TRANSFORMED, images, annotations, partitions, transform_id)
    summary = {
        "status": "complete", "images": len(names), "elapsed_seconds": time.monotonic() - started,
        "contract": transform_contract, "transform_id": transform_id,
        "source_hashes_digest": digest_json(source_hashes),
        "output_hashes_digest": digest_json(output_hashes),
        "diagnostics": diagnostics,
        "diagnostic_summary": {
            key: {"min": float(np.min([row[key] for row in diagnostics])),
                  "median": float(np.median([row[key] for row in diagnostics])),
                  "max": float(np.max([row[key] for row in diagnostics]))}
            for key in ("cx", "cy", "radius", "disk_fraction", "raw_std",
                        "transformed_std", "changed_fraction")
        },
        "populations": {"all": EXPECTED, "train": TRAIN_EXPECTED,
                        "held_files": len(partitions["valid_files"])},
    }
    write_json(BASE / "preprocessing_manifest.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "images", "transform_id",
                                              "diagnostic_summary", "populations")}, indent=2))


def training_arguments(epochs: int = 20, fraction: float = 1.0, seed: int = SEED) -> dict:
    return dict(
        epochs=epochs, patience=0, batch=GLOBAL_BATCH, imgsz=TRAIN_RESOLUTION, workers=4,
        resume=False, optimizer="AdamW", lr0=1e-4, lrf=0.1, momentum=0.9,
        weight_decay=5e-4, cos_lr=False, warmup_epochs=1.0,
        warmup_bias_lr=0.0, nbs=64, seed=seed, deterministic=True,
        single_cls=True, mask_ratio=1, overlap_mask=False, amp=True, rect=False,
        multi_scale=0.0, fraction=fraction, freeze=None,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=10, translate=0.1,
        scale=0.3, shear=0.0, perspective=0.0, fliplr=0.0, flipud=0.0,
        bgr=0.0, mosaic=0.3, close_mosaic=5 if epochs > 1 else 0,
        mixup=0.0, cutmix=0.0, copy_paste=0.0, save_period=5,
        val=False, plots=False,
    )


def verify() -> None:
    if sha256(SOURCE_DETECTOR) != DETECTOR_SHA256:
        raise RuntimeError("V17 detector identity changed")
    if sha256(PROVENANCE) != PROVENANCE_SHA256:
        raise RuntimeError("V6 provenance identity changed")
    manifest = json.loads((BASE / "preprocessing_manifest.json").read_text())
    if manifest["images"] != EXPECTED["files"]:
        raise RuntimeError("transformed population mismatch")
    _, images, annotations = load_population()
    partitions = dataset_partitions(images, annotations, json.loads(PROVENANCE.read_text()))
    payload = {
        "status": "verified", "source_detector_sha256": sha256(SOURCE_DETECTOR),
        "annotation_sha256": sha256(ANNOTATIONS),
        "raw_dataset": verify_dataset(RAW_DATASET, partitions),
        "arm_dataset": verify_dataset(ARM_DATASET, partitions),
        "transform_id": manifest["transform_id"],
        "training_arguments": training_arguments(),
        "submits": False,
    }
    write_json(BASE / "verification.json", payload)
    print(json.dumps(payload, indent=2))


def train(arm: str, smoke: bool, seed: int = SEED) -> None:
    import pandas as pd
    import torch
    from ultralytics import YOLO

    verify()
    if arm not in ("control", "transformed"):
        raise ValueError(f"unknown arm {arm}")
    dataset = RAW_DATASET if arm == "control" else ARM_DATASET
    seed_label = "seed1" if seed == SEED else "seed2"
    output = OUTPUT / ("smoke" if smoke else "train") / arm / seed_label
    marker = output / "result.json"
    final = output / "final.pt"
    args = training_arguments(epochs=1 if smoke else 20, fraction=1.0, seed=seed)
    signature = {
        "arm": arm, "seed": seed, "seed_label": seed_label, "smoke": smoke,
        "source_sha256": sha256(SOURCE_DETECTOR),
        "dataset_manifest": json.loads((dataset / "manifest.json").read_text()),
        "arguments": args,
    }
    if marker.is_file() and final.is_file():
        saved = json.loads(marker.read_text())
        if saved["signature"] != signature or saved["checkpoint_sha256"] != sha256(final):
            raise RuntimeError(f"cannot resume changed {arm} stage")
        print(json.dumps(saved, indent=2)); return
    if output.exists():
        raise RuntimeError(f"refusing to overwrite partial stage {output}")
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = YOLO(str(SOURCE_DETECTOR))
    model.train(
        data=str(dataset / "data.yaml"),
        device="0",
        project=str(output), name="run", exist_ok=False,
        **args,
    )
    elapsed = time.monotonic() - started
    source = Path(model.trainer.last if not smoke else model.trainer.last)
    shutil.copy2(source, final)
    results_path = Path(model.trainer.save_dir) / "results.csv"
    results = pd.read_csv(results_path)
    peak = max(torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count()))
    record = {
        "signature": signature, "elapsed_seconds": elapsed,
        "epochs_completed": len(results), "checkpoint": str(final),
        "checkpoint_sha256": sha256(final), "results_sha256": sha256(results_path),
        "peak_memory_bytes_rank_process": int(peak),
        "projected_hours": elapsed * (20 if smoke else 1) / 3600.0,
        "finite_results": bool(np.isfinite(results.select_dtypes(include=[np.number]).to_numpy()).all()),
    }
    if not record["finite_results"]:
        raise RuntimeError("non-finite training result")
    write_json(marker, record)
    print(json.dumps(record, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify", "smoke-control", "smoke-arm",
                                         "train-control", "train-arm"))
    parser.add_argument("--seed", type=int, choices=(SEED, REPLICATION_SEED), default=SEED)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); cv2.setNumThreads(2)
    if args.mode == "prepare":
        prepare()
    elif args.mode == "verify":
        verify()
    elif args.mode == "smoke-control":
        train("control", True, args.seed)
    elif args.mode == "smoke-arm":
        train("transformed", True, args.seed)
    elif args.mode == "train-control":
        train("control", False, args.seed)
    else:
        train("transformed", False, args.seed)


if __name__ == "__main__":
    main()
