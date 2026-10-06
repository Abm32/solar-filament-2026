"""V27 slicing-aided fine-tuning: deterministic matched datasets and four-GPU training.

Modes: prepare, verify, smoke-control, smoke-sliced, train-control, train-sliced.
No mode evaluates held-out labels, reads test images, exports, or submits.
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
BASE = Path(os.environ.get("V27_BASE", "/mnt/solar-v27-sliced-finetune"))
DATA_ROOT = Path(os.environ.get("V27_DATA_ROOT", "/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"))
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
RAW_IMAGES = DATA_ROOT / "train/train_images"
INPUTS = BASE / "inputs"
SOURCE_DETECTOR = INPUTS / "v17_detector.pt"
PROVENANCE = INPUTS / "v6_source/v6_provenance.json"
CONTROL_DATASET = BASE / "datasets/control"
SLICED_DATASET = BASE / "datasets/sliced"
TILE_IMAGES = BASE / "tile_images"
OUTPUT = BASE / "outputs"

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
DETECTOR_SHA256 = "ae476e663039fbbbfda77ec519635ca15e2d50d1fe9f1af54735f84dbdc251ef"
PROVENANCE_SHA256 = "ceb09f980d5be195db35f2954c26fd19ea7ffc0f318b2bae5b7c803504b17122"
EXPECTED_ALL = {"files": 707, "records": 1154, "annotations": 8199}
EXPECTED_TRAIN = {"files": 566, "records": 937, "annotations": 6606}
TILE_SIZE = 1024
TILE_STARTS = (0, 512, 1024)
OWNER_BOUNDARIES = (768, 1280)
EMPTY_TILES = 540
RECORDS_PER_EPOCH = 5622
CONTROL_REPETITIONS = 6
EPOCHS = 2
BATCH = 1
TRAIN_SIZE = 1024
PRIMARY_SEED = 42
REPLICA_SEED = 20260912
MEMORY_GATE_MIB = 15872


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_population():
    if sha256(ANNOTATIONS) != ANNOTATION_SHA256:
        raise RuntimeError("canonical annotations changed")
    coco = json.loads(ANNOTATIONS.read_text())
    images = {entry["id"]: entry for entry in coco["images"]}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    actual = {"files": len({entry["file_name"] for entry in images.values()}),
              "records": len(images), "annotations": len(coco["annotations"])}
    if actual != EXPECTED_ALL:
        raise RuntimeError(f"population changed {actual}")
    return images, annotations


def best_ids(images, annotations):
    by_file = defaultdict(list)
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    def completeness(image_id):
        entries = annotations.get(image_id, [])
        return len(entries), sum(entry.get("area", 0) for entry in entries)
    return {name: max(ids, key=completeness) for name, ids in by_file.items()}, by_file


def partitions(images, annotations, provenance):
    best, by_file = best_ids(images, annotations)
    files = sorted(best); random.Random(42).shuffle(files)
    held, train = sorted(files[:141]), sorted(files[141:])
    if train != sorted(provenance["training_files"]) or held != sorted(provenance["validation_files"]):
        raise RuntimeError("split differs from V6 provenance")
    train_ids = [image_id for name in train for image_id in sorted(by_file[name])]
    actual = {"files": len(train), "records": len(train_ids),
              "annotations": sum(len(annotations[i]) for i in train_ids)}
    if actual != EXPECTED_TRAIN:
        raise RuntimeError(f"training population changed {actual}")
    return {"train_files": train, "held_files": held, "train_ids": train_ids,
            "valid_ids": [best[name] for name in held],
            "by_file": {name: sorted(ids) for name, ids in by_file.items()}}


def owner_axis(value: float) -> int:
    return 0 if value < OWNER_BOUNDARIES[0] else 1 if value < OWNER_BOUNDARIES[1] else 2


def owner_origin(annotation: dict) -> tuple[int, int]:
    x, y, width, height = map(float, annotation["bbox"])
    return TILE_STARTS[owner_axis(x + width / 2)], TILE_STARTS[owner_axis(y + height / 2)]


def owner_contains(annotation: dict, x0: int, y0: int) -> bool:
    x, y, width, height = map(float, annotation["bbox"])
    return x >= x0 and y >= y0 and x + width <= x0 + TILE_SIZE and y + height <= y0 + TILE_SIZE


def polygon_lines(annotation: dict, width: int, height: int, offset=(0, 0)) -> list[str]:
    x0, y0 = offset
    lines = []
    for polygon in annotation["segmentation"]:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        points[:, 0] = np.clip((points[:, 0] - x0) / width, 0, 1)
        points[:, 1] = np.clip((points[:, 1] - y0) / height, 0, 1)
        lines.append("0 " + " ".join(f"{value:.6f}" for value in points.ravel()))
    return lines


def link_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def build_validation(root: Path, images, annotations, split):
    for image_id in split["valid_ids"]:
        info = images[image_id]; name = Path(info["file_name"])
        link_image(RAW_IMAGES / name, root / "images/val" / name)
        lines = [line for annotation in annotations[image_id]
                 for line in polygon_lines(annotation, info["width"], info["height"])]
        write_label(root / "labels/val" / f"{name.stem}.txt", lines)


def prepare() -> None:
    if sha256(PROVENANCE) != PROVENANCE_SHA256 or sha256(SOURCE_DETECTOR) != DETECTOR_SHA256:
        raise RuntimeError("pinned input changed")
    if CONTROL_DATASET.exists() or SLICED_DATASET.exists() or TILE_IMAGES.exists():
        raise RuntimeError("prepared outputs already exist; use verify")
    images, annotations = load_population(); provenance = json.loads(PROVENANCE.read_text())
    split = partitions(images, annotations, provenance)
    # Control: six exact repetitions of every original record.
    control_labels = 0
    for image_id in split["train_ids"]:
        info = images[image_id]; original = Path(info["file_name"])
        lines = [line for annotation in annotations[image_id]
                 for line in polygon_lines(annotation, info["width"], info["height"])]
        control_labels += len(lines) * CONTROL_REPETITIONS
        for repetition in range(CONTROL_REPETITIONS):
            stem = f"full_ann{image_id}_rep{repetition}"
            link_image(RAW_IMAGES / original, CONTROL_DATASET / f"images/train/{stem}.jpeg")
            write_label(CONTROL_DATASET / f"labels/train/{stem}.txt", lines)
    build_validation(CONTROL_DATASET, images, annotations, split)
    # Arm: all full records once plus all positive tiles and deterministic empty subset.
    sliced_labels = 0; positive = []; empty = []; excluded = []
    for image_id in split["train_ids"]:
        info = images[image_id]; original = Path(info["file_name"])
        full_lines = [line for annotation in annotations[image_id]
                      for line in polygon_lines(annotation, info["width"], info["height"])]
        sliced_labels += len(full_lines)
        full_stem = f"full_ann{image_id}"
        link_image(RAW_IMAGES / original, SLICED_DATASET / f"images/train/{full_stem}.jpeg")
        write_label(SLICED_DATASET / f"labels/train/{full_stem}.txt", full_lines)
        per_tile = {(x0, y0): [] for y0 in TILE_STARTS for x0 in TILE_STARTS}
        for annotation in annotations[image_id]:
            x0, y0 = owner_origin(annotation)
            if owner_contains(annotation, x0, y0):
                per_tile[(x0, y0)].append(annotation)
            else:
                excluded.append({"file_name": info["file_name"], "image_id": image_id,
                                 "annotation_id": str(annotation["id"]), "area": annotation["area"]})
        for (x0, y0), entries in per_tile.items():
            key = f"{info['file_name']}|{image_id}|{x0}|{y0}"
            record = (hashlib.sha256(key.encode()).hexdigest(), info["file_name"], image_id,
                      x0, y0, entries)
            (positive if entries else empty).append(record)
    selected = positive + sorted(empty, key=lambda record: record[0])[:EMPTY_TILES]
    selected_keys = []
    for _, file_name, image_id, x0, y0, entries in selected:
        tile_name = f"{Path(file_name).stem}_x{x0}_y{y0}.png"
        tile_path = TILE_IMAGES / tile_name
        if not tile_path.is_file():
            image = cv2.imread(str(RAW_IMAGES / file_name))
            if image is None or image.shape[:2] != (2048, 2048):
                raise RuntimeError(f"cannot read tile source {file_name}")
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(tile_path), image[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]):
                raise RuntimeError(f"failed writing tile {tile_path}")
        stem = f"tile_ann{image_id}_x{x0}_y{y0}"
        link_image(tile_path, SLICED_DATASET / f"images/train/{stem}.png")
        lines = [line for annotation in entries
                 for line in polygon_lines(annotation, TILE_SIZE, TILE_SIZE, (x0, y0))]
        sliced_labels += len(lines)
        write_label(SLICED_DATASET / f"labels/train/{stem}.txt", lines)
        selected_keys.append({"file_name": file_name, "image_id": image_id, "x0": x0, "y0": y0,
                              "positive": bool(entries), "labels": len(entries), "tile_sha256": sha256(tile_path)})
    build_validation(SLICED_DATASET, images, annotations, split)
    for root in (CONTROL_DATASET, SLICED_DATASET):
        (root / "data.yaml").write_text(
            f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    payload = {"status": "complete", "train_files": 566, "train_records": 937,
               "control": {"records": RECORDS_PER_EPOCH, "polygon_lines": control_labels,
                           "repetitions": CONTROL_REPETITIONS},
               "sliced": {"records": 937 + len(selected), "full_records": 937,
                          "positive_tiles": len(positive), "empty_tiles": EMPTY_TILES,
                          "polygon_lines": sliced_labels, "excluded_labels": excluded,
                          "selected_tiles": selected_keys},
               "dataset_identity": digest_json({"selected": selected_keys, "excluded": excluded,
                                                 "control_labels": control_labels,
                                                 "sliced_labels": sliced_labels})}
    if payload["control"]["records"] != RECORDS_PER_EPOCH or payload["sliced"]["records"] != RECORDS_PER_EPOCH:
        raise RuntimeError("matched record budget failed")
    if len(positive) != 4145 or sliced_labels != 6606 + 6593 or len(excluded) != 13:
        raise RuntimeError("audited sliced population changed")
    write_json(BASE / "dataset_manifest.json", payload)
    print(json.dumps({k: payload[k] for k in ("status", "control", "dataset_identity")}
                     | {"sliced": {k: v for k, v in payload["sliced"].items()
                                    if k not in ("selected_tiles", "excluded_labels")}}, indent=2))


def training_arguments(epochs=EPOCHS, seed=PRIMARY_SEED):
    return dict(epochs=epochs, patience=0, batch=BATCH, imgsz=TRAIN_SIZE, workers=4,
                resume=False, optimizer="AdamW", lr0=1e-4, lrf=0.1, momentum=0.9,
                weight_decay=5e-4, cos_lr=False, warmup_epochs=0.2, warmup_bias_lr=0.0,
                nbs=64, seed=seed, deterministic=True, single_cls=True, mask_ratio=1,
                overlap_mask=False, amp=True, rect=False, multi_scale=0.0, fraction=1.0,
                freeze=None, hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=10, translate=0.1,
                scale=0.3, shear=0.0, perspective=0.0, fliplr=0.0, flipud=0.0, bgr=0.0,
                mosaic=0.3, close_mosaic=0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                save_period=1, val=False, plots=False)


def verify():
    if sha256(SOURCE_DETECTOR) != DETECTOR_SHA256 or sha256(PROVENANCE) != PROVENANCE_SHA256:
        raise RuntimeError("pinned input changed")
    manifest = json.loads((BASE / "dataset_manifest.json").read_text())
    for root in (CONTROL_DATASET, SLICED_DATASET):
        images = [path for path in (root / "images/train").iterdir()
                  if path.suffix.lower() in {".jpeg", ".jpg", ".png"}]
        labels = list((root / "labels/train").glob("*.txt"))
        if len(images) != RECORDS_PER_EPOCH or len(labels) != RECORDS_PER_EPOCH:
            raise RuntimeError(f"dataset count mismatch {root}")
        if {p.stem for p in images} != {p.stem for p in labels}:
            raise RuntimeError("dataset stems differ")
    train_files = set(json.loads(PROVENANCE.read_text())["training_files"])
    if any(entry["file_name"] not in train_files for entry in manifest["sliced"]["selected_tiles"]):
        raise RuntimeError("held-out filename leaked into tiles")
    payload = {"status": "verified", "detector_sha256": sha256(SOURCE_DETECTOR),
               "annotation_sha256": sha256(ANNOTATIONS), "dataset_identity": manifest["dataset_identity"],
               "records_per_epoch": RECORDS_PER_EPOCH, "steps": RECORDS_PER_EPOCH * EPOCHS,
               "training_arguments": training_arguments(), "submits": False}
    write_json(BASE / "verification.json", payload); print(json.dumps(payload, indent=2)); return payload


def train(policy: str, smoke: bool, seed: int):
    import pandas as pd
    import torch
    from ultralytics import YOLO
    verification = verify()
    dataset = CONTROL_DATASET if policy == "control" else SLICED_DATASET
    seed_label = "seed1" if seed == PRIMARY_SEED else "seed2"
    stage = "smoke" if smoke else "train"; output = OUTPUT / stage / policy / seed_label
    marker, final = output / "result.json", output / "final.pt"
    epochs = 1 if smoke else EPOCHS; args = training_arguments(epochs, seed)
    signature = {"policy": policy, "seed": seed, "smoke": smoke, "dataset_identity": verification["dataset_identity"],
                 "source_sha256": sha256(SOURCE_DETECTOR), "arguments": args}
    if marker.is_file() and final.is_file():
        saved=json.loads(marker.read_text()); assert saved["signature"]==signature and saved["checkpoint_sha256"]==sha256(final); print(json.dumps(saved,indent=2)); return
    if output.exists(): raise RuntimeError(f"partial stage exists {output}")
    output.mkdir(parents=True); torch.cuda.reset_peak_memory_stats(); started=time.monotonic(); model=YOLO(str(SOURCE_DETECTOR))
    model.train(data=str(dataset/"data.yaml"),device="0",project=str(output),name="run",exist_ok=False,**args)
    elapsed=time.monotonic()-started; shutil.copy2(Path(model.trainer.last),final); results_path=Path(model.trainer.save_dir)/"results.csv"; results=pd.read_csv(results_path)
    record={"signature":signature,"elapsed_seconds":elapsed,"epochs_completed":len(results),"expected_steps":RECORDS_PER_EPOCH*epochs,
            "checkpoint_sha256":sha256(final),"results_sha256":sha256(results_path),"peak_allocated_mib":int(torch.cuda.max_memory_allocated(0))/2**20,
            "projected_two_epoch_hours":elapsed*(2 if smoke else 1)/3600,"finite_results":bool(np.isfinite(results.select_dtypes(include=[np.number]).to_numpy()).all())}
    if not record["finite_results"] or record["epochs_completed"]!=epochs: raise RuntimeError("training incomplete")
    write_json(marker,record); print(json.dumps(record,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("mode",choices=("prepare","verify","smoke-control","smoke-sliced","train-control","train-sliced")); parser.add_argument("--seed",type=int,choices=(PRIMARY_SEED,REPLICA_SEED),default=PRIMARY_SEED); args=parser.parse_args(); random.seed(args.seed);np.random.seed(args.seed);cv2.setNumThreads(2)
    if args.mode=="prepare":prepare()
    elif args.mode=="verify":verify()
    elif args.mode=="smoke-control":train("control",True,args.seed)
    elif args.mode=="smoke-sliced":train("sliced",True,args.seed)
    elif args.mode=="train-control":train("control",False,args.seed)
    else:train("sliced",False,args.seed)
if __name__=="__main__":main()
