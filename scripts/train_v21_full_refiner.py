"""V21 full-data refiner chain for the frozen V17 recipe.

Stages, all fixed-duration with no held-out selection:
1. 8-epoch ImageNet-initialized GT-crop base refiner (V6 schedule).
2. 3-epoch low-LR canonical continuation (the selected E014 duration).
3. Build proposal-centered positives from the V21 full-data detector.
4. 4-epoch proposal-centered continuation (V11 selected duration).

All 707 approved files / 1,154 annotator records / 8,199 annotations are used.
No test inference or Kaggle submission occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

WORK = Path(os.environ.get("V21_WORK", "/mnt/solar-v21-full-refit"))
REPO = Path(os.environ.get("V21_REPO", "/home/azureuser/solar-filament-2026"))
DATA_ROOT = Path(os.environ.get(
    "V21_DATA_ROOT",
    "/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
))
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_DIR = DATA_ROOT / "train/train_images"
DETECTOR = WORK / "detector/mask_ratio1/final.pt"
OUTPUT = WORK / "refiner"
BASE_SOURCE = WORK / "inputs/solar_base.py"
EXPECTED_ANNOTATION_SHA = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
EXPECTED_DETECTOR_SHA = "5cc803e4f3ca334c471fc1012ef1e27a142b59b085831fc09d5e35d621ca5629"
EXPECTED = {"files": 707, "records": 1154, "annotations": 8199}
SEED = 42
BATCH = 16

sys.path.insert(0, str(WORK / "runtime"))


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


def load_modules():
    import optimize_inference as core
    import detector_tta as tta
    import v10_quality as v10
    import v11_refiner as v11
    base = core.load_base(BASE_SOURCE)
    base.CFG.root = DATA_ROOT
    base.CFG.train_dir = IMAGE_DIR
    base.CFG.test_dir = DATA_ROOT / "test/test_images"
    base.CFG.train_json = ANNOTATIONS
    base.CFG.crop_size = 256
    base.CFG.crop_context = 1.8
    base.CFG.min_crop = 96
    base.CFG.imgsz = 1024
    base.CFG.proposal_cache = OUTPUT / "proposal_cache"
    v11.core, v11.tta, v11.v10 = core, tta, v10
    return base, core, tta, v10, v11


def load_population(base):
    if sha256(ANNOTATIONS) != EXPECTED_ANNOTATION_SHA:
        raise RuntimeError("approved annotation identity changed")
    coco, images, annotations = base.load_coco()
    actual = {
        "files": len({info["file_name"] for info in images.values()}),
        "records": len(images),
        "annotations": sum(len(value) for value in annotations.values()),
    }
    if actual != EXPECTED:
        raise RuntimeError(f"population mismatch: {actual} != {EXPECTED}")
    if sha256(DETECTOR) != EXPECTED_DETECTOR_SHA:
        raise RuntimeError("V21 detector identity changed")
    return coco, images, annotations


def seed_everything(base):
    base.seed_everything(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    cv2.setNumThreads(2)


def gt_crop_arrays(base, images, annotations):
    ids = sorted(images, key=str)
    image_arrays, mask_arrays = base.build_crop_arrays(ids, images, annotations)
    if len(image_arrays) != EXPECTED["annotations"]:
        raise RuntimeError(
            f"expected {EXPECTED['annotations']} GT crops, built {len(image_arrays)}"
        )
    return image_arrays, mask_arrays


def save_weights(path: Path, model) -> str:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)
    return sha256(path)


def train_gt_stage(base, name: str, source: Path | None, arrays, epochs: int,
                   learning_rate: float, amp: bool) -> dict:
    import torch

    stage = OUTPUT / name
    marker = stage / "stage_result.json"
    checkpoint = stage / "final.pt"
    source_sha = sha256(source) if source else "imagenet-resnet18"
    signature = {
        "stage": name, "source_sha256": source_sha, "epochs": epochs,
        "learning_rate": learning_rate, "weight_decay": 1e-4, "batch": BATCH,
        "loss": "0.5 BCEWithLogits + 0.5 soft Dice", "amp": amp,
        "crops": len(arrays[0]), "population": EXPECTED,
    }
    if marker.exists() and checkpoint.exists():
        saved = json.loads(marker.read_text())
        if saved["signature"] != signature or saved["checkpoint_sha256"] != sha256(checkpoint):
            raise RuntimeError(f"cannot resume changed refiner stage {name}")
        return saved

    seed_everything(base)
    model = base.CropUNet(pretrained=source is None).to("cuda:0")
    if source:
        model.load_state_dict(torch.load(source, map_location="cuda:0", weights_only=True))
    dataset = base.CropDataset(arrays[0], arrays[1], augment=True)
    generator = torch.Generator().manual_seed(SEED)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=True, generator=generator,
        num_workers=2, pin_memory=True, drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, started = [], time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train(); loss_sum = 0.0; samples = 0; updates = 0
        for batch, target in loader:
            batch, target = batch.to("cuda:0"), target.to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(batch)
                loss = (
                    0.5 * torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                    + 0.5 * base.soft_dice_loss(logits, target)
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {name} epoch {epoch} loss")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.detach()) * len(batch)
            samples += len(batch); updates += 1
        if samples != len(dataset) or updates != math.ceil(len(dataset) / BATCH):
            raise RuntimeError("GT refiner update budget mismatch")
        history.append({
            "epoch": epoch, "samples": samples, "updates": updates,
            "mean_loss": loss_sum / samples,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        scheduler.step()
        write_json(stage / "history.json", history)
        print(f"refiner stage={name} epoch={epoch}/{epochs} loss={history[-1]['mean_loss']:.6f}", flush=True)
    digest = save_weights(checkpoint, model)
    record = {
        "signature": signature, "elapsed_seconds": time.monotonic() - started,
        "epochs_completed": epochs, "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest, "history": history,
        "selection": "fixed duration; all approved labels used for fitting",
    }
    write_json(marker, record)
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return record


def build_proposal_cohort(base, core, tta, v10, v11, images, annotations) -> dict:
    import torch
    from ultralytics import YOLO

    destination = OUTPUT / "proposal_cohort.json"
    cache_root = OUTPUT / "proposal_cache"
    signature = {
        "detector_sha256": sha256(DETECTOR), "population": EXPECTED,
        "views": 4, "resolution": 1024, "confidence_floor": 0.1,
        "policy": "flip_consensus", "box_iou": 0.7,
        "matching": {"min_iou": 0.30, "max_second": 0.20, "min_margin": 0.15},
    }
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["signature"] != signature:
            raise RuntimeError("cannot resume changed proposal cohort")
        for entry in saved["sources"].values():
            if sha256(OUTPUT / entry["cache"]) != entry["cache_sha256"]:
                raise RuntimeError("proposal cache changed")
        return saved

    cache_root.mkdir(parents=True, exist_ok=True)
    base.CFG.proposal_cache = cache_root
    detector = YOLO(str(DETECTOR))
    by_file_ids: dict[str, list] = {}
    for image_id, info in images.items():
        by_file_ids.setdefault(info["file_name"], []).append(image_id)
    records, sources, annotation_count = [], {}, 0
    names = sorted(by_file_ids)
    for file_index, name in enumerate(names, 1):
        path = IMAGE_DIR / name
        views = tta.proposal_views(base, detector, path)
        proposals, scores = tta.cluster_views(views, "flip_consensus")
        cache = cache_root / (hashlib.sha256(str(path).encode()).hexdigest()[:16] + ".json.gz")
        sources[name] = {
            "image_sha256": sha256(path), "cache": str(cache.relative_to(OUTPUT)),
            "cache_sha256": sha256(cache),
        }
        for image_id in sorted(by_file_ids[name], key=str):
            info = images[image_id]
            targets = [
                (annotation, core.annotation_rle(info, annotation))
                for annotation in annotations.get(image_id, [])
            ]
            targets = [(annotation, rle) for annotation, rle in targets if mask_utils.area(rle) > 0]
            annotation_count += len(targets)
            iou = (
                mask_utils.iou([rle for _, rle in targets], proposals, [0] * len(proposals))
                if targets and proposals else np.zeros((len(targets), len(proposals)))
            )
            for target_index, proposal_index in v11.match_positive_proposals(iou, scores):
                annotation, target = targets[target_index]
                proposal = proposals[proposal_index]
                target_mask = mask_utils.decode(target)
                seed_mask = mask_utils.decode(proposal)
                records.append({
                    "file_name": name, "image_id": image_id,
                    "annotation_id": annotation["id"], "category_id": annotation["category_id"],
                    "proposal_id": proposal_index,
                    "matched_iou": float(iou[target_index, proposal_index]),
                    "confidence": float(scores[proposal_index]),
                    "gt_bounds": list(base.square_bounds(target_mask)),
                    "proposal_bounds": list(base.square_bounds(seed_mask)),
                    "target": v10.serialized_rle(target),
                    "seed": v10.serialized_rle(proposal),
                })
        if file_index % 20 == 0 or file_index == len(names):
            print(f"proposal cohort files={file_index}/{len(names)} matched={len(records)}", flush=True)
    del detector; torch.cuda.empty_cache()
    if annotation_count != EXPECTED["annotations"]:
        raise RuntimeError(f"proposal cohort saw {annotation_count} annotations")
    keys = {(record["image_id"], record["annotation_id"]) for record in records}
    if not records or len(keys) != len(records):
        raise RuntimeError("proposal cohort empty or duplicate targets")
    saved = {
        "signature": signature, "sources": sources, "records": records,
        "matched_samples": len(records), "all_training_annotations": annotation_count,
        "cohort_sha256": digest_json(records),
    }
    write_json(destination, saved)
    return saved


def train_proposal_stage(base, v11, source: Path, cohort: dict, epochs: int = 4) -> dict:
    import torch

    name = "proposal_centered"
    stage = OUTPUT / name
    marker = stage / "stage_result.json"
    checkpoint = stage / "final.pt"
    signature = {
        "stage": name, "source_sha256": sha256(source), "epochs": epochs,
        "learning_rate": 5e-5, "weight_decay": 1e-4, "batch": BATCH,
        "precision": "float32", "cohort_sha256": cohort["cohort_sha256"],
        "samples": cohort["matched_samples"], "population": EXPECTED,
    }
    if marker.exists() and checkpoint.exists():
        saved = json.loads(marker.read_text())
        if saved["signature"] != signature or saved["checkpoint_sha256"] != sha256(checkpoint):
            raise RuntimeError("cannot resume changed proposal-centered stage")
        return saved

    seed_everything(base)
    model = base.CropUNet(pretrained=False).to("cuda:0")
    model.load_state_dict(torch.load(source, map_location="cuda:0", weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history, started = [], time.monotonic()
    for epoch in range(1, epochs + 1):
        dataset = v11.ProposalDataset(cohort["records"], IMAGE_DIR, "proposal_centered", epoch)
        order = v11.epoch_sample_order(cohort["records"], epoch)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH, sampler=order, num_workers=0,
            pin_memory=True, drop_last=False,
        )
        model.train(); loss_sum = 0.0; samples = 0; updates = 0
        for batch, target in loader:
            batch, target = batch.to("cuda:0"), target.to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = (
                0.5 * torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                + 0.5 * base.soft_dice_loss(logits, target)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite proposal epoch {epoch} loss")
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
            samples += len(batch); updates += 1
            if updates % 100 == 0:
                print(f"proposal train epoch={epoch}/{epochs} updates={updates}/{len(loader)} loss={loss_sum/samples:.6f}", flush=True)
        if samples != len(cohort["records"]):
            raise RuntimeError("proposal-centered sample budget mismatch")
        history.append({
            "epoch": epoch, "samples": samples, "updates": updates,
            "mean_loss": loss_sum / samples,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "sample_order_sha256": v11.digest_json(order),
        })
        scheduler.step(); write_json(stage / "history.json", history)
        print(f"refiner stage=proposal_centered epoch={epoch}/{epochs} loss={history[-1]['mean_loss']:.6f}", flush=True)
    digest = save_weights(checkpoint, model)
    record = {
        "signature": signature, "elapsed_seconds": time.monotonic() - started,
        "epochs_completed": epochs, "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest, "history": history,
        "selection": "fixed duration; all approved labels used for fitting",
    }
    write_json(marker, record)
    del model, optimizer, scheduler; torch.cuda.empty_cache()
    return record


def run() -> None:
    import torch

    base, core, tta, v10, v11 = load_modules()
    _, images, annotations = load_population(base)
    seed_everything(base)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("building all-data GT crop arrays", flush=True)
    arrays = gt_crop_arrays(base, images, annotations)
    base_stage = train_gt_stage(base, "base", None, arrays, epochs=8, learning_rate=3e-4, amp=True)
    # E014 selected epoch 3; preserve that fitted duration in the final refit.
    canonical_stage = train_gt_stage(
        base, "canonical", Path(base_stage["checkpoint"]), arrays,
        epochs=3, learning_rate=5e-5, amp=True,
    )
    del arrays
    cohort = build_proposal_cohort(base, core, tta, v10, v11, images, annotations)
    proposal_stage = train_proposal_stage(
        base, v11, Path(canonical_stage["checkpoint"]), cohort, epochs=4,
    )
    result = {
        "status": "complete", "population": EXPECTED,
        "detector_sha256": sha256(DETECTOR),
        "base": base_stage, "canonical": canonical_stage,
        "proposal_cohort": {
            "matched_samples": cohort["matched_samples"],
            "all_training_annotations": cohort["all_training_annotations"],
            "cohort_sha256": cohort["cohort_sha256"],
        },
        "proposal_centered": proposal_stage,
        "selected_checkpoint": proposal_stage["checkpoint"],
        "selected_checkpoint_sha256": proposal_stage["checkpoint_sha256"],
        "validation": "none: all approved labels used for fitting",
    }
    write_json(WORK / "refiner_full_result.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
