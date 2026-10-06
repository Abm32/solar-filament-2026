"""V28 feasibility: exact-serving-population proposal utility learning.

Generate exact V26 iou030 hybrid proposals on frozen training files, refine each
proposal independently, label against every annotator, and run file-grouped OOF
scoring. This script never reads tune/audit/test images and never exports/submits.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

BASE = Path(os.environ.get("V28_BASE", "/mnt/solar-v28-utility"))
DATA_ROOT = Path(os.environ.get(
    "V28_DATA_ROOT",
    "/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/"
    "MAGFiLO_1.0_Kaggle_2026",
))
INPUTS = BASE / "inputs"
RUNTIME = BASE / "runtime"
OUTPUT = BASE / "cohort"
RAW_IMAGES = DATA_ROOT / "train/train_images"
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
PROVENANCE = INPUTS / "v6_source/v6_provenance.json"
SOLAR_BASE = INPUTS / "v6_source/solar_base.py"
DETECTOR = INPUTS / "v17_detector.pt"
CANONICAL_REFINER = INPUTS / "best_refiner_coco.pt"
V11_REFINER = INPUTS / "proposal_centered_e4.pt"
V26_ENGINE = RUNTIME / "evaluate_v26_sliced.py"

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
PROVENANCE_SHA256 = "ceb09f980d5be195db35f2954c26fd19ea7ffc0f318b2bae5b7c803504b17122"
DETECTOR_SHA256 = "ae476e663039fbbbfda77ec519635ca15e2d50d1fe9f1af54735f84dbdc251ef"
CANONICAL_REFINER_SHA256 = "8d00805318233a4196c706c1ec4eba9136ca1cb80d62298f082e82f7a2d01fb1"
V11_REFINER_SHA256 = "fae496b9d00819f49c5278e81e41a7917b31d18ea80f827e162746d840d14cd2"
CELL = {"name": "iou030", "retina_masks": True, "agnostic_nms": True, "iou": 0.3}
REFINE_THRESHOLD = 0.45
CROP_SIZE = 64
SHARDS = 4
FOLDS = 4
SEED = 20260912


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def serialized_rle(rle: dict) -> dict:
    counts = rle["counts"].decode() if isinstance(rle["counts"], bytes) else rle["counts"]
    return {"size": list(rle["size"]), "counts": counts}


def restored_rle(raw: dict) -> dict:
    counts = raw["counts"].encode() if isinstance(raw["counts"], str) else raw["counts"]
    return {"size": list(raw["size"]), "counts": counts}


def ranking_auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique), dtype=float)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) /
                 (positive * negative))


def file_fold(name: str, folds: int = FOLDS) -> int:
    return int(hashlib.sha256(f"v28|{name}".encode()).hexdigest()[:8], 16) % folds


def expected_utility(iou: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Annotator-aware expected thresholded-IoU utility for each proposal.

    iou has shape [n_annotators, n_proposals, variable_gt] represented as a list
    upstream; this helper accepts [annotators, proposals] best-IoU values.
    """
    if iou.ndim != 2:
        raise ValueError("best-IoU matrix must be annotators x proposals")
    positive = iou > 0.5
    thresholded = np.where(positive, iou, 0.0)
    return thresholded.mean(axis=0), positive.mean(axis=0), iou.mean(axis=0)


def independent_refined_masks(runtime, rles, image_path: Path):
    if not rles:
        return [], [], [], []
    masks = [mask_utils.decode(rle).astype(np.uint8) for rle in rles]
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (2048, 2048):
        raise RuntimeError(f"cannot read {image_path}")
    bounds, _, probabilities = runtime["core"].crop_probabilities(
        runtime["base"], runtime["refiner"], gray, masks, True,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    refined, crop_arrays = [], []
    for mask, bound, probability in zip(masks, bounds, probabilities):
        x0, y0, x1, y1 = bound
        proposal_crop = mask[y0:y1, x0:x1]
        probability_native = cv2.resize(probability, (x1 - x0, y1 - y0))
        crop = probability_native >= REFINE_THRESHOLD
        crop &= cv2.dilate(proposal_crop, kernel) > 0
        if not crop.any():
            crop = proposal_crop.astype(bool)
        refined.append(runtime["core"].encode_crop(crop, bound, mask.shape))
        image_crop = gray[y0:y1, x0:x1]
        crop_arrays.append(np.stack([
            cv2.resize(image_crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA),
            cv2.resize(proposal_crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST) * 255,
            cv2.resize(crop.astype(np.uint8), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST) * 255,
            np.clip(cv2.resize(probability_native, (CROP_SIZE, CROP_SIZE)) * 255, 0, 255).astype(np.uint8),
        ]).astype(np.uint8))
    return refined, masks, bounds, crop_arrays


def load_runtime():
    import importlib.util
    import torch

    sys.path.insert(0, str(RUNTIME))
    spec = importlib.util.spec_from_file_location("v26_engine", V26_ENGINE)
    v26 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v26)
    v26.BASE = BASE
    v26.DATA_ROOT = DATA_ROOT
    v26.INPUTS = INPUTS
    v26.RUNTIME = RUNTIME
    v26.OUTPUT = BASE / "proposal_cache"
    v26.RAW_IMAGES = RAW_IMAGES
    v26.ANNOTATIONS = ANNOTATIONS
    v26.PROVENANCE = PROVENANCE
    v26.SOLAR_BASE = SOLAR_BASE
    v26.DETECTOR = DETECTOR
    v26.CANONICAL_REFINER = CANONICAL_REFINER
    v26.V11_REFINER = V11_REFINER
    v26.HISTORICAL_DIAGNOSTICS = INPUTS / "proposal_diagnostics.csv"
    v26.DETECTOR_SHA256 = DETECTOR_SHA256
    runtime = v26.runtime_state()
    provenance = json.loads(PROVENANCE.read_text())
    train_files = sorted(provenance["training_files"])
    if len(train_files) != 566:
        raise RuntimeError("expected 566 frozen training files")
    # Full-file annotation index, unlike v26 runtime's held-out-only by_file.
    by_file = {name: [] for name in train_files}
    for image_id, info in runtime["images"].items():
        if info["file_name"] in by_file:
            by_file[info["file_name"]].append(image_id)
    runtime.update(v26=v26, train_files=train_files, train_by_file=by_file)
    if not torch.cuda.is_available():
        raise RuntimeError("proposal generation requires CUDA")
    return runtime


def generate_shard(shard: int, limit: int = 0):
    from ultralytics import YOLO

    runtime = load_runtime()
    names = runtime["train_files"][shard::SHARDS]
    if limit:
        names = names[:limit]
    detector = YOLO(str(DETECTOR))
    shard_root = OUTPUT / f"shard_{shard}"
    shard_root.mkdir(parents=True, exist_ok=True)
    summary = []
    started = time.monotonic()
    for position, name in enumerate(names, 1):
        proposals, proposal_diag = runtime["v26"].proposals_for_image(
            runtime, detector, RAW_IMAGES / name, CELL,
        )
        hybrid_rles, scores = proposals["hybrid"]
        refined, masks, bounds, crops = independent_refined_masks(
            runtime, hybrid_rles, RAW_IMAGES / name,
        )
        best_rows = []
        for image_id in sorted(runtime["train_by_file"][name]):
            info = runtime["images"][image_id]
            gt = [runtime["core"].annotation_rle(info, entry)
                  for entry in runtime["annotations"].get(image_id, [])]
            if refined and gt:
                matrix = np.asarray(mask_utils.iou(refined, gt, [0] * len(gt)), dtype=float)
                best = matrix.max(axis=1) if matrix.size else np.zeros(len(refined))
            else:
                best = np.zeros(len(refined), dtype=float)
            best_rows.append(best)
        best_matrix = np.stack(best_rows) if best_rows else np.zeros((0, len(refined)))
        utility, positive_fraction, mean_iou = expected_utility(best_matrix)
        source_control = {json.dumps(serialized_rle(r), sort_keys=True) for r in proposals["control"][0]}
        source_sliced = {json.dumps(serialized_rle(r), sort_keys=True) for r in proposals["sliced"][0]}
        rows = []
        for index, (rle, refined_rle, score, bound, crop) in enumerate(
            zip(hybrid_rles, refined, scores, bounds, crops)
        ):
            key = json.dumps(serialized_rle(rle), sort_keys=True)
            area = float(mask_utils.area(rle))
            refined_area = float(mask_utils.area(refined_rle))
            x0, y0, x1, y1 = bound
            rows.append({
                "proposal_index": index, "confidence": float(score),
                "utility": float(utility[index]),
                "positive_fraction": float(positive_fraction[index]),
                "mean_iou": float(mean_iou[index]),
                "annotator_records": len(best_rows),
                "gt_instances": int(sum(len(runtime["annotations"].get(image_id, []))
                                        for image_id in runtime["train_by_file"][name])),
                "area": area, "refined_area": refined_area,
                "log_area": float(math.log1p(area)),
                "area_ratio": float(refined_area / max(area, 1.0)),
                "crop_width": x1 - x0, "crop_height": y1 - y0,
                "source_full": key in source_control,
                "source_tile": key in source_sliced,
                "proposal_rle": serialized_rle(rle),
                "refined_rle": serialized_rle(refined_rle),
            })
        np.savez_compressed(
            shard_root / f"{Path(name).stem}.npz",
            crops=np.stack(crops) if crops else np.zeros((0, 4, CROP_SIZE, CROP_SIZE), np.uint8),
        )
        with gzip.open(shard_root / f"{Path(name).stem}.json.gz", "wt") as handle:
            json.dump({"file_name": name, "fold": file_fold(name),
                       "annotator_records": len(best_rows),
                       "proposal_diagnostics": proposal_diag, "proposals": rows}, handle)
        summary.append({"file_name": name, "proposals": len(rows),
                        "positive_any": int((positive_fraction > 0).sum()),
                        "utility_sum": float(utility.sum())})
        if position % 20 == 0 or position == len(names):
            print(f"v28 cohort shard={shard} {position}/{len(names)} "
                  f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    pd.DataFrame(summary).to_csv(shard_root / "summary.csv", index=False)
    result = {"status": "complete", "shard": shard, "files": names,
              "file_count": len(names), "proposals": int(sum(r["proposals"] for r in summary)),
              "elapsed_seconds": time.monotonic() - started}
    write_json(shard_root / "result.json", result)
    print(json.dumps(result, indent=2))


def merge_cohort():
    files, summaries = [], []
    for shard in range(SHARDS):
        result = json.loads((OUTPUT / f"shard_{shard}/result.json").read_text())
        files.extend(result["files"])
        summaries.append(pd.read_csv(OUTPUT / f"shard_{shard}/summary.csv"))
    if len(files) != len(set(files)) or len(files) != 566:
        raise RuntimeError("cohort shards do not cover 566 unique train files")
    frame = pd.concat(summaries, ignore_index=True)
    result = {"status": "complete", "files": len(files),
              "proposals": int(frame.proposals.sum()),
              "positive_any": int(frame.positive_any.sum()),
              "positive_fraction": float(frame.positive_any.sum() / max(frame.proposals.sum(), 1)),
              "utility_sum": float(frame.utility_sum.sum()),
              "fold_files": {str(fold): sum(file_fold(name) == fold for name in files)
                             for fold in range(FOLDS)}}
    frame.to_csv(OUTPUT / "summary.csv", index=False)
    write_json(OUTPUT / "cohort_manifest.json", result)
    print(json.dumps(result, indent=2))


def load_all_samples():
    metadata, crop_blocks, file_indices = [], [], []
    for shard in range(SHARDS):
        root = OUTPUT / f"shard_{shard}"
        for path in sorted(root.glob("*.json.gz")):
            with gzip.open(path, "rt") as handle:
                record = json.load(handle)
            crops = np.load(root / f"{path.name[:-8]}.npz")["crops"]
            if len(crops) != len(record["proposals"]):
                raise RuntimeError("crop/proposal alignment failed")
            base = len(metadata)
            for row in record["proposals"]:
                metadata.append({"file_name": record["file_name"], "fold": record["fold"], **row})
            crop_blocks.append(crops)
    return pd.DataFrame(metadata), np.concatenate(crop_blocks) if crop_blocks else np.zeros((0, 4, CROP_SIZE, CROP_SIZE), np.uint8)


def utility_pq(frame: pd.DataFrame, score: str, threshold: float) -> float:
    """Expected global PQ surrogate, preserving per-file annotator weighting."""
    selected = frame[frame[score] >= threshold]
    numerator = float((selected.utility * selected.annotator_records).sum())
    tp = float((selected.positive_fraction * selected.annotator_records).sum())
    predictions = float(selected.annotator_records.sum())
    fp = predictions - tp
    gt = float(frame.groupby("file_name").gt_instances.first().sum())
    fn = max(gt - tp, 0.0)
    denominator = tp + 0.5 * (fp + fn)
    return numerator / denominator if denominator else 0.0


def cross_fitted_utility_pq(frame: pd.DataFrame, score: str) -> dict:
    thresholds = np.linspace(0.05, 0.95, 37)
    fold_rows = []
    for fold in range(FOLDS):
        training = frame[frame.fold != fold]
        validation = frame[frame.fold == fold]
        chosen = max(thresholds, key=lambda value: (utility_pq(training, score, value), value))
        fold_rows.append({"fold": fold, "threshold": float(chosen),
                          "validation_pq": float(utility_pq(validation, score, chosen))})
    return {"mean_fold_pq": float(np.mean([row["validation_pq"] for row in fold_rows])),
            "folds": fold_rows}


def train_oof(epochs: int = 6):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    frame, crops = load_all_samples()
    scalar_columns = ["confidence", "log_area", "area_ratio", "crop_width", "crop_height",
                      "source_full", "source_tile"]
    scalar = frame[scalar_columns].astype(float).to_numpy(np.float32)
    scalar[:, 3:5] /= 2048.0
    predictions = np.zeros((len(frame), 2), dtype=np.float32)

    class UtilityNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            self.head = nn.Sequential(nn.Linear(64 + len(scalar_columns), 64), nn.ReLU(),
                                      nn.Dropout(0.15), nn.Linear(64, 2))

        def forward(self, image, values):
            feature = self.conv(image).flatten(1)
            return self.head(torch.cat((feature, values), 1))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    for fold in range(FOLDS):
        train_mask = frame.fold.to_numpy() != fold
        valid_mask = ~train_mask
        model = UtilityNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        dataset = TensorDataset(
            torch.from_numpy(crops[train_mask].astype(np.float32) / 255.0),
            torch.from_numpy(scalar[train_mask]),
            torch.from_numpy(frame.utility.to_numpy(np.float32)[train_mask]),
            torch.from_numpy(frame.positive_fraction.to_numpy(np.float32)[train_mask]))
        loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2,
                            generator=torch.Generator().manual_seed(SEED + fold))
        for _ in range(epochs):
            model.train()
            for image, values, utility, fraction in loader:
                image, values = image.to(device), values.to(device)
                utility, fraction = utility.to(device), fraction.to(device)
                output = model(image, values)
                predicted_utility = output[:, 0].sigmoid()
                predicted_fraction = output[:, 1].sigmoid()
                loss = (torch.nn.functional.smooth_l1_loss(predicted_utility, utility)
                        + torch.nn.functional.binary_cross_entropy(predicted_fraction, fraction))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_crops = crops[valid_mask].astype(np.float32) / 255.0
            valid_scalar = scalar[valid_mask]
            blocks = []
            for start in range(0, len(valid_crops), 256):
                image = torch.from_numpy(valid_crops[start:start + 256]).to(device)
                values = torch.from_numpy(valid_scalar[start:start + 256]).to(device)
                blocks.append(model(image, values).sigmoid().cpu().numpy())
            predictions[valid_mask] = np.concatenate(blocks)
        del model
        torch.cuda.empty_cache()

    frame["oof_utility"] = predictions[:, 0]
    frame["oof_fraction"] = predictions[:, 1]
    frame["oof_product"] = frame.confidence * frame.oof_utility
    labels = frame.positive_fraction.to_numpy() > 0
    metrics = {}
    for name in ("confidence", "oof_utility", "oof_fraction", "oof_product"):
        values = frame[name].to_numpy(float)
        metrics[name] = {
            "auc": ranking_auc(values, labels),
            "utility_correlation": float(pd.Series(values).corr(frame.utility, method="spearman")),
            "cross_fitted_pq": cross_fitted_utility_pq(frame, name),
        }
    baseline = metrics["confidence"]
    candidate = metrics["oof_product"]
    learnable = bool(
        candidate["auc"] > baseline["auc"] + 0.01
        and candidate["utility_correlation"] > baseline["utility_correlation"] + 0.01
        and candidate["cross_fitted_pq"]["mean_fold_pq"]
        > baseline["cross_fitted_pq"]["mean_fold_pq"] + 0.005
    )
    frame.to_csv(OUTPUT / "oof_predictions.csv", index=False)
    result = {"status": "complete", "samples": len(frame), "positive": int(labels.sum()),
              "folds": FOLDS, "epochs": epochs, "metrics": metrics,
              "learnability_gates": {"auc_delta": 0.01, "correlation_delta": 0.01,
                                     "cross_fitted_pq_delta": 0.005},
              "learnable": learnable,
              "action": "freeze V28 model/evaluation" if learnable
                        else "close utility scorer without tune/audit"}
    write_json(OUTPUT / "oof_report.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "merge", "oof"))
    parser.add_argument("--shard", type=int, choices=range(SHARDS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.mode == "generate":
        if args.shard is None: parser.error("generate requires --shard")
        generate_shard(args.shard, args.limit)
    elif args.mode == "merge": merge_cohort()
    else: train_oof(args.epochs)


if __name__ == "__main__":
    main()
