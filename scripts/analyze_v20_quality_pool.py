"""Diagnose the V20 low-confidence candidate pool for a quality-aware V21.

Runs one frozen checkpoint over a requested split, retains the exact score-ordered
non-overlapping candidates accepted at confidence 0.05, and records per-candidate
image/mask geometry plus IoU against every annotator target. It computes:
- current detector-score ordering/cutoffs;
- an optimistic per-annotator oracle (upper bound; annotator-specific selection);
- a deployable file-level oracle utility ranking shared across annotators;
- feature/label tables for training a scorer on train and testing on tune.
No model is trained and no submission is produced.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

BASE = Path(os.environ.get("V20_BASE", "/mnt/solar-v20-condinst"))
sys.path.insert(0, str(BASE))
import train_v20_condinst as v20  # noqa: E402


def targets_by_file(path: Path):
    data = v20.read_json(path)
    anns = defaultdict(list)
    for ann in data["annotations"]:
        anns[ann["image_id"]].append(ann)
    targets, infos = defaultdict(list), {}
    for image in data["images"]:
        rles = [mask_utils.merge(mask_utils.frPyObjects(
            ann["segmentation"], image["height"], image["width"]
        )) for ann in anns[image["id"]]]
        targets[image["file_name"]].append(rles)
        infos[image["file_name"]] = image
    return dict(targets), infos


def encode_masks(masks):
    if not masks:
        return []
    value = mask_utils.encode(np.asfortranarray(np.stack(masks, -1).astype(np.uint8)))
    return value if isinstance(value, list) else [value]


def ring_contrast(gray: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & ~mask
    inside = gray[mask]
    outside = gray[ring]
    mean_in = float(inside.mean()) if inside.size else 0.0
    mean_ring = float(outside.mean()) if outside.size else mean_in
    return mean_in, mean_ring, mean_ring - mean_in


def candidate_features(mask: np.ndarray, gray: np.ndarray, score: float,
                       box: np.ndarray, image_index: int, candidate_index: int,
                       file_name: str) -> dict:
    ys, xs = np.where(mask)
    area = int(len(xs))
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width, height = x1 - x0, y1 - y0
    perimeter = float(cv2.arcLength(max(cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )[0], key=cv2.contourArea), True))
    mean_in, mean_ring, contrast = ring_contrast(gray, mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    radius = ((cx - 1023.5) ** 2 + (cy - 1023.5) ** 2) ** 0.5 / 1024.0
    return {
        "file_name": file_name, "image_index": image_index,
        "candidate_index": candidate_index, "score": float(score), "area": area,
        "log_area": float(np.log1p(area)), "bbox_width": width, "bbox_height": height,
        "bbox_area": width * height, "fill_ratio": area / max(width * height, 1),
        "aspect_ratio": max(width, height) / max(min(width, height), 1),
        "perimeter": perimeter,
        "compactness": 4.0 * np.pi * area / max(perimeter * perimeter, 1.0),
        "center_x": cx / 2048.0, "center_y": cy / 2048.0, "disk_radius": radius,
        "mean_intensity": mean_in, "ring_intensity": mean_ring,
        "dark_contrast": contrast,
        "box_x0": float(box[0]), "box_y0": float(box[1]),
        "box_x1": float(box[2]), "box_y1": float(box[3]),
    }


def evaluate_file_candidates(pred_rles: list[dict], annotator_targets: list[list[dict]]):
    matrices = []
    for targets in annotator_targets:
        if pred_rles and targets:
            matrices.append(np.asarray(mask_utils.iou(pred_rles, targets, [0] * len(targets))).T)
        else:
            matrices.append(np.zeros((len(targets), len(pred_rles)), dtype=float))
    return matrices


def totals_for_indices(indices_by_file, file_payloads):
    total = [0.0, 0, 0, 0]
    for file_name, payload in file_payloads.items():
        indices = indices_by_file.get(file_name, [])
        for matrix in payload["iou_matrices"]:
            selected = matrix[:, indices] if indices else np.zeros((matrix.shape[0], 0))
            hits = selected > 0.5
            total[0] += float(selected[hits].sum())
            total[1] += int(hits.sum())
            total[2] += int((~hits.any(0)).sum()) if selected.shape[1] else 0
            total[3] += int((~hits.any(1)).sum()) if selected.shape[0] else 0
    return total


def pq(total):
    denominator = total[1] + 0.5 * (total[2] + total[3])
    return total[0] / denominator if denominator else 0.0


def oracle_summaries(file_payloads):
    # Shared file-level utility: retain a candidate if it matches any GT for at
    # least one annotator. This is optimistic but deployable as one prediction
    # set per physical file, unlike the per-annotator oracle below.
    shared = {}
    perfect_total = [0.0, 0, 0, 0]
    all_indices = {}
    for file_name, payload in file_payloads.items():
        n = len(payload["features"])
        all_indices[file_name] = list(range(n))
        keep = []
        for index in range(n):
            if any(matrix.shape[0] and float(matrix[:, index].max()) > 0.5
                   for matrix in payload["iou_matrices"]):
                keep.append(index)
        shared[file_name] = keep
        # Impossible per-annotator selection upper bound: choose every prediction
        # that hits that annotator, with zero FP among retained predictions.
        for matrix in payload["iou_matrices"]:
            if not matrix.size:
                perfect_total[3] += matrix.shape[0]
                continue
            useful = np.where((matrix > 0.5).any(0))[0]
            selected = matrix[:, useful]
            hits = selected > 0.5
            perfect_total[0] += float(selected[hits].sum())
            perfect_total[1] += int(hits.sum())
            perfect_total[2] += int((~hits.any(0)).sum())
            perfect_total[3] += int((~hits.any(1)).sum())
    all_total = totals_for_indices(all_indices, file_payloads)
    shared_total = totals_for_indices(shared, file_payloads)
    return {
        "all_candidates": {"pq": pq(all_total), "counts": all_total},
        "shared_any_annotator_oracle": {"pq": pq(shared_total), "counts": shared_total},
        "per_annotator_oracle_upper_bound": {"pq": pq(perfect_total), "counts": perfect_total},
    }


def run(checkpoint: Path, split_path: Path, output_dir: Path):
    from detectron2.engine import DefaultPredictor

    v20.register_datasets()
    cfg = v20.build_cfg()
    cfg.defrost(); cfg.MODEL.WEIGHTS = str(checkpoint); cfg.freeze()
    predictor = DefaultPredictor(cfg)
    targets, _ = targets_by_file(split_path)
    rows, payloads = [], {}
    started = time.monotonic()
    for image_index, file_name in enumerate(sorted(targets), 1):
        image = cv2.imread(str(v20.IMAGE_ROOT / file_name))
        output = predictor(image)["instances"].to("cpu")
        masks = output.pred_masks.numpy() if len(output) else np.zeros((0, 2048, 2048), bool)
        scores = output.scores.numpy() if len(output) else np.zeros(0, np.float32)
        boxes = output.pred_boxes.tensor.numpy() if len(output) else np.zeros((0, 4), np.float32)
        order = np.argsort(-scores)
        occupied = np.zeros((2048, 2048), bool)
        accepted_masks, features = [], []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for raw_index in order:
            if float(scores[raw_index]) < 0.05:
                continue
            residual = masks[raw_index].astype(bool) & ~occupied
            if residual.sum() < 20:
                continue
            candidate_index = len(accepted_masks)
            feature = candidate_features(
                residual, gray, scores[raw_index], boxes[raw_index], image_index,
                candidate_index, file_name,
            )
            features.append(feature); rows.append(feature); accepted_masks.append(residual)
            occupied |= residual
        pred_rles = encode_masks(accepted_masks)
        matrices = evaluate_file_candidates(pred_rles, targets[file_name])
        # Add labels/utility shared across annotators to table rows.
        for index, feature in enumerate(features):
            ious = [float(matrix[:, index].max()) if matrix.shape[0] else 0.0 for matrix in matrices]
            feature["max_iou_any_annotator"] = max(ious, default=0.0)
            feature["mean_max_iou_annotator"] = float(np.mean(ious)) if ious else 0.0
            feature["annotator_hit_count"] = sum(value > 0.5 for value in ious)
            feature["annotator_record_count"] = len(ious)
            feature["is_positive_any"] = int(any(value > 0.5 for value in ious))
        payloads[file_name] = {"features": features, "iou_matrices": matrices}
        if image_index % 20 == 0 or image_index == len(targets):
            print(f"diagnosed {image_index}/{len(targets)}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "candidates.csv", index=False)
    oracle = oracle_summaries(payloads)
    report = {
        "checkpoint": checkpoint.name, "checkpoint_sha256": v20.sha256(checkpoint),
        "split": split_path.name, "files": len(targets), "candidates": len(frame),
        "positive_any": int(frame.is_positive_any.sum()) if len(frame) else 0,
        "candidate_positive_rate": float(frame.is_positive_any.mean()) if len(frame) else 0.0,
        "detector_score_auc": None,
        "oracle": oracle, "seconds": time.monotonic() - started,
    }
    if len(frame) and frame.is_positive_any.nunique() == 2:
        # Mann-Whitney rank AUC with tie averaging; avoids a heavyweight
        # sklearn dependency in the isolated Azure runtime.
        labels = frame.is_positive_any.to_numpy(dtype=np.int8)
        scores = frame.score.to_numpy(dtype=np.float64)
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and scores[order[end]] == scores[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
            start = end
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        rank_sum_pos = float(ranks[labels == 1].sum())
        report["detector_score_auc"] = (
            rank_sum_pos - n_pos * (n_pos + 1) / 2.0
        ) / (n_pos * n_neg)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "tune"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    split_path = {"train": v20.TRAIN_JSON, "tune": v20.TUNE_JSON}[args.split]
    run(args.checkpoint, split_path, args.output_dir)


if __name__ == "__main__":
    main()
