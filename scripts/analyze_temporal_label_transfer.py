"""Evaluate split-safe temporal/cross-site propagation of competition labels.

Uses only the frozen training partition as label donors. Policies are selected on
tune and evaluated once on audit. This is a CPU-only diagnostic: it never reads
test labels, writes a submission, or downloads external annotations.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_v19_mask2former import load_split  # noqa: E402
from src.coco import index_coco, load_coco  # noqa: E402

TIME_LIMITS_HOURS = (1.0, 6.0, 24.0, 72.0)
REGISTRATION_SIZE = 512
MIN_CORRELATION = 0.80


def timestamp(name: str) -> datetime:
    return datetime.strptime(Path(name).stem[:14], "%Y%m%d%H%M%S")


def nearest_donor(query: str, donors: list[str]) -> tuple[str, float]:
    donor = min(donors, key=lambda name: abs((timestamp(name) - timestamp(query)).total_seconds()))
    hours = abs((timestamp(donor) - timestamp(query)).total_seconds()) / 3600.0
    return donor, hours


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (REGISTRATION_SIZE, REGISTRATION_SIZE), interpolation=cv2.INTER_AREA)
    disk = image > np.percentile(image, 12)
    values = image[disk]
    lo, hi = np.percentile(values, (2, 98))
    normalized = np.clip((image.astype(np.float32) - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    return normalized


def register_donor_to_query(donor: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, float]:
    donor_small = normalize_image(donor)
    query_small = normalize_image(query)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    correlation, warp = cv2.findTransformECC(
        query_small, donor_small, warp, cv2.MOTION_AFFINE, criteria, None, 3,
    )
    scale = donor.shape[1] / REGISTRATION_SIZE
    full = warp.copy()
    full[:, 2] *= scale
    return full, float(correlation)


def rasterize(annotations: list[dict], height: int, width: int) -> list[np.ndarray]:
    masks = []
    for annotation in annotations:
        rle = mask_utils.merge(mask_utils.frPyObjects(annotation["segmentation"], height, width))
        masks.append(mask_utils.decode(rle).astype(np.uint8))
    return masks


def warp_masks(masks: list[np.ndarray], warp: np.ndarray, shape: tuple[int, int]) -> list[np.ndarray]:
    height, width = shape
    output = []
    for mask in masks:
        warped = cv2.warpAffine(
            mask, warp, (width, height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        if warped.sum() >= 20:
            output.append(warped.astype(bool))
    return output


def counts(predictions: list[np.ndarray], targets: list[np.ndarray]) -> list[float]:
    if not predictions or not targets:
        return [0.0, 0, len(predictions), len(targets)]
    pred_rles = mask_utils.encode(np.asfortranarray(np.stack(predictions, axis=-1).astype(np.uint8)))
    gt_rles = mask_utils.encode(np.asfortranarray(np.stack(targets, axis=-1).astype(np.uint8)))
    iou = np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(targets))).T
    hits = iou > 0.5
    return [float(iou[hits].sum()), int(hits.sum()), int((~hits.any(0)).sum()), int((~hits.any(1)).sum())]


def pq(total: list[float]) -> float:
    return total[0] / (total[1] + 0.5 * (total[2] + total[3])) if sum(total[1:]) else 0.0


def add(total: list[float], row: list[float]) -> None:
    for index, value in enumerate(row):
        total[index] += value


def build_records(coco_path: Path) -> tuple[dict[str, list[list[dict]]], dict[str, dict]]:
    coco = load_coco(coco_path)
    images, annotations = index_coco(coco)
    records: dict[str, list[list[dict]]] = {}
    info: dict[str, dict] = {}
    for image_id, image in images.items():
        records.setdefault(image["file_name"], []).append(annotations.get(image_id, []))
        info[image["file_name"]] = image
    return records, info


def evaluate_group(names: list[str], donors: list[str], image_dir: Path,
                   records: dict[str, list[list[dict]]], info: dict[str, dict]) -> dict:
    policies = {hours: [0.0, 0, 0, 0] for hours in TIME_LIMITS_HOURS}
    diagnostics = []
    for index, name in enumerate(names, 1):
        donor_name, delta_hours = nearest_donor(name, donors)
        query = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)
        donor = cv2.imread(str(image_dir / donor_name), cv2.IMREAD_GRAYSCALE)
        if query is None or donor is None:
            raise RuntimeError(f"Missing image for pair {name} <- {donor_name}")
        try:
            warp, correlation = register_donor_to_query(donor, query)
        except cv2.error:
            warp, correlation = np.eye(2, 3, dtype=np.float32), -1.0
        donor_masks = rasterize(records[donor_name][0], info[donor_name]["height"], info[donor_name]["width"])
        predictions = warp_masks(donor_masks, warp, query.shape)
        per_target = []
        for target_annotations in records[name]:
            targets = rasterize(target_annotations, info[name]["height"], info[name]["width"])
            per_target.append(counts(predictions, targets))
        for hours, total in policies.items():
            accepted = delta_hours <= hours and correlation >= MIN_CORRELATION
            for row in per_target:
                add(total, row if accepted else [0.0, 0, 0, row[1] + row[3]])
        diagnostics.append({
            "file_name": name, "donor": donor_name, "delta_hours": delta_hours,
            "correlation": correlation, "predictions": len(predictions),
            "annotator_records": len(records[name]),
        })
        if index % 20 == 0:
            print(f"evaluated {index}/{len(names)}", flush=True)
    return {
        "policies": {str(hours): {"pq": pq(total), "counts": total} for hours, total in policies.items()},
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/temporal_transfer_2026-09-11.json")
    args = parser.parse_args()
    base = args.data_root / "filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    coco_path = base / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    image_dir = base / "train/train_images"
    split = load_split(coco_path)
    records, info = build_records(coco_path)
    tune = evaluate_group(split.tune_files, split.train_files, image_dir, records, info)
    selected = max(TIME_LIMITS_HOURS, key=lambda hours: tune["policies"][str(hours)]["pq"])
    audit = evaluate_group(split.audit_files, split.train_files, image_dir, records, info)
    result = {
        "method": "nearest frozen-train observation; 512px affine ECC; first donor annotator; no model predictions",
        "registration_size": REGISTRATION_SIZE,
        "min_correlation": MIN_CORRELATION,
        "time_limits_hours": list(TIME_LIMITS_HOURS),
        "selection": "maximum tune PQ; audit read only at selected time limit",
        "selected_time_limit_hours": selected,
        "selected_tune": tune["policies"][str(selected)],
        "selected_audit": audit["policies"][str(selected)],
        "tune": tune,
        "audit": audit,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["selected_time_limit_hours", "selected_tune", "selected_audit"]}, indent=2))


if __name__ == "__main__":
    main()
