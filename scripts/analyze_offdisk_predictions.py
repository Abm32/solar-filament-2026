"""Measure how much of our prediction mass lies outside the solar disk.

Filaments are on-disk H-alpha features, so any predicted instance outside the
solar limb is a false positive by construction, independent of ground truth.
This decides whether disk-based FP suppression (plan item E-B) is worth any
effort before spending GPU time on it.

Calibration matters more than the suppression itself: the canonical GT
annotations define how close to the limb real filaments actually appear, so the
GT margin distribution sets the safe erosion radius. A suppression rule that
discards true positives near the limb would cost more PQ than the FP it removes.

CPU only; consumes the submitted CSV, the canonical annotations, and the test
JPEGs. Runs no inference and writes no predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


def fit_disk(gray, blur=5):
    """Otsu threshold, largest filled component, then a least-squares circle.

    The solar disk dominates an H-alpha frame, so the largest bright component
    is the disk. A circle fit is used rather than the raw component because
    filaments and off-limb prominences distort the component boundary.
    """
    work = cv2.GaussianBlur(gray, (blur, blur), 0)
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, [largest], -1, 255, cv2.FILLED)
    coverage = float((filled > 0).mean())
    return dict(cx=float(x), cy=float(y), radius=float(radius), disk_fraction=coverage)


def radial_stats(mask, disk):
    """Fraction of mask pixels beyond the fitted limb, and the worst excursion."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    distance = np.hypot(xs - disk["cx"], ys - disk["cy"]) / disk["radius"]
    return dict(area=int(len(ys)), max_radius=float(distance.max()),
                mean_radius=float(distance.mean()),
                fraction_outside=float((distance > 1.0).mean()),
                centroid_radius=float(np.hypot(xs.mean() - disk["cx"],
                                               ys.mean() - disk["cy"]) / disk["radius"]))


def load_gt_by_file(annotation_path):
    payload = json.loads(Path(annotation_path).read_text())
    images = {image["id"]: image for image in payload["images"]}
    grouped = {}
    for annotation in payload["annotations"]:
        image = images[annotation["image_id"]]
        grouped.setdefault(image["file_name"], []).append((image, annotation))
    return grouped


def rasterize(annotation, height, width):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    else:
        rle = segmentation
    return mask_utils.decode(rle).astype(bool)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path,
                        default=root / "submission_ready/v11_2026-09-06/submission_v11.csv")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/offdisk_2026-09-06")
    parser.add_argument("--gt-limit", type=int, default=120,
                        help="Training files sampled to calibrate the safe limb margin")
    args = parser.parse_args()
    data_root = root / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    test_dir = data_root / "test/test_images"
    train_dir = data_root / "train/train_images"
    annotations = data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    args.output.mkdir(parents=True, exist_ok=True)

    # Calibrate against real annotations first: this bounds how aggressive any
    # erosion may be without deleting genuine near-limb filaments.
    grouped = load_gt_by_file(annotations)
    gt_rows = []
    for file_name in sorted(grouped)[: args.gt_limit]:
        image_path = train_dir / file_name
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        disk = fit_disk(gray)
        if disk is None:
            continue
        for image, annotation in grouped[file_name]:
            mask = rasterize(annotation, image["height"], image["width"])
            stats = radial_stats(mask, disk)
            if stats:
                gt_rows.append(dict(file_name=file_name, category_id=annotation["category_id"],
                                    **stats))
    gt = pd.DataFrame(gt_rows)

    frame = pd.read_csv(args.submission, dtype=str, keep_default_na=False)
    frame["image"] = frame.filament_id.str.rsplit("_", n=1, expand=True)[0]
    prediction_rows = []
    disks = {}
    for stem, rows in frame.groupby("image"):
        gray = cv2.imread(str(test_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Missing test image for {stem}")
        disk = fit_disk(gray)
        if disk is None:
            raise ValueError(f"Could not fit a disk for {stem}")
        disks[stem] = disk
        for row in rows.itertuples(index=False):
            mask = mask_utils.decode({"size": [2048, 2048],
                                      "counts": row.segmentation_rle.encode()}).astype(bool)
            stats = radial_stats(mask, disk)
            prediction_rows.append(dict(filament_id=row.filament_id, image=stem, **stats))
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(args.output / "prediction_radial.csv", index=False)
    gt.to_csv(args.output / "gt_radial.csv", index=False)

    def counts(frame_, column, thresholds):
        return {str(t): int((frame_[column] > t).sum()) for t in thresholds}

    summary = dict(
        submission=str(args.submission),
        predictions=len(predictions), gt_instances=len(gt), gt_files=int(gt.file_name.nunique()),
        disk_fraction=dict(median=float(np.median([d["disk_fraction"] for d in disks.values()])),
                           min=float(min(d["disk_fraction"] for d in disks.values())),
                           max=float(max(d["disk_fraction"] for d in disks.values()))),
        gt_centroid_radius=dict(median=float(gt.centroid_radius.median()),
                                p95=float(gt.centroid_radius.quantile(0.95)),
                                p99=float(gt.centroid_radius.quantile(0.99)),
                                max=float(gt.centroid_radius.max())),
        gt_max_radius=dict(p95=float(gt.max_radius.quantile(0.95)),
                           p99=float(gt.max_radius.quantile(0.99)),
                           max=float(gt.max_radius.max())),
        gt_beyond_limb=counts(gt, "centroid_radius", [0.95, 1.0, 1.05]),
        gt_any_pixel_beyond_limb=int((gt.fraction_outside > 0).sum()),
        prediction_centroid_beyond=counts(predictions, "centroid_radius", [0.95, 1.0, 1.05]),
        prediction_fully_outside=int((predictions.fraction_outside > 0.999).sum()),
        prediction_majority_outside=int((predictions.fraction_outside > 0.5).sum()),
        prediction_any_pixel_outside=int((predictions.fraction_outside > 0).sum()),
        note=("Off-disk predictions are false positives by construction. GT radial "
              "statistics bound the safe erosion; a rule must not discard genuine "
              "near-limb filaments. Circle fit is least-squares on the largest "
              "Otsu component, so radii near 1.0 carry fit error."))
    (args.output / "offdisk_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
