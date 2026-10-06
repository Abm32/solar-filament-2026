"""V40 tune-only multiscale dark-ridge signal opportunity audit.

The CPU-only gate measures fixed Frangi response on expert-spine pixels versus
clean local background for canonical tune instances. It reads only official
tune train-images and labels, never fit/audit/test images, trains, or exports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata
from skimage.filters import frangi
from skimage.morphology import disk

try:
    from scripts import train_v23_radial_clahe as v23
    from scripts import v31_target_oracle as canonical
except ModuleNotFoundError:  # Direct execution from scripts/.
    import train_v23_radial_clahe as v23
    import v31_target_oracle as canonical

ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMAGES = (
    ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/"
    "train/train_images"
)
V39_RESULT = ROOT / "outputs/v39_fragment_graph_opportunity/result.json"
V39_SHA256 = "bf224e9de46324f8f7849b7d524f28f9a07d4a5f3fdde6ad370a85d4705a5233"
FRANGI_SIGMAS = (1, 2, 4, 8)
FRANGI_ALPHA = 0.5
FRANGI_BETA = 0.5
FRANGI_GAMMA = None
BLACK_RIDGES = True
CROP_MARGIN = 48
SPINE_DILATION_RADIUS = 2
BACKGROUND_INNER = 8.0
BACKGROUND_OUTER = 24.0
MIN_POSITIVE_PIXELS = 10
MIN_BACKGROUND_PIXELS = 100
AUC_GATE = 0.70
Q90_COVERAGE_GATE = 0.25
MISS_ADDRESSABLE_GATE = 0.65
ZERO_FRAGMENT_ADDRESSABLE_GATE = 0.65
FOUND_ADDRESSABLE_GATE = 0.80
RECOVERY_IOU = 0.65
PROJECTED_PQ_GATE = 0.50
ZERO_FRAGMENT_MEDIAN_AUC_GATE = 0.75
ZERO_FRAGMENT_MEDIAN_Q90_GATE = 0.35
EXPECTED_FOUND = 922
EXPECTED_MISSED = 212
EXPECTED_ZERO_FRAGMENT = 113
V17_COUNTS = (481.00931292355114, 704, 306, 430)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def average_rank_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64).ravel()
    negative = np.asarray(negative, dtype=np.float64).ravel()
    if not positive.size or not negative.size:
        raise ValueError("AUC requires positive and negative samples")
    values = np.concatenate([positive, negative])
    if not np.isfinite(values).all():
        raise ValueError("AUC values must be finite")
    ranks = rankdata(values, method="average")
    rank_sum = float(ranks[: positive.size].sum())
    return (
        rank_sum - positive.size * (positive.size + 1) / 2.0
    ) / (positive.size * negative.size)


def ridge_response(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 2 or value.size == 0:
        raise ValueError("Frangi input must be a nonempty 2D image")
    response = frangi(
        value,
        sigmas=FRANGI_SIGMAS,
        alpha=FRANGI_ALPHA,
        beta=FRANGI_BETA,
        gamma=FRANGI_GAMMA,
        black_ridges=BLACK_RIDGES,
        mode="reflect",
    )
    response = np.asarray(response, dtype=np.float32)
    if response.shape != value.shape or not np.isfinite(response).all():
        raise RuntimeError("Frangi response is invalid")
    return response


def crop_bounds(mask: np.ndarray, margin: int = CROP_MARGIN) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        raise ValueError("cannot crop an empty GT mask")
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(mask.shape[1], int(xs.max()) + margin + 1),
        min(mask.shape[0], int(ys.max()) + margin + 1),
    )


def measure_signal(
    transformed: np.ndarray,
    gt_mask: np.ndarray,
    spine_mask: np.ndarray,
    all_annotations_mask: np.ndarray,
) -> dict:
    """Measure the frozen local ridge signal for one annotated instance."""
    shape = np.asarray(transformed).shape
    if shape != gt_mask.shape or shape != spine_mask.shape or shape != all_annotations_mask.shape:
        raise ValueError("V40 image and masks must share shape")
    x0, y0, x1, y1 = crop_bounds(gt_mask)
    image_crop = transformed[y0:y1, x0:x1]
    gt_crop = gt_mask[y0:y1, x0:x1]
    spine_crop = spine_mask[y0:y1, x0:x1]
    annotations_crop = all_annotations_mask[y0:y1, x0:x1]

    positive = ndi.binary_dilation(spine_crop, structure=disk(SPINE_DILATION_RADIUS))
    positive &= gt_crop
    distance = ndi.distance_transform_edt(~gt_crop)
    background = (
        (distance >= BACKGROUND_INNER)
        & (distance <= BACKGROUND_OUTER)
        & ~annotations_crop
    )
    positive_count = int(positive.sum())
    background_count = int(background.sum())
    if positive_count < MIN_POSITIVE_PIXELS:
        raise RuntimeError(f"V40 positive region too small: {positive_count}")
    if background_count < MIN_BACKGROUND_PIXELS:
        raise RuntimeError(f"V40 clean background too small: {background_count}")

    response = ridge_response(image_crop)
    positive_values = response[positive].astype(np.float64)
    background_values = response[background].astype(np.float64)
    auc = average_rank_auc(positive_values, background_values)
    background_q90 = float(np.quantile(background_values, 0.90))
    q90_coverage = float(np.mean(positive_values > background_q90))
    positive_median = float(np.median(positive_values))
    background_median = float(np.median(background_values))
    median_separation = positive_median - background_median
    addressable = (
        auc >= AUC_GATE
        and q90_coverage >= Q90_COVERAGE_GATE
        and median_separation > 0.0
    )
    return {
        "crop": [x0, y0, x1, y1],
        "positive_pixels": positive_count,
        "clean_background_pixels": background_count,
        "auc": auc,
        "background_q90": background_q90,
        "q90_coverage": q90_coverage,
        "positive_median": positive_median,
        "background_median": background_median,
        "median_separation": median_separation,
        "ridge_addressable": addressable,
    }


def projected_v17(recovered: int) -> dict:
    iou_sum, tp, fp, fn = V17_COUNTS
    if recovered < 0 or recovered > fn:
        raise ValueError("recovered count outside V17 FN budget")
    projected_iou = iou_sum + recovered * RECOVERY_IOU
    projected_tp = tp + recovered
    projected_fn = fn - recovered
    denominator = projected_tp + 0.5 * fp + 0.5 * projected_fn
    return {
        "assumed_recovered_iou": RECOVERY_IOU,
        "recovered": recovered,
        "iou_sum": projected_iou,
        "tp": projected_tp,
        "fp": fp,
        "fn": projected_fn,
        "pq": projected_iou / denominator if denominator else 0.0,
    }


def synthetic_checks() -> dict:
    size = 129
    flat = np.full((size, size), 180, dtype=np.uint8)
    dark = flat.copy()
    bright = flat.copy()
    cv2.line(dark, (18, 64), (110, 64), 60, 3, cv2.LINE_AA)
    cv2.line(bright, (18, 64), (110, 64), 250, 3, cv2.LINE_AA)
    region = np.zeros_like(flat, dtype=bool)
    region[62:67, 24:105] = True
    dark_score = float(np.mean(ridge_response(dark)[region]))
    bright_score = float(np.mean(ridge_response(bright)[region]))
    flat_score = float(np.mean(ridge_response(flat)[region]))
    auc = average_rank_auc(np.asarray([2.0, 3.0, 4.0]), np.asarray([0.0, 1.0, 2.0]))
    tied_auc = average_rank_auc(np.ones(3), np.ones(4))
    checks = {
        "dark_exceeds_bright": dark_score > bright_score,
        "dark_exceeds_flat": dark_score > flat_score,
        "known_auc": math.isclose(auc, 17.0 / 18.0),
        "tied_auc_is_half": math.isclose(tied_auc, 0.5),
        "finite": all(math.isfinite(value) for value in (dark_score, bright_score, flat_score)),
    }
    checks["passed"] = all(checks.values())
    return {**checks, "dark_score": dark_score, "bright_score": bright_score,
            "flat_score": flat_score, "auc_example": auc}


def all_annotations_by_file(runtime: dict, name: str) -> np.ndarray:
    shape = (2048, 2048)
    union = np.zeros(shape, dtype=bool)
    for image_id in runtime["by_file"][name]:
        info = runtime["images"][image_id]
        for annotation in runtime["annotations"][image_id]:
            union |= canonical.annotation_mask(info, annotation)
    return union


def summarize(rows: list[dict]) -> dict:
    addressable = [row for row in rows if row["ridge_addressable"]]
    return {
        "instances": len(rows),
        "ridge_addressable": len(addressable),
        "ridge_addressable_rate": len(addressable) / len(rows) if rows else 0.0,
        "median_auc": float(np.median([row["auc"] for row in rows])) if rows else 0.0,
        "median_q90_coverage": (
            float(np.median([row["q90_coverage"] for row in rows])) if rows else 0.0
        ),
        "median_auc_addressable": (
            float(np.median([row["auc"] for row in addressable])) if addressable else 0.0
        ),
        "median_q90_coverage_addressable": (
            float(np.median([row["q90_coverage"] for row in addressable]))
            if addressable else 0.0
        ),
    }


def run(v39_result: Path, image_root: Path, output: Path) -> dict:
    runtime = canonical.load_population(canonical.CANONICAL_ANNOTATIONS)
    if runtime["populations"]["tune"] != canonical.EXPECTED_TUNE:
        raise RuntimeError("V40 tune population changed")
    if sha256(v39_result) != V39_SHA256:
        raise RuntimeError("V40 pinned V39 result identity changed")
    v39_report = json.loads(v39_result.read_text())
    v39_rows = {
        (str(row["image_id"]), int(row["gt_index"])): row
        for row in v39_report["diagnostics"]
    }
    if len(v39_rows) != canonical.EXPECTED_TUNE["annotations"]:
        raise RuntimeError("V40 V39 diagnostic population changed")
    found_count = sum(row["detector_group"] == "found_by_any" for row in v39_rows.values())
    missed_count = sum(row["detector_group"] == "missed_by_all" for row in v39_rows.values())
    zero_count = sum(
        row["detector_group"] == "missed_by_all"
        and row["admissible_fragment_masks"] == 0
        for row in v39_rows.values()
    )
    if (found_count, missed_count, zero_count) != (
        EXPECTED_FOUND, EXPECTED_MISSED, EXPECTED_ZERO_FRAGMENT
    ):
        raise RuntimeError("V40 V39 group regression failed")

    diagnostics = []
    opened_images = []
    transform_diagnostics = []
    for name in runtime["tune"]:
        path = image_root / name
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (2048, 2048):
            raise RuntimeError(f"V40 invalid tune JPEG: {path}")
        opened_images.append(name)
        transformed, transform_report = v23.transform_image(gray)
        transform_diagnostics.append({"file_name": name, **transform_report})
        exclusion = all_annotations_by_file(runtime, name)
        for image_id in runtime["by_file"][name]:
            info = runtime["images"][image_id]
            for gt_index, annotation in enumerate(runtime["annotations"][image_id]):
                key = (str(image_id), gt_index)
                prior = v39_rows.get(key)
                if prior is None:
                    raise RuntimeError(f"V40 missing V39 key {key}")
                gt_mask = canonical.annotation_mask(info, annotation)
                spine_mask, spine_report = canonical.project_spine(gt_mask, annotation["spine"])
                if not spine_report["valid"]:
                    raise RuntimeError(f"V40 invalid projected spine {key}")
                signal = measure_signal(transformed, gt_mask, spine_mask, exclusion)
                diagnostics.append({
                    "file_name": name,
                    "image_id": str(image_id),
                    "gt_index": gt_index,
                    "detector_group": prior["detector_group"],
                    "admissible_fragment_masks": prior["admissible_fragment_masks"],
                    "zero_fragment_miss": (
                        prior["detector_group"] == "missed_by_all"
                        and prior["admissible_fragment_masks"] == 0
                    ),
                    **signal,
                })

    if len(opened_images) != 94 or len(set(opened_images)) != 94:
        raise RuntimeError("V40 did not open exactly 94 distinct tune images")
    if len(diagnostics) != canonical.EXPECTED_TUNE["annotations"]:
        raise RuntimeError("V40 diagnostic population changed")
    diagnostics.sort(key=lambda row: (row["file_name"], row["image_id"], row["gt_index"]))
    groups = {
        "found_by_any": [row for row in diagnostics if row["detector_group"] == "found_by_any"],
        "missed_by_all": [row for row in diagnostics if row["detector_group"] == "missed_by_all"],
        "zero_fragment_misses": [row for row in diagnostics if row["zero_fragment_miss"]],
    }
    summaries = {name: summarize(rows) for name, rows in groups.items()}
    recovered = summaries["missed_by_all"]["ridge_addressable"]
    projection = projected_v17(recovered)
    synthetic = synthetic_checks()
    zero = summaries["zero_fragment_misses"]
    finite = all(
        math.isfinite(row[key])
        for row in diagnostics
        for key in ("auc", "q90_coverage", "positive_median", "background_median")
    )
    gates = {
        "g0_synthetic": synthetic["passed"],
        "g0_v39_regression": True,
        "g0_image_population": len(opened_images) == 94 and len(set(opened_images)) == 94,
        "g0_regions_valid": all(
            row["positive_pixels"] >= MIN_POSITIVE_PIXELS
            and row["clean_background_pixels"] >= MIN_BACKGROUND_PIXELS
            for row in diagnostics
        ),
        "miss_addressable": (
            summaries["missed_by_all"]["ridge_addressable_rate"] >= MISS_ADDRESSABLE_GATE
        ),
        "zero_fragment_addressable": (
            zero["ridge_addressable_rate"] >= ZERO_FRAGMENT_ADDRESSABLE_GATE
        ),
        "found_control": (
            summaries["found_by_any"]["ridge_addressable_rate"] >= FOUND_ADDRESSABLE_GATE
        ),
        "projected_pq": projection["pq"] >= PROJECTED_PQ_GATE,
        "zero_fragment_median_auc": zero["median_auc_addressable"] >= ZERO_FRAGMENT_MEDIAN_AUC_GATE,
        "zero_fragment_median_q90": (
            zero["median_q90_coverage_addressable"] >= ZERO_FRAGMENT_MEDIAN_Q90_GATE
        ),
        "finite": finite,
        "firewall": True,
    }
    passed = all(gates.values())
    report = {
        "status": "complete",
        "experiment": "v40_multiscale_ridge_signal",
        "annotation_sha256": canonical.sha256(runtime["path"]),
        "v39_sha256": sha256(v39_result),
        "population": runtime["populations"]["tune"],
        "opened_images": len(opened_images),
        "constants": {
            "frangi_sigmas": list(FRANGI_SIGMAS),
            "frangi_alpha": FRANGI_ALPHA,
            "frangi_beta": FRANGI_BETA,
            "frangi_gamma": FRANGI_GAMMA,
            "black_ridges": BLACK_RIDGES,
            "crop_margin": CROP_MARGIN,
            "spine_dilation_radius": SPINE_DILATION_RADIUS,
            "background_ring": [BACKGROUND_INNER, BACKGROUND_OUTER],
            "auc_gate": AUC_GATE,
            "q90_coverage_gate": Q90_COVERAGE_GATE,
            "v23_transform": {
                "profile_size": v23.PROFILE_SIZE,
                "radial_bins": v23.RADIAL_BINS,
                "profile_sigma_bins": v23.PROFILE_SIGMA_BINS,
                "disk_limit": v23.DISK_LIMIT,
                "clahe_clip": v23.CLAHE_CLIP,
                "clahe_grid": list(v23.CLAHE_GRID),
            },
        },
        "synthetic_checks": synthetic,
        "v39_regression": {
            "found_by_any": found_count,
            "missed_by_all": missed_count,
            "zero_fragment_misses": zero_count,
        },
        "groups": summaries,
        "projected_v17": projection,
        "gates": gates,
        "g1_passed": passed,
        "g2_model_design_authorized": passed,
        "transform_diagnostics": transform_diagnostics,
        "diagnostics": diagnostics,
        "reads_tune_images": True,
        "reads_fit_images": False,
        "reads_audit": False,
        "reads_test": False,
        "trains": False,
        "exports": False,
        "submits": False,
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v39-result", type=Path, default=V39_RESULT)
    parser.add_argument("--image-root", type=Path, default=TRAIN_IMAGES)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/v40_multiscale_ridge_signal/result.json",
    )
    args = parser.parse_args()
    report = run(args.v39_result, args.image_root, args.output)
    print(json.dumps({
        "g1_passed": report["g1_passed"],
        "groups": report["groups"],
        "projected_v17": report["projected_v17"],
        "gates": report["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
