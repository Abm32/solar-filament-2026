"""V34 CPU oracle for ordered expert spines plus bilateral width profiles.

Reads only the hash-pinned official competition training annotations and evaluates
fit+tune geometry. It never reads image pixels, audit, test, checkpoints, or CSVs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

try:
    from scripts import v31_target_oracle as canonical
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import v31_target_oracle as canonical

RAY_STEP = 0.25
RAY_CAP = 128.0
TANGENT_RADIUS = 3
MIN_AREA = 20
VALIDITY_GATE = 0.995
COMBINED_PQ_GATE = 0.95
SPLIT_PQ_GATE = 0.94
RECALL_GATE = 0.99
ZERO_RECORD_GATE = 0.01
MEDIAN_IOU_GATE = 0.90
BILATERAL_DELTA_GATE = 0.005
BILATERAL_NEAR_PERFECT = 0.975
CAPPED_RAY_GATE = 0.0001


def ordered_projected_points(mask: np.ndarray, spine: list[float]) -> tuple[np.ndarray, dict]:
    """Project ordered dense spine samples with the frozen deterministic tie rule."""
    value = np.asarray(mask, dtype=bool)
    dense = canonical.densify_polyline(spine)
    points: list[tuple[int, int]] = []
    distances: list[float] = []
    height, width = value.shape
    radius = canonical.SPINE_PROJECTION_RADIUS
    for x, y in dense:
        best = None
        for dy in range(-radius, radius + 1):
            yy = int(y + dy)
            if yy < 0 or yy >= height:
                continue
            for dx in range(-radius, radius + 1):
                xx = int(x + dx)
                if xx < 0 or xx >= width or not value[yy, xx]:
                    continue
                candidate = (dx * dx + dy * dy, yy, xx)
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            distance2, yy, xx = best
            point = (yy, xx)
            if not points or point != points[-1]:
                points.append(point)
                distances.append(float(np.sqrt(distance2)))
    output = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    return output, {
        "dense_points": len(dense), "projected_points": len(points),
        "valid": len(points) >= 2,
        "max_projection_distance": max(distances, default=0.0),
    }


def tangent_at(points_yx: np.ndarray, index: int) -> np.ndarray | None:
    """Return frozen local tangent in xy coordinate order."""
    count = len(points_yx)
    for radius in range(TANGENT_RADIUS, count):
        low, high = max(0, index - radius), min(count - 1, index + radius)
        delta_yx = points_yx[high].astype(np.float64) - points_yx[low].astype(np.float64)
        delta_xy = delta_yx[::-1]
        norm = float(np.linalg.norm(delta_xy))
        if norm > 0:
            return delta_xy / norm
        if low == 0 and high == count - 1:
            break
    return None


def ray_width(mask: np.ndarray, center_yx: np.ndarray, normal_xy: np.ndarray) -> tuple[float, bool]:
    """Measure one side by frozen quarter-pixel nearest-neighbour ray casting."""
    center_y, center_x = map(float, center_yx)
    height, width = mask.shape
    last = 0.0
    steps = int(round(RAY_CAP / RAY_STEP))
    for step in range(1, steps + 1):
        distance = step * RAY_STEP
        xx = int(np.rint(center_x + normal_xy[0] * distance))
        yy = int(np.rint(center_y + normal_xy[1] * distance))
        if yy < 0 or yy >= height or xx < 0 or xx >= width or not mask[yy, xx]:
            return last, False
        last = distance
    return last, True


def measure_profile(mask: np.ndarray, points_yx: np.ndarray) -> dict:
    left, right, normals = [], [], []
    capped = 0
    for index, point in enumerate(points_yx):
        tangent = tangent_at(points_yx, index)
        if tangent is None:
            return {"valid": False, "left": [], "right": [], "normals": [],
                    "capped_rays": capped, "rays": 2 * len(left)}
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        left_width, left_capped = ray_width(mask, point, normal)
        right_width, right_capped = ray_width(mask, point, -normal)
        left.append(left_width); right.append(right_width); normals.append(normal)
        capped += int(left_capped) + int(right_capped)
    return {
        "valid": len(points_yx) >= 2, "left": np.asarray(left, dtype=np.float64),
        "right": np.asarray(right, dtype=np.float64),
        "normals": np.asarray(normals, dtype=np.float64),
        "capped_rays": capped, "rays": 2 * len(points_yx),
    }


def reconstruct_ribbon(
    shape: tuple[int, int], points_yx: np.ndarray, normals_xy: np.ndarray,
    left_widths: np.ndarray, right_widths: np.ndarray, symmetric: bool = False,
) -> np.ndarray:
    """Rasterize the frozen sequence of cross-sections as one geometric ribbon."""
    if len(points_yx) < 2:
        return np.zeros(shape, dtype=bool)
    points_xy = points_yx[:, ::-1].astype(np.float64)
    if symmetric:
        mean = (left_widths + right_widths) / 2.0
        left_widths = right_widths = mean
    left = np.rint(points_xy + normals_xy * left_widths[:, None]).astype(np.int32)
    right = np.rint(points_xy - normals_xy * right_widths[:, None]).astype(np.int32)
    all_x = np.concatenate([left[:, 0], right[:, 0], points_xy[:, 0]])
    all_y = np.concatenate([left[:, 1], right[:, 1], points_xy[:, 1]])
    margin = 2
    x0 = max(0, int(np.floor(all_x.min())) - margin)
    y0 = max(0, int(np.floor(all_y.min())) - margin)
    x1 = min(shape[1], int(np.ceil(all_x.max())) + margin + 1)
    y1 = min(shape[0], int(np.ceil(all_y.max())) + margin + 1)
    local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    offset = np.asarray([x0, y0], dtype=np.int32)
    local_left, local_right = left - offset, right - offset
    for index in range(len(points_yx) - 1):
        quad = np.asarray([
            local_left[index], local_left[index + 1],
            local_right[index + 1], local_right[index],
        ], dtype=np.int32)
        cv2.fillPoly(local, [quad], color=1, lineType=cv2.LINE_8)
    for first, second in zip(local_left, local_right):
        cv2.line(local, tuple(first), tuple(second), color=1, thickness=1, lineType=cv2.LINE_8)
    full = np.zeros(shape, dtype=bool)
    full[y0:y1, x0:x1] = local.astype(bool)
    return full


def instance_reconstructions(info: dict, annotation: dict) -> dict:
    source = canonical.annotation_mask(info, annotation)
    points, projection = ordered_projected_points(source, annotation["spine"])
    if not projection["valid"]:
        return {"valid": False, "projection": projection, "rays": 0, "capped_rays": 0}
    profile = measure_profile(source, points)
    if not profile["valid"]:
        return {"valid": False, "projection": projection,
                "rays": profile["rays"], "capped_rays": profile["capped_rays"]}
    bilateral = reconstruct_ribbon(
        source.shape, points, profile["normals"], profile["left"], profile["right"], False
    )
    symmetric = reconstruct_ribbon(
        source.shape, points, profile["normals"], profile["left"], profile["right"], True
    )
    return {
        "valid": True, "projection": projection, "rays": profile["rays"],
        "capped_rays": profile["capped_rays"], "bilateral": bilateral,
        "symmetric": symmetric, "source": source,
    }


def rle(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))


def counts_and_ious(predictions: list[dict], targets: list[dict]) -> tuple[np.ndarray, list[float]]:
    """Canonical strict-IoU counts with explicit one-to-one maximum matching.

    Most project masks are disjoint, where every IoU>0.5 edge is unique by
    construction. V34 intentionally preserves rare source/prediction overlaps,
    so use assignment rather than the historical disjoint-mask shortcut.
    """
    if not predictions or not targets:
        return np.asarray([0.0, 0, len(predictions), len(targets)], dtype=np.float64), []
    from scipy.optimize import linear_sum_assignment

    iou = np.asarray(mask_utils.iou(predictions, targets, [0] * len(targets)), dtype=np.float64)
    pred_indices, target_indices = linear_sum_assignment(-iou)
    matched = [
        float(iou[pred_index, target_index])
        for pred_index, target_index in zip(pred_indices, target_indices)
        if iou[pred_index, target_index] > 0.5
    ]
    tp = len(matched)
    counts = np.asarray([
        float(sum(matched)), tp, len(predictions) - tp, len(targets) - tp,
    ], dtype=np.float64)
    return counts, matched


def evaluate_record(task: tuple[str, str, str, dict, list[dict]]) -> dict:
    group, name, image_id, info, entries = task
    targets, predictions = [], {"bilateral": [], "symmetric": []}
    valid = rays = capped_rays = 0
    instance_ious = {"bilateral": [], "symmetric": []}
    for annotation in entries:
        targets.append(canonical.annotation_rle(info, annotation))
        result = instance_reconstructions(info, annotation)
        valid += int(result["valid"]); rays += result["rays"]; capped_rays += result["capped_rays"]
        if result["valid"]:
            for mode in predictions:
                mask = result[mode]
                if int(mask.sum()) >= MIN_AREA:
                    predictions[mode].append(rle(mask))
                source = result["source"]
                union = int(np.logical_or(mask, source).sum())
                instance_ious[mode].append(int(np.logical_and(mask, source).sum()) / union if union else 0.0)
    counts, matched = {}, {}
    for mode in predictions:
        counts[mode], matched[mode] = counts_and_ious(predictions[mode], targets)
    overlap = 0
    if len(entries) > 1:
        occupancy = sum(
            (canonical.annotation_mask(info, item).astype(np.uint8) for item in entries),
            np.zeros((int(info["height"]), int(info["width"])), dtype=np.uint8),
        )
        overlap = int((occupancy > 1).any())
    return {
        "group": group, "file_name": name, "image_id": image_id,
        "annotations": len(entries), "valid": valid, "rays": rays,
        "capped_rays": capped_rays, "overlap": overlap,
        "counts": {mode: value.tolist() for mode, value in counts.items()},
        "matched_ious": matched, "instance_ious": instance_ious,
    }


def metric(counts: np.ndarray) -> dict:
    return canonical.metric_record(counts)


def synthetic_checks() -> dict:
    shape = (64, 96)
    source = np.zeros(shape, dtype=bool)
    source[20:32, 10:86] = True
    points = np.asarray([[26, x] for x in range(10, 86)], dtype=np.int32)
    profile = measure_profile(source, points)
    bilateral = reconstruct_ribbon(shape, points, profile["normals"], profile["left"], profile["right"])
    reversed_points = points[::-1].copy()
    reversed_profile = measure_profile(source, reversed_points)
    reversed_mask = reconstruct_ribbon(
        shape, reversed_points, reversed_profile["normals"],
        reversed_profile["left"], reversed_profile["right"],
    )
    symmetric_source = np.zeros(shape, dtype=bool); symmetric_source[20:33, 10:86] = True
    symmetric_points = np.asarray([[26, x] for x in range(10, 86)], dtype=np.int32)
    symmetric_profile = measure_profile(symmetric_source, symmetric_points)
    symmetric_bilateral = reconstruct_ribbon(
        shape, symmetric_points, symmetric_profile["normals"],
        symmetric_profile["left"], symmetric_profile["right"], False,
    )
    symmetric_control = reconstruct_ribbon(
        shape, symmetric_points, symmetric_profile["normals"],
        symmetric_profile["left"], symmetric_profile["right"], True,
    )
    repeated = reconstruct_ribbon(shape, points, profile["normals"], profile["left"], profile["right"])
    checks = {
        "asymmetric_exact": bool(np.array_equal(bilateral, source)),
        "reverse_invariant": bool(np.array_equal(bilateral, reversed_mask)),
        "symmetric_parity": bool(np.array_equal(symmetric_bilateral, symmetric_control)),
        "deterministic": bool(np.array_equal(bilateral, repeated)),
    }
    checks["passed"] = all(checks.values())
    return checks


def run_oracle(annotation_path: Path, output: Path, limit: int | None = None, workers: int = 1) -> dict:
    runtime = canonical.load_population(annotation_path)
    synthetic = synthetic_checks()
    if not synthetic["passed"]:
        raise RuntimeError(f"V34 synthetic parity failed: {synthetic}")
    groups = ("fit", "tune")
    tasks = []
    for group in groups:
        names = runtime[group][:limit] if limit else runtime[group]
        for name in names:
            for image_id in runtime["by_file"][name]:
                tasks.append((group, name, image_id, runtime["images"][image_id],
                              runtime["annotations"][image_id]))
    totals = {group: {mode: np.zeros(4, dtype=np.float64) for mode in ("bilateral", "symmetric")}
              for group in groups}
    all_counts = {mode: np.zeros(4, dtype=np.float64) for mode in ("bilateral", "symmetric")}
    matched_ious = {mode: [] for mode in ("bilateral", "symmetric")}
    source_ious = {mode: [] for mode in ("bilateral", "symmetric")}
    records = annotations = valid = rays = capped_rays = overlap_records = zero_records = 0
    diagnostics = []
    if workers == 1:
        results = map(evaluate_record, tasks); pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers)
        results = pool.imap(evaluate_record, tasks, chunksize=1)
    try:
        for result in results:
            records += 1; annotations += result["annotations"]; valid += result["valid"]
            rays += result["rays"]; capped_rays += result["capped_rays"]
            overlap_records += result["overlap"]
            row = {key: result[key] for key in (
                "group", "file_name", "image_id", "annotations", "valid",
                "rays", "capped_rays", "overlap",
            )}
            for mode in ("bilateral", "symmetric"):
                value = np.asarray(result["counts"][mode], dtype=np.float64)
                totals[result["group"]][mode] += value; all_counts[mode] += value
                matched_ious[mode].extend(result["matched_ious"][mode])
                source_ious[mode].extend(result["instance_ious"][mode])
                row[f"{mode}_pq"] = metric(value)["pq"]
            zero_records += int(row["bilateral_pq"] == 0.0)
            diagnostics.append(row)
            if records % 100 == 0:
                print(f"v34 oracle processed {records}/{len(tasks)} records", flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    split_metrics = {group: {mode: metric(totals[group][mode]) for mode in totals[group]}
                     for group in groups}
    aggregate = {mode: metric(all_counts[mode]) for mode in all_counts}
    for mode in aggregate:
        aggregate[mode]["median_matched_iou"] = float(np.median(matched_ious[mode])) if matched_ious[mode] else 0.0
        aggregate[mode]["median_source_instance_iou"] = float(np.median(source_ious[mode])) if source_ious[mode] else 0.0
    validity = valid / annotations if annotations else 0.0
    capped_fraction = capped_rays / rays if rays else 0.0
    zero_fraction = zero_records / records if records else 1.0
    delta = aggregate["bilateral"]["pq"] - aggregate["symmetric"]["pq"]
    gates = {
        "synthetic": synthetic["passed"], "validity": validity >= VALIDITY_GATE,
        "combined_pq": aggregate["bilateral"]["pq"] >= COMBINED_PQ_GATE,
        "fit_pq": split_metrics["fit"]["bilateral"]["pq"] >= SPLIT_PQ_GATE,
        "tune_pq": split_metrics["tune"]["bilateral"]["pq"] >= SPLIT_PQ_GATE,
        "recall": aggregate["bilateral"]["recall"] >= RECALL_GATE,
        "zero_records": zero_fraction <= ZERO_RECORD_GATE,
        "median_matched_iou": aggregate["bilateral"]["median_matched_iou"] >= MEDIAN_IOU_GATE,
        "bilateral_value": delta >= BILATERAL_DELTA_GATE or aggregate["bilateral"]["pq"] >= BILATERAL_NEAR_PERFECT,
        "capped_rays": capped_fraction <= CAPPED_RAY_GATE,
    }
    passed = all(gates.values()) and limit is None
    diagnostics.sort(key=lambda row: (row["group"], row["file_name"], row["image_id"]))
    report = {
        "status": "complete", "experiment": "v34_bilateral_width_profile_oracle",
        "annotation_sha256": canonical.sha256(runtime["path"]), "limit_per_group": limit,
        "workers": workers, "populations": {g: runtime["populations"][g] for g in groups},
        "records_evaluated": records, "annotations_evaluated": annotations,
        "valid_instances": valid, "validity": validity, "rays": rays,
        "capped_rays": capped_rays, "capped_ray_fraction": capped_fraction,
        "overlap_records": overlap_records, "zero_pq_records": zero_records,
        "zero_pq_record_fraction": zero_fraction, "synthetic": synthetic,
        "split_metrics": split_metrics, "aggregate": aggregate,
        "bilateral_pq_delta": delta, "gates": gates, "g1_passed": passed,
        "gpu_authorized": passed,
        "action": "freeze V34 G2 design" if passed else "close V34 before GPU work",
        "reads_images": False, "reads_audit": False, "reads_test": False,
        "exports": False, "submits": False, "diagnostics": diagnostics,
    }
    canonical.write_json(output, report)
    print(json.dumps({key: report[key] for key in (
        "records_evaluated", "annotations_evaluated", "validity", "capped_ray_fraction",
        "zero_pq_record_fraction", "split_metrics", "aggregate", "bilateral_pq_delta",
        "gates", "g1_passed", "action",
    )}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=canonical.CANONICAL_ANNOTATIONS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    run_oracle(args.annotations, args.output, args.limit, args.workers)


if __name__ == "__main__":
    main()
