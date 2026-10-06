"""V31 CPU target generation and exact-target oracle.

This module is deliberately CPU-only. It reads only the hash-pinned official
competition training annotation file, reconstructs the canonical physical-file
split, builds matched DT and expert-spine SDT targets, and tests the frozen
watershed decoder on fit and tune records. It never opens audit images, test
images, model checkpoints, exports, or submissions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi
from skimage.segmentation import watershed

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ANNOTATIONS = (
    ROOT
    / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)
ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
EXPECTED_ALL = {"files": 707, "records": 1154, "annotations": 8199}
EXPECTED_FIT = {"files": 566, "records": 937, "annotations": 6606}
EXPECTED_TUNE = {"files": 94, "records": 150, "annotations": 1134}
EXPECTED_AUDIT = {"files": 47, "records": 67, "annotations": 459}
SEED_THRESHOLDS = (0.65, 0.75, 0.85)
ENERGY_BINS = 10
SDT_ALPHA = 0.8
SPINE_PROJECTION_RADIUS = 3
MIN_AREA = 20
TARGET_PQ_GATE = 0.97
TARGET_RECALL_GATE = 0.99
SPINE_VALIDITY_GATE = 0.95
FORBIDDEN_PATH_TOKENS = (
    "magfilo-v1-0-segmentation-of-solar-filaments",
    "public_magfilo",
    "test/test_images",
    "test_images",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def assert_canonical_annotation_path(path: Path) -> Path:
    """Accept only the exact local competition-train JSON and its pinned bytes."""
    resolved = path.expanduser().resolve()
    lowered = str(resolved).lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"V31 data firewall rejected annotation path: {resolved}")
    if resolved != CANONICAL_ANNOTATIONS.resolve():
        raise RuntimeError(f"V31 accepts only canonical competition annotations: {resolved}")
    if not resolved.is_file() or sha256(resolved) != ANNOTATION_SHA256:
        raise RuntimeError("canonical competition annotation identity changed")
    return resolved


def population(names: list[str], by_file: dict[str, list[str]], annotations: dict[str, list[dict]]) -> dict:
    ids = [image_id for name in names for image_id in by_file[name]]
    return {
        "files": len(names),
        "records": len(ids),
        "annotations": sum(len(annotations[image_id]) for image_id in ids),
    }


def load_population(annotation_path: Path = CANONICAL_ANNOTATIONS) -> dict:
    path = assert_canonical_annotation_path(annotation_path)
    document = json.loads(path.read_text())
    images = {str(row["id"]): row for row in document["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for row in document["annotations"]:
        annotations[str(row["image_id"])].append(row)
        if not row.get("spine") or len(row["spine"]) < 4:
            raise RuntimeError(f"annotation {row.get('id')} has no usable expert spine")
    by_file: dict[str, list[str]] = defaultdict(list)
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    by_file = {name: sorted(ids) for name, ids in by_file.items()}
    actual = {
        "files": len(by_file), "records": len(images),
        "annotations": len(document["annotations"]),
    }
    if actual != EXPECTED_ALL:
        raise RuntimeError(f"complete population changed: {actual}")
    files = sorted(by_file)
    random.Random(42).shuffle(files)
    held = files[:141]
    tune = sorted(held[:94])
    audit = sorted(held[94:])
    fit = sorted(files[141:])
    populations = {
        "all": actual,
        "fit": population(fit, by_file, annotations),
        "tune": population(tune, by_file, annotations),
        "audit": population(audit, by_file, annotations),
    }
    expected = {
        "all": EXPECTED_ALL, "fit": EXPECTED_FIT,
        "tune": EXPECTED_TUNE, "audit": EXPECTED_AUDIT,
    }
    if populations != expected:
        raise RuntimeError(f"canonical split populations changed: {populations}")
    if set(fit) & set(tune) or set(fit) & set(audit) or set(tune) & set(audit):
        raise RuntimeError("physical-file split overlap")
    return {
        "path": path, "images": images, "annotations": dict(annotations),
        "by_file": by_file, "fit": fit, "tune": tune, "audit": audit,
        "populations": populations,
    }


def annotation_rle(info: dict, annotation: dict) -> dict:
    return mask_utils.merge(mask_utils.frPyObjects(
        annotation["segmentation"], int(info["height"]), int(info["width"]),
    ))


def annotation_mask(info: dict, annotation: dict) -> np.ndarray:
    return mask_utils.decode(annotation_rle(info, annotation)).astype(bool)


def densify_polyline(spine: list[float]) -> np.ndarray:
    points = np.asarray(spine, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("spine must contain at least two finite points")
    dense: list[np.ndarray] = []
    for first, second in zip(points[:-1], points[1:]):
        steps = max(1, int(np.ceil(np.max(np.abs(second - first)))))
        dense.extend(first + (second - first) * (index / steps) for index in range(steps))
    dense.append(points[-1])
    return np.rint(np.asarray(dense)).astype(np.int32)


def project_spine(mask: np.ndarray, spine: list[float], radius: int = SPINE_PROJECTION_RADIUS) -> tuple[np.ndarray, dict]:
    """Project dense polyline samples to nearest foreground with stable ties."""
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or not value.any():
        raise ValueError("spine projection requires a nonempty 2D mask")
    dense = densify_polyline(spine)
    projected: list[tuple[int, int]] = []
    distances: list[float] = []
    height, width = value.shape
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
                distance2 = dx * dx + dy * dy
                candidate = (distance2, yy, xx)
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            distance2, yy, xx = best
            point = (yy, xx)
            if not projected or point != projected[-1]:
                projected.append(point)
                distances.append(float(np.sqrt(distance2)))
    spine_mask = np.zeros(value.shape, dtype=np.uint8, order="C")
    for yy, xx in projected:
        spine_mask[yy, xx] = 1
    for first, second in zip(projected[:-1], projected[1:]):
        cv2.line(
            spine_mask,
            (int(first[1]), int(first[0])),
            (int(second[1]), int(second[0])),
            color=1,
            thickness=1,
            lineType=cv2.LINE_8,
        )
    spine_mask = spine_mask.astype(bool) & value
    valid = len(projected) >= 2 and int(spine_mask.sum()) >= 2
    return spine_mask, {
        "dense_points": len(dense), "projected_points": len(projected),
        "valid": valid, "max_projection_distance": max(distances, default=0.0),
        "mean_projection_distance": float(np.mean(distances)) if distances else 0.0,
    }


def crop_bounds(mask: np.ndarray, margin: int = SPINE_PROJECTION_RADIUS + 2) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        raise ValueError("cannot crop an empty mask")
    return (
        max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
        min(mask.shape[1], int(xs.max()) + margin + 1),
        min(mask.shape[0], int(ys.max()) + margin + 1),
    )


def instance_energy(mask: np.ndarray, mode: str, spine_mask: np.ndarray | None = None) -> np.ndarray:
    """Construct standard DT or expert-spine SDT on the active instance crop."""
    value = np.asarray(mask, dtype=bool)
    distance_boundary = ndi.distance_transform_edt(value)
    if mode == "dt":
        maximum = float(distance_boundary.max())
        return (distance_boundary / maximum if maximum else distance_boundary).astype(np.float32)
    if mode != "sdt":
        raise ValueError(f"unknown target mode: {mode}")
    if spine_mask is None or not np.asarray(spine_mask, dtype=bool).any():
        raise ValueError("SDT requires a projected expert spine")
    distance_spine = ndi.distance_transform_edt(~np.asarray(spine_mask, dtype=bool))
    energy = (
        distance_boundary / (distance_spine + distance_boundary + np.float64(1e-6))
    ) ** SDT_ALPHA
    energy[~value] = 0
    return energy.astype(np.float32)


def quantize_energy(energy: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    classes = np.zeros(np.asarray(energy).shape, dtype=np.uint8)
    values = np.clip(np.floor(np.asarray(energy)[foreground] * ENERGY_BINS).astype(int) + 1, 1, ENERGY_BINS)
    classes[foreground] = values
    return classes


def dequantize_energy(classes: np.ndarray) -> np.ndarray:
    values = np.asarray(classes, dtype=np.uint8)
    energy = np.zeros(values.shape, dtype=np.float32)
    foreground = values > 0
    energy[foreground] = (values[foreground].astype(np.float32) - 0.5) / ENERGY_BINS
    return energy


def build_record_target(info: dict, entries: list[dict], mode: str) -> dict:
    shape = (int(info["height"]), int(info["width"]))
    foreground = np.zeros(shape, dtype=bool)
    energy = np.zeros(shape, dtype=np.float32)
    occupancy = np.zeros(shape, dtype=np.uint8)
    spine_records = []
    targets = []
    for annotation in entries:
        mask = annotation_mask(info, annotation)
        targets.append(annotation_rle(info, annotation))
        occupancy += mask.astype(np.uint8)
        x0, y0, x1, y1 = crop_bounds(mask)
        local_mask = mask[y0:y1, x0:x1]
        projected, projection = project_spine(mask, annotation["spine"])
        projection["annotation_id"] = str(annotation["id"])
        spine_records.append(projection)
        local_spine = projected[y0:y1, x0:x1]
        local_energy = instance_energy(local_mask, mode, local_spine if mode == "sdt" else None)
        destination = energy[y0:y1, x0:x1]
        destination[local_mask] = np.maximum(destination[local_mask], local_energy[local_mask])
        foreground |= mask
    classes = quantize_energy(energy, foreground)
    return {
        "classes": classes,
        "energy": dequantize_energy(classes),
        "foreground": foreground,
        "ignore": occupancy > 1,
        "targets": targets,
        "spines": spine_records,
    }


def decode_target(classes: np.ndarray, seed_threshold: float, min_area: int = MIN_AREA) -> list[dict]:
    values = np.asarray(classes, dtype=np.uint8)
    foreground = values > 0
    if not foreground.any():
        return []
    ys, xs = np.nonzero(foreground)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    local_values = values[y0:y1, x0:x1]
    local_foreground = foreground[y0:y1, x0:x1]
    local_energy = dequantize_energy(local_values)
    markers, count = ndi.label((local_energy >= seed_threshold) & local_foreground)
    if count == 0:
        return []
    labels = watershed(-local_energy, markers=markers, mask=local_foreground)
    predictions = []
    for label in range(1, count + 1):
        local_mask = labels == label
        if int(local_mask.sum()) < min_area:
            continue
        full_mask = np.zeros(values.shape, dtype=np.uint8, order="F")
        full_mask[y0:y1, x0:x1] = local_mask
        predictions.append(mask_utils.encode(full_mask))
    return predictions


def pq_counts(predictions: list[dict], targets: list[dict]) -> np.ndarray:
    if not predictions or not targets:
        return np.asarray([0.0, 0, len(predictions), len(targets)], dtype=np.float64)
    iou = np.asarray(mask_utils.iou(predictions, targets, [0] * len(targets)), dtype=np.float64)
    hits = iou > 0.5
    return np.asarray([
        float(iou[hits].sum()), int(hits.sum()),
        int((~hits.any(axis=1)).sum()), int((~hits.any(axis=0)).sum()),
    ], dtype=np.float64)


def metric_record(counts: np.ndarray) -> dict:
    iou_sum, tp, fp, fn = map(float, counts)
    denominator = tp + 0.5 * (fp + fn)
    return {
        "pq": iou_sum / denominator if denominator else 0.0,
        "sq": iou_sum / tp if tp else 0.0,
        "rq": tp / denominator if denominator else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "iou_sum": iou_sum, "tp": int(tp), "fp": int(fp), "fn": int(fn),
    }


def evaluate_record(task: tuple[str, str, str, dict, list[dict]]) -> dict:
    group, name, image_id, info, entries = task
    row = {"group": group, "file_name": name, "image_id": image_id, "annotations": len(entries)}
    counts_by_mode = {}
    spine_total = spine_valid = overlap = 0
    for mode in ("dt", "sdt"):
        target = build_record_target(info, entries, mode)
        if mode == "sdt":
            spine_total = len(target["spines"])
            spine_valid = sum(item["valid"] for item in target["spines"])
            overlap = int(target["ignore"].any())
        counts_by_mode[mode] = {}
        for threshold in SEED_THRESHOLDS:
            counts = pq_counts(decode_target(target["classes"], threshold), target["targets"])
            counts_by_mode[mode][threshold] = counts.tolist()
            row[f"{mode}_{threshold:.2f}_pq"] = metric_record(counts)["pq"]
    return {
        "row": row, "counts": counts_by_mode,
        "spine_total": spine_total, "spine_valid": spine_valid, "overlap": overlap,
    }


def run_oracle(
    annotation_path: Path,
    output: Path,
    groups: tuple[str, ...] = ("fit", "tune"),
    limit: int | None = None,
    workers: int = 1,
) -> dict:
    runtime = load_population(annotation_path)
    totals = {
        group: {mode: {threshold: np.zeros(4, dtype=np.float64) for threshold in SEED_THRESHOLDS}
                for mode in ("dt", "sdt")}
        for group in groups
    }
    tasks = []
    for group in groups:
        names = runtime[group][:limit] if limit else runtime[group]
        for name in names:
            for image_id in runtime["by_file"][name]:
                tasks.append((group, name, image_id, runtime["images"][image_id], runtime["annotations"][image_id]))
    projection_total = projection_valid = overlap_records = records = 0
    diagnostics = []
    if workers == 1:
        results = map(evaluate_record, tasks)
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(processes=workers)
        results = pool.imap(evaluate_record, tasks, chunksize=1)
    try:
        for result in results:
            records += 1
            projection_total += result["spine_total"]
            projection_valid += result["spine_valid"]
            overlap_records += result["overlap"]
            diagnostics.append(result["row"])
            group = result["row"]["group"]
            for mode in ("dt", "sdt"):
                for threshold in SEED_THRESHOLDS:
                    totals[group][mode][threshold] += np.asarray(
                        result["counts"][mode][threshold], dtype=np.float64
                    )
            if records % 100 == 0:
                print(f"v31 target oracle processed {records}/{len(tasks)} records", flush=True)
    finally:
        if workers != 1:
            pool.close()
            pool.join()
    diagnostics.sort(key=lambda row: (row["group"], row["file_name"], row["image_id"]))
    metrics = {
        group: {
            mode: {f"{threshold:.2f}": metric_record(counts) for threshold, counts in values.items()}
            for mode, values in modes.items()
        }
        for group, modes in totals.items()
    }
    aggregate = {}
    for mode in ("dt", "sdt"):
        aggregate[mode] = {}
        for threshold in SEED_THRESHOLDS:
            counts = sum((totals[group][mode][threshold] for group in groups), np.zeros(4, dtype=np.float64))
            aggregate[mode][f"{threshold:.2f}"] = metric_record(counts)
        selected = max(aggregate[mode], key=lambda key: (aggregate[mode][key]["pq"], -float(key)))
        aggregate[mode]["selected"] = selected
        aggregate[mode]["selected_metrics"] = aggregate[mode][selected]
    spine_validity = projection_valid / projection_total if projection_total else 0.0
    gates = {
        "dt_pq": aggregate["dt"]["selected_metrics"]["pq"] >= TARGET_PQ_GATE,
        "dt_recall": aggregate["dt"]["selected_metrics"]["recall"] >= TARGET_RECALL_GATE,
        "sdt_pq": aggregate["sdt"]["selected_metrics"]["pq"] >= TARGET_PQ_GATE,
        "sdt_recall": aggregate["sdt"]["selected_metrics"]["recall"] >= TARGET_RECALL_GATE,
        "spine_validity": spine_validity >= SPINE_VALIDITY_GATE,
    }
    report = {
        "status": "complete", "annotation_sha256": sha256(runtime["path"]),
        "groups": list(groups), "limit_per_group": limit, "workers": workers,
        "populations": {group: runtime["populations"][group] for group in groups},
        "records_evaluated": records, "overlap_records": overlap_records,
        "spine_projection": {
            "annotations": projection_total, "valid_annotations": projection_valid,
            "validity": spine_validity,
        },
        "metrics": metrics, "aggregate": aggregate, "gates": gates,
        "gpu_authorized": all(gates.values()),
        "action": "proceed to V31 numerical overfit gate" if all(gates.values())
                  else "stop V31 before GPU work; target/decoder oracle failed",
        "reads_audit": False, "reads_test": False, "exports": False, "submits": False,
        "diagnostics": diagnostics,
    }
    write_json(output, report)
    print(json.dumps({
        "records_evaluated": records, "spine_projection": report["spine_projection"],
        "aggregate": aggregate, "gates": gates,
        "gpu_authorized": report["gpu_authorized"], "action": report["action"],
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=CANONICAL_ANNOTATIONS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=("fit", "tune"), default=("fit", "tune"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    run_oracle(args.annotations, args.output, tuple(args.groups), args.limit, args.workers)


if __name__ == "__main__":
    main()
