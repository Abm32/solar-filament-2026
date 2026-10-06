"""Worker for the V32 pinned Omnipose v1.0.6 exact-field oracle.

Run only through ``run_v32_omnipose_oracle.py`` with its disposable interpreter.
This file imports untouched upstream Omnipose by reference. It never reads images,
audit records, test data, model checkpoints, or submission artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
import types
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
EXPECTED_FIT = {"files": 566, "records": 937, "annotations": 6606}
EXPECTED_TUNE = {"files": 94, "records": 150, "annotations": 1134}
CROP_MARGIN = 16
MIN_AREA = 20
DIAM_THRESHOLD = 12.0
EPS = 2 ** 0.5
PQ_GATE = 0.90
SPLIT_PQ_GATE = 0.89
RECALL_GATE = 0.99
FP_RATIO_GATE = 0.25
ZERO_RECORD_RATE_GATE = 0.01
UPSTREAM_CORE = None


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


def load_upstream(upstream: Path):
    """Load untouched v1.0.6 core while bypassing unrelated eager imports."""
    global UPSTREAM_CORE
    if UPSTREAM_CORE is not None:
        return UPSTREAM_CORE
    source = upstream.resolve() / "omnipose"
    package = types.ModuleType("omnipose")
    package.__path__ = [str(source)]
    package.__package__ = "omnipose"
    sys.modules["omnipose"] = package
    core = importlib.import_module("omnipose.core")
    core.torch.set_num_threads(1)
    try:
        core.torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    UPSTREAM_CORE = core
    return core


def initialize_worker(upstream: str) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    load_upstream(Path(upstream))


def annotation_rle(info: dict, annotation: dict) -> dict:
    return mask_utils.merge(mask_utils.frPyObjects(
        annotation["segmentation"], int(info["height"]), int(info["width"]),
    ))


def annotation_mask(info: dict, annotation: dict) -> np.ndarray:
    return mask_utils.decode(annotation_rle(info, annotation)).astype(bool)


def metric_counts(predictions: list[dict], targets: list[dict]) -> np.ndarray:
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


def population(names: list[str], by_file: dict[str, list[str]], annotations: dict[str, list[dict]]) -> dict:
    ids = [image_id for name in names for image_id in by_file[name]]
    return {"files": len(names), "records": len(ids),
            "annotations": sum(len(annotations[i]) for i in ids)}


def load_population(path: Path) -> dict:
    if sha256(path) != ANNOTATION_SHA256:
        raise RuntimeError("V32 annotation identity changed")
    document = json.loads(path.read_text())
    images = {str(row["id"]): row for row in document["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for row in document["annotations"]:
        annotations[str(row["image_id"])].append(row)
    by_file: dict[str, list[str]] = defaultdict(list)
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    by_file = {name: sorted(ids) for name, ids in by_file.items()}
    files = sorted(by_file); random.Random(42).shuffle(files)
    fit = sorted(files[141:]); tune = sorted(files[:94])
    actual_fit = population(fit, by_file, annotations)
    actual_tune = population(tune, by_file, annotations)
    if actual_fit != EXPECTED_FIT or actual_tune != EXPECTED_TUNE:
        raise RuntimeError(f"V32 populations changed: {actual_fit}/{actual_tune}")
    return {"images": images, "annotations": dict(annotations), "by_file": by_file,
            "fit": fit, "tune": tune,
            "populations": {"fit": actual_fit, "tune": actual_tune}}


def crop_record(info: dict, entries: list[dict]) -> dict:
    masks = [annotation_mask(info, annotation) for annotation in entries]
    foreground = np.logical_or.reduce(masks)
    ys, xs = np.nonzero(foreground)
    y0 = max(0, int(ys.min()) - CROP_MARGIN)
    y1 = min(foreground.shape[0], int(ys.max()) + CROP_MARGIN + 1)
    x0 = max(0, int(xs.min()) - CROP_MARGIN)
    x1 = min(foreground.shape[1], int(xs.max()) + CROP_MARGIN + 1)
    local_masks = [mask[y0:y1, x0:x1] for mask in masks]
    occupancy = sum((mask.astype(np.uint8) for mask in local_masks),
                    np.zeros(local_masks[0].shape, dtype=np.uint8))
    labels = np.zeros(local_masks[0].shape, dtype=np.int32)
    unique = occupancy == 1
    for label, mask in enumerate(local_masks, 1):
        labels[mask & unique] = label
    return {"labels": labels, "bounds": (x0, y0, x1, y1),
            "overlap_pixels": int((occupancy > 1).sum()),
            "targets": [annotation_rle(info, annotation) for annotation in entries]}


def labels_to_predictions(labels: np.ndarray, info: dict, bounds: tuple[int, int, int, int]) -> list[dict]:
    x0, y0, x1, y1 = bounds
    predictions = []
    for label in range(1, int(labels.max()) + 1):
        local = labels == label
        if int(local.sum()) < MIN_AREA:
            continue
        full = np.zeros((int(info["height"]), int(info["width"])), dtype=np.uint8, order="F")
        full[y0:y1, x0:x1] = local
        predictions.append(mask_utils.encode(full))
    return predictions


def reconstruct(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    core = UPSTREAM_CORE
    target = core.masks_to_flows(
        labels.copy(), use_gpu=False, device=core.torch_CPU, omni=True, dim=2,
    )
    distance = target[3]
    flow = target[4]
    if hasattr(distance, "cpu"):
        distance = distance.cpu().numpy()
    if hasattr(flow, "cpu"):
        flow = flow.cpu().numpy()
    distance = np.asarray(distance, dtype=np.float32)
    flow = np.asarray(flow, dtype=np.float32)
    positive = distance[labels > 0]
    niter = max(1, int(2 * (labels.ndim + 1) * float(np.mean(positive))))
    effective_diameter = float(core.dist_to_diam(np.abs(positive), labels.ndim))
    output = core.compute_masks(
        flow, distance, bd=np.zeros_like(distance), iscell=labels > 0,
        niter=niter, flow_threshold=0, omni=True, use_gpu=False,
        device=core.torch_CPU, nclasses=3, dim=2, min_size=MIN_AREA,
        cluster=False, suppress=True, diam_threshold=DIAM_THRESHOLD, eps=EPS,
        affinity_seg=False, hdbscan=False,
    )
    reconstructed = np.asarray(output[0], dtype=np.int32)
    return reconstructed, {
        "niter": niter, "effective_diameter": effective_diameter,
        "automatic_dbscan": effective_diameter <= DIAM_THRESHOLD,
        "distance_min": float(distance.min()), "distance_max": float(distance.max()),
        "flow_finite": bool(np.isfinite(flow).all()),
    }


def evaluate_task(task: tuple[str, str, str, dict, list[dict]]) -> dict:
    group, name, image_id, info, entries = task
    cropped = crop_record(info, entries)
    started = time.monotonic()
    reconstructed, dynamics = reconstruct(cropped["labels"])
    predictions = labels_to_predictions(reconstructed, info, cropped["bounds"])
    counts = metric_counts(predictions, cropped["targets"])
    metric = metric_record(counts)
    return {
        "group": group, "file_name": name, "image_id": image_id,
        "annotations": len(entries), "crop_shape": list(cropped["labels"].shape),
        "overlap_pixels": cropped["overlap_pixels"],
        "predictions": len(predictions), "elapsed_seconds": time.monotonic() - started,
        **dynamics, **metric, "counts": counts.tolist(),
    }


def synthetic_parity() -> dict:
    labels = np.zeros((96, 96), dtype=np.int32)
    labels[10:22, 8:52] = 1
    labels[58:74, 35:86] = 2
    first, dynamics_first = reconstruct(labels)
    second, dynamics_second = reconstruct(labels)
    return {
        "labels": 2, "reconstructed_labels": int(first.max()),
        "foreground_equal": bool(np.array_equal(first > 0, labels > 0)),
        "repeat_equal": bool(np.array_equal(first, second)),
        "dynamics_equal": dynamics_first == dynamics_second,
        "dynamics": dynamics_first,
    }


def run_oracle(annotations: Path, upstream: Path, output: Path,
               workers: int, limit: int | None) -> dict:
    load_upstream(upstream)
    parity = synthetic_parity()
    if not all((parity["foreground_equal"], parity["repeat_equal"], parity["dynamics_equal"])):
        raise RuntimeError("V32 synthetic upstream parity failed")
    runtime = load_population(annotations)
    tasks = []
    for group in ("fit", "tune"):
        names = runtime[group][:limit] if limit else runtime[group]
        for name in names:
            for image_id in runtime["by_file"][name]:
                tasks.append((group, name, image_id, runtime["images"][image_id],
                              runtime["annotations"][image_id]))
    determinism_first = evaluate_task(tasks[0])
    determinism_second = evaluate_task(tasks[0])
    determinism_first.pop("elapsed_seconds", None)
    determinism_second.pop("elapsed_seconds", None)
    real_record_determinism = determinism_first == determinism_second
    if not real_record_determinism:
        raise RuntimeError("V32 real-record exact-field reconstruction is nondeterministic")
    if workers == 1:
        results = map(evaluate_task, tasks)
        pool = None
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(workers, initializer=initialize_worker, initargs=(str(upstream),))
        results = pool.imap(evaluate_task, tasks, chunksize=1)
    rows = []
    totals = {"fit": np.zeros(4, dtype=np.float64),
              "tune": np.zeros(4, dtype=np.float64)}
    try:
        for index, row in enumerate(results, 1):
            rows.append(row)
            totals[row["group"]] += np.asarray(row.pop("counts"), dtype=np.float64)
            if index % 50 == 0:
                print(f"v32 Omnipose oracle processed {index}/{len(tasks)} records", flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    rows.sort(key=lambda row: (row["group"], row["file_name"], row["image_id"]))
    metrics = {group: metric_record(counts) for group, counts in totals.items()}
    combined_counts = totals["fit"] + totals["tune"]
    combined = metric_record(combined_counts)
    zero_records = sum(row["pq"] == 0 for row in rows)
    overlap_rows = [row for row in rows if row["overlap_pixels"] > 0]
    gt = int(combined["tp"] + combined["fn"])
    gates = {
        "combined_pq": combined["pq"] >= PQ_GATE,
        "fit_pq": metrics["fit"]["pq"] >= SPLIT_PQ_GATE,
        "tune_pq": metrics["tune"]["pq"] >= SPLIT_PQ_GATE,
        "recall": combined["recall"] >= RECALL_GATE,
        "fp_ratio": combined["fp"] / gt <= FP_RATIO_GATE,
        "zero_record_rate": zero_records / len(rows) <= ZERO_RECORD_RATE_GATE,
        "synthetic_parity": all((parity["foreground_equal"], parity["repeat_equal"],
                                  parity["dynamics_equal"])),
        "real_record_determinism": real_record_determinism,
        "finite": all(np.isfinite(row["pq"]) and row["flow_finite"] for row in rows),
    }
    report = {
        "status": "complete", "semantic_reference": "Omnipose v1.0.6",
        "upstream_commit": "21e4d0199cc03662669c150f6151e3365dc201fd",
        "annotation_sha256": sha256(annotations), "crop_margin": CROP_MARGIN,
        "min_area": MIN_AREA, "workers": workers, "limit_per_group": limit,
        "populations": runtime["populations"], "records_evaluated": len(rows),
        "synthetic_parity": parity, "real_record_determinism": real_record_determinism,
        "metrics": metrics, "combined": combined,
        "zero_pq_records": zero_records,
        "zero_pq_record_rate": zero_records / len(rows),
        "overlap_records": len(overlap_rows),
        "overlap_metrics": metric_record(sum(
            (np.asarray([row["iou_sum"], row["tp"], row["fp"], row["fn"]])
             for row in overlap_rows), np.zeros(4, dtype=np.float64))),
        "automatic_dbscan_records": sum(row["automatic_dbscan"] for row in rows),
        "niter": {"min": min(row["niter"] for row in rows),
                  "mean": float(np.mean([row["niter"] for row in rows])),
                  "max": max(row["niter"] for row in rows)},
        "gates": gates, "gpu_authorized": all(gates.values()),
        "action": "proceed to V32 numerical overfit gate" if all(gates.values())
                  else "close V32 before GPU work; exact-field oracle failed",
        "reads_audit": False, "reads_test": False, "exports": False, "submits": False,
        "diagnostics": rows,
    }
    write_json(output, report)
    print(json.dumps({key: report[key] for key in (
        "records_evaluated", "synthetic_parity", "real_record_determinism",
        "metrics", "combined",
        "zero_pq_records", "overlap_records", "automatic_dbscan_records",
        "niter", "gates", "gpu_authorized", "action",
    )}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    initialize_worker(str(args.upstream))
    run_oracle(args.annotations.resolve(), args.upstream.resolve(), args.output,
               args.workers, args.limit)


if __name__ == "__main__":
    main()
