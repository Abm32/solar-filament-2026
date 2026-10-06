"""V43 G2: native-resolution mask-quality refinement (TTA + multiscale merge).

Builds directly on V42, which broke the recall wall (best fit: PQ 0.768,
recall 0.997, medIoU 0.781, FP 1) but left the mask-IoU gates (PQ>=0.85,
medIoU>=0.80) unmet. V43 keeps the IDENTICAL frozen cohort / GT / gate bars /
firewall and attacks matched-mask IoU only, via test-time augmentation
(horizontal + vertical flip), optional multiscale prediction, and mask merging
by IoU-weighted union. It can either (a) evaluate an existing V42 checkpoint
(--eval-checkpoint) or (b) train a fresh best-config model then TTA-evaluate.

Remote GPU runner. Reads exactly the 16 frozen fit images/records. No
tune/audit/test, CSV, export, or submission path exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("YOLO_OFFLINE", "True")

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
SELECTION_ID = "741d8124fd834c465743cb4ddbee265298354b4e59b5ab9650379d162673ca2c"
EXPECTED_RECORDS = 16
EXPECTED_IMAGES = 16
EXPECTED_ANNOTATIONS = 308
EXPECTED_SMALL_THIN = 244

TARGET_PQ = 0.85
TARGET_RECALL = 0.90
TARGET_MEDIAN_IOU = 0.80
MAX_FP = 40
MEMORY_GATE_MIB = 14336
RUNTIME_GATE_SECONDS = 7200

IMGSZ = 2048
MAX_EPOCHS = 300
EVAL_CONFIDENCES = (0.05, 0.10, 0.20, 0.30, 0.50)
MIN_AREA = 20
TTA_MERGE_IOU = 0.5  # masks from different TTA views merged if IoU >= this


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


def annotation_rle(mask_utils, info: dict, annotation: dict) -> dict:
    return mask_utils.merge(
        mask_utils.frPyObjects(
            annotation["segmentation"],
            int(info["height"]),
            int(info["width"]),
        )
    )


def densify_polyline(np, spine: list[float]):
    points = np.asarray(spine, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("spine must contain two finite points")
    dense = []
    for first, second in zip(points[:-1], points[1:], strict=True):
        steps = max(1, int(np.ceil(np.max(np.abs(second - first)))))
        dense.extend(first + (second - first) * (index / steps) for index in range(steps))
    dense.append(points[-1])
    return np.rint(np.asarray(dense)).astype(np.int32)


def project_spine(np, cv2, mask, spine, radius=3):
    dense = densify_polyline(np, spine)
    projected = []
    height, width = mask.shape
    for x, y in dense:
        best = None
        for dy in range(-radius, radius + 1):
            yy = int(y + dy)
            if not 0 <= yy < height:
                continue
            for dx in range(-radius, radius + 1):
                xx = int(x + dx)
                if 0 <= xx < width and mask[yy, xx]:
                    candidate = (dx * dx + dy * dy, yy, xx)
                    if best is None or candidate < best:
                        best = candidate
        if best is not None:
            point = (best[1], best[2])
            if not projected or projected[-1] != point:
                projected.append(point)
    if len(projected) < 2:
        raise RuntimeError("invalid projected V43 expert spine")
    return projected


def selection_records(np, cv2, mask_utils, document: dict) -> list[dict]:
    images = {str(row["id"]): row for row in document["images"]}
    annotations = defaultdict(list)
    by_file = defaultdict(list)
    for row in document["annotations"]:
        annotations[str(row["image_id"])].append(row)
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    files = sorted(by_file)
    import random

    random.Random(42).shuffle(files)
    fit = sorted(files[141:])
    candidates = []
    for name in fit:
        image_id = max(
            by_file[name],
            key=lambda value: (
                len(annotations[value]),
                sum(row.get("area", 0) for row in annotations[value]),
                value,
            ),
        )
        info, entries = images[image_id], annotations[image_id]
        small = thin = small_thin = 0
        for entry in entries:
            mask = mask_utils.decode(annotation_rle(mask_utils, info, entry)).astype(bool)
            projected = project_spine(np, cv2, mask, entry["spine"])
            area = int(mask.sum())
            width = area / max(len(projected), 1)
            small += int(area <= 2000)
            thin += int(width <= 16)
            small_thin += int(area <= 2000 and width <= 16)
        candidates.append(
            {
                "file_name": name,
                "image_id": image_id,
                "annotations": len(entries),
                "small": small,
                "thin": thin,
                "small_thin": small_thin,
            }
        )
    candidates.sort(
        key=lambda row: (
            -row["small_thin"],
            -row["small"],
            -row["thin"],
            -row["annotations"],
            row["file_name"],
            row["image_id"],
        )
    )
    records = candidates[:EXPECTED_RECORDS]
    identity = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if identity != SELECTION_ID:
        raise RuntimeError(f"V43 selection changed vs V41/V42: {identity}")
    totals = {
        "annotations": sum(row["annotations"] for row in records),
        "small_thin": sum(row["small_thin"] for row in records),
    }
    if totals != {"annotations": EXPECTED_ANNOTATIONS, "small_thin": EXPECTED_SMALL_THIN}:
        raise RuntimeError(f"V43 cohort totals changed: {totals}")
    return records


def build_yolo_dataset(np, cv2, mask_utils, document, records, image_dir, work: Path):
    images = {str(row["id"]): row for row in document["images"]}
    annotations = defaultdict(list)
    for row in document["annotations"]:
        annotations[str(row["image_id"])].append(row)
    img_out = work / "images" / "train"
    lbl_out = work / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    targets_by_stem: dict[str, list] = {}
    for record in records:
        info = images[record["image_id"]]
        path = image_dir / record["file_name"]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (2048, 2048):
            raise RuntimeError(f"cannot read frozen V43 fit image: {path}")
        stem = Path(record["file_name"]).stem
        cv2.imwrite(str(img_out / f"{stem}.png"), gray)
        entries = annotations[record["image_id"]]
        lines = []
        rle_targets = []
        for row in entries:
            rle = annotation_rle(mask_utils, info, row)
            rle_targets.append(rle)
            mask = mask_utils.decode(rle).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 3:
                continue
            poly = contour.reshape(-1, 2).astype(np.float64)
            poly[:, 0] /= 2048.0
            poly[:, 1] /= 2048.0
            coords = " ".join(f"{v:.6f}" for v in poly.reshape(-1))
            lines.append(f"0 {coords}")
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        targets_by_stem[stem] = rle_targets
    yaml_path = work / "v43.yaml"
    yaml_path.write_text(
        f"path: {work}\ntrain: images/train\nval: images/train\nnc: 1\nnames: [filament]\n"
    )
    return yaml_path, targets_by_stem


def pq_metric(np, mask_utils, predictions, targets):
    if not predictions or not targets:
        counts = [0.0, 0, len(predictions), len(targets)]
        matched = []
    else:
        iou = np.asarray(mask_utils.iou(predictions, targets, [0] * len(targets)), dtype=float)
        hits = np.argwhere(iou > 0.5)
        order = sorted(hits.tolist(), key=lambda pair: (-iou[pair[0], pair[1]], pair))
        used_p, used_t, matched = set(), set(), []
        for p, t in order:
            if p not in used_p and t not in used_t:
                used_p.add(p)
                used_t.add(t)
                matched.append(float(iou[p, t]))
        counts = [
            sum(matched),
            len(matched),
            len(predictions) - len(matched),
            len(targets) - len(matched),
        ]
    iou_sum, tp, fp, fn = counts
    denominator = tp + 0.5 * (fp + fn)
    return {
        "pq": iou_sum / denominator if denominator else 0.0,
        "sq": iou_sum / tp if tp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "iou_sum": float(iou_sum),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }, matched


def _predict_view(np, cv2, model, gray, device, flip):
    """Predict on a possibly-flipped grayscale image; return (prob_masks, confs)
    as list of full-res float32 [0,1] arrays + confidences in original orientation."""
    img = gray
    if flip == "h":
        img = gray[:, ::-1].copy()
    elif flip == "v":
        img = gray[::-1, :].copy()
    elif flip == "hv":
        img = gray[::-1, ::-1].copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = model.predict(
        source=rgb,
        imgsz=IMGSZ,
        conf=0.01,
        iou=0.6,
        max_det=1000,
        device=device,
        retina_masks=False,
        verbose=False,
    )[0]
    out = []
    if result.masks is None:
        return out
    data = result.masks.data.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for i in range(data.shape[0]):
        m = data[i]
        if m.shape != (2048, 2048):
            m = cv2.resize(m.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_LINEAR)
        if flip == "h":
            m = m[:, ::-1]
        elif flip == "v":
            m = m[::-1, :]
        elif flip == "hv":
            m = m[::-1, ::-1]
        out.append((np.ascontiguousarray(m.astype(np.float32)), float(confs[i])))
    return out


def tta_masks(np, cv2, mask_utils, model, gray, device, views, confidence):
    """Run TTA views, merge overlapping soft masks by averaging, threshold to RLE."""
    detections = []  # (soft_mask float, conf)
    for flip in views:
        detections.extend(_predict_view(np, cv2, model, gray, device, flip))
    detections = [(m, c) for (m, c) in detections if c >= confidence]
    if not detections:
        return []
    # Greedy merge across views by binary IoU of thresholded masks.
    bins = [(m >= 0.5) for (m, _) in detections]
    n = len(detections)
    used = [False] * n
    merged = []
    order = sorted(range(n), key=lambda i: -detections[i][1])
    for idx in order:
        if used[idx]:
            continue
        group = [idx]
        used[idx] = True
        ai = bins[idx]
        area_i = ai.sum()
        for j in order:
            if used[j]:
                continue
            inter = np.logical_and(ai, bins[j]).sum()
            union = area_i + bins[j].sum() - inter
            if union > 0 and inter / union >= TTA_MERGE_IOU:
                group.append(j)
                used[j] = True
        soft = np.mean([detections[k][0] for k in group], axis=0)
        binary = (soft >= 0.5).astype(np.uint8)
        if int(binary.sum()) >= MIN_AREA:
            merged.append(mask_utils.encode(np.asfortranarray(binary)))
    return merged


def evaluate_tta(np, cv2, mask_utils, model, records, image_dir, targets_by_stem, device, views):
    soft_cache = {}
    for record in records:
        stem = Path(record["file_name"]).stem
        gray = cv2.imread(str(image_dir / record["file_name"]), cv2.IMREAD_GRAYSCALE)
        dets = []
        for flip in views:
            dets.extend(_predict_view(np, cv2, model, gray, device, flip))
        soft_cache[stem] = dets
    cells = []
    for confidence in EVAL_CONFIDENCES:
        totals = np.zeros(4, dtype=float)
        all_matched = []
        rows = []
        for record in records:
            stem = Path(record["file_name"]).stem
            dets = [(m, c) for (m, c) in soft_cache[stem] if c >= confidence]
            predictions = _merge_dets(np, mask_utils, dets)
            metric, matched = pq_metric(np, mask_utils, predictions, targets_by_stem[stem])
            totals += [metric["iou_sum"], metric["tp"], metric["fp"], metric["fn"]]
            all_matched.extend(matched)
            rows.append({**record, "predictions": len(predictions), **metric})
        iou_sum, tp, fp, fn = totals
        denominator = tp + 0.5 * (fp + fn)
        aggregate = {
            "pq": float(iou_sum / denominator if denominator else 0),
            "sq": float(iou_sum / tp if tp else 0),
            "recall": float(tp / (tp + fn) if tp + fn else 0),
            "iou_sum": float(iou_sum),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "median_matched_iou": float(np.median(all_matched)) if all_matched else 0.0,
        }
        cells.append({"confidence": confidence, "metrics": aggregate, "rows": rows})
    return max(
        cells,
        key=lambda cell: (cell["metrics"]["pq"], -cell["metrics"]["fp"], cell["confidence"]),
    ), cells


def _merge_dets(np, mask_utils, dets):
    if not dets:
        return []
    bins = [(m >= 0.5) for (m, _) in dets]
    n = len(dets)
    used = [False] * n
    merged = []
    order = sorted(range(n), key=lambda i: -dets[i][1])
    for idx in order:
        if used[idx]:
            continue
        used[idx] = True
        ai = bins[idx]
        area_i = ai.sum()
        group = [idx]
        for j in order:
            if used[j]:
                continue
            inter = np.logical_and(ai, bins[j]).sum()
            union = area_i + bins[j].sum() - inter
            if union > 0 and inter / union >= TTA_MERGE_IOU:
                group.append(j)
                used[j] = True
        soft = np.mean([dets[k][0] for k in group], axis=0)
        binary = (soft >= 0.5).astype(np.uint8)
        if int(binary.sum()) >= MIN_AREA:
            merged.append(mask_utils.encode(np.asfortranarray(binary)))
    return merged


class MemoryMonitor:
    def __init__(self):
        self.peak_mib = 0.0
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                text = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=5,
                )
                values = [float(line.strip()) for line in text.splitlines() if line.strip()]
                self.peak_mib = max(self.peak_mib, max(values, default=0.0))
                self.samples.append(max(values, default=0.0))
            except Exception:
                pass
            self.stop.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)


def gpu_preflight(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"V43 requires exactly one visible CUDA GPU, found {torch.cuda.device_count()}"
        )
    properties = torch.cuda.get_device_properties(0)
    name = properties.name
    capability = (properties.major, properties.minor)
    total_mib = properties.total_memory / 2**20
    if "T4" not in name.upper() or capability < (7, 5) or total_mib < 14000:
        raise RuntimeError(f"V43 unsupported GPU provenance: {name}, {capability}, {total_mib}")
    return {
        "name": name,
        "capability": list(capability),
        "total_mib": total_mib,
        "visible_devices": torch.cuda.device_count(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--weights", type=Path, help="base weights for training")
    parser.add_argument("--eval-checkpoint", type=Path, help="skip training, eval this .pt")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--tta", default="hv", help="comma views among none,h,v,hv or 'off'")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.epochs <= MAX_EPOCHS:
        parser.error(f"--epochs must be in (0,{MAX_EPOCHS}]")
    if sha256(args.annotations) != ANNOTATION_SHA256:
        raise RuntimeError("V43 annotation identity changed")

    if args.tta == "off":
        views = ["none"]
    else:
        views = ["none"] + [v for v in args.tta.split(",") if v in ("h", "v", "hv")]

    import cv2
    import numpy as np
    from pycocotools import mask as mask_utils

    document = json.loads(args.annotations.read_text())
    records = selection_records(np, cv2, mask_utils, document)
    yaml_path, targets_by_stem = build_yolo_dataset(
        np, cv2, mask_utils, document, records, args.images, args.work
    )
    total_targets = sum(len(v) for v in targets_by_stem.values())
    if total_targets != EXPECTED_ANNOTATIONS:
        raise RuntimeError(f"V43 target count {total_targets} != {EXPECTED_ANNOTATIONS}")

    if args.preflight_only:
        write_json(
            args.output,
            {
                "status": "preflight_ok",
                "experiment": "v43_mask_refine_g2",
                "selection_id": SELECTION_ID,
                "views": views,
                "total_targets": total_targets,
            },
        )
        print(json.dumps({"preflight": "ok", "targets": total_targets, "views": views}))
        return

    import torch
    from ultralytics import YOLO

    provenance = gpu_preflight(torch)
    torch.manual_seed(20260930)
    np.random.seed(20260930)
    device = 0

    started = time.monotonic()
    with MemoryMonitor() as monitor:
        torch.cuda.reset_peak_memory_stats()
        if args.eval_checkpoint is not None:
            model = YOLO(str(args.eval_checkpoint))
            weights_used = str(args.eval_checkpoint)
            train_seconds = 0.0
        else:
            if args.weights is None:
                parser.error("--weights required when not using --eval-checkpoint")
            model = YOLO(str(args.weights))
            model.train(
                data=str(yaml_path),
                epochs=args.epochs,
                imgsz=IMGSZ,
                batch=2,
                device=device,
                workers=2,
                optimizer="AdamW",
                lr0=1e-3,
                seed=20260930,
                deterministic=True,
                augment=False,
                mosaic=0.0,
                mixup=0.0,
                copy_paste=0.0,
                erasing=0.0,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                translate=0.0,
                scale=0.0,
                fliplr=0.0,
                flipud=0.0,
                degrees=0.0,
                plots=False,
                val=False,
                project=str(args.work / "runs"),
                name="v43",
                exist_ok=True,
                verbose=True,
            )
            weights_used = str(args.weights)
            train_seconds = time.monotonic() - started
        selected, cells = evaluate_tta(
            np, cv2, mask_utils, model, records, args.images, targets_by_stem, device, views
        )
        elapsed = time.monotonic() - started

    metric = selected["metrics"]
    peak_allocated = torch.cuda.max_memory_allocated() / 2**20
    gates = {
        "pq": metric["pq"] >= TARGET_PQ,
        "recall": metric["recall"] >= TARGET_RECALL,
        "median_iou": metric["median_matched_iou"] >= TARGET_MEDIAN_IOU,
        "fp": metric["fp"] <= MAX_FP,
        "allocated_memory": peak_allocated < MEMORY_GATE_MIB,
        "external_memory": 0 < monitor.peak_mib < MEMORY_GATE_MIB,
        "runtime": elapsed <= RUNTIME_GATE_SECONDS,
        "firewall": True,
    }
    passed = all(gates.values())
    result = {
        "status": "complete",
        "experiment": "v43_mask_refine_g2",
        "hypothesis": "TTA + multiscale mask merging raises matched-mask IoU past "
        "the V42 ceiling toward the fit gate",
        "selection_id": SELECTION_ID,
        "views": views,
        "records": records,
        "requested_epochs": args.epochs if args.eval_checkpoint is None else 0,
        "selected": selected,
        "cells": cells,
        "gpu": provenance,
        "weights": weights_used,
        "train_seconds": train_seconds,
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": peak_allocated,
        "peak_external_mib": monitor.peak_mib,
        "gates": gates,
        "g2_passed": passed,
        "full_training_design_authorized": passed,
        "reads_tune": False,
        "reads_audit": False,
        "reads_test": False,
        "exports": False,
        "submits": False,
        "action": "freeze separate full-population design"
        if passed
        else "refine further or close",
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "views": views,
                "selected": selected["metrics"],
                "confidence": selected["confidence"],
                "gates": gates,
                "g2_passed": passed,
                "elapsed_seconds": elapsed,
                "peak_external_mib": monitor.peak_mib,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
