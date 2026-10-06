"""Tune-only direct-mask evaluation for V29 native-2048 YOLOv8l.

Evaluates the public upstream inference family on the canonical 94 tune files
and every annotator record. It contains no audit/test path and cannot export.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

IMAGE_SIZE = 2048
DETECTOR_FLOOR = 0.10
CONFIDENCES = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
NMS_IOUS = (0.0, 0.3, 0.7)
ORDER_POLICIES = ("public_class_id", "confidence")
MIN_AREA = 5
EXPECTED_TUNE = {"files": 94, "records": 150, "gt": 1134}
V17_TUNE_PQ = 0.44870271727943206
PROMOTION_PQ = 0.50
COUNTS = ("iou_sum", "tp", "fp", "fn")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def annotation_rle(info: dict, annotation: dict) -> dict:
    return mask_utils.merge(mask_utils.frPyObjects(
        annotation["segmentation"], int(info["height"]), int(info["width"]),
    ))


def clean_masks(masks: list[np.ndarray], scores: np.ndarray, policy: str) -> tuple[list[dict], np.ndarray]:
    if len(masks) != len(scores):
        raise ValueError("mask/score count mismatch")
    if not masks:
        return [], np.zeros(0, dtype=float)
    if policy == "confidence":
        order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    elif policy == "public_class_id":
        # HDJoJo passes single-class IDs (all zero) into argsort()[::-1].
        order = list(range(len(scores) - 1, -1, -1))
    else:
        raise ValueError(policy)
    occupied = np.zeros(masks[0].shape, dtype=bool)
    output, output_scores = [], []
    for index in order:
        clean = masks[index].astype(bool) & ~occupied
        if int(clean.sum()) < MIN_AREA:
            continue
        occupied |= clean
        output.append(mask_utils.encode(np.asfortranarray(clean.astype(np.uint8))))
        output_scores.append(float(scores[index]))
    return output, np.asarray(output_scores, dtype=float)


def pq_counts(predictions: list[dict], targets: list[dict]) -> np.ndarray:
    if not predictions or not targets:
        return np.asarray([0.0, 0, len(predictions), len(targets)], dtype=float)
    iou = np.asarray(mask_utils.iou(predictions, targets, [0] * len(targets)), dtype=float)
    hits = iou > 0.5
    return np.asarray([
        float(iou[hits].sum()), int(hits.sum()),
        int((~hits.any(axis=1)).sum()), int((~hits.any(axis=0)).sum()),
    ], dtype=float)


def metric_record(counts) -> dict:
    iou_sum, tp, fp, fn = map(float, counts)
    denominator = tp + 0.5 * (fp + fn)
    return {
        "pq": iou_sum / denominator if denominator else 0.0,
        "sq": iou_sum / tp if tp else 0.0,
        "rq": tp / denominator if denominator else 0.0,
        "iou_sum": iou_sum, "tp": int(tp), "fp": int(fp), "fn": int(fn),
    }


def load_population(annotation_path: Path):
    document = json.loads(annotation_path.read_text())
    images = {str(row["id"]): row for row in document["images"]}
    annotations: dict[str, list[dict]] = {}
    for row in document["annotations"]:
        annotations.setdefault(str(row["image_id"]), []).append(row)
    by_file: dict[str, list[str]] = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)
    files = sorted(by_file)
    random.Random(42).shuffle(files)
    tune = sorted(files[:94])
    population = {
        "files": len(tune),
        "records": sum(len(by_file[name]) for name in tune),
        "gt": sum(len(annotations.get(image_id, [])) for name in tune for image_id in by_file[name]),
    }
    if population != EXPECTED_TUNE:
        raise RuntimeError(f"canonical tune population changed: {population}")
    return images, annotations, by_file, tune


def candidate_key(iou: float, policy: str, confidence: float) -> str:
    return f"iou{iou:.1f}/{policy}/c{confidence:.2f}"


def evaluate(checkpoint: Path, annotation_path: Path, image_dir: Path, output: Path) -> dict:
    from ultralytics import YOLO

    images, annotations, by_file, tune = load_population(annotation_path)
    model = YOLO(str(checkpoint))
    totals = {
        candidate_key(iou, policy, confidence): np.zeros(4, dtype=float)
        for iou in NMS_IOUS for policy in ORDER_POLICIES for confidence in CONFIDENCES
    }
    diagnostics = []
    for position, name in enumerate(tune, 1):
        path = image_dir / name
        for nms_iou in NMS_IOUS:
            result = model.predict(
                str(path), imgsz=IMAGE_SIZE, conf=DETECTOR_FLOOR, iou=nms_iou,
                max_det=100, retina_masks=True, agnostic_nms=True, verbose=False,
            )[0]
            if result.masks is None:
                masks, scores = [], np.zeros(0, dtype=float)
            else:
                masks = [value.cpu().numpy().astype(np.uint8) for value in result.masks.data]
                scores = result.boxes.conf.detach().cpu().numpy().astype(float)
                if masks and masks[0].shape != (IMAGE_SIZE, IMAGE_SIZE):
                    masks = [cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                             for mask in masks]
            for policy in ORDER_POLICIES:
                cleaned, cleaned_scores = clean_masks(masks, scores, policy)
                for confidence in CONFIDENCES:
                    predictions = [rle for rle, score in zip(cleaned, cleaned_scores) if score >= confidence]
                    key = candidate_key(nms_iou, policy, confidence)
                    for image_id in sorted(by_file[name]):
                        info = images[image_id]
                        targets = [annotation_rle(info, row) for row in annotations.get(image_id, [])]
                        count = pq_counts(predictions, targets)
                        totals[key] += count
                        diagnostics.append({
                            "file_name": name, "image_id": image_id, "candidate": key,
                            **dict(zip(COUNTS, count)), "predictions": len(predictions),
                        })
        print(f"v29 tune {checkpoint.name} {position}/{len(tune)}", flush=True)
    metrics = {key: metric_record(value) for key, value in totals.items()}
    selected = max(sorted(metrics), key=lambda key: metrics[key]["pq"])
    report = {
        "status": "complete", "checkpoint": str(checkpoint),
        "population": EXPECTED_TUNE, "selected": selected,
        "selected_metrics": metrics[selected], "all_metrics": metrics,
        "public_operating_point": metrics[candidate_key(0.0, "public_class_id", 0.30)],
        "delta_vs_v17": metrics[selected]["pq"] - V17_TUNE_PQ,
        "promotion_bar": PROMOTION_PQ,
        "audit_authorized": bool(metrics[selected]["pq"] >= PROMOTION_PQ),
        "reads_audit": False, "reads_test": False, "exports": False,
    }
    write_json(output, report)
    # Compact diagnostics are intentionally omitted; counts remain independently recomputable.
    print(json.dumps({key: report[key] for key in (
        "checkpoint", "selected", "selected_metrics", "public_operating_point",
        "delta_vs_v17", "audit_authorized",
    )}, indent=2))
    return report


def merge(inputs: list[Path], output: Path) -> dict:
    reports = [json.loads(path.read_text()) for path in inputs]
    if any(report["population"] != EXPECTED_TUNE for report in reports):
        raise RuntimeError("checkpoint evaluations used different populations")
    winner = max(reports, key=lambda report: (report["selected_metrics"]["pq"], report["checkpoint"]))
    result = {
        "status": "complete", "selected_on": "tune only",
        "checkpoint_reports": reports,
        "winner": {
            "checkpoint": winner["checkpoint"], "candidate": winner["selected"],
            **winner["selected_metrics"],
        },
        "delta_vs_v17": winner["selected_metrics"]["pq"] - V17_TUNE_PQ,
        "audit_authorized": bool(winner["selected_metrics"]["pq"] >= PROMOTION_PQ),
        "action": "evaluate frozen winner on audit once" if winner["selected_metrics"]["pq"] >= PROMOTION_PQ
                  else "close V29 without audit/test/export",
    }
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate", "merge"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="*")
    args = parser.parse_args()
    if args.mode == "evaluate":
        if not all((args.checkpoint, args.annotations, args.images)):
            parser.error("evaluate requires checkpoint, annotations, and images")
        evaluate(args.checkpoint, args.annotations, args.images, args.output)
    else:
        if not args.inputs:
            parser.error("merge requires inputs")
        merge(args.inputs, args.output)


if __name__ == "__main__":
    main()
