"""Tune-only V17-refiner evaluation on frozen V29 native-2048 proposals."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

IMAGE_SIZE = 2048
DETECTOR_FLOOR = 0.10
NMS_IOU = 0.0
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
CONFIDENCES = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
EXPECTED_TUNE = {"files": 94, "records": 150, "gt": 1134}
V17_TUNE_PQ = 0.44870271727943206
PROMOTION_PQ = 0.50
COUNTS = ("iou_sum", "tp", "fp", "fn")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def population(runtime):
    _, _, _, _, by_file = runtime["split"]
    names = sorted(runtime["split"][1])
    actual = {
        "files": len(names), "records": sum(len(by_file[name]) for name in names),
        "gt": sum(len(runtime["annotations"].get(image_id, []))
                  for name in names for image_id in by_file[name]),
    }
    if actual != EXPECTED_TUNE:
        raise RuntimeError(f"canonical tune population changed: {actual}")
    return names, by_file


def metric_record(counts) -> dict:
    iou_sum, tp, fp, fn = map(float, counts)
    denominator = tp + 0.5 * (fp + fn)
    return {"pq": iou_sum / denominator if denominator else 0.0,
            "sq": iou_sum / tp if tp else 0.0,
            "rq": tp / denominator if denominator else 0.0,
            "iou_sum": iou_sum, "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def key(threshold, confidence):
    return f"t{threshold:.2f}_c{confidence:.2f}"


def evaluate(checkpoint: Path, output: Path) -> dict:
    from ultralytics import YOLO
    import analyze_v28_utility as cohort

    runtime = cohort.load_runtime()
    names, by_file = population(runtime)
    model = YOLO(str(checkpoint))
    totals = {key(t, c): np.zeros(4, dtype=float) for t in THRESHOLDS for c in CONFIDENCES}
    raw_total = np.zeros(4, dtype=float)
    for position, name in enumerate(names, 1):
        image_path = cohort.RAW_IMAGES / name
        result = model.predict(
            str(image_path), imgsz=IMAGE_SIZE, conf=DETECTOR_FLOOR, iou=NMS_IOU,
            max_det=100, retina_masks=True, agnostic_nms=True, verbose=False,
        )[0]
        if result.masks is None:
            masks, scores = [], []
        else:
            masks = [value.cpu().numpy().astype(np.uint8) for value in result.masks.data]
            scores = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
        proposal_rles = [runtime["core"].encode(mask) for mask in masks]
        refined = runtime["v26"].refine(runtime, proposal_rles, scores, image_path)
        for image_id in sorted(by_file[name]):
            info = runtime["images"][image_id]
            targets = [runtime["core"].annotation_rle(info, row)
                       for row in runtime["annotations"].get(image_id, [])]
            raw_matrix = (mask_utils.iou(proposal_rles, targets, [0] * len(targets)).T
                          if proposal_rles and targets else np.zeros((len(targets), len(proposal_rles))))
            raw_total += runtime["core"].pq_counts(raw_matrix, len(targets), len(proposal_rles))
            for threshold in THRESHOLDS:
                rles, kept_scores = refined[threshold]
                for confidence in CONFIDENCES:
                    selected = [rle for rle, score in zip(rles, kept_scores) if score >= confidence]
                    matrix = (mask_utils.iou(selected, targets, [0] * len(targets)).T
                              if selected and targets else np.zeros((len(targets), len(selected))))
                    totals[key(threshold, confidence)] += runtime["core"].pq_counts(
                        matrix, len(targets), len(selected))
        print(f"v29 refiner tune {position}/{len(names)}", flush=True)
    metrics = {name: metric_record(value) for name, value in totals.items()}
    selected = max(sorted(metrics), key=lambda name: metrics[name]["pq"])
    report = {
        "status": "complete", "checkpoint": str(checkpoint), "population": EXPECTED_TUNE,
        "raw_floor": metric_record(raw_total), "selected": selected,
        "selected_metrics": metrics[selected], "all_metrics": metrics,
        "delta_vs_v17": metrics[selected]["pq"] - V17_TUNE_PQ,
        "audit_authorized": bool(metrics[selected]["pq"] >= PROMOTION_PQ),
        "reads_audit": False, "reads_test": False, "exports": False,
    }
    write_json(output, report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
