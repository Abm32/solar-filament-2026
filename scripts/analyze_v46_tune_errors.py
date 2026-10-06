"""V46 analysis: decompose held-out tune PQ loss into actionable buckets.

For the V44 model on the 141-file canonical tune split, for every GT instance find
its best-overlapping prediction and bucket:
  TP        : best IoU > 0.5   (already scored)
  near_miss : 0.3 < best IoU <= 0.5   (mask-quality lever: promote to TP)
  hard_fn   : best IoU <= 0.3   (recall lever: essentially undetected)
Count FP (predictions matching no GT above 0.5). Break down by instance size.
Compute PQ under counterfactuals:
  baseline, +promote near-misses to IoU 0.55, +remove all FP, +both.
This localizes the highest-expected-value lever and the realizable ceiling.

Uses tune labels for analysis ONLY. No training, no test, no export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
IMGSZ = 2048
MIN_AREA = 20
CONF = 0.20  # V44 best tune confidence region
SMALL_AREA = 2000


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as s:
        for b in iter(lambda: s.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def annotation_rle(mask_utils, info, ann):
    return mask_utils.merge(
        mask_utils.frPyObjects(ann["segmentation"], int(info["height"]), int(info["width"]))
    )


def canonical_tune(document):
    by_file = defaultdict(list)
    images = {str(r["id"]): r for r in document["images"]}
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    files = sorted(by_file)
    import random

    random.Random(42).shuffle(files)
    tune = sorted(files[:141])
    annotations = defaultdict(list)
    for r in document["annotations"]:
        annotations[str(r["image_id"])].append(r)
    recs = []
    for name in tune:
        iid = max(
            by_file[name],
            key=lambda v: (len(annotations[v]), sum(r.get("area", 0) for r in annotations[v]), v),
        )
        recs.append((name, iid))
    return recs, images, annotations


def disjoint_instances(np, cv2, mask_utils, result, confidence):
    if result.masks is None:
        return []
    data = result.masks.data.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)
    occupied = np.zeros((2048, 2048), dtype=bool)
    out = []
    for i in order:
        if confs[i] < confidence:
            continue
        m = data[i]
        if m.shape != (2048, 2048):
            m = cv2.resize(m.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_LINEAR)
        b = (m >= 0.5) & (~occupied)
        if int(b.sum()) < MIN_AREA:
            continue
        occupied |= b
        out.append(mask_utils.encode(np.asfortranarray(b.astype(np.uint8))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if sha256(args.annotations) != ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")

    import cv2
    import numpy as np
    from pycocotools import mask as mask_utils
    from ultralytics import YOLO

    document = json.loads(args.annotations.read_text())
    recs, images, annotations = canonical_tune(document)
    model = YOLO(str(args.weights))

    buckets = {"tp": 0, "near_miss": 0, "hard_fn": 0}
    buckets_small = {"tp": 0, "near_miss": 0, "hard_fn": 0}
    fp_total = 0
    tp_iou_sum = 0.0
    near_miss_best_iou = []
    n_gt = 0
    n_pred = 0

    for name, iid in recs:
        info = images[iid]
        gray = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = model.predict(
            source=rgb, imgsz=IMGSZ, conf=0.01, iou=0.6, max_det=1000,
            device=0, retina_masks=False, verbose=False,
        )[0]
        preds = disjoint_instances(np, cv2, mask_utils, result, CONF)
        targets = [annotation_rle(mask_utils, info, r) for r in annotations[iid]]
        gt_areas = [int(mask_utils.area(t)) for t in targets]
        n_gt += len(targets)
        n_pred += len(preds)
        if not targets:
            continue
        if not preds:
            for a in gt_areas:
                buckets["hard_fn"] += 1
                if a <= SMALL_AREA:
                    buckets_small["hard_fn"] += 1
            continue
        iou = np.asarray(mask_utils.iou(preds, targets, [0] * len(targets)), dtype=float)
        # per-GT best prediction IoU
        best_pred_for_gt = iou.max(axis=0)  # over preds, for each gt
        # greedy TP matching at 0.5 for FP accounting
        hits = np.argwhere(iou > 0.5)
        ordered = sorted(hits.tolist(), key=lambda pr: (-iou[pr[0], pr[1]], pr))
        up, ut = set(), set()
        for p, t in ordered:
            if p not in up and t not in ut:
                up.add(p)
                ut.add(t)
                tp_iou_sum += float(iou[p, t])
        fp_total += len(preds) - len(up)
        for ti in range(len(targets)):
            bi = float(best_pred_for_gt[ti])
            small = gt_areas[ti] <= SMALL_AREA
            if ti in ut:
                buckets["tp"] += 1
                if small:
                    buckets_small["tp"] += 1
            elif bi > 0.3:
                buckets["near_miss"] += 1
                near_miss_best_iou.append(bi)
                if small:
                    buckets_small["near_miss"] += 1
            else:
                buckets["hard_fn"] += 1
                if small:
                    buckets_small["hard_fn"] += 1

    def pq(tp_sum, tp, fp, fn):
        denom = tp + 0.5 * (fp + fn)
        return tp_sum / denom if denom else 0.0

    tp = buckets["tp"]
    nm = buckets["near_miss"]
    hf = buckets["hard_fn"]
    fn = nm + hf
    baseline = pq(tp_iou_sum, tp, fp_total, fn)
    # counterfactual A: promote near-misses to TP at IoU 0.55
    cf_promote = pq(tp_iou_sum + 0.55 * nm, tp + nm, fp_total, hf)
    # counterfactual B: remove all FP
    cf_nofp = pq(tp_iou_sum, tp, 0, fn)
    # counterfactual C: both
    cf_both = pq(tp_iou_sum + 0.55 * nm, tp + nm, 0, hf)
    # counterfactual D: raise matched IoU to 0.80 (mask-quality on existing TP)
    cf_iou80 = pq(0.80 * tp, tp, fp_total, fn)

    result = {
        "status": "complete",
        "experiment": "v46_tune_error_decomposition",
        "confidence": CONF,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "buckets": buckets,
        "buckets_small": buckets_small,
        "fp_total": fp_total,
        "tp_iou_sum": tp_iou_sum,
        "mean_matched_iou": tp_iou_sum / tp if tp else 0.0,
        "mean_near_miss_iou": float(np.mean(near_miss_best_iou)) if near_miss_best_iou else 0.0,
        "pq_baseline": baseline,
        "pq_if_promote_nearmiss": cf_promote,
        "pq_if_remove_fp": cf_nofp,
        "pq_if_promote_and_nofp": cf_both,
        "pq_if_matched_iou_0p80": cf_iou80,
        "v17_tune_pq": 0.448703,
        "reads_tune": True,
        "reads_test": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "buckets", "buckets_small", "fp_total", "mean_matched_iou", "mean_near_miss_iou",
        "pq_baseline", "pq_if_promote_nearmiss", "pq_if_remove_fp",
        "pq_if_promote_and_nofp", "pq_if_matched_iou_0p80", "v17_tune_pq",
    )}, indent=2))


if __name__ == "__main__":
    main()
