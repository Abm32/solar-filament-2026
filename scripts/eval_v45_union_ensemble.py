"""V45: multi-model union-ensemble complementarity + tune PQ.

Measures whether combining several full-population YOLO-seg models recovers more
held-out tune instances than any single model (the ensemble hypothesis that gave
V17 its 0.4487). For each tune image it gathers instances from every model,
applies cross-model NMS (highest-confidence wins overlapping detections), enforces
non-overlap, and scores the project PQ. Reports per-model recall, union recall,
and ensemble PQ vs the V17 bar.

Uses tune labels for EVALUATION only. No training here.
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
CROSS_MODEL_NMS_IOU = 0.5
CONF_SWEEP = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
V17_TUNE_PQ = 0.448703


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as s:
        for b in iter(lambda: s.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


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


def model_instances(np, cv2, mask_utils, result, confidence):
    """Return list of (binary_mask_bool, conf) above confidence (not yet disjoint)."""
    if result.masks is None:
        return []
    data = result.masks.data.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    out = []
    for i in range(data.shape[0]):
        if confs[i] < confidence:
            continue
        m = data[i]
        if m.shape != (2048, 2048):
            m = cv2.resize(m.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_LINEAR)
        b = m >= 0.5
        if int(b.sum()) >= MIN_AREA:
            out.append((b, float(confs[i])))
    return out


def union_disjoint(np, mask_utils, dets, min_agree=1, high_conf=1.01):
    """Cross-model NMS with agreement voting.

    dets: list of (mask_bool, conf). An instance is KEPT only if it is corroborated
    by at least `min_agree` distinct detections (via IoU-overlap clustering) OR its
    confidence is >= `high_conf`. Highest-conf wins pixels; output is non-overlap.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i][1])
    # Cluster detections that overlap (>= NMS IoU); a cluster's size = agreement count.
    assigned = [False] * len(dets)
    clusters = []
    for i in order:
        if assigned[i]:
            continue
        assigned[i] = True
        members = [i]
        bi = dets[i][0]
        for j in order:
            if assigned[j]:
                continue
            inter = np.logical_and(bi, dets[j][0]).sum()
            union = bi.sum() + dets[j][0].sum() - inter
            if union > 0 and inter / union >= CROSS_MODEL_NMS_IOU:
                assigned[j] = True
                members.append(j)
        clusters.append(members)
    kept = []
    occupied = np.zeros((2048, 2048), dtype=bool)
    # process clusters by their top confidence
    clusters.sort(key=lambda c: -max(dets[m][1] for m in c))
    for members in clusters:
        top = max(members, key=lambda m: dets[m][1])
        agree = len(members)
        conf = dets[top][1]
        if agree < min_agree and conf < high_conf:
            continue
        b = dets[top][0] & (~occupied)
        if int(b.sum()) < MIN_AREA:
            continue
        occupied |= b
        kept.append(b)
    return [mask_utils.encode(np.asfortranarray(k.astype(np.uint8))) for k in kept]


def pq_from_preds(np, mask_utils, preds, targets):
    if not preds or not targets:
        return 0.0, 0, len(preds), len(targets), []
    iou = np.asarray(mask_utils.iou(preds, targets, [0] * len(targets)), dtype=float)
    hits = np.argwhere(iou > 0.5)
    ordered = sorted(hits.tolist(), key=lambda pr: (-iou[pr[0], pr[1]], pr))
    up, ut, matched = set(), set(), []
    for p, t in ordered:
        if p not in up and t not in ut:
            up.add(p)
            ut.add(t)
            matched.append(float(iou[p, t]))
    return sum(matched), len(up), len(preds) - len(up), len(targets) - len(ut), matched


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--weights", type=Path, nargs="+", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if sha256(args.annotations) != ANNOTATION_SHA256:
        raise RuntimeError("V45 annotation identity changed")

    import cv2
    import numpy as np
    from pycocotools import mask as mask_utils
    from ultralytics import YOLO

    document = json.loads(args.annotations.read_text())
    recs, images, annotations = canonical_tune(document)
    models = [YOLO(str(w)) for w in args.weights]

    # Pre-run every model on every tune image at conf 0.01; cache instances.
    cache = defaultdict(dict)  # stem -> model_idx -> [(mask,conf)]
    targets_by = {}
    for name, iid in recs:
        info = images[iid]
        gray = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for mi, model in enumerate(models):
            res = model.predict(
                source=rgb, imgsz=IMGSZ, conf=0.01, iou=0.6, max_det=1000,
                device=0, retina_masks=False, verbose=False,
            )[0]
            cache[name][mi] = model_instances(np, cv2, mask_utils, res, 0.01)
        targets_by[name] = [annotation_rle(mask_utils, info, r) for r in annotations[iid]]

    def score(model_subset, confidence, min_agree=1, high_conf=1.01):
        iou_sum = tp = fp = fn = 0
        matched_all = []
        for name, _ in recs:
            dets = []
            for mi in model_subset:
                dets.extend([(b, c) for (b, c) in cache[name][mi] if c >= confidence])
            preds = union_disjoint(np, mask_utils, dets, min_agree=min_agree, high_conf=high_conf)
            s, t, f, n, m = pq_from_preds(np, mask_utils, preds, targets_by[name])
            iou_sum += s
            tp += t
            fp += f
            fn += n
            matched_all.extend(m)
        denom = tp + 0.5 * (fp + fn)
        return {
            "confidence": confidence,
            "min_agree": min_agree,
            "high_conf": high_conf,
            "pq": float(iou_sum / denom) if denom else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "median_matched_iou": float(np.median(matched_all)) if matched_all else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
        }

    # Per-model best (single model, no agreement filter).
    per_model = {}
    for mi in range(len(models)):
        cells = [score([mi], c) for c in CONF_SWEEP]
        per_model[str(args.weights[mi])] = max(cells, key=lambda c: c["pq"])

    # Ensemble sweep: confidence x agreement policy.
    all_idx = list(range(len(models)))
    ensemble_cells = []
    # policy A: plain union (min_agree=1)
    for c in CONF_SWEEP:
        ensemble_cells.append(score(all_idx, c, min_agree=1))
    # policy B: require >=2 models to agree, with high-conf rescue at various floors
    if len(models) >= 2:
        for c in (0.05, 0.10, 0.20):
            for hc in (0.5, 0.7, 0.9):
                ensemble_cells.append(score(all_idx, c, min_agree=2, high_conf=hc))
    ensemble_best = max(ensemble_cells, key=lambda c: c["pq"])

    result = {
        "status": "complete",
        "experiment": "v45_union_ensemble",
        "weights": [str(w) for w in args.weights],
        "per_model_best": per_model,
        "ensemble_cells": ensemble_cells,
        "ensemble_best": ensemble_best,
        "v17_tune_pq": V17_TUNE_PQ,
        "ensemble_beats_v17": ensemble_best["pq"] > V17_TUNE_PQ,
        "best_single_pq": max(m["pq"] for m in per_model.values()),
        "ensemble_gain_over_best_single": ensemble_best["pq"]
        - max(m["pq"] for m in per_model.values()),
        "reads_tune": True,
        "reads_test": False,
    }
    write_json(args.output, result)
    print(json.dumps({
        "per_model_best_pq": {Path(k).parent.parent.name if "runs" in k else Path(k).stem:
                              round(v["pq"], 4) for k, v in per_model.items()},
        "ensemble_best": ensemble_best,
        "v17_tune_pq": V17_TUNE_PQ,
        "ensemble_beats_v17": result["ensemble_beats_v17"],
        "gain_over_best_single": round(result["ensemble_gain_over_best_single"], 4),
    }, indent=2))


if __name__ == "__main__":
    main()
