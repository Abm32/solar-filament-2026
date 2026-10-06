"""V47: realizable precision + mask-cleanup sweep to beat V17 on held-out tune.

The V46 decomposition showed FP removal + near-miss promotion can reach tune PQ
0.470 (> V17 0.4487) as an oracle. V47 tests how much of that a REALIZABLE
inference-side filter captures on the V44 model, sweeping:
  - confidence threshold (drop low-conf false positives)
  - min-area gate (drop tiny spurious masks)
  - morphological cleanup (close + fill holes to push near-miss IoU past 0.5)
No retraining. Uses tune labels for evaluation only; no test/export/submit here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
IMGSZ = 2048
V17_TUNE_PQ = 0.448703
CONF_GRID = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
AREA_GRID = (20, 60, 120, 200)
CLEANUP_GRID = ("none", "close", "close_fill")


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


def cleanup_mask(np, cv2, ndi, m, mode):
    if mode == "none":
        return m
    b = m.astype(np.uint8)
    if mode in ("close", "close_fill"):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    if mode == "close_fill":
        b = ndi.binary_fill_holes(b.astype(bool)).astype(np.uint8)
    return b.astype(bool)


def build_preds(np, cv2, ndi, mask_utils, raw, confidence, min_area, cleanup):
    """raw = list of (softmask float32 2048, conf). Apply gates + cleanup, disjoint."""
    order = sorted(range(len(raw)), key=lambda i: -raw[i][1])
    occupied = np.zeros((2048, 2048), dtype=bool)
    out = []
    for i in order:
        m, c = raw[i]
        if c < confidence:
            continue
        b = m >= 0.5
        b = cleanup_mask(np, cv2, ndi, b, cleanup)
        b = b & (~occupied)
        if int(b.sum()) < min_area:
            continue
        occupied |= b
        out.append(mask_utils.encode(np.asfortranarray(b.astype(np.uint8))))
    return out


def pq(np, mask_utils, preds, targets):
    if not preds or not targets:
        return 0.0, 0, len(preds), len(targets)
    iou = np.asarray(mask_utils.iou(preds, targets, [0] * len(targets)), dtype=float)
    hits = np.argwhere(iou > 0.5)
    ordered = sorted(hits.tolist(), key=lambda pr: (-iou[pr[0], pr[1]], pr))
    up, ut, s = set(), set(), 0.0
    for p, t in ordered:
        if p not in up and t not in ut:
            up.add(p)
            ut.add(t)
            s += float(iou[p, t])
    return s, len(up), len(preds) - len(up), len(targets) - len(ut)


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
    from scipy import ndimage as ndi
    from ultralytics import YOLO

    document = json.loads(args.annotations.read_text())
    recs, images, annotations = canonical_tune(document)
    model = YOLO(str(args.weights))

    # Cache raw soft masks + confs once per image (expensive part).
    cache = {}
    targets_by = {}
    for name, iid in recs:
        info = images[iid]
        gray = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        res = model.predict(
            source=rgb, imgsz=IMGSZ, conf=0.01, iou=0.6, max_det=1000,
            device=0, retina_masks=False, verbose=False,
        )[0]
        raw = []
        if res.masks is not None:
            data = res.masks.data.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for i in range(data.shape[0]):
                m = data[i]
                if m.shape != (2048, 2048):
                    m = cv2.resize(m.astype(np.float32), (2048, 2048),
                                   interpolation=cv2.INTER_LINEAR)
                raw.append((m.astype(np.float32), float(confs[i])))
        cache[name] = raw
        targets_by[name] = [annotation_rle(mask_utils, info, r) for r in annotations[iid]]

    results = []
    for conf in CONF_GRID:
        for area in AREA_GRID:
            for cleanup in CLEANUP_GRID:
                s = tp = fp = fn = 0
                for name, _ in recs:
                    preds = build_preds(np, cv2, ndi, mask_utils, cache[name], conf, area, cleanup)
                    ss, t, f, n = pq(np, mask_utils, preds, targets_by[name])
                    s += ss
                    tp += t
                    fp += f
                    fn += n
                denom = tp + 0.5 * (fp + fn)
                results.append({
                    "confidence": conf, "min_area": area, "cleanup": cleanup,
                    "pq": float(s / denom) if denom else 0.0,
                    "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                    "tp": tp, "fp": fp, "fn": fn,
                })
    best = max(results, key=lambda r: r["pq"])
    out = {
        "status": "complete",
        "experiment": "v47_precision_cleanup_sweep",
        "weights": str(args.weights),
        "all_results": results,
        "best": best,
        "v17_tune_pq": V17_TUNE_PQ,
        "beats_v17": best["pq"] > V17_TUNE_PQ,
        "reads_tune": True,
        "reads_test": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"best": best, "v17_tune_pq": V17_TUNE_PQ,
                      "beats_v17": out["beats_v17"]}, indent=2))


if __name__ == "__main__":
    main()
