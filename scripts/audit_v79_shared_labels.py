"""V79 shared-component audit: label fidelity, widths, multi-annotator oracles, annotator style.

CPU only. Reads only the train annotation json. Writes outputs/v79_audit/labels_*.{pkl,json}.
"""
from __future__ import annotations

import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
from skimage.morphology import skeletonize
from ultralytics.data.utils import polygon2mask

ROOT = Path.home() / "solar-filament-2026"
ANN = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
OUT = ROOT / "outputs/v79_audit"
S = 2048
PAD = 24
SHIFTS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def up(m, n):  # low-res (n x n) mask/prob -> 2048 bool via bilinear >= 0.5 (retina-style)
    return cv2.resize(m.astype(np.float32), (S, S), interpolation=cv2.INTER_LINEAR) >= 0.5


def square_bounds(mask, context=1.8, minimum=96):  # solar_base.square_bounds (V11/r3 refiner crop)
    ys, xs = np.where(mask)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), S)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x0, y0 = min(max(x0, 0), S - side), min(max(y0, 0), S - side)
    return x0, y0, x0 + side, y0 + side


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def gt_paths(poly):
    gt = mu.decode(mu.frPyObjects([poly], S, S)).max(axis=2).astype(bool)
    ys, xs = np.where(gt)
    y0, y1, x0, x1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, S), max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, S)
    win = (slice(y0, y1), slice(x0, x1))
    g = gt[win]
    # polygon_line normalisation (6 decimals), then ultralytics polygon2mask at 1024 training res
    seg = np.round(np.clip(np.asarray(poly, np.float64).reshape(-1, 2) / S, 0, 1), 6) * 1024
    res = {}
    for name, ratio in (("yolo1024", 1), ("yolo512", 2), ("yolo256", 4)):
        m = polygon2mask((1024, 1024), [seg.reshape(-1)], downsample_ratio=ratio)
        p = up(m, 1024 // ratio)
        res[name] = iou(p[win], g)
        if name == "yolo1024":
            res["yolo1024_area_ratio"] = float(p.sum() / gt.sum())
            res["yolo1024_shift"] = [iou(np.roll(p, (dy, dx), (0, 1))[win], g) for dx, dy in SHIFTS]
    # ideal 1024 (area-average down, bilinear up): inherent half-res loss without truncation artefact
    res["ideal1024"] = iou(up(cv2.resize(gt.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_AREA), 1024)[win], g)
    res["ideal512"] = iou(up(cv2.resize(gt.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA), 512)[win], g)
    # refiner crop target: square_bounds -> nearest 256 -> linear back >= 0.5
    bx0, by0, bx1, by1 = square_bounds(gt)
    crop = cv2.resize(gt[by0:by1, bx0:bx1].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
    back = cv2.resize(crop.astype(np.float32), (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_LINEAR) >= 0.5
    p = np.zeros_like(gt)
    p[by0:by1, bx0:bx1] = back
    res["refiner256"] = iou(p[win], g)
    res["refiner_scale"] = (bx1 - bx0) / 256
    # width: area / skeleton length (2048 px); also 2*max distance transform
    sk = skeletonize(g)
    res["area"] = int(g.sum())
    res["skel"] = int(sk.sum())
    res["width"] = float(g.sum() / max(sk.sum(), 1))
    dt = cv2.distanceTransform(np.pad(g, 1).astype(np.uint8), cv2.DIST_L2, 5)
    res["width_dt_med"] = float(2 * np.median(dt[1:-1, 1:-1][sk])) if sk.any() else 0.0
    return res


def file_job(job):
    name, records = job
    rows = []
    for ri, rec in enumerate(records):
        for ai, a in enumerate(rec["anns"]):
            r = gt_paths(a["segmentation"][0])
            r.update(file=name, rec=ri, rec_id=rec["rec_id"], ann=ai, n_rec=len(records), cat=a["category_id"])
            rows.append(r)
    # cross-record IoU matrices (full masks, RLE)
    rles = [[mu.frPyObjects(a["segmentation"], S, S)[0] for a in rec["anns"]] for rec in records]
    cross = {}
    for i in range(len(records)):
        for j in range(len(records)):
            if rles[i] and rles[j]:
                cross[(i, j)] = mu.iou(rles[i], rles[j], [0] * len(rles[j]))
    multi = None
    if len(records) >= 2:
        multi = multi_oracles(rles, cross)
    return name, rows, cross, multi


def multi_oracles(rles, cross):
    """Cluster GT across records (IoU>0.5, one per record), build vote masks, score vs every record."""
    R = len(rles)
    nodes = [(r, k) for r in range(R) for k in range(len(rles[r]))]
    cl = []  # list of dict r->k
    assigned = set()
    for r, k in nodes:
        if (r, k) in assigned:
            continue
        c = {r: k}
        assigned.add((r, k))
        for r2 in range(R):
            if r2 == r or not rles[r2]:
                continue
            row = cross[(r, r2)][k]
            order = np.argsort(-row)
            for k2 in order:
                if row[k2] > 0.5 and (r2, k2) not in assigned:
                    c[r2] = int(k2)
                    assigned.add((r2, int(k2)))
                    break
        cl.append(c)
    cands = []
    for c in cl:
        ms = [mu.decode(rles[r][k]) for r, k in c.items()]
        vote = (np.mean(ms, axis=0) >= 0.5).astype(np.uint8) if len(ms) > 1 else ms[0]
        if vote.sum() == 0:
            vote = ms[0]
        v = mu.encode(np.asfortranarray(vote))
        ious = []
        for r in range(R):
            if rles[r]:
                x = mu.iou([v], rles[r], [0] * len(rles[r]))[0]
                ious.append(x)
            else:
                ious.append(np.zeros(0))
        cands.append(dict(k=len(c), ious=ious))
    return dict(R=R, n_gt=[len(x) for x in rles], cands=cands)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = json.loads(ANN.read_text())
    per = {}
    for a in doc["annotations"]:
        per.setdefault(a["image_id"], []).append(a)
    files = {}
    for info in doc["images"]:
        files.setdefault(info["file_name"], []).append(dict(rec_id=info["id"].split("-")[0], date=info["date_captured"],
                                                            anns=per.get(info["id"], [])))
    jobs = sorted(files.items())
    if len(sys.argv) > 1:
        jobs = jobs[: int(sys.argv[1])]
    with Pool(16) as pool:
        out = pool.map(file_job, jobs, chunksize=2)
    pickle.dump(dict(out=out, meta={n: [dict(rec_id=r["rec_id"], date=r["date"], n=len(r["anns"])) for r in recs]
                                    for n, recs in files.items()}),
                open(OUT / "labels_raw.pkl", "wb"))
    print("done", len(out))


if __name__ == "__main__":
    main()
