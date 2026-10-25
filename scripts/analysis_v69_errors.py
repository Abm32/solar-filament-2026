"""V69 error analysis (CPU only): where is PQ lost for the p2first 5-fold OOF direct detector.

Stage `collect` (parallel over 707 files) -> outputs/v69_analysis/{gt,fp,files,pairs}.pkl
Stage `report` -> outputs/v69_analysis/report.json + printed tables.

Scoring replicates eval_v61_oof.file_counts exactly (consensus -> ownership -> cut -> per-record
pq_counts); the collect stage asserts its per-record counts equal K.pq_counts.
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import eval_v61_oof as E  # noqa: E402
import v61_toolkit as K  # noqa: E402

RECIPE = os.environ.get("V69_RECIPE", "p2first")
CUT = 0.35
OUT = E.ROOT / "outputs/v69_analysis" / RECIPE
OUT.mkdir(parents=True, exist_ok=True)


def iou_mat(a, b):
    from pycocotools import mask as mu
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou(a, b, [0] * len(b)))


def areas(rles):
    from pycocotools import mask as mu
    return np.asarray([float(mu.area(r)) for r in rles]) if rles else np.zeros(0)


def inter_from_iou(iou, aa, ab):
    # |A∩B| = iou*(|A|+|B|)/(1+iou)
    return iou * (aa[:, None] + ab[None, :]) / (1.0 + iou)


def disk_geometry(img):
    import cv2
    small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    _, th = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.moments(th, binaryImage=True)
    cx, cy = m["m10"] / m["m00"] * 4, m["m01"] / m["m00"] * 4
    r = np.sqrt(m["m00"] / np.pi) * 4
    disk = cv2.resize(th, (2048, 2048), interpolation=cv2.INTER_NEAREST) > 0
    vals = img[disk].astype(float)
    lap = cv2.Laplacian(small.astype(np.float32), cv2.CV_32F)[th > 0]
    return cx, cy, r, disk, dict(disk_mean=float(vals.mean()), disk_std=float(vals.std()),
                                 sharp=float(lap.var()), disk_r=float(r))


def gt_features(img, rle, all_gt_union, cx, cy, r):
    import cv2
    from pycocotools import mask as mu
    m = mu.decode(rle).astype(bool)
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = max(ys.min() - 20, 0), ys.max() + 21, max(xs.min() - 20, 0), xs.max() + 21
    mc = m[y0:y1, x0:x1].astype(np.uint8)
    ring = cv2.dilate(mc, np.ones((21, 21), np.uint8)).astype(bool) & ~all_gt_union[y0:y1, x0:x1]
    crop = img[y0:y1, x0:x1].astype(float)
    fg = crop[mc.astype(bool)].mean()
    bg = crop[ring].mean() if ring.any() else np.nan
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    length = max(w, h, 1.0)
    dist = np.hypot(xs.mean() - cx, ys.mean() - cy) / r
    return dict(fg=float(fg), bg=float(bg), contrast=float((bg - fg) / max(bg, 1.0)),
                length=float(length), rdist=float(dist))


def spine_len(sp):
    p = np.asarray(sp, float).reshape(-1, 2)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def load_records():
    from pycocotools import mask as mu
    doc = json.loads(E.ANNOTATIONS.read_text())
    per_image = {}
    for a in doc["annotations"]:
        per_image.setdefault(a["image_id"], []).append(a)
    recs = {}
    for info in doc["images"]:
        items = []
        for a in per_image.get(info["id"], []):
            r = mu.merge(mu.frPyObjects(a["segmentation"], info["height"], info["width"]))
            if mu.area(r) > 0:
                items.append(dict(rle=r, spine=spine_len(a.get("spine") or []),
                                  cat=a["category_id"]))
        recs.setdefault(info["file_name"], []).append(
            dict(rec_id=info["id"].split("-")[0], date=info["date_captured"], items=items))
    return recs


def work(job):
    import cv2
    from pycocotools import mask as mu
    fold, name, records = job
    views = K.read_cache_views(E.cache_path(RECIPE, fold, name))
    raw = [r for rl, _ in views for r in rl]
    raw_s = [s for _, sl in views for s in sl]
    cons, cons_s = K.cluster_views(views)
    own, own_s = K.own_pixels(cons, cons_s)
    own_s = np.asarray(own_s, float)
    keep = [r for r, s in zip(own, own_s) if s >= CUT]
    keep_s = own_s[own_s >= CUT]
    low = [r for r, s in zip(own, own_s) if s < CUT]
    cons_hi = [r for r, s in zip(cons, cons_s) if s >= CUT]
    a_keep = areas(keep)

    img = cv2.imread(str(E.IMAGE_DIR / name), cv2.IMREAD_GRAYSCALE)
    cx, cy, rad, disk, fstats = disk_geometry(img)

    gts = [[it["rle"] for it in rec["items"]] for rec in records]
    gt_rows, fp_rows, counts = [], [], []
    # cross-record matching of GT
    for ri, rec in enumerate(records):
        g = gts[ri]
        if g:
            union = mu.decode(mu.merge(g)).astype(bool)
        n_g = len(g)
        iou_k = iou_mat(g, keep)            # (G, P)
        cnt = K.pq_counts(iou_k, n_g, len(keep))
        counts.append(cnt)
        a_g = areas(g)
        inter = inter_from_iou(iou_k, a_g, a_keep) if n_g and keep else np.zeros((n_g, len(keep)))
        hits = iou_k > 0.5
        gt_matched = hits.any(1) if keep else np.zeros(n_g, bool)
        pred_matched = hits.any(0) if n_g else np.zeros(len(keep), bool)
        iou_low = iou_mat(g, low)
        iou_cons = iou_mat(g, cons_hi)
        iou_raw = iou_mat(g, raw)
        other = [gts[j] for j in range(len(records)) if j != ri]
        for gi, it in enumerate(rec["items"]):
            row = dict(file=name, fold=fold, rec=ri, rec_id=rec["rec_id"], n_rec=len(records),
                       area=a_g[gi], spine=it["spine"], cat=it["cat"], n_gt_rec=n_g,
                       matched=bool(gt_matched[gi]))
            row.update(gt_features(img, it["rle"], union, cx, cy, rad))
            row["best_keep_iou"] = float(iou_k[gi].max()) if keep else 0.0
            if keep:
                j = int(iou_k[gi].argmax())
                row["best_keep_cov"] = float(inter[gi, j] / a_g[gi])
                row["best_keep_prec"] = float(inter[gi, j] / a_keep[j])
                row["best_keep_matched_other"] = bool(pred_matched[j] and not hits[gi, j])
                row["match_score"] = float(keep_s[j]) if hits[gi, j] else np.nan
                # split: >=2 kept preds mostly inside this GT, each covering >=10% of it
                inside = (inter[gi] / np.maximum(a_keep, 1) >= 0.5) & (inter[gi] / a_g[gi] >= 0.1)
                row["n_frag"] = int(inside.sum())
            else:
                row.update(best_keep_cov=0.0, best_keep_prec=0.0, best_keep_matched_other=False,
                           match_score=np.nan, n_frag=0)
            row["best_low_iou"] = float(iou_low[gi].max()) if low else 0.0
            row["best_cons_iou"] = float(iou_cons[gi].max()) if cons_hi else 0.0
            row["best_raw_iou"] = float(iou_raw[gi].max()) if raw else 0.0
            if raw:
                j = int(iou_raw[gi].argmax())
                row["best_raw_score"] = float(raw_s[j])
            # agreement: how many OTHER records have a GT at IoU>0.5 with this one
            row["n_other_agree"] = int(sum(
                (iou_mat([it["rle"]], o).max() > 0.5) if o else 0 for o in other))
            gt_rows.append(row)
        # FP rows
        a_other = [areas(o) for o in other]
        for pj in np.nonzero(~pred_matched)[0] if keep else []:
            row = dict(file=name, fold=fold, rec=ri, rec_id=rec["rec_id"], n_rec=len(records),
                       score=float(keep_s[pj]), area=a_keep[pj], n_gt_rec=n_g)
            if n_g:
                col_iou = iou_k[:, pj]
                cov_gt = inter[:, pj] / a_g
                prec = inter[:, pj] / a_keep[pj]
                row["best_iou"] = float(col_iou.max())
                row["n_gt_cov30"] = int((cov_gt >= 0.3).sum())
                gi = int(prec.argmax())
                row["max_inside"] = float(prec[gi])
                row["inside_gt_matched"] = bool(gt_matched[gi])
                row["any_overlap"] = bool((inter[:, pj] > 0).any())
            else:
                row.update(best_iou=0.0, n_gt_cov30=0, max_inside=0.0, inside_gt_matched=False,
                           any_overlap=False)
            row["other_rec_match"] = bool(any(o and iou_mat([keep[pj]], o).max() > 0.5
                                              for o in other))
            row["other_rec_overlap"] = bool(any(
                o and (inter_from_iou(iou_mat([keep[pj]], o), a_keep[[pj]], ao) > 0).any()
                for o, ao in zip(other, a_other)))
            pm = mu.decode(keep[pj]).astype(bool)
            ys, xs = np.nonzero(pm)
            row["rdist"] = float(np.hypot(xs.mean() - cx, ys.mean() - cy) / rad)
            row["fg"] = float(img[pm].mean())
            fp_rows.append(row)
    # inter-annotator: ordered pairs (pred=record a, gt=record b)
    pairs = []
    for a in range(len(records)):
        for b in range(len(records)):
            if a != b:
                pairs.append((a, b, K.pq_counts(iou_mat(gts[b], gts[a]), len(gts[b]),
                                                len(gts[a]))))
    # sanity: equals the reference evaluator at CUT
    ref = K.score_file(own, own_s, gts, confidences=(CUT,))[CUT]
    assert np.allclose(ref, np.sum(counts, 0)), (name, ref, np.sum(counts, 0))
    frow = dict(file=name, fold=fold, n_rec=len(records), date=records[0]["date"],
                site=name[-7], n_keep=len(keep), counts=np.sum(counts, 0),
                rec_counts=counts, rec_ids=[r["rec_id"] for r in records],
                n_gt_mean=float(np.mean([len(x) for x in gts])), **fstats)
    return gt_rows, fp_rows, frow, (name, pairs)


def collect():
    recs = load_records()
    folds = E.load_folds()
    jobs = [(f, n, recs[n]) for f, names in sorted(folds.items()) for n in names]
    jobs = jobs[: int(os.environ.get("V69_LIMIT", len(jobs)))]
    with Pool(int(os.environ.get("V69_PROCS", "48"))) as p:
        res = p.map(work, jobs, chunksize=2)
    import pandas as pd
    gt = pd.DataFrame([r for g, _, _, _ in res for r in g])
    fp = pd.DataFrame([r for _, f, _, _ in res for r in f])
    files = pd.DataFrame([f for _, _, f, _ in res])
    pairs = {n: p for _, _, _, (n, p) in res}
    for k, v in dict(gt=gt, fp=fp, files=files, pairs=pairs).items():
        with open(OUT / f"{k}.pkl", "wb") as fh:
            pickle.dump(v, fh)
    tot = np.sum(np.stack(files["counts"].values), 0)
    print("pooled", K.pq_value(tot), tot, "gt rows", len(gt), "fp rows", len(fp))


if __name__ == "__main__":
    collect()
