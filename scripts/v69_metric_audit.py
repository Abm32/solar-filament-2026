"""V69 metric audit (CPU only, train OOF caches only; never reads test).

1. data facts relevant to the host metric (empty records, area-0 GT, overlapping GT in a record)
2. host self-evaluation code (verbatim logic) vs v61_toolkit on the same OOF predictions
3. per-prediction table (conf, area, R, s = sum over records of matched IoU) for base and
   post-processing variants -> any selection rule can be scored exactly offline.
Usage: python v69_metric_audit.py <repo> <recipe> <outdir> [nproc]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
RECIPE = sys.argv[2]
OUT = Path(sys.argv[3])
sys.path.insert(0, str(REPO / "scripts"))
import eval_v61_oof as E  # noqa: E402
import v61_toolkit as K  # noqa: E402

import os
VARIANTS = os.environ.get("VARIANTS", "base,erode1,dilate1,dx+1,dx-1,dy+1,dy-1,open_cc").split(",")


def host_counts(gt_records, pred_counts):
    """Host Self_Evaluation_Notebook logic (torch float32 matmul, iou>0.5), per file."""
    import pandas as pd
    import torch
    from pycocotools import mask as mu

    def layers(rles):
        if not rles:
            return np.zeros((0, 2048, 2048), np.float32)
        m = mu.decode([{"size": [2048, 2048], "counts": r} for r in rles])
        return m.transpose(2, 0, 1).astype(np.float32)

    tot = np.zeros(4)
    pl = torch.from_numpy(layers(pred_counts)).reshape(len(pred_counts), 2048 * 2048)
    for gt in gt_records:  # host only sees records that have >=1 GT row
        if not gt:
            continue
        gl = torch.from_numpy(layers(gt)).reshape(len(gt), 2048 * 2048)
        n_gt, n_pred = len(gt), len(pred_counts)
        if n_pred == 0:
            tot += (0, 0, 0, n_gt)
            continue
        inter = gl @ pl.t()
        union = gl.sum(1).view(-1, 1) + pl.sum(1).view(1, -1) - inter
        iou = torch.where(union == 0, torch.tensor(0.0), inter / union)
        hit = iou > 0.5
        tot += (float(iou[hit].sum()), int(hit.sum()), int((hit.sum(0) == 0).sum()),
                int((hit.sum(1) == 0).sum()))
    del pd
    return tot


def variant_masks(masks, scores, name):
    import cv2
    from scipy import ndimage

    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    out = []
    for m in masks:
        if name == "base":
            v = m
        elif name == "erode1":
            v = cv2.erode(m.astype(np.uint8), k).astype(bool)
        elif name == "erode2":
            v = cv2.erode(m.astype(np.uint8), k, iterations=2).astype(bool)
        elif name == "erodeSq":
            v = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        elif name == "erode1_keepsmall":  # erode only masks wider than the cross (no vanishing)
            e = cv2.erode(m.astype(np.uint8), k).astype(bool)
            v = e if e.sum() >= 0.5 * m.sum() else m
        elif name == "dilate1":
            v = cv2.dilate(m.astype(np.uint8), k).astype(bool)
        elif name == "open_cc":  # keep only largest connected component
            lab, n = ndimage.label(m)
            v = m if n <= 1 else lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
        else:
            ax = 1 if name[1] == "x" else 0
            v = np.roll(m, 1 if name[2] == "+" else -1, axis=ax)
        out.append(v)
    if name == "dilate1":  # re-own in descending score order (masks already score-sorted)
        occ = np.zeros_like(out[0]) if out else None
        for i, v in enumerate(out):
            out[i] = v & ~occ
            occ |= out[i]
    return out


def work(item):
    from pycocotools import mask as mu

    fold, name, gt_records, gt_counts = item
    views = K.read_cache_views(E.cache_path(RECIPE, fold, name))
    rles, scores = K.cluster_views(views)
    rles, scores = K.own_pixels(rles, scores)  # score-desc order, min area 20
    res = {"file": name, "fold": fold, "R": len(gt_records),
           "R_host": sum(1 for g in gt_records if g), "n_gt": sum(len(g) for g in gt_records)}
    # equivalence at the primary cut
    keep = [r for r, s in zip(rles, scores) if s >= 0.35]
    res["toolkit_035"] = K.score_file(rles, scores, gt_records, (0.35,))[0.35].tolist()
    res["host_035"] = host_counts(gt_counts, [r["counts"] for r in keep]).tolist()
    # round-trip of export encoding (bytes counts -> str -> decode)
    rt = [mu.decode({"size": [2048, 2048], "counts": r["counts"].decode().encode()}) for r in keep]
    res["rle_roundtrip_ok"] = all(
        np.array_equal(a, mu.decode(r)) for a, r in zip(rt, keep))
    masks = [mu.decode(r).astype(bool) for r in rles]
    preds = {}
    for v in VARIANTS:
        vm = variant_masks(masks, scores, v)
        enc = [K.encode(m) for m in vm]
        rows = []
        for j, (r, s) in enumerate(zip(enc, scores)):
            area = int(mu.area(r))
            ssum, nhit, ious = 0.0, 0, []
            for gt in gt_records:
                if gt and area:
                    iou = mu.iou([r], gt, [0] * len(gt))[0]
                    b = float(iou.max())
                    ious.append(b)
                    if b > 0.5:
                        ssum += b
                        nhit += int((iou > 0.5).sum())
                else:
                    ious.append(0.0)
            rows.append([float(s), area, ssum, nhit, max(ious) if ious else 0.0])
        preds[v] = rows
    res["preds"] = preds
    # centroid radius (disk centre 1023.1,1024.1 R 899.7) for base masks
    rad = []
    for m in masks:
        ys, xs = np.nonzero(m)
        rad.append(float(np.hypot(xs.mean() - 1023.1, ys.mean() - 1024.1) / 899.7) if len(xs) else 0)
    res["radius"] = rad
    return res


def main():
    from pycocotools import mask as mu

    doc = json.loads(E.ANNOTATIONS.read_text())
    gts = E.gt_by_file(doc)
    # host-format GT (all annotation rows, incl. area-0) per file, per record
    id_info = {i["id"]: i for i in doc["images"]}
    per_rec = {}
    area0, n_ann = 0, 0
    for a in doc["annotations"]:
        info = id_info[a["image_id"]]
        r = mu.merge(mu.frPyObjects(a["segmentation"], 2048, 2048))
        n_ann += 1
        area0 += int(mu.area(r) == 0)
        per_rec.setdefault(a["image_id"], []).append(r)
    host_gt = {}
    for iid, info in id_info.items():
        host_gt.setdefault(info["file_name"], []).append(
            [r["counts"] for r in per_rec.get(iid, [])])
    # overlaps between GT within one record
    ov_pairs, ov_hi = 0, 0
    for recs in per_rec.values():
        if len(recs) > 1:
            m = np.asarray(mu.iou(recs, recs, [0] * len(recs)))
            np.fill_diagonal(m, 0)
            ov_pairs += int((m > 0).sum() // 2)
            ov_hi += int((m > 0.2).sum() // 2)
    facts = {"records": len(id_info), "records_empty": sum(1 for i in id_info if i not in per_rec),
             "annotations": n_ann, "annotations_area0": area0,
             "gt_pairs_overlapping_within_record": ov_pairs, "gt_pairs_iou_gt_0.2": ov_hi,
             "files": len(gts),
             "records_per_file": {str(k): int(v) for k, v in zip(*np.unique(
                 [len(v) for v in gts.values()], return_counts=True))}}
    print(json.dumps(facts), flush=True)
    folds = E.load_folds()
    items = [(f, n, gts[n], host_gt[n]) for f, names in sorted(folds.items()) for n in names]
    import os
    items = items[: int(os.environ.get("LIMIT", len(items)))]
    OUT.mkdir(parents=True, exist_ok=True)
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    with Pool(nproc) as pool, open(OUT / f"{RECIPE}_perfile.jsonl", "w") as f:
        for i, res in enumerate(pool.imap_unordered(work, items, chunksize=2), 1):
            f.write(json.dumps(res) + "\n")
            if i % 50 == 0:
                print(i, flush=True)
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
