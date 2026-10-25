"""V91 G1 diagnostic: where does the GT-target decode lose PQ? (CPU, train labels only)"""
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v91_spine as V  # noqa: E402


def hard(rec):
    from pycocotools import mask as mu
    fg = np.zeros((V.SIZE, V.SIZE), bool)
    line = np.zeros((V.SIZE, V.SIZE), np.uint8)
    for a in rec:
        fg |= mu.decode(mu.merge(mu.frPyObjects(V.parse(a["segmentation"]), V.SIZE, V.SIZE))).astype(bool)
        p = np.round(np.asarray(V.parse(a["spine"]), float).reshape(-1, 2)).astype(np.int32)
        cv2.polylines(line, [p], False, 1)
    return fg.astype(float), np.exp(-ndi.distance_transform_edt(line == 0) ** 2 / (2 * V.SIGMA ** 2))


def one(i):
    G = pickle.load(open(V.DATA / "gt.pkl", "rb"))["gt"][i]
    recs = RECS[V.NAMES[i]]
    out = {}
    # per-record hard oracle: decode record r's own targets, score vs record r only
    s = np.zeros(4)
    for r, rec in zip(G, recs):
        pf, ps = hard(rec)
        lab, f = V.decode(pf, ps, 0.5, 0.5)
        s += V.stats([(len(r), [V.hits(lab, len(f), V.gt_masks(r))])], [np.ones(len(f), bool)])
    out["per_record"] = s
    pf, ps = V.mm("fg.u8")[i] / 255.0, V.mm("sp.u8")[i] / 255.0
    for tf, ts in [(0.5, 0.5), (0.3, 0.3), (0.5, 0.3), (0.3, 0.5)]:
        lab, f = V.decode(pf, ps, tf, ts)
        out[f"soft_{tf}_{ts}"] = V.stats([(len(r), [V.hits(lab, len(f), V.gt_masks(r))]) for r in G],
                                         [np.ones(len(f), bool)])
    out["nrec"] = len(G)
    return out


RECS = V.records()
if __name__ == "__main__":
    with Pool(48) as p:
        R = p.map(one, range(len(V.NAMES)), chunksize=2)
    res = {}
    for k in R[0]:
        if k == "nrec":
            continue
        S = np.array([r[k] for r in R])
        res[k] = {"pq": V.pq(S), "tp_fp_fn": S.sum(0)[1:].astype(int).tolist()}
        for n in (1, 2, 3):
            sel = [j for j, r in enumerate(R) if r["nrec"] == n]
            res[k][f"pq_nrec{n}"] = (V.pq(S[sel]), len(sel))
    print(json.dumps(res, indent=1))
