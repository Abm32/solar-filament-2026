"""V69 probe: replace the strongest-mask representative of each flip-consensus group with a
pixel vote of its member masks. Same grouping/scores/ownership/scoring as eval_v61_oof.
Post-hoc diagnostic (not pre-registered); direct detector, CPU only."""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import eval_v61_oof as E  # noqa: E402
import v61_toolkit as K  # noqa: E402

RECIPE = os.environ.get("V69_RECIPE", "p2first")
VARIANTS = tuple(os.environ.get("V69_VARIANTS", "rep,vote50,vote50w,vote75").split(","))


def groups_of(views, match_iou=K.MATCH_IOU):
    from pycocotools import mask as mu
    props = sorted(((r, float(s), v) for v, (rl, sl) in enumerate(views)
                    for r, s in zip(rl, sl)), key=lambda x: -x[1])
    groups = []
    for rle, s, v in props:
        elig = [i for i, g in enumerate(groups) if v not in g["views"]]
        match = None
        if elig:
            ov = mu.iou([rle], [groups[i]["rle"] for i in elig], [0] * len(elig))[0]
            b = int(np.argmax(ov))
            if ov[b] >= match_iou:
                match = elig[b]
        if match is None:
            groups.append(dict(rle=rle, scores=[s], views={v}, members=[rle]))
        else:
            groups[match]["scores"].append(s)
            groups[match]["views"].add(v)
            groups[match]["members"].append(rle)
    return groups


def fuse(g, variant):
    from pycocotools import mask as mu
    if variant.startswith("erode"):  # control: shrink the strongest mask, no voting
        import cv2
        k = int(variant[5:])
        m = cv2.erode(mu.decode(g["rle"]), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        return K.encode(m) if m.sum() >= K.MIN_AREA else g["rle"]
    if variant == "rep" or len(g["members"]) == 1:
        return g["rle"]
    if variant == "vote100":
        ms = mu.decode(g["members"]).all(2)
        return K.encode(ms) if ms.sum() >= K.MIN_AREA else g["rle"]
    ms = mu.decode(g["members"]).astype(np.float32)  # H,W,n
    w = np.asarray(g["scores"], np.float32) if variant == "vote50w" else np.ones(ms.shape[2])
    frac = (ms * w).sum(2) / w.sum()
    thr = 0.75 if variant == "vote75" else 0.5
    m = frac >= thr - 1e-6
    return K.encode(m) if m.sum() >= K.MIN_AREA else g["rle"]


def work(job):
    fold, name, gt = job
    views = K.read_cache_views(E.cache_path(RECIPE, fold, name))
    groups = groups_of(views)
    scores = [sum(g["scores"]) / len(views) for g in groups]
    out = {}
    for var in VARIANTS:
        rles = [fuse(g, var) for g in groups]
        own, own_s = K.own_pixels(rles, scores)
        out[var] = {c: v.tolist() for c, v in K.score_file(own, own_s, gt).items()}
    return name, out


def main():
    doc = json.loads(E.ANNOTATIONS.read_text())
    gts = E.gt_by_file(doc)
    jobs = [(f, n, gts[n]) for f, ns in sorted(E.load_folds().items()) for n in ns]
    with Pool(int(os.environ.get("V69_PROCS", "48"))) as p:
        res = dict(p.map(work, jobs, chunksize=2))
    names = [n for _, n, _ in jobs]
    rep = {}
    for var in VARIANTS:
        rep[var] = {}
        for c in K.CONFIDENCES:
            arr = np.asarray([res[n][var][c] for n in names])
            t = arr.sum(0)
            rep[var][f"{c:.2f}"] = dict(pq=K.pq_value(t), sq=t[0] / t[1], tp=int(t[1]),
                                       fp=int(t[2]), fn=int(t[3]))
    # paired bootstrap vs rep at 0.35
    rng = np.random.default_rng(69)
    a = np.asarray([res[n]["rep"][0.35] for n in names])
    for var in VARIANTS[1:]:
        b = np.asarray([res[n][var][0.35] for n in names])
        ds = []
        for _ in range(2000):
            i = rng.integers(0, len(a), len(a))
            ds.append(K.pq_value(b[i].sum(0)) - K.pq_value(a[i].sum(0)))
        rep[var]["delta@0.35"] = [K.pq_value(b.sum(0)) - K.pq_value(a.sum(0)),
                                  float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
    out = E.ROOT / "outputs/v69_analysis" / RECIPE / os.environ.get("V69_OUT", "fusion.json")
    out.write_text(json.dumps(rep, indent=1))
    for var in VARIANTS:
        print(var, json.dumps(rep[var].get("0.35")), rep[var].get("delta@0.35"))


if __name__ == "__main__":
    main()
