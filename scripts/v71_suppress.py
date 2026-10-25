"""V71 post-refiner suppression. Pre-registration:
docs/v71_postrefiner_suppression_preregistration_2026-10-10.md. CPU only.
  python scripts/v71_suppress.py cache   # r3 refiner on CPU -> outputs/v71/cache.pkl
  python scripts/v71_suppress.py eval
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import analysis_v69_refined_probe as P  # noqa: E402  (CPU patches)

E, K, F = P.O, P.K, P.F
OUT = E.ROOT / "outputs/v71"
CUT = 0.30
R1 = [("drop", 0.25), ("drop", 0.5), ("merge", 0.25), ("merge", 0.5)]
R2 = [(150, 0.5), (300, 0.5), (150, 1.01), (300, 1.01)]
R3 = [0.02, 0.04, 0.06]
GRIDS = {"R1": [None] + R1, "R2": [None] + R2, "R3": [None] + R3}


def stats(m, gray):
    import cv2
    ys, xs = np.nonzero(m)
    area = len(xs)
    cov = np.cov(np.vstack([xs, ys])) if area > 2 else np.eye(2)
    w, v = np.linalg.eigh(cov)
    angle = float(np.arctan2(v[1, 1], v[0, 1]))
    elong = float(w[1] / max(w[0], 1e-6))
    y0, y1, x0, x1 = max(ys.min() - 15, 0), ys.max() + 16, max(xs.min() - 15, 0), xs.max() + 16
    mc = m[y0:y1, x0:x1].astype(np.uint8)
    ring = cv2.dilate(mc, np.ones((21, 21), np.uint8)).astype(bool) & ~mc.astype(bool)
    crop = gray[y0:y1, x0:x1].astype(float)
    bg = crop[ring].mean() if ring.any() else crop.mean()
    contrast = float((bg - crop[mc.astype(bool)].mean()) / max(bg, 1.0))
    return dict(area=area, angle=angle, elong=elong, contrast=contrast)


def cache_work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = P.RT, P.RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    gts = []
    for image_id in sorted(rt["split"][4][name]):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        gts.append([r for r in rl if mu.area(r) > 0])
    views = K.read_cache_views(E.cache_path("p2first", fold, name))
    groups = F.groups_of(views)
    sc = [sum(g["scores"]) / len(views) for g in groups]
    order = sorted(range(len(groups)), key=lambda i: -sc[i])
    preds = []
    if order:
        masks = [mu.decode(groups[i]["rle"]) for i in order]
        bounds, _, probs = core.crop_probabilities(rt["base"], rt["model"], gray, masks, True)
        rles, fs = core.clean_refined_predictions(masks, [sc[i] for i in order], bounds, probs,
                                                  0.45, P.v50.MODE)
        for r, s in zip(rles, fs):
            if s >= CUT:
                preds.append(dict(rle=r, score=float(s),
                                  **stats(mu.decode(r).astype(bool), gray)))
    return name, dict(fold=fold, gts=gts, preds=preds)


def angle_diff(a, b):
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def apply(preds, cfg):
    """cfg = dict(R1=..., R2=..., R3=...) -> list of (rle, score)."""
    import cv2
    from pycocotools import mask as mu
    keep = list(range(len(preds)))
    masks = {}
    if cfg.get("R1"):
        action, rho = cfg["R1"]
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dec = [mu.decode(p["rle"]).astype(bool) for p in preds]
        merge_into, drop = {}, set()
        for qi, q in enumerate(preds):
            ys, xs = np.nonzero(dec[qi])
            y0, y1, x0, x1 = max(ys.min() - 6, 0), ys.max() + 7, max(xs.min() - 6, 0), xs.max() + 7
            dil = cv2.dilate(dec[qi][y0:y1, x0:x1].astype(np.uint8), ker).astype(bool)
            best = None
            for pi, p in enumerate(preds):
                if (pi == qi or p["score"] <= q["score"] or p["area"] <= q["area"]
                        or q["area"] / p["area"] > rho):
                    continue
                if not (dil & dec[pi][y0:y1, x0:x1]).any():
                    continue
                if q["elong"] >= 2 and angle_diff(q["angle"], p["angle"]) >= np.pi / 6:
                    continue
                if best is None or p["score"] > preds[best]["score"]:
                    best = pi
            if best is not None:
                drop.add(qi)
                if action == "merge":
                    merge_into.setdefault(best, []).append(qi)
        for pi, qs in merge_into.items():
            if pi in drop:  # P itself is a fragment of something else: keep as dropped
                continue
            m = dec[pi].copy()
            for qi in qs:
                m |= dec[qi]
            masks[pi] = K.encode(m)
        keep = [i for i in keep if i not in drop]
    if cfg.get("R2"):
        a, s = cfg["R2"]
        keep = [i for i in keep if not (preds[i]["area"] < a and preds[i]["score"] < s)]
    if cfg.get("R3"):
        t = cfg["R3"]
        keep = [i for i in keep if not (preds[i]["contrast"] < t and preds[i]["score"] < 0.5)]
    return [masks.get(i, preds[i]["rle"]) for i in keep], [preds[i]["score"] for i in keep]


def count(d, cfg):
    r, s = apply(d["preds"], cfg)
    return K.score_file(r, s, d["gts"], (CUT,))[CUT]


def cmd_cache():
    from multiprocessing import set_start_method
    set_start_method("fork")
    ref = json.loads((E.ROOT / "outputs/v63/r3.json").read_text())["per_file"]
    fmap = P.R.fold_of_file()
    names = sorted(ref)
    with Pool(int(os.environ.get("V71_PROCS", "40")), initializer=P.init) as p:
        data = dict(p.map(cache_work, [(n, fmap[n]) for n in names], chunksize=1))
    for n in names:
        assert np.allclose(count(data[n], {}), ref[n]["t0.45_c0.30"]), n
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "cache.pkl", "wb") as fh:
        pickle.dump(data, fh)
    print("cached", len(data), "files; baseline reproduced")


DATA = None


def cfg_work(args):
    rule, point = args
    return (rule, str(point)), {n: count(d, {rule: point}).tolist() for n, d in DATA.items()}


def combo_work(args):
    fold, cfg = args
    return fold, {n: count(d, cfg).tolist() for n, d in DATA.items() if d["fold"] == fold}


def boot(a, b, seed=71):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(a), len(a))
        d.append(K.pq_value(b[i].sum(0)) - K.pq_value(a[i].sum(0)))
    d = np.asarray(d)
    return dict(delta=K.pq_value(b.sum(0)) - K.pq_value(a.sum(0)),
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                p_gt_0=float((d > 0).mean()))


def summ(c):
    t = c.sum(0)
    return dict(pq=K.pq_value(t), sq=t[0] / t[1], tp=int(t[1]), fp=int(t[2]), fn=int(t[3]))


def cmd_eval():
    global DATA
    with open(OUT / "cache.pkl", "rb") as fh:
        DATA = pickle.load(fh)
    names = sorted(DATA)
    fo = np.asarray([DATA[n]["fold"] for n in names])
    jobs = [(r, p) for r, g in GRIDS.items() for p in g]
    with Pool(len(jobs)) as pool:
        per = dict(pool.map(cfg_work, jobs))
    arr = {k: np.asarray([v[n] for n in names]) for k, v in per.items()}
    base = arr[("R1", "None")]
    rep = {"files": len(names), "baseline": summ(base),
           "grid_pooled_descriptive": {f"{r}={p}": dict(summ(a), dpq=K.pq_value(a.sum(0)) -
                                                        K.pq_value(base.sum(0)))
                                       for (r, p), a in arr.items()}}
    chosen = {}
    for k in range(5):
        tr = fo != k
        chosen[k] = {}
        for r, g in GRIDS.items():
            best = max(g, key=lambda p: (round(K.pq_value(arr[(r, str(p))][tr].sum(0)), 12),
                                         -g.index(p)) if p is not None else
                       (round(K.pq_value(base[tr].sum(0)), 12), 1))
            chosen[k][r] = best
    rep["chosen_per_fold"] = {k: {r: str(p) for r, p in v.items()} for k, v in chosen.items()}
    # each rule alone, cross-fitted
    for r in GRIDS:
        out = np.zeros_like(base)
        for k in range(5):
            out[fo == k] = arr[(r, str(chosen[k][r]))][fo == k]
        rep[f"alone_{r}"] = dict(summ(out), vs_r3=boot(base, out))
    # combination
    with Pool(5) as pool:
        res = dict(pool.map(combo_work, [(k, chosen[k]) for k in range(5)]))
    comb = np.asarray([res[DATA[n]["fold"]][n] for n in names])
    rep["combined"] = dict(summ(comb), vs_r3=boot(base, comb),
                           per_fold_delta={k: K.pq_value(comb[fo == k].sum(0)) -
                                           K.pq_value(base[fo == k].sum(0)) for k in range(5)})
    g = rep["combined"]["vs_r3"]
    rep["verdict"] = ("CONTINUE" if g["delta"] >= 0.010 and g["p_gt_0"] >= 0.90 and
                      g["ci95"][0] > 0 else "KILL" if g["delta"] < 0.005 else "INCONCLUSIVE")
    (OUT / "result.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    {"cache": cmd_cache, "eval": cmd_eval}[sys.argv[1]]()
