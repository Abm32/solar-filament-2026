"""V88 stage 1 (CPU): V84 soft refiner (t0.50) once per file, then every ownership arm of
docs/v88_ownership_preregistration_2026-10-12.md on the same pre-ownership crops. Per arm and file it stores the
V84-extract format (rows, iou, n_gt, scores) with features recomputed on that arm's masks.
Asserts arm G equals outputs/v84/extract_soft.pkl[0.5] (IoU columns and feature rows).
  CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/v88_extract.py [limit]
"""
from __future__ import annotations

import os
import pickle
import sys
from multiprocessing import Pool, set_start_method
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import audit_v80_extract as E  # noqa: E402
import train_v73_refiner as V73  # noqa: E402
import v83_extract as X  # noqa: E402
import v83_rule as V  # noqa: E402  (FEATS only; lightgbm imported lazily in workers)

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v88"
T, CUT = 0.50, 0.30
XS = (0.1, 0.2, 0.3, 0.5, 0.7)
ARMS = ["G", "A_argmax", "B_pin", "B_yhat", "D_small"] + [f"C_{x}" for x in XS]
BOOST = {}


def feat(crop, a_pre, seed, P, fallback, bnd, gray, disk):
    """v83_extract.refine_feats feature body for one final crop (bbox-local bool)."""
    import cv2
    x0, y0, x1, y1 = bnd
    k5 = np.ones((5, 5), np.uint8)
    c8 = crop.astype(np.uint8)
    a = int(crop.sum())
    sb = seed.astype(bool)
    inner = crop & ~(cv2.erode(c8, k5) > 0)
    outer = (cv2.dilate(c8, k5) > 0) & ~crop
    pts = np.column_stack(np.where(crop))[:, ::-1].astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    L = max(w, h, 1.0)
    g = gray[y0:y1, x0:x1].astype(np.float32)
    ring = (cv2.dilate(c8, np.ones((25, 25), np.uint8)) > 0) & ~(cv2.dilate(c8, np.ones((7, 7), np.uint8)) > 0)
    fg = float(g[crop].mean())
    bg = float(g[ring].mean()) if ring.any() else fg
    ys, xs = np.where(crop)
    return dict(
        p_in=float(P[crop].mean()), p_max=float(P[crop].max()),
        p_seed=float(P[sb].mean()) if sb.any() else 0.0,
        margin=float(P[inner].mean() - (P[outer].mean() if outer.any() else 0.0)),
        fallback=float(fallback), seed_area=float(sb.sum()),
        ratio_pre_seed=a_pre / max(sb.sum(), 1), retain=a / max(a_pre, 1),
        iou_seed_post=float((sb & crop).sum() / max((sb | crop).sum(), 1)),
        log_area=float(np.log(a)), elong=L * L / a, width=a / L,
        rdist=float(np.hypot(xs.mean() + x0 - disk["cx"], ys.mean() + y0 - disk["cy"]) / disk["r"]),
        contrast=(bg - fg) / max(bg, 1), dark=fg / max(disk["disk_mean"], 1),
        disk_mean=disk["disk_mean"], disk_std=disk["disk_std"], lowfreq_cv=disk["lowfreq_cv"],
        bbox=(x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1))


def greedy(own, order, pre, bounds, MIN):
    for i in order:
        x0, y0, x1, y1 = bounds[i]
        o = own[y0:y1, x0:x1]
        c = pre[i] & (o < 0)
        if c.sum() >= MIN:
            o[c] = i


def argmax_own(own, S, pre, P, bounds, scores, MIN):
    alive = list(S)
    while True:
        best = np.full(own.shape, -np.inf, np.float32)
        win = np.full(own.shape, -1, np.int32)
        for i in sorted(alive, key=lambda j: -scores[j]):  # strict > : tie -> higher score
            x0, y0, x1, y1 = bounds[i]
            v = np.where(pre[i], P[i], -np.inf).astype(np.float32)
            b, w = best[y0:y1, x0:x1], win[y0:y1, x0:x1]
            u = v > b
            b[u], w[u] = v[u], i
        kill = [i for i in alive if (win[bounds[i][1]:bounds[i][3], bounds[i][0]:bounds[i][2]] == i).sum() < MIN]
        if not kill:
            break
        alive = [i for i in alive if i not in kill]
    own[:] = win


def trim_own(own, S, pre, a_pre, bounds, x, MIN):
    for i in S:
        x0, y0, x1, y1 = bounds[i]
        o = own[y0:y1, x0:x1]
        c = pre[i].copy()
        cont = c & (o >= 0)
        if cont.sum() > x * a_pre[i]:
            for j in np.unique(o[cont]):
                if not a_pre[i] < a_pre[j]:  # occupier is smaller (or equal): it keeps its pixels
                    c &= ~(o == j)
        else:
            c &= ~cont
        if c.sum() >= MIN:
            o[c] = i
    for i in S:
        x0, y0, x1, y1 = bounds[i]
        o = own[y0:y1, x0:x1]
        if (o == i).sum() < MIN:
            o[o == i] = -1


def pre_yhat(fold, groups, scores, pre, P, seeds, fb, a_pre, bounds, gray, disk):
    import lightgbm as lgb
    if fold not in BOOST:
        BOOST[fold] = lgb.Booster(model_file=str(OUT / f"premodel_fold{fold}.txt"))
    n = len(groups)
    feats = [feat(pre[i], a_pre[i], seeds[i], P[i], fb[i], bounds[i], gray, disk) for i in range(n)]
    cand, rows = X.candidate_rows(groups, scores, list(range(n)), feats)
    yh = np.full(n, -1.0)
    if rows:
        yh[cand] = BOOST[fold].predict(np.asarray([[r[k] for k in V.FEATS] for r in rows], float), num_threads=1)
    return yh


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = E.RT, E.RT["core"]
    MIN = core.MIN_AREA
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    recs = []
    for image_id in sorted(rt["by_file"][name], key=str):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        recs.append([r for r in rl if mu.area(r) > 0])
    views = V73.R.K.read_cache_views(V73.R.O.cache_path(V73.RECIPE, fold, name))
    groups = E.groups_with_ids(views)
    scores = [g["score"] for g in groups]
    disk = E.disk_fit(gray)
    n = len(groups)
    out = {}
    if n:
        masks = [mu.decode(g["rle"]) for g in groups]
        bounds, _, probs = core.crop_probabilities(rt["base"], E.MODELS[fold], gray, masks, True)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (X.KS, X.KS))
        pre, P, seeds, fb = [], [], [], []
        for i, m in enumerate(masks):  # as refine_feats before the occupied clip
            x0, y0, x1, y1 = bounds[i]
            seed = m[y0:y1, x0:x1]
            p = cv2.resize(probs[i], (x1 - x0, y1 - y0))
            crop = (p >= T) & (cv2.dilate(seed, kern) > 0)
            fb.append(not crop.any())
            pre.append(seed.astype(bool) if fb[-1] else crop)
            P.append(p)
            seeds.append(seed)
        a_pre = [int(c.sum()) for c in pre]
        order = sorted(range(n), key=lambda j: -scores[j])  # == refine_feats order
        S = [i for i in order if scores[i] >= CUT]
        rest = [i for i in order if scores[i] < CUT]
        shape = masks[0].shape
        for arm in ARMS:
            own = np.full(shape, -1, np.int32)
            if arm == "G":
                greedy(own, S, pre, bounds, MIN)
            elif arm == "A_argmax":
                argmax_own(own, S, pre, P, bounds, scores, MIN)
            elif arm == "B_pin":
                greedy(own, sorted(S, key=lambda j: -float(P[j][pre[j]].mean())), pre, bounds, MIN)
            elif arm == "B_yhat":
                yh = pre_yhat(fold, groups, scores, pre, P, seeds, fb, a_pre, bounds, gray, disk)
                greedy(own, sorted(S, key=lambda j: -yh[j]), pre, bounds, MIN)
            elif arm == "D_small":
                greedy(own, sorted(S, key=lambda j: a_pre[j]), pre, bounds, MIN)
            else:
                trim_own(own, S, pre, a_pre, bounds, float(arm[2:]), MIN)
            greedy(own, rest, pre, bounds, MIN)
            post, feats = [None] * n, [None] * n
            for i in range(n):
                x0, y0, x1, y1 = bounds[i]
                c = own[y0:y1, x0:x1] == i
                if c.sum() >= MIN:
                    post[i] = core.encode_crop(c, bounds[i], shape)
                    feats[i] = feat(c, a_pre[i], seeds[i], P[i], fb[i], bounds[i], gray, disk)
            alive = [i for i in range(n) if post[i] is not None]
            cand, rows = X.candidate_rows(groups, scores, alive, feats) if alive else ([], [])
            for c, f in zip(cand, rows):  # as v84_extract
                f["area"] = float(mu.area(post[alive[c]]))
            post_l = [post[i] for i in alive]
            iou = [E.iou_mat(g, post_l)[:, cand] if g else np.zeros((0, len(cand))) for g in recs]
            out[arm] = dict(name=name, fold=fold, rows=rows, iou=iou, n_gt=[len(g) for g in recs],
                            scores=[scores[alive[c]] for c in cand])
    else:
        for arm in ARMS:
            out[arm] = dict(name=name, fold=fold, rows=[], iou=[np.zeros((len(g), 0)) for g in recs],
                            n_gt=[len(g) for g in recs], scores=[])
    return out


def main():
    import torch
    set_start_method("fork")
    V73.DEV = "cpu"
    E.RT = V73.runtime()
    for f in range(5):
        m = V73.build_model(E.RT["base"], "imagenet", "resnet18")
        m.load_state_dict(torch.load(ROOT / f"outputs/v84/soft_fold{f}_refiner.pt", map_location="cpu",
                                     weights_only=True)["model"])
        E.MODELS[f] = m.eval()
    fmap = V73.R.fold_of_file()
    names = sorted(fmap)
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if lim:
        names = names[:: max(1, len(names) // lim)][:lim]
    with torch.no_grad(), Pool(16, initializer=E.init) as p:
        res = p.map(work, [(n, fmap[n]) for n in names], chunksize=1)
    ref = {r["name"]: r for r in pickle.load(open(ROOT / "outputs/v84/extract_soft.pkl", "rb"))[0.5]}
    wi, wf = 0.0, 0.0
    for o in res:
        r, q = o["G"], ref[o["G"]["name"]]
        assert len(r["rows"]) == len(q["rows"]) and len(r["iou"]) == len(q["iou"]), r["name"]
        for a, b in zip(r["iou"], q["iou"]):
            if b.size:
                wi = max(wi, float(np.abs(a - b).max()))
        for a, b in zip(r["rows"], q["rows"]):
            wf = max(wf, max(abs(a[k] - b[k]) for k in V.FEATS + ["area"]))
    print("files", len(res), "G vs v84: max |iou|", wi, "max |feat|", wf, flush=True)
    assert wi < 1e-6 and wf < 1e-6
    OUT.mkdir(parents=True, exist_ok=True)
    data = {arm: [o[arm] for o in res] for arm in ARMS}
    pickle.dump(data, open(OUT / ("extract.pkl" if not lim else "extract_smoke.pkl"), "wb"))


if __name__ == "__main__":
    main()
