"""V100 arm D (CPU): split kept masks that swallow a second proposal. Prereg: docs/v100_preregistration_2026-10-25.md.
Pipeline = V84 soft extract path (p2first OOF views -> flip consensus -> soft fold refiner t0.50 -> cut 0.30 ->
MIN200), reproduced per file against outputs/v84/extract_soft.pkl (cut-0.30 counts asserted). For every kept mask
the best inner proposal is recorded with the PQ count vectors of the unsplit and split versions per threshold tau,
so the cross-fold tau selection is exact arithmetic afterwards. Train images only; never reads test.
  CUDA_VISIBLE_DEVICES= PROCS=22 python scripts/v100_split.py [limit]   -> outputs/v100/split.pkl, split.json
"""
from __future__ import annotations

import json
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
import v84_extract as X84  # noqa: E402
import v83_rule as V  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v100"
T, CUT, MINA, JFLOOR = 0.50, 0.30, 200, 0.10
TAUS = (0.10, 0.15, 0.20, 0.25, 0.30)
CONT, SHARE = 0.70, (0.15, 0.60)


def counts_for(recs_rles, kept_rles, core):
    tot = np.zeros(4)
    for g in recs_rles:
        I = E.iou_mat(g, kept_rles)
        tot += core.pq_counts(I if g else np.zeros((0, len(kept_rles))), len(g), len(kept_rles))
    return tot


def largest_cc(m):
    import cv2
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m, n - 1
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab == k, n - 1


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = E.RT, E.RT["core"]
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
    masks, bounds, probs = [], [], []
    if groups:
        masks = [mu.decode(g["rle"]) for g in groups]
        bounds, _, probs = core.crop_probabilities(rt["base"], E.MODELS[fold], gray, masks, True)
    X.T = T
    post, _ = X.refine_feats(masks, scores, bounds, probs, gray, disk) if groups else ([], [])
    alive = [i for i in range(len(groups)) if post[i] is not None]
    keep030 = [i for i in alive if scores[i] >= CUT]
    tot030 = counts_for(recs, [post[i] for i in keep030], core)  # reproduction check vs extract_soft
    kept = [i for i in keep030 if mu.area(post[i]) >= MINA]
    base_rles = [post[i] for i in kept]
    base = counts_for(recs, base_rles, core)
    # candidate splits: for each kept i the best inner proposal j
    dec = {i: mu.decode(post[i]).astype(bool) for i in kept}
    cands = []  # (i, score_j, rle_a, rle_b)
    kept_set = set(kept)
    for i in kept:
        mi = dec[i]
        ai = mi.sum()
        best = None
        for j in range(len(groups)):
            if j == i or j in kept_set or scores[j] < JFLOOR:
                continue
            mj = masks[j].astype(bool)
            inter = int((mj & mi).sum())
            if inter == 0:
                continue
            if inter / mj.sum() < CONT or not (SHARE[0] <= inter / ai <= SHARE[1]):
                continue
            if best is None or scores[j] > best[0]:
                best = (scores[j], j)
        if best is None:
            continue
        sj, j = best
        mj = masks[j].astype(bool)
        a = mi & mj
        b, ncc = largest_cc(mi & ~mj)
        if a.sum() < MINA or b.sum() < MINA:
            continue
        enc = lambda m: mu.encode(np.asfortranarray(m.astype(np.uint8)))  # noqa: E731
        cands.append((i, float(sj), enc(a), enc(b)))
    # per tau: count vectors with every candidate of score >= tau split
    per_tau = {}
    for tau in TAUS:
        split = {i: (ra, rb) for i, sj, ra, rb in cands if sj >= tau}
        rles = []
        for i in kept:
            if i in split:
                rles += list(split[i])
            else:
                rles.append(post[i])
        per_tau[tau] = (counts_for(recs, rles, core).tolist(), len(split))
    return dict(name=name, fold=fold, tot030=tot030.tolist(), base=base.tolist(), n_kept=len(kept),
                n_cands=len(cands), per_tau=per_tau)


def main():
    import torch
    set_start_method("fork")
    V73.DEV = "cpu"
    E.RT = V73.runtime()
    for f in range(5):
        m = V73.build_model(E.RT["base"], "imagenet", "resnet18")
        m.load_state_dict(torch.load(ROOT / (X84.CKPT["soft"].format(f) + "_refiner.pt"), map_location="cpu",
                                     weights_only=True)["model"])
        E.MODELS[f] = m.eval()
    fmap = V73.R.fold_of_file()
    names = sorted(fmap)
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if lim:
        names = names[:: max(1, len(names) // lim)][:lim]
    ref = {r["name"]: r for r in pickle.load(open(ROOT / "outputs/v84/extract_soft.pkl", "rb"))[0.5]}
    with torch.no_grad(), Pool(int(os.environ.get("PROCS", "22")), initializer=E.init) as p:
        res = p.map(work, [(n, fmap[n]) for n in names], chunksize=1)
    worst = max(max(abs(a - b) for a, b in zip(r["tot030"], ref[r["name"]]["tot"])) for r in res)
    print("files", len(res), "max cut-0.30 count diff vs extract_soft", worst, flush=True)
    assert worst <= 1e-6
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / f"split{'_smoke' if lim else ''}.pkl", "wb"))
    # cross-selected tau
    fold = np.asarray([r["fold"] for r in res])
    base = np.asarray([r["base"] for r in res])
    C = {tau: np.asarray([r["per_tau"][tau][0] for r in res]) for tau in TAUS}
    nsplit = {tau: int(sum(r["per_tau"][tau][1] for r in res)) for tau in TAUS}
    insample = {tau: V.pq(C[tau].sum(0)) for tau in TAUS}
    chosen = base.copy(); taus = []
    for k in range(5):
        tr = fold != k
        best = max(TAUS, key=lambda t: V.pq(C[t][tr].sum(0)))
        taus.append(best); chosen[fold == k] = C[best][fold == k]
    rep = dict(files=len(res), base=V.summ(base), insample={str(t): dict(pq=insample[t], splits=nsplit[t]) for t in TAUS},
               cross_tau=taus, cross=V.summ(chosen), d_cross=V.boot(base, chosen, seed=100),
               n_kept=int(sum(r["n_kept"] for r in res)), n_cands=int(sum(r["n_cands"] for r in res)))
    d = rep["d_cross"]
    rep["verdict"] = "CONTINUE" if d["delta"] >= 0.008 and d["p_gt_0"] >= 0.95 else "INCONCLUSIVE" if d["delta"] >= 0.003 else "KILL"
    json.dump(rep, open(OUT / f"split{'_smoke' if lim else ''}.json", "w"), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
