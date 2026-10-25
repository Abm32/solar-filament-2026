"""V75: E_p2x2 voting ensemble (p2first + p2first_cp) through the r3 refiner. CPU only.
Pre-registration: docs/v75_ensemble_refined_preregistration_2026-10-10.md
  python scripts/v75_ensemble_refined.py selfcheck   # members=[p2first] reproduces outputs/v71 r3
  python scripts/v75_ensemble_refined.py run
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v71_suppress as S  # noqa: E402

P, K, E, F = S.P, S.K, S.E, S.F
OUT = E.ROOT / "outputs/v75"
MEMBERS = ["p2first", "p2first_cp"]


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold, members = job
    rt, core = P.RT, P.RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    views = []
    for m in members:
        views.extend(K.read_cache_views(E.cache_path(m, fold, name)))
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
            if s >= S.CUT:
                preds.append(dict(rle=r, score=float(s), area=int(mu.area(r))))
    return name, preds


def counts(data, names, preds_of, min_area=0):
    return np.asarray([K.score_file([p["rle"] for p in preds_of[n] if p["area"] >= min_area],
                                    [p["score"] for p in preds_of[n] if p["area"] >= min_area],
                                    data[n]["gts"], (S.CUT,))[S.CUT] for n in names])


def main(mode):
    set_start_method("fork")
    with open(E.ROOT / "outputs/v71/cache.pkl", "rb") as fh:
        ref = pickle.load(fh)
    names = sorted(ref)
    if len(names) != 141:
        raise RuntimeError("canonical population changed")
    members = ["p2first"] if mode == "selfcheck" else MEMBERS
    sub = names[:12] if mode == "selfcheck" else names
    with Pool(int(os.environ.get("V75_PROCS", "40")), initializer=P.init) as p:
        got = dict(p.map(work, [(n, ref[n]["fold"], members) for n in sub], chunksize=1))
    if mode == "selfcheck":
        for n in sub:
            assert np.allclose(counts(ref, [n], got)[0], S.count(ref[n], {})), n
        print("selfcheck ok:", len(sub), "files reproduce r3")
        return
    base = np.asarray([S.count(ref[n], {}) for n in names])
    assert (int(base.sum(0)[1]), int(base.sum(0)[2]), int(base.sum(0)[3])) == (1020, 448, 573)
    ens = counts(ref, names, got)
    ref_preds = {n: [dict(rle=p["rle"], score=p["score"], area=p["area"]) for p in ref[n]["preds"]]
                 for n in names}
    fo = np.asarray([ref[n]["fold"] for n in names])
    d = S.boot(base, ens, seed=75)
    verdict = ("continue" if d["delta"] >= 0.010 and d["p_gt_0"] >= 0.90
               else "kill" if d["delta"] < 0.005 else "inconclusive")
    rep = {"members": MEMBERS, "files": len(names), "r3": S.summ(base), "E_p2x2": S.summ(ens),
           "primary": d, "verdict": verdict,
           "per_fold_delta": {k: K.pq_value(ens[fo == k].sum(0)) - K.pq_value(base[fo == k].sum(0))
                              for k in range(5)},
           "secondary_min200": S.boot(counts(ref, names, ref_preds, 200),
                                      counts(ref, names, got, 200), seed=75)}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "cache.pkl", "wb") as fh:
        pickle.dump(got, fh)
    (OUT / "result.json").write_text(json.dumps(rep, indent=2, default=float) + "\n")
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
