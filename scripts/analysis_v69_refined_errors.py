"""V69: FN/FP categories for the REAL pipeline (p2first OOF -> r3 refiner, V17 cell t0.45/c0.30)
on the 141 canonical held-out files. CPU only (reuses the patched runtime of the probe)."""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import analysis_v69_refined_probe as P  # noqa: E402  (sets CPU patches)

A, K, O, R, F = P.F.E, P.K, P.O, P.R, P.F
import analysis_v69_errors as AE  # noqa: E402

CUT = 0.30


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = P.RT, P.RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    recs = []
    for image_id in sorted(rt["split"][4][name]):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        recs.append([r for r in rl if mu.area(r) > 0])
    views = K.read_cache_views(O.cache_path("p2first", fold, name))
    raw = [r for rl, _ in views for r in rl]
    groups = F.groups_of(views)
    sc = [sum(g["scores"]) / len(views) for g in groups]
    order = sorted(range(len(groups)), key=lambda i: -sc[i])
    props, sc = [groups[i]["rle"] for i in order], [sc[i] for i in order]
    if props:
        masks = [mu.decode(r) for r in props]
        bounds, _, probs = core.crop_probabilities(rt["base"], rt["model"], gray, masks, True)
        rles, fs = core.clean_refined_predictions(masks, sc, bounds, probs, 0.45, P.v50.MODE)
    else:
        rles, fs = [], []
    fs = np.asarray(fs, float)
    keep = [r for r, s in zip(rles, fs) if s >= CUT]
    low = [r for r, s in zip(rles, fs) if s < CUT]
    a_keep = AE.areas(keep)
    gts, fps, tot = [], [], np.zeros(4)
    for ri, g in enumerate(recs):
        iou_k = AE.iou_mat(g, keep)
        tot += K.pq_counts(iou_k, len(g), len(keep))
        hits = iou_k > 0.5
        gm = hits.any(1) if keep else np.zeros(len(g), bool)
        pm = hits.any(0) if g else np.zeros(len(keep), bool)
        il, ir = AE.iou_mat(g, low), AE.iou_mat(g, raw)
        other = [recs[j] for j in range(len(recs)) if j != ri]
        a_g = AE.areas(g)
        for gi in range(len(g)):
            gts.append(dict(matched=bool(gm[gi]), n_rec=len(recs),
                            iou=float(iou_k[gi].max()) if keep else 0.0,
                            low=float(il[gi].max()) if low else 0.0,
                            raw=float(ir[gi].max()) if raw else 0.0,
                            agree=int(sum((AE.iou_mat([g[gi]], o).max() > 0.5) if o else 0
                                          for o in other))))
        for pj in (np.nonzero(~pm)[0] if keep else []):
            col = iou_k[:, pj] if g else np.zeros(0)
            inter = AE.inter_from_iou(col[:, None], a_g, a_keep[[pj]])[:, 0] if g else col
            fps.append(dict(score=float(fs[fs >= CUT][pj]),
                            best_iou=float(col.max()) if g else 0.0,
                            n_cov30=int((inter / np.maximum(a_g, 1) >= 0.3).sum()) if g else 0,
                            inside=float((inter / a_keep[pj]).max()) if g else 0.0,
                            overlap=bool((inter > 0).any()) if g else False,
                            other=bool(any(o and AE.iou_mat([keep[pj]], o).max() > 0.5
                                           for o in other))))
    return name, gts, fps, tot.tolist()


def main():
    from multiprocessing import set_start_method
    set_start_method("fork")
    import pandas as pd
    ref = json.loads((O.ROOT / "outputs/v63/r3.json").read_text())["per_file"]
    fmap = R.fold_of_file()
    names = sorted(ref)
    with Pool(int(os.environ.get("V69_PROCS", "40")), initializer=P.init) as p:
        res = p.map(work, [(n, fmap[n]) for n in names], chunksize=1)
    for n, _, _, t in res:
        assert np.allclose(t, ref[n]["t0.45_c0.30"]), n
    gt = pd.DataFrame([r for _, g, _, _ in res for r in g])
    fp = pd.DataFrame([r for _, _, f, _ in res for r in f])
    tot = np.sum([t for *_, t in res], 0)
    out = {"pq": K.pq_value(tot), "sq": tot[0] / tot[1], "counts": tot.tolist()}
    fn = gt[~gt.matched]
    cat = np.select([fn.low > 0.5, fn.raw > 0.5, fn.iou >= 0.3,
                     np.maximum(fn.low, fn.raw) >= 0.3,
                     np.maximum.reduce([fn.iou, fn.low, fn.raw]) >= 0.05],
                    ["a score (<cut refined pred >0.5)", "c pool-only (a view mask >0.5)",
                     "d near-miss kept 0.3-0.5", "e near-miss below cut/views",
                     "f weak 0.05-0.3"], "g none")
    out["fn_cats"] = pd.Series(cat).value_counts().sort_index().to_dict()
    m = gt[gt.n_rec >= 2]
    out["multi_recall_by_agree"] = m.groupby(m.agree.clip(upper=1)).matched.agg(
        ["size", "mean"]).to_dict("index")
    fcat = np.select([fp.n_cov30 >= 2, fp.best_iou >= 0.3, fp.inside >= 0.5, fp.overlap],
                     ["a merge", "b near-miss 0.3-0.5", "c/d fragment", "e weak overlap"],
                     "f no overlap")
    out["fp_cats"] = pd.crosstab(pd.Series(fcat, name="cat"),
                                 fp.other.rename("other_rec_match")).to_dict()
    out["fp_other_rec_match"] = int(fp.other.sum())
    a = fn[fn.low > 0.5]
    t = np.asarray(tot, float)
    out["W_score_oracle"] = K.pq_value(t + [a.low.sum(), len(a), 0, -len(a)])
    out["W_fp_avoidable_removed"] = K.pq_value(t - [0, 0, int((~fp.other).sum()), 0])
    out["W_fp_halved"] = K.pq_value(t * [1, 1, 0.5, 1])
    out["W_fn_halved_at_SQ"] = K.pq_value(t + [out["sq"] * t[3] / 2, t[3] / 2, 0, -t[3] / 2])
    out["W_SQ0.9"] = 0.9 * t[1] / (t[1] + 0.5 * (t[2] + t[3]))
    print(json.dumps(out, indent=1, default=str))
    (O.ROOT / "outputs/v69_analysis/p2first/refined_errors.json").write_text(
        json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
