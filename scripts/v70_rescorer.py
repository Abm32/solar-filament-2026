"""V70 re-scorer over the full flip-view pool. Pre-registration:
docs/v70_rescorer_preregistration_2026-10-10.md (written before any V70 number). CPU only.

  PYTHONPATH=outputs/v70/pylib CUDA_VISIBLE_DEVICES="" python scripts/v70_rescorer.py features
  ... train | direct | refined | report
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
import analysis_v69_errors as AE  # noqa: E402
import analysis_v69_fusion as F  # noqa: E402
import eval_v61_oof as E  # noqa: E402
import v61_toolkit as K  # noqa: E402

OUT = E.ROOT / "outputs/v70"
CAND = OUT / "cands"
GRID = tuple(round(0.05 + 0.025 * i, 3) for i in range(19))
FEATS = ["cand_score", "is_fused", "is_anchor", "grp_n_views", "grp_cons_score", "grp_max_score",
         "grp_min_score", "grp_score_std", "cand_mean_iou_members", "grp_mean_pairwise_iou",
         "log_area", "elong", "width", "contrast", "darkness", "rdist", "nbr_max_iou",
         "nbr_n_overlap", "nbr_cov_stronger", "file_n_groups_035", "disk_mean"]
PARAMS = dict(num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=70,
              n_jobs=16, verbose=-1)
PROCS = int(os.environ.get("V70_PROCS", "40"))


def jobs():
    return [(f, n) for f, ns in sorted(E.load_folds().items()) for n in ns]


# ---------------------------------------------------------------- features
def feat_work(job):
    import cv2
    from pycocotools import mask as mu
    fold, name = job
    recs = GTS[name]
    views = K.read_cache_views(E.cache_path("p2first", fold, name))
    groups = F.groups_of(views)
    img = cv2.imread(str(E.IMAGE_DIR / name), cv2.IMREAD_GRAYSCALE)
    cx, cy, rad, _, fst = AE.disk_geometry(img)
    cons = np.asarray([sum(g["scores"]) / len(views) for g in groups])
    order = np.argsort(-cons, kind="stable")
    rank = np.empty(len(groups), int)
    rank[order] = np.arange(len(groups))
    anchors = [g["rle"] for g in groups]
    a_anch = AE.areas(anchors)
    big = len(groups) + 1
    rank_map = np.full((2048, 2048), big, np.int32)
    for gi in order:
        m = mu.decode(anchors[gi]).astype(bool)
        rank_map[m & (rank_map == big)] = rank[gi]
    n035 = int((cons >= 0.35).sum())
    rows, rles = [], []
    for gi, g in enumerate(groups):
        mem, sc = g["members"], np.asarray(g["scores"], float)
        mi = AE.iou_mat(mem, mem)
        n = len(mem)
        pair = float((mi.sum() - n) / (n * (n - 1))) if n > 1 else 0.0
        cands = [(r, float(s), 0, j == 0, j) for j, (r, s) in enumerate(zip(mem, sc))]
        if n >= 2:
            fm = mu.decode(mem).sum(2) >= 0.75 * n - 1e-6
            if fm.sum() >= K.MIN_AREA:
                cands.append((K.encode(fm), float(sc.max()), 1, False, -1))
        for rle, s, fused, anchor, j in cands:
            m = mu.decode(rle).astype(bool)
            area = float(m.sum())
            gf = AE.gt_features(img, rle, m, cx, cy, rad)
            iou_a = AE.iou_mat([rle], anchors)[0]
            inter = AE.inter_from_iou(iou_a[None], np.asarray([area]), a_anch)[0]
            others = np.arange(len(groups)) != gi
            if j >= 0:
                miou = float((mi[j].sum() - 1) / (n - 1)) if n > 1 else 0.0
            else:
                miou = float(AE.iou_mat([rle], mem)[0].mean())
            row = dict(file=name, fold=fold, group=gi, cand_score=s, is_fused=fused,
                       is_anchor=int(anchor), grp_n_views=n, grp_cons_score=float(cons[gi]),
                       grp_max_score=float(sc.max()), grp_min_score=float(sc.min()),
                       grp_score_std=float(sc.std()), cand_mean_iou_members=miou,
                       grp_mean_pairwise_iou=pair, log_area=float(np.log(area)),
                       elong=gf["length"] ** 2 / area, width=area / gf["length"],
                       contrast=gf["contrast"], darkness=gf["fg"] / fst["disk_mean"],
                       rdist=gf["rdist"],
                       nbr_max_iou=float(iou_a[others].max()) if others.any() else 0.0,
                       nbr_n_overlap=int((inter[others] > 0).sum()),
                       nbr_cov_stronger=float((rank_map[m] < rank[gi]).mean()),
                       file_n_groups_035=n035, disk_mean=fst["disk_mean"])
            ms = [AE.iou_mat([rle], gt)[0].max() if gt else 0.0 for gt in recs]
            hit = [x for x in ms if x > 0.5]
            row["p"] = len(hit) / len(recs)
            row["e"] = float(np.mean(hit)) if hit else np.nan
            rows.append(row)
            rles.append(rle)
    with open(CAND / f"{Path(name).stem}.pkl", "wb") as fh:
        pickle.dump(rles, fh)
    return rows


def cmd_features():
    import pandas as pd
    global GTS
    CAND.mkdir(parents=True, exist_ok=True)
    GTS = E.gt_by_file(json.loads(E.ANNOTATIONS.read_text()))
    with Pool(PROCS) as p:
        res = p.map(feat_work, jobs(), chunksize=2)
    df = pd.DataFrame([r for rs in res for r in rs])
    df["cand_idx"] = df.groupby("file").cumcount()
    df.to_pickle(OUT / "features.pkl")
    print(len(df), "candidates;", df.file.nunique(), "files; p>0:", (df.p > 0).mean())


GTS = {}


# ---------------------------------------------------------------- train
def date_folds(names):
    days = {}
    for n in names:
        days.setdefault(n[:8], []).append(n)
    load, fold = [0] * 5, {}
    for d in sorted(sorted(days), key=lambda d: -len(days[d])):
        k = int(np.argmin(load))
        load[k] += len(days[d])
        for n in days[d]:
            fold[n] = k
    return fold


def fit(x, y, objective):
    """Native LightGBM API (sklearn absent); PARAMS mapped 1:1 to booster params."""
    import lightgbm as lgb
    p = dict(objective=objective, num_leaves=PARAMS["num_leaves"],
             learning_rate=PARAMS["learning_rate"], min_data_in_leaf=PARAMS["min_child_samples"],
             bagging_fraction=PARAMS["subsample"], bagging_freq=PARAMS["subsample_freq"],
             feature_fraction=PARAMS["colsample_bytree"], seed=PARAMS["random_state"],
             num_threads=PARAMS["n_jobs"], verbose=-1, deterministic=True)
    return lgb.train(p, lgb.Dataset(x, y), num_boost_round=PARAMS["n_estimators"])


def cmd_train():
    import pandas as pd
    df = pd.read_pickle(OUT / "features.pkl")
    dmap = date_folds(sorted(df.file.unique()))
    df["dfold"] = df.file.map(dmap)
    imp = {}
    for scheme, col in (("v61", "fold"), ("date", "dfold")):
        s = np.zeros(len(df))
        for k in range(5):
            tr, te = df[col] != k, df[col] == k
            mp = fit(df.loc[tr, FEATS], df.loc[tr, "p"], "cross_entropy")
            pos = tr & (df.p > 0)
            me = fit(df.loc[pos, FEATS], df.loc[pos, "e"], "regression")
            s[te.values] = (np.clip(mp.predict(df.loc[te, FEATS]), 0, 1) *
                            np.clip(me.predict(df.loc[te, FEATS]), 0, 1))
            if scheme == "v61":
                imp[k] = dict(zip(FEATS, mp.feature_importance("gain").tolist()))
        df[f"s_{scheme}"] = s
    df[["file", "fold", "dfold", "group", "cand_idx", "s_v61", "s_date", "p", "e",
        "cand_score", "grp_cons_score", "is_anchor"]].to_pickle(OUT / "scores.pkl")
    gi = pd.DataFrame(imp).mean(1).sort_values(ascending=False)
    (OUT / "importance.json").write_text(json.dumps((gi / gi.sum()).round(4).to_dict(), indent=1))
    from scipy.stats import spearmanr
    print("spearman(s_v61,p)", spearmanr(df.s_v61, df.p)[0],
          "spearman(cons,p)", spearmanr(df.grp_cons_score, df.p)[0])
    print(gi.head(10))


# ---------------------------------------------------------------- selection
def select(name, sc_rows, scheme):
    """-> (rles, scores) one candidate per cluster (max s), clusters in descending s."""
    with open(CAND / f"{Path(name).stem}.pkl", "rb") as fh:
        rles = pickle.load(fh)
    best = sc_rows.sort_values(f"s_{scheme}", ascending=False, kind="stable") \
        .drop_duplicates("group")
    return [rles[i] for i in best.cand_idx], best[f"s_{scheme}"].to_numpy()


def own_skip(rles, scores):
    from pycocotools import mask as mu
    occ = np.zeros((2048, 2048), bool)
    out, out_s = [], []
    for r, s in zip(rles, scores):
        m = mu.decode(r).astype(bool)
        a = m.sum()
        if a == 0 or (m & occ).sum() > 0.5 * a:
            continue
        m &= ~occ
        if m.sum() < K.MIN_AREA:
            continue
        occ |= m
        out.append(K.encode(m))
        out_s.append(s)
    return out, np.asarray(out_s)


SC = None


def direct_work(job):
    fold, name = job
    gt = GTS[name]
    res = {}
    rows = SC[SC.file == name]
    for scheme in ("v61", "date"):
        r, s = select(name, rows, scheme)
        r, s = own_skip(r, s)
        res[scheme] = {c: v.tolist() for c, v in K.score_file(r, s, gt, GRID).items()}
    views = K.read_cache_views(E.cache_path("p2first", fold, name))
    r, s = K.own_pixels(*K.cluster_views(views))
    res["cons"] = {c: v.tolist() for c, v in K.score_file(r, s, gt).items()}
    return name, res


def crossfit(per, names, fold_of, grid):
    """Cut per fold chosen on the other folds; -> (F,4) counts, {fold: cut}."""
    arr = {c: np.asarray([per[n][c] for n in names]) for c in grid}
    fo = np.asarray([fold_of[n] for n in names])
    out, cuts = np.zeros((len(names), 4)), {}
    for k in sorted(set(fo)):
        c = max(grid, key=lambda c: K.pq_value(arr[c][fo != k].sum(0)))
        cuts[int(k)] = c
        out[fo == k] = arr[c][fo == k]
    return out, cuts


def boot(a, b, seed=70):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(a), len(a))
        d.append(K.pq_value(b[i].sum(0)) - K.pq_value(a[i].sum(0)))
    d = np.asarray(d)
    return dict(delta=K.pq_value(b.sum(0)) - K.pq_value(a.sum(0)),
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                p_gt_0=float((d > 0).mean()))


def summary(c):
    t = c.sum(0)
    return dict(pq=K.pq_value(t), sq=t[0] / t[1], tp=int(t[1]), fp=int(t[2]), fn=int(t[3]))


def cmd_direct():
    import pandas as pd
    global GTS, SC
    GTS = E.gt_by_file(json.loads(E.ANNOTATIONS.read_text()))
    SC = pd.read_pickle(OUT / "scores.pkl")
    with Pool(PROCS) as p:
        per = dict(p.map(direct_work, jobs(), chunksize=2))
    names = [n for _, n in jobs()]
    fmap = {n: f for f, n in jobs()}
    dmap = date_folds(names)
    cons_fixed = np.asarray([per[n]["cons"][0.35] for n in names])
    assert abs(K.pq_value(cons_fixed.sum(0)) - 0.40602092947818985) < 1e-9
    cons_cf, cons_cuts = crossfit({n: per[n]["cons"] for n in names}, names, fmap, K.CONFIDENCES)
    rep = {"consensus_fixed_0.35": summary(cons_fixed),
           "consensus_crossfit": dict(summary(cons_cf), cuts=cons_cuts)}
    for scheme, fo in (("v61", fmap), ("date", dmap)):
        arr, cuts = crossfit({n: per[n][scheme] for n in names}, names, fo, GRID)
        rep[f"rescorer_{scheme}"] = dict(summary(arr), cuts=cuts,
                                         vs_consensus_fixed=boot(cons_fixed, arr),
                                         vs_consensus_crossfit=boot(cons_cf, arr))
        rep[f"rescorer_{scheme}_grid"] = {
            str(c): K.pq_value(np.sum([per[n][scheme][c] for n in names], 0)) for c in GRID}
    (OUT / "direct.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("grid")}, indent=1,
                     default=float))


# ---------------------------------------------------------------- refined (CPU r3 refiner)
def refined_work(job):
    import analysis_v69_refined_probe as P
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = P.RT, P.RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    gt = {}
    for image_id in sorted(rt["split"][4][name]):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        gt[image_id] = [r for r in rl if mu.area(r) > 0]
    gts = list(gt.values())

    def run(props, scores, grid):
        if not props:
            return K.score_file([], [], gts, grid)
        masks = [mu.decode(r) for r in props]
        bounds, _, probs = core.crop_probabilities(rt["base"], rt["model"], gray, masks, True)
        rles, fs = core.clean_refined_predictions(masks, list(scores), bounds, probs, 0.45,
                                                  P.v50.MODE)
        return K.score_file(rles, fs, gts, grid)

    views = K.read_cache_views(E.cache_path("p2first", fold, name))
    groups = F.groups_of(views)
    sc = [sum(g["scores"]) / len(views) for g in groups]
    order = sorted(range(len(groups)), key=lambda i: -sc[i])
    res = {"base": run([groups[i]["rle"] for i in order], [sc[i] for i in order],
                       P.v50.CONFIDENCES)}
    rows = SC[SC.file == name]
    for scheme in ("v61", "date"):
        r, s = select(name, rows, scheme)
        res[scheme] = run(r, s, GRID)
    return name, {k: {c: v.tolist() for c, v in d.items()} for k, d in res.items()}


def cmd_refined():
    import analysis_v69_refined_probe as P
    import pandas as pd
    from multiprocessing import set_start_method
    global SC
    set_start_method("fork")
    SC = pd.read_pickle(OUT / "scores.pkl")
    ref = json.loads((E.ROOT / "outputs/v63/r3.json").read_text())["per_file"]
    fmap = {n: f for f, n in jobs()}
    names = sorted(ref)[: int(os.environ.get("V70_LIMIT", len(ref)))]
    with Pool(PROCS, initializer=P.init) as p:
        per = dict(p.map(refined_work, [(n, fmap[n]) for n in names], chunksize=1))
    for n in names:
        assert np.allclose(per[n]["base"][0.30], ref[n]["t0.45_c0.30"]), n
    base_fixed = np.asarray([per[n]["base"][0.30] for n in names])
    base_cf, base_cuts = crossfit({n: per[n]["base"] for n in names}, names, fmap,
                                  P.v50.CONFIDENCES)
    dmap = date_folds([n for _, n in jobs()])
    rep = {"files": len(names), "r3_fixed_v17_cell": summary(base_fixed),
           "r3_crossfit": dict(summary(base_cf), cuts=base_cuts)}
    for scheme, fo in (("v61", fmap), ("date", dmap)):
        arr, cuts = crossfit({n: per[n][scheme] for n in names}, names, fo, GRID)
        fo_arr = np.asarray([fmap[n] for n in names])
        rep[f"rescorer_{scheme}"] = dict(
            summary(arr), cuts=cuts, vs_r3_crossfit=boot(base_cf, arr),
            vs_r3_fixed=boot(base_fixed, arr),
            per_v61_fold_delta={int(k): K.pq_value(arr[fo_arr == k].sum(0)) -
                                K.pq_value(base_cf[fo_arr == k].sum(0))
                                for k in sorted(set(fo_arr))})
        rep[f"rescorer_{scheme}_grid"] = {
            str(c): K.pq_value(np.sum([per[n][scheme][c] for n in names], 0)) for c in GRID}
    g = rep["rescorer_v61"]["vs_r3_crossfit"]
    rep["verdict"] = ("CONTINUE" if g["delta"] >= 0.010 and g["p_gt_0"] >= 0.90 and
                      g["ci95"][0] > 0 else "KILL" if g["delta"] < 0.005 else "INCONCLUSIVE")
    (OUT / "refined.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("grid")}, indent=1,
                     default=float))


if __name__ == "__main__":
    {"features": cmd_features, "train": cmd_train, "direct": cmd_direct,
     "refined": cmd_refined}[sys.argv[1]]()
