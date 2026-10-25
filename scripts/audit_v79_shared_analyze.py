"""Analyse labels_raw.pkl -> outputs/v79_audit/labels_report.json (printed too)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "solar-filament-2026"
OUT = ROOT / "outputs/v79_audit"
d = pickle.load(open(OUT / "labels_raw.pkl", "rb"))
rep = {}

df = pd.DataFrame([r for o in d["out"] for r in o[1]])
PATHS = ["yolo1024", "yolo512", "yolo256", "ideal1024", "ideal512", "refiner256"]


def self_pq(x):
    hit = x > 0.5
    return float(x[hit].sum() / (hit.sum() + (~hit).sum()))  # TP + 0.5(FP+FN), FP=FN=misses


rep["fidelity"] = {p: dict(mean_iou=round(df[p].mean(), 4), frac_le_0_5=round((df[p] <= 0.5).mean(), 4),
                           self_pq=round(self_pq(df[p].values), 4)) for p in PATHS}
rep["yolo1024_area_ratio_median"] = float(df.yolo1024_area_ratio.median())
sh = np.array(df.yolo1024_shift.tolist()).mean(0)
rep["yolo1024_shift_mean_iou"] = {f"dx{dx}dy{dy}": round(float(v), 4) for (dx, dy), v in
                                  zip([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], sh)}

bins = [0, 4, 6, 8, 12, 16, 1e9]
df["wbin"] = pd.cut(df.width, bins, right=False)
rep["width_quantiles_2048"] = {q: round(float(df.width.quantile(q)), 2) for q in (0.05, 0.1, 0.25, 0.5, 0.75)}
rep["frac_width_lt"] = {f"<{w}px@2048 (<{w/2}@1024)": round(float((df.width < w).mean()), 4) for w in (4, 6, 8, 12)}
rep["frac_width_dt_lt"] = {f"<{w}px@2048": round(float((df.width_dt_med < w).mean()), 4) for w in (4, 6, 8)}

# join to v69 p2first direct OOF per-GT outcomes (same json order)
g = pd.read_pickle(ROOT / "outputs/v69_analysis/p2first/gt.pkl")
g["ann"] = g.groupby(["file", "rec"]).cumcount()
m = df.merge(g[["file", "rec", "ann", "area", "matched", "n_other_agree"]], on=["file", "rec", "ann"], suffixes=("", "_v69"))
rep["join"] = dict(rows=len(m), area_equal=float((m.area == m.area_v69).mean()))
tab = m.groupby("wbin", observed=True).agg(n=("matched", "size"), recall=("matched", "mean"),
                                           yolo1024=("yolo1024", "mean"), yolo512=("yolo512", "mean"),
                                           ideal1024=("ideal1024", "mean"), refiner256=("refiner256", "mean"))
tab["share_of_FN"] = m[~m.matched].groupby("wbin", observed=True).size() / (~m.matched).sum()
rep["by_width"] = json.loads(tab.round(3).to_json(orient="index"))
mm = m[m.n_rec > 1]
rep["by_width_agreed_only"] = json.loads(mm.groupby("wbin", observed=True).apply(
    lambda x: pd.Series(dict(n=len(x), recall_agreed=x[x.n_other_agree > 0].matched.mean(),
                             recall_single=x[x.n_other_agree == 0].matched.mean(),
                             frac_single=(x.n_other_agree == 0).mean()))).round(3).to_json(orient="index"))

# ---- multi-annotator oracles ----
R_tot = {}
oracle = {k: np.zeros(4) for k in ("self_mean", "self_best", "self_worst", "union", "k>=2", "majority",
                                   "dinkelbach")}
multi = [(name, mo, cross) for name, rows, cross, mo in d["out"] if mo]


def counts_from_pred(pred_ious_per_rec, n_gt):  # pred_ious_per_rec: list over rec of (n_pred, n_gt) arrays
    t = np.zeros(4)
    for x, ng in zip(pred_ious_per_rec, n_gt):
        npred = x.shape[0]
        if not ng or not npred:
            t += (0, 0, npred, ng)
            continue
        h = x > 0.5
        t += (x[h].sum(), h.sum(), (~h.any(1)).sum(), (~h.any(0)).sum())
    return t


def pqv(t):
    return t[0] / (t[1] + 0.5 * (t[2] + t[3]))


cand_rows = []
for name, mo, cross in multi:
    R, ng = mo["R"], mo["n_gt"]
    per_r = []
    for r in range(R):
        xs = [cross.get((r, s), np.zeros((ng[r], ng[s]))) for s in range(R)]
        per_r.append(counts_from_pred(xs, ng))
    per_r = np.array(per_r)
    oracle["self_mean"] += per_r.mean(0)
    vals = [pqv(c) if c[1] + c[2] + c[3] else 0 for c in per_r]
    oracle["self_best"] += per_r[int(np.argmax(vals))]
    oracle["self_worst"] += per_r[int(np.argmin(vals))]
    for c in mo["cands"]:
        hits = [float(x[x > 0.5].max()) if len(x) and (x > 0.5).any() else 0.0 for x in c["ious"]]
        cand_rows.append(dict(file=name, R=R, k=c["k"], s=sum(hits), h=sum(h > 0 for h in hits)))
    for key, rule in (("union", lambda c: True), ("k>=2", lambda c: c["k"] >= 2), ("majority", lambda c: c["k"] > R / 2)):
        keep = [c for c in mo["cands"] if rule(c)]
        xs = [np.array([c["ious"][r] for c in keep]).reshape(len(keep), ng[r]) for r in range(R)]
        oracle[key] += counts_from_pred(xs, ng)

cr = pd.DataFrame(cand_rows)
NG = sum(sum(mo["n_gt"]) for _, mo, _ in multi)
# Dinkelbach: keep cand iff s/R > tau, tau = PQ/2 (exact: candidates are disjoint clusters)
tau = 0.2
for _ in range(30):
    keep = cr[cr.s / cr.R > tau]
    pq = keep.s.sum() / (0.5 * keep.R.sum() + 0.5 * NG)
    tau = pq / 2
oracle_d = dict(pq=round(float(pq), 4), tau=round(float(tau), 4), kept=int(len(keep)), cands=int(len(cr)))
rep["multi_record_files"] = len(multi)
rep["oracles_multi_record_files"] = {k: round(float(pqv(v)), 4) for k, v in oracle.items() if k != "dinkelbach"}
rep["oracle_dinkelbach_vote_masks"] = oracle_d
rep["oracle_rq_sq"] = {k: dict(rq=round(float(v[1] / (v[1] + 0.5 * (v[2] + v[3]))), 4), sq=round(float(v[0] / v[1]), 4)) for k, v in oracle.items() if k != "dinkelbach"}
kh = keep[keep.h > 0]
rep["oracle_dinkelbach_sq"] = round(float(kh.s.sum() / kh.h.sum()), 4)
rep["oracle_dinkelbach_rq"] = round(float(kh.h.sum() / (kh.h.sum() + 0.5 * ((keep.R - keep.h).sum() + NG - kh.h.sum()))), 4)
rep["cluster_support"] = json.loads(cr.groupby(["R", "k"]).size().to_json())
# model on same subset, from v69 files.pkl
f = pd.read_pickle(ROOT / "outputs/v69_analysis/p2first/files.pkl")
c = np.array(f[f.n_rec > 1].counts.tolist()).sum(0)
rep["model_direct_p2first_same_files"] = round(float(pqv(c)), 4)
c1 = np.array(f[f.n_rec == 1].counts.tolist()).sum(0)
rep["model_direct_single_record_files"] = round(float(pqv(c1)), 4)

# ---- annotator style ----
rec_ids = sorted(df.rec_id.unique())
st = df.groupby("rec_id").agg(n_gt=("area", "size"), med_area=("area", "median"), med_width=("width", "median"),
                              med_len=("skel", "median"))
st["records"] = df.groupby("rec_id").apply(lambda x: x[["file", "rec"]].drop_duplicates().shape[0])
st["gt_per_rec"] = st.n_gt / st.records
st["recall_direct"] = m.groupby("rec_id").matched.mean()
# agreement: fraction of this annotator's GT matched (>0.5) in the other records of the file
agree, merge_ratio = {}, {}
for name, rows, cross, mo in d["out"]:
    if not mo:
        continue
    ids = {r["rec"]: r["rec_id"] for r in rows}
    for (i, j), x in cross.items():
        if i == j or i not in ids:
            continue
        a = agree.setdefault(ids[i], [0, 0, 0])
        a[0] += int((x > 0.5).any(1).sum())
        a[1] += x.shape[0]
        a[2] += int(((x > 0.1).sum(1) >= 2).sum())  # one GT overlapping >=2 of the other's: coarser
st["agree_frac"] = pd.Series({k: v[0] / v[1] for k, v in agree.items()})
st["merges_per_gt"] = pd.Series({k: v[2] / v[1] for k, v in agree.items()})
st["prefix"] = [s[:2] for s in st.index]
rep["annotators"] = json.loads(st.round(3).to_json(orient="index"))
rep["prefix"] = json.loads(st.groupby("prefix").apply(lambda x: pd.Series(dict(
    records=x.records.sum(), gt_per_rec=x.n_gt.sum() / x.records.sum(), med_width=np.median(x.med_width),
    recall_direct=(x.recall_direct * x.n_gt).sum() / x.n_gt.sum(), agree=x.agree_frac.mean()))).round(3).to_json(orient="index"))

# annotator date ranges vs test dates
meta = d["meta"]
ad = {}
for name, recs in meta.items():
    for r in recs:
        ad.setdefault(r["rec_id"][:2], []).append(name[:8])
test = sorted(p.name[:8] for p in (ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images").iterdir())
rep["prefix_date_range"] = {k: [min(v), max(v), len(v)] for k, v in sorted(ad.items())}
rep["test_date_range"] = [test[0], test[-1], len(test)]
tr_years = pd.Series([n[:4] for n in meta]).value_counts().sort_index()
te_years = pd.Series([t[:4] for t in test]).value_counts().sort_index()
rep["years_train_vs_test"] = pd.concat([tr_years, te_years], axis=1, keys=["train", "test"]).fillna(0).astype(int).to_dict(orient="index")
rep["prefix_by_year"] = pd.crosstab(pd.Series([n[:4] for k, v in ad.items() for n in v]),
                                    pd.Series([k for k, v in ad.items() for n in v])).to_dict(orient="index")

(OUT / "labels_report.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps(rep, indent=1, default=str))
