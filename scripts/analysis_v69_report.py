"""V69 error analysis tables from outputs/v69_analysis/<recipe>/*.pkl (CPU, seconds)."""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import v61_toolkit as K  # noqa: E402

ROOT = SCRIPTS.parent
RECIPE = os.environ.get("V69_RECIPE", "p2first")
D = ROOT / "outputs/v69_analysis" / RECIPE
gt, fp, files, pairs = (pickle.load(open(D / f"{k}.pkl", "rb"))
                        for k in ("gt", "fp", "files", "pairs"))
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
R = {}


def pq(t):
    return K.pq_value(t)


def show(title, df):
    print(f"\n### {title}\n{df.to_string()}")


tot = np.sum(np.stack(files.counts.values), 0)
iou, tp, nfp, nfn = tot
SQ, RQ = iou / tp, tp / (tp + 0.5 * (nfp + nfn))
R["totals"] = dict(pq=pq(tot), sq=SQ, rq=RQ, tp=tp, fp=nfp, fn=nfn, iou=iou)
assert len(gt) == tp + nfn and int(gt.matched.sum()) == tp and len(fp) == nfp
sc = {
    "now": pq(tot),
    "SQ=0.9 (all TP IoU 0.9)": 0.9 * tp / (tp + 0.5 * (nfp + nfn)),
    "SQ+0.05": (SQ + 0.05) * tp / (tp + 0.5 * (nfp + nfn)),
    "FN halved -> TP at current SQ": pq([iou + SQ * nfn / 2, tp + nfn / 2, nfp, nfn / 2]),
    "FP halved (removed)": pq([iou, tp, nfp / 2, nfn]),
    "FP=0": pq([iou, tp, 0, nfn]),
    "FN=0 (all -> TP at SQ)": pq([iou + SQ * nfn, tp + nfn, nfp, 0]),
}
R["scenarios"] = sc
show("Q1 totals", pd.Series(R["totals"]))
show("Q1 scenarios", pd.Series(sc))
tpi = gt[gt.matched].best_keep_iou
show("TP IoU distribution", pd.cut(tpi, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).value_counts().sort_index())

# ---------- FN categories
fn = gt[~gt.matched].copy()
anyp = np.maximum.reduce([fn.best_keep_iou, fn.best_low_iou, fn.best_raw_iou, fn.best_cons_iou])


def fn_cat(r):
    if r.best_low_iou > 0.5:
        return "a score: owned proposal <cut at IoU>0.5"
    if r.best_cons_iou > 0.5:
        return "b ownership-clipped (consensus >=cut matched)"
    if r.best_raw_iou > 0.5:
        return "c view-only (single flip view >0.5, consensus lost it)"
    if r.best_keep_iou >= 0.3:
        return "d near-miss kept pred IoU 0.3-0.5"
    if max(r.best_low_iou, r.best_raw_iou) >= 0.3:
        return "e near-miss only below cut/in views"
    if max(r.best_keep_iou, r.best_low_iou, r.best_raw_iou) >= 0.05:
        return "f weak overlap 0.05-0.3"
    return "g no proposal (IoU<0.05 everywhere)"


fn["cat_err"] = fn.apply(fn_cat, axis=1)
t = fn.cat_err.value_counts().sort_index()
show("Q2 FN categories", pd.DataFrame({"n": t, "share": t / len(fn)}))
R["fn_cats"] = t.to_dict()
nm = fn[fn.cat_err.str.startswith("d")]
shape = np.select([nm.best_keep_matched_other, nm.best_keep_prec >= 0.8, nm.best_keep_cov >= 0.8],
                  ["pred belongs to another GT (merge)", "pred inside GT (under-seg/fragment)",
                   "pred covers GT (over-seg/too fat)"], "boundary disagreement")
show("near-miss (d) shape", pd.Series(shape).value_counts())
R["nearmiss_shape"] = pd.Series(shape).value_counts().to_dict()

# ---------- FN rate by GT properties
gt["elong"] = gt.length ** 2 / gt.area.clip(lower=1)
gt["width"] = gt.area / gt.length.clip(lower=1)


def by(col, bins=None, q=None, labels=None):
    s = gt[col]
    g = pd.qcut(s, q, duplicates="drop") if q else (pd.cut(s, bins) if bins else s)
    a = gt.groupby(g, observed=True).agg(n=("matched", "size"), recall=("matched", "mean"),
                                         n_fn=("matched", lambda x: int((~x).sum())))
    a["fn_share"] = a.n_fn / (~gt.matched).sum()
    a["mIoU_tp"] = gt[gt.matched].groupby(g[gt.matched], observed=True).best_keep_iou.mean()
    return a.round(3)


for col, kw in [("area", dict(bins=[0, 500, 1000, 2000, 4000, 8000, 1e7])),
                ("width", dict(q=5)), ("elong", dict(q=5)), ("contrast", dict(q=5)),
                ("rdist", dict(bins=[0, 0.5, 0.7, 0.85, 0.95, 1.2])),
                ("n_rec", {}), ("cat", {}), ("n_other_agree", {}),
                ("n_gt_rec", dict(bins=[0, 5, 10, 15, 25, 100]))]:
    t = by(col, **kw)
    R[f"fn_by_{col}"] = {str(k): v for k, v in t.to_dict("index").items()}
    show(f"Q2 recall by {col}", t)
show("FN categories x area", pd.crosstab(pd.cut(fn.area, [0, 1000, 4000, 1e7]), fn.cat_err))
show("FN categories x contrast quintile",
     pd.crosstab(pd.qcut(gt.contrast, 5)[~gt.matched], fn.cat_err))

# ---------- FP categories


def fp_cat(r):
    if r.n_gt_cov30 >= 2:
        return "a merge (covers >=30% of >=2 GTs)"
    if r.best_iou >= 0.3:
        return "b near-miss IoU 0.3-0.5"
    if r.max_inside >= 0.5 and r.inside_gt_matched:
        return "c fragment/duplicate of a MATCHED GT"
    if r.max_inside >= 0.5:
        return "d fragment of an UNMATCHED GT"
    if r.any_overlap:
        return "e weak overlap with GT"
    return "f no overlap with this record's GT"


fp["cat_err"] = fp.apply(fp_cat, axis=1)
t = pd.crosstab(fp.cat_err, fp.other_rec_match, margins=True)
show("Q2 FP categories x matches GT of ANOTHER record (IoU>0.5)", t)
R["fp_cats"] = fp.cat_err.value_counts().sort_index().to_dict()
R["fp_other_rec_match"] = int(fp.other_rec_match.sum())
hal = fp[fp.cat_err.str.startswith("f")]
R["halluc_other_overlap"] = int(hal.other_rec_overlap.sum())
show("pure no-overlap FP: overlaps another record's GT?", hal.other_rec_overlap.value_counts())
show("FP by score", pd.cut(fp.score, [0.35, 0.45, 0.55, 0.65, 0.8, 1.01]).value_counts().sort_index())
show("TP score", pd.cut(gt.match_score.dropna(), [0.35, 0.45, 0.55, 0.65, 0.8, 1.01])
     .value_counts().sort_index())
show("FP by rdist", pd.cut(fp.rdist, [0, 0.5, 0.7, 0.85, 0.95, 1.2]).value_counts().sort_index())
show("FP rate by n_rec (FP per record)", fp.groupby("n_rec").size() /
     files.groupby("n_rec").n_rec.sum())

# ---------- Q3 annotator disagreement
multi = files[files.n_rec >= 2]
pair_tot = np.zeros(4)
best_or, mean_or = np.zeros(4), np.zeros(4)
for name in multi.file:
    ps = pairs[name]
    n_rec = files.set_index("file").loc[name, "n_rec"]
    ngt = [len(gt[(gt.file == name) & (gt.rec == r)]) for r in range(n_rec)]
    per_pred = []
    for j in range(n_rec):
        c = np.array([ngt[j], ngt[j], 0, 0], float)
        for a, b, cnt in ps:
            if a == j:
                c += np.asarray(cnt, float)
        per_pred.append(c)
    for _, _, cnt in ps:
        pair_tot += np.asarray(cnt, float)
    per_pred = np.asarray(per_pred)
    best_or += per_pred[int(np.argmax([pq(c) for c in per_pred]))]
    mean_or += per_pred.mean(0)
model_multi = np.sum(np.stack(multi.counts.values), 0)
R["interannotator"] = dict(
    n_multi_files=len(multi), n_pairs=int(sum(len(pairs[n]) for n in multi.file)),
    pairwise_pq=pq(pair_tot), pairwise_counts=pair_tot.tolist(),
    pairwise_sq=pair_tot[0] / pair_tot[1],
    record_as_pred_mean_pq=pq(mean_or), record_as_pred_best_pq=pq(best_or),
    model_pq_same_files=pq(model_multi), model_counts=model_multi.tolist(),
    model_pq_single_record_files=pq(np.sum(np.stack(files[files.n_rec == 1].counts.values), 0)))
show("Q3 inter-annotator", pd.Series(R["interannotator"]))
g2 = gt[gt.n_rec >= 2]
t = g2.groupby(g2.n_other_agree.clip(upper=1)).agg(n=("matched", "size"),
                                                    recall=("matched", "mean"))
show("multi-record GT: agreed by >=1 other record?", t)
R["agree_recall"] = t.to_dict("index")
# per record-id PQ
rows = []
for _, f in files.iterrows():
    for rid, c in zip(f.rec_ids, f.rec_counts):
        rows.append(dict(rec_id=rid, grp=rid[:2], **dict(zip(["iou", "tp", "fp", "fn"], c))))
rc = pd.DataFrame(rows)
for key in ("grp", "rec_id"):
    a = rc.groupby(key)[["iou", "tp", "fp", "fn"]].sum()
    a["n_rec"] = rc.groupby(key).size()
    a["pq"] = a.apply(lambda r: pq(r[["iou", "tp", "fp", "fn"]].values), axis=1)
    a["gt_per_rec"] = (a.tp + a.fn) / a.n_rec
    a["fp_per_rec"] = a.fp / a.n_rec
    a = a.sort_values("n_rec", ascending=False)
    show(f"PQ by annotator-record id ({key})", a.round(3).head(40))
    R[f"pq_by_{key}"] = a.round(4).to_dict("index")

# ---------- Q4 fragmentation/merging
R["split_gt"] = dict(gt_ge2_frag=int((gt.n_frag >= 2).sum()),
                     gt_ge2_frag_fn=int(((gt.n_frag >= 2) & ~gt.matched).sum()),
                     fn_absorbed_by_other_gt_pred=int((fn.best_keep_matched_other &
                                                       (fn.best_keep_cov >= 0.5)).sum()),
                     fp_merge=int(fp.cat_err.str.startswith("a").sum()),
                     fp_frag_matched=int(fp.cat_err.str.startswith("c").sum()),
                     fp_frag_unmatched=int(fp.cat_err.str.startswith("d").sum()))
show("Q4 split/merge", pd.Series(R["split_gt"]))

# ---------- Q5 per-file
cnt = np.stack(files.counts.values)
files["pq"] = [pq(c) for c in cnt]
files["fp"], files["fn"], files["tp"], files["iou"] = cnt[:, 2], cnt[:, 3], cnt[:, 1], cnt[:, 0]
files["n_gt"] = files.tp + files.fn
gain = []
for i in range(len(files)):
    c = cnt.copy()
    c[i] = [files.n_gt[i], files.n_gt[i], 0, 0]
    gain.append(pq(c.sum(0)) - pq(tot))
files["gain_if_perfect"] = gain
files["err"] = files.fp + files.fn
s = files.sort_values("gain_if_perfect", ascending=False)
cum = s.gain_if_perfect.cumsum() / s.gain_if_perfect.sum()
R["concentration"] = {f"top{p}%": float(cum.iloc[int(len(s) * p / 100) - 1]) for p in (5, 10, 20, 50)}
show("Q5 share of 'perfect-file' gain held by top files", pd.Series(R["concentration"]))
show("worst 15 files", s[["file", "n_rec", "n_gt", "tp", "fp", "fn", "pq", "gain_if_perfect",
                          "disk_std", "sharp"]].head(15).round(3))
files["year"] = files.date.str[:4].astype(int)
for col, kw in [("site", {}), ("year", {}), ("n_rec", {}),
                ("n_gt_mean", dict(bins=[-1, 3, 6, 10, 15, 100])),
                ("disk_std", dict(q=5)), ("sharp", dict(q=5)), ("disk_mean", dict(q=5)),
                ("disk_r", dict(q=4))]:
    s_ = files[col]
    g = pd.qcut(s_, kw["q"], duplicates="drop") if "q" in kw else (
        pd.cut(s_, kw["bins"]) if "bins" in kw else s_)
    a = files.groupby(g, observed=True)[["iou", "tp", "fp", "fn"]].sum()
    a["files"] = files.groupby(g, observed=True).size()
    a["pq"] = a.apply(lambda r: pq(r[["iou", "tp", "fp", "fn"]].values), axis=1)
    a["fp_per_file"] = a.fp / a.files
    a["recall"] = a.tp / (a.tp + a.fn)
    show(f"Q5 pooled PQ by {col}", a.round(3))
    R[f"file_by_{col}"] = {str(k): v for k, v in a.round(4).to_dict("index").items()}
from scipy.stats import spearmanr  # noqa: E402
R["spearman_filepq"] = {c: spearmanr(files[c], files.pq, nan_policy="omit")[0]
                        for c in ("n_gt", "n_rec", "disk_std", "sharp", "disk_mean", "year",
                                  "n_keep")}
show("Spearman(file PQ, x)", pd.Series(R["spearman_filepq"]))

# ---------- refined pipeline (fold 0, r3) vs direct on the same 141 files
r3p = ROOT / "outputs/v65/refined/r3.json"
if r3p.exists():
    r3 = json.loads(r3p.read_text())["per_file"]
    f0 = files[files.file.isin(r3)]
    d0 = np.sum(np.stack(f0.counts.values), 0)
    r0 = np.sum([r3[n]["t0.45_c0.30"] for n in f0.file], 0)
    R["refined_fold0"] = dict(files=len(f0), direct=dict(pq=pq(d0), sq=d0[0] / d0[1],
                                                       counts=d0.tolist()),
                              r3=dict(pq=pq(r0), sq=r0[0] / r0[1], counts=list(r0)))
    show("refined r3 vs direct (fold 0)", pd.DataFrame(R["refined_fold0"]).T)


def default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


(D / "report.json").write_text(json.dumps(R, indent=1, default=default))
fn[["file", "rec", "area", "contrast", "rdist", "cat_err"]].to_csv(D / "fn_rows.csv", index=False)
