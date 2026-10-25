"""V80 forensics stage 2: categorise missed AGREED GT instances, PQ gains, cut-sweep FP cost, images.
Reads outputs/v80_forensics/extract.pkl -> report.json (printed)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path.home() / "solar-filament-2026/outputs/v80_forensics"
res = pickle.load(open(OUT / "extract.pkl", "rb"))
BASE = np.sum([r["tot"] for r in res], 0)


def pq(t):
    return float(t[0] / (t[1] + 0.5 * (t[2] + t[3])))


P0, SQ = pq(BASE), BASE[0] / BASE[1]
gt = pd.DataFrame([g for r in res for g in r["gt"]])
for c in ("raw", "g_rep", "g_score", "g_views", "g_mean", "g_pre", "g_post", "post_any", "pre_any",
          "frag_union", "frag_n", "absorb_fp"):
    gt[c] = gt[c].astype(float).fillna(0.0) if c in gt else 0.0
for c in ("merge_nbr", "oversize", "best_unmatched"):
    gt[c] = gt[c].fillna(False).astype(bool) if c in gt else False
gt["site"] = gt.file.str[-7]
gt["year"] = gt.file.str[:4].astype(int)
gt["agreed"] = gt.agree >= 1
rep = dict(base=dict(pq=P0, sq=SQ, counts=BASE.tolist(), files=len(res)),
           instances=dict(total=len(gt), multi_rec=int((gt.n_rec >= 2).sum()),
                          agreed=int(gt.agreed.sum()), agreed_recall=float(gt[gt.agreed].matched.mean()),
                          single_multi_recall=float(gt[(gt.n_rec >= 2) & ~gt.agreed].matched.mean())))
m = gt[gt.agreed & ~gt.matched].copy()
d_flag = (m.frag_union > 0.5) | m.merge_nbr
cat = np.select(
    [m.raw <= 0.5,
     m.g_rep <= 0.5,                                   # good raw member absorbed by a worse anchor
     (m.g_score < 0.30) & (m.g_mean >= 0.30),
     m.g_score < 0.30,
     m.g_pre <= 0.5],
    ["A", "D_grouping", "b_votes", "c_conf", "f_refiner"], "e_ownership")
sub = np.select([d_flag, m.final >= 0.3], ["d_merge_frag", "g_boundary"], "a_no_proposal")
m["cat"] = np.where(cat == "A", sub, cat)
m.loc[m.cat == "D_grouping", "cat"] = "d_grouping"
rep["missed_agreed"] = dict(n=len(m), n_files=int(m.file.nunique()),
                            unique_filaments_approx=float((1 / (m.agree + 1)).sum()))


def est_iou(r):
    if r["cat"] in ("b_votes", "c_conf"):
        return r.g_pre if r.g_pre > 0.5 else r.g_rep
    if r["cat"] == "e_ownership":
        return r.g_pre
    if r["cat"] == "f_refiner":
        return r.g_rep
    if r["cat"] == "d_merge_frag" and r.frag_union > 0.5:
        return r.frag_union
    return SQ


m["est"] = m.apply(est_iou, axis=1)
m["fp_absorb"] = np.where(m.cat.isin(["g_boundary", "f_refiner", "d_merge_frag"]), m.absorb_fp, 0)
cats = {}
for c, x in m.groupby("cat"):
    t = BASE + [x.est.sum(), len(x), -x.fp_absorb.sum(), -len(x)]
    cats[c] = dict(n=len(x), share=round(len(x) / len(m), 3), pq_if_fixed=round(pq(t), 4),
                   gain=round(pq(t) - P0, 4), est_iou=round(float(x.est.mean()), 3),
                   final_iou_mean=round(float(x.final.mean()), 3), raw_mean=round(float(x.raw.mean()), 3))
rep["categories"] = dict(sorted(cats.items(), key=lambda kv: -kv[1]["n"]))
t = BASE + [m.est.sum(), len(m), -m.fp_absorb.sum(), -len(m)]
rep["all_missed_agreed_fixed"] = dict(pq=pq(t), gain=pq(t) - P0)
rep["overlays"] = dict(final_0p3_0p5=int((m.final >= 0.3).sum()), post_any_gt0p5=int((m.post_any > 0.5).sum()),
                       pre_any_gt0p5=int((m.pre_any > 0.5).sum()), raw_gt0p5=int((m.raw > 0.5).sum()),
                       frag_union_gt0p5=int((m.frag_union > 0.5).sum()), merge_nbr=int(m.merge_nbr.sum()),
                       oversize=int(m.oversize.sum()))
bc = m[m.cat.isin(["b_votes", "c_conf"])]
rep["bc_detail"] = dict(views_hist=bc.g_views.value_counts().sort_index().to_dict(),
                        score_q=bc.g_score.quantile([.1, .25, .5, .75, .9]).round(3).to_dict(),
                        mean_member_q=bc.g_mean.quantile([.1, .5, .9]).round(3).to_dict())
f = m[m.cat == "f_refiner"]
rep["f_detail"] = dict(rep_iou_mean=float(f.g_rep.mean()), pre_iou_mean=float(f.g_pre.mean()),
                       pre_lt_0p3=int((f.g_pre < 0.3).sum()))

# ---- (a) breakdown vs all agreed instances ----
ag = gt[gt.agreed].copy()
ag["is_a"] = False
ag.loc[m.index[m.cat == "a_no_proposal"], "is_a"] = True
ag["miss"] = ~ag.matched
feat_bins = dict(width=[0, 5, 6, 8, 10, 12, 16, 1e9], length=[0, 30, 60, 100, 200, 400, 1e9],
                 area=[0, 300, 600, 1200, 2500, 5000, 1e9], rnorm=[0, .5, .7, .8, .9, .95, 1.0, 9],
                 contrast=[-9, 0.02, 0.05, 0.08, 0.12, 0.18, 9])
rep["a_breakdown"] = {}
for k, b in feat_bins.items():
    tab = ag.groupby(pd.cut(ag[k], b, right=False), observed=True).agg(
        n=("matched", "size"), recall=("matched", "mean"), a_rate=("is_a", "mean"), a_n=("is_a", "sum"))
    tab["a_share"] = tab.a_n / ag.is_a.sum()
    rep["a_breakdown"][k] = {str(i): {c: round(float(v), 3) for c, v in r.items()} for i, r in tab.iterrows()}
rep["a_vs_matched_median"] = {k: dict(a=round(float(ag[ag.is_a][k].median()), 3),
                                      matched=round(float(ag[ag.matched][k].median()), 3))
                              for k in feat_bins}

# ---- cut sweep / extra proposals ----
pr = pd.DataFrame([dict(p, file=r["name"]) for r in res for p in r["preds"]])
ext = pr[pr.score < 0.30].copy()
ext["cls"] = np.select([ext.m_agreed > 0, ext.m_single > 0], ["agreed", "single"], "none")
sweep = {}
for c in (0.05, 0.10, 0.15, 0.20, 0.225, 0.25, 0.275):
    x = ext[ext.score >= c]
    t = BASE + x[["iou_sum", "tp", "fp", "fn"]].sum().values
    sweep[f"{c:.3f}"] = dict(pq=round(pq(t), 5), delta=round(pq(t) - P0, 5), extra_preds=len(x),
                             rec_tp=int(x.tp.sum()), rec_fp=int(x.fp.sum()),
                             rec_precision=round(float(x.tp.sum() / max(x.tp.sum() + x.fp.sum(), 1)), 3),
                             pred_cls=x.cls.value_counts().to_dict())
rep["cut_sweep"] = sweep
ext["sbin"] = pd.cut(ext.score, [0, .1, .15, .2, .25, .3], right=False)
cells = {}
for (v, s), x in ext.groupby(["n_views", "sbin"], observed=True):
    t = BASE + x[["iou_sum", "tp", "fp", "fn"]].sum().values
    cells[f"v{v}_{s}"] = dict(n=len(x), agreed=int((x.cls == "agreed").sum()), single=int((x.cls == "single").sum()),
                              none=int((x.cls == "none").sum()),
                              rec_prec=round(float(x.tp.sum() / max(x.tp.sum() + x.fp.sum(), 1)), 3),
                              dpq=round(pq(t) - P0, 5))
rep["extra_by_views_score"] = cells
for name, x in (("kept_ge_0.30", pr[pr.score >= 0.30]),):
    cl = np.select([x.m_agreed > 0, x.m_single > 0], ["agreed", "single"], "none")
    rep[name] = dict(n=len(x), cls=pd.Series(cl).value_counts().to_dict(),
                     rec_prec=round(float(x.tp.sum() / (x.tp.sum() + x.fp.sum())), 3))
# oracle: add only extras that hit agreed GT (ceiling of a perfect re-scorer below the cut)
x = ext[ext.cls == "agreed"]
rep["oracle_rescore_agreed_extras"] = dict(n=len(x), gain=pq(BASE + x[["iou_sum", "tp", "fp", "fn"]].sum().values) - P0)

# ---- image concentration ----
img = ag.groupby("file").agg(n=("matched", "size"), miss=("miss", "sum"), site=("site", "first"),
                             year=("year", "first"), contrast=("contrast", "median"))
dk = pd.DataFrame([dict(file=r["name"], **r["disk"]) for r in res]).set_index("file")
img = img.join(dk)
s = img.miss.sort_values(ascending=False)
rep["images"] = dict(files_with_agreed=len(img), files_with_miss=int((img.miss > 0).sum()),
                     top10pct_files_share=round(float(s.iloc[:len(s) // 10].sum() / s.sum()), 3),
                     top20pct_files_share=round(float(s.iloc[:len(s) // 5].sum() / s.sum()), 3),
                     by_site=img.groupby("site").apply(lambda x: dict(files=len(x), n=int(x.n.sum()),
                                                                     recall=round(1 - x.miss.sum() / x.n.sum(), 3))).to_dict(),
                     by_year=img.groupby("year").apply(lambda x: dict(n=int(x.n.sum()),
                                                                     recall=round(1 - x.miss.sum() / x.n.sum(), 3))).to_dict())
for k in ("disk_std", "lowfreq_cv", "disk_mean", "contrast", "disk_frac"):
    q = pd.qcut(img[k], 5, duplicates="drop")
    rep["images"][f"recall_by_{k}_quintile"] = {str(i): round(float(1 - x.miss.sum() / x.n.sum()), 3)
                                                for i, x in img.groupby(q, observed=True)}
rep["images"]["worst10"] = [dict(file=f, n=int(r.n), miss=int(r.miss), site=r.site,
                                 lowfreq_cv=round(r.lowfreq_cv, 3), contrast=round(r.contrast, 3))
                            for f, r in img.sort_values("miss", ascending=False).head(10).iterrows()]
rep["images"]["miss_rate_by_cat_site"] = pd.crosstab(m.site, m.cat).to_dict()
(OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps(rep, indent=1, default=str))
