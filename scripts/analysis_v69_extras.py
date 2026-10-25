"""V69 extras: lever what-ifs, view-only FN breakdown, leave-one-annotator-out human ceiling."""
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
import analysis_v69_errors as A  # noqa: E402
import v61_toolkit as K  # noqa: E402

RECIPE = os.environ.get("V69_RECIPE", "p2first")
D = A.E.ROOT / "outputs/v69_analysis" / RECIPE
gt, fp, files = (pickle.load(open(D / f"{k}.pkl", "rb")) for k in ("gt", "fp", "files"))
tot = np.sum(np.stack(files.counts.values), 0)
iou, tp, nfp, nfn = tot
pq = K.pq_value
R = {"now": pq(tot)}
fn = gt[~gt.matched]
a = fn[fn.best_low_iou > 0.5]
b = fn[(fn.best_low_iou <= 0.5) & (fn.best_cons_iou > 0.5)]
c = fn[(fn.best_low_iou <= 0.5) & (fn.best_cons_iou <= 0.5) & (fn.best_raw_iou > 0.5)]
rest = fn[(fn.best_low_iou <= 0.5) & (fn.best_cons_iou <= 0.5) & (fn.best_raw_iou <= 0.5)]
d = rest[rest.best_keep_iou >= 0.3]
d_free = d[~d.best_keep_matched_other]   # its best kept pred is itself an FP in this record


def add_tp(t, n, s, fp_drop=0):
    return pq([t[0] + s, t[1] + n, t[2] - fp_drop, t[3] - n])


R["W_a_score_oracle (owned <cut proposals promoted, no new FP)"] = pq(
    tot + [a.best_low_iou.sum(), len(a), 0, -len(a)])
R["W_a_half"] = pq(tot + [a.best_low_iou.sum() / 2, len(a) / 2, 0, -len(a) / 2])
R["W_abc_pool_oracle (any proposal >0.5 picked, no new FP)"] = pq(
    tot + [a.best_low_iou.sum() + b.best_cons_iou.sum() + c.best_raw_iou.sum(),
           len(a) + len(b) + len(c), 0, -(len(a) + len(b) + len(c))])
R["W_d_nearmiss_fixed_to_0.55 (kept near-miss masks repaired)"] = add_tp(
    tot, len(d), 0.55 * len(d), fp_drop=len(d_free))
R["W_fp_hallucination_removed (no overlap any record + frag of matched)"] = pq(
    tot - [0, 0, int(((fp.n_gt_cov30 < 2) & (fp.best_iou < 0.3) & ~fp.any_overlap &
                      ~fp.other_rec_overlap).sum()), 0])
R["W_fp_all_avoidable_removed (FP not matching another record)"] = pq(
    tot - [0, 0, int((~fp.other_rec_match).sum()), 0])
R["n"] = dict(a=len(a), b=len(b), c=len(c), d=len(d), d_free=len(d_free))
# view-only breakdown
cc = pd.DataFrame(dict(raw_score=pd.cut(c.best_raw_score, [0, 0.2, 0.3, 0.4, 0.6, 1.01]),
                       kept_nm=c.best_keep_iou >= 0.3, low_nm=c.best_low_iou >= 0.3))
R["c_raw_score"] = {str(k): int(v) for k, v in cc.raw_score.value_counts().sort_index().items()}
R["c_has_kept_iou>=0.3"] = int(cc.kept_nm.sum())
R["c_has_owned_low_iou>=0.3"] = int(cc.low_nm.sum())
R["a_low_iou_mean"] = float(a.best_low_iou.mean())

# leave-one-annotator-out human ceiling on multi-record files
recs = A.load_records()
multi = files[files.n_rec >= 2]
hum_u, hum_ag, mod = np.zeros(4), np.zeros(4), np.zeros(4)
mod3, hum_u3, hum_ag3 = np.zeros(4), np.zeros(4), np.zeros(4)
for _, f in multi.iterrows():
    gts = [[it["rle"] for it in r["items"]] for r in recs[f.file]]
    for r in range(len(gts)):
        others = [gts[j] for j in range(len(gts)) if j != r]
        union = list(others[0])
        for o in others[1:]:
            for m in o:
                if not union or A.iou_mat([m], union).max() < 0.5:
                    union.append(m)
        if len(others) >= 2:
            agreed = [m for m in others[0]
                      if others[1] and A.iou_mat([m], others[1]).max() > 0.5]
        else:
            agreed = list(others[0])
        cu = K.pq_counts(A.iou_mat(gts[r], union), len(gts[r]), len(union))
        ca = K.pq_counts(A.iou_mat(gts[r], agreed), len(gts[r]), len(agreed))
        cm = np.asarray(f.rec_counts[r], float)
        hum_u += cu
        hum_ag += ca
        mod += cm
        if f.n_rec == 3:
            hum_u3 += cu
            hum_ag3 += ca
            mod3 += cm
R["loo_human"] = dict(
    all_multi=dict(union_others=pq(hum_u), agreed_others=pq(hum_ag), model=pq(mod),
                   union_counts=hum_u.tolist(), agreed_counts=hum_ag.tolist(),
                   model_counts=mod.tolist(), union_sq=hum_u[0] / hum_u[1]),
    three_rec=dict(union_others=pq(hum_u3), agreed_others=pq(hum_ag3), model=pq(mod3),
                   union_counts=hum_u3.tolist(), agreed_counts=hum_ag3.tolist(),
                   model_counts=mod3.tolist()))
print(json.dumps(R, indent=1, default=float))
(D / "extras.json").write_text(json.dumps(R, indent=1, default=float))
