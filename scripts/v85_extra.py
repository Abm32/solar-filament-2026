"""V85 error analysis, stage 3 (CPU): exact what-ifs on the V84 OOF kept set + ownership drill-down + LB-gap checks.
Reads outputs/v85_analysis/{extract,rows}.pkl, outputs/v84/extract_soft.pkl; writes outputs/v85_analysis/extra.json.
  .venv/bin/python scripts/v85_extra.py
"""
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v84_bundle as B  # noqa: E402

OUT = ROOT / "outputs/v85_analysis"
ref = pickle.load(open(ROOT / "outputs/v84/extract_soft.pkl", "rb"))[0.5]
A = B.arrays(ref)
_, keep, _ = B.run_arm(ref, A, B.crossfit(A), 1, 1)
start = np.searchsorted(A["fi"], np.arange(len(ref)))
keep_by = {r["name"]: keep[start[f]:start[f] + len(r["rows"])] for f, r in enumerate(ref)}
res = pickle.load(open(OUT / "extract.pkl", "rb"))
rows = pickle.load(open(OUT / "rows.pkl", "rb"))
pq = B.V.pq


def counts(r, kept):  # exact pq_counts semantics on alive indices
    c = np.zeros(4)
    for ri, n in enumerate(r["n_gt"]):
        if not n or not kept:
            c += (0, 0, len(kept), n)
            continue
        I = r["I_post"][ri][:, kept]
        h = I > 0.5
        c += (I[h].sum(), h.sum(), (~h.any(0)).sum(), (~h.any(1)).sum())
    return c


base = {}
for r in res:
    base[r["name"]] = [r["cand"][c] for c in np.flatnonzero(keep_by[r["name"]])]
C0 = np.array([counts(r, base[r["name"]]) for r in res])
T0 = C0.sum(0)
assert abs(pq(T0) - 0.45898) < 5e-5
out = dict(pq=pq(T0))

# 1) add dropped alive candidates (non-overlapping by construction) that are TP-quality for >=1 GT of some record
fn = [g for g in rows["gts"] if not g["tp"]]
drop = defaultdict(set)
for g in fn:
    for a in g.get("drop_alive", []):
        drop[g["file"]].add(a)
byname = {r["name"]: r for r in res}


def add_arm(sel):
    C = C0.copy()
    for i, r in enumerate(res):
        extra = [a for a in drop[r["name"]] if a not in base[r["name"]] and sel(r, a)]
        if extra:
            C[i] = counts(r, base[r["name"]] + extra)
    return C


def frac(r, a):  # fraction of records with a match for alive candidate a
    R = len(r["n_gt"])
    return sum(bool(r["n_gt"][ri]) and (r["I_post"][ri][:, a] > 0.5).any() for ri in range(R)) / R


for name, sel in [("all_dropped_tp_quality", lambda r, a: True),
                  ("dropped_with_match_frac>=0.5", lambda r, a: frac(r, a) >= 0.5),
                  ("dropped_with_match_frac>0.34", lambda r, a: frac(r, a) > 0.34),
                  ("dropped_score<0.30_only", lambda r, a: r["g_score"][r["alive"][a]] < 0.30),
                  ("dropped_score>=0.30_only(demote/min200)", lambda r, a: r["g_score"][r["alive"][a]] >= 0.30)]:
    C = add_arm(sel)
    out[f"add_{name}"] = dict(pq=pq(C.sum(0)), delta=pq(C.sum(0)) - pq(T0),
                              n_added=int(sum(len([a for a in drop[r["name"]] if a not in base[r["name"]] and sel(r, a)]) for r in res)))

# realistic: candidates a score/ranker could actually separate — distribution of their scores vs kept FP scores
dsc = [byname[f]["g_score"][byname[f]["alive"][a]] for f, s in drop.items() for a in s]
out["dropped_score_quantiles"] = np.quantile(dsc, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist()
nk_fp_sc = []  # alive non-kept candidates that match nothing anywhere (what a looser cut also lets in)
for r in res:
    kb = set(base[r["name"]])
    for a in range(len(r["alive"])):
        if a in kb:
            continue
        if not any(r["n_gt"][ri] and (r["I_post"][ri][:, a] > 0.5).any() for ri in range(len(r["n_gt"]))):
            nk_fp_sc.append(r["g_score"][r["alive"][a]])
nk_fp_sc = np.asarray(nk_fp_sc)
out["nonkept_junk_alive"] = dict(n=len(nk_fp_sc), n_score_ge_0_15=int((nk_fp_sc >= 0.15).sum()),
                                 n_score_0_20_0_30=int(((nk_fp_sc >= 0.2) & (nk_fp_sc < 0.3)).sum()))
dsc = np.asarray(dsc)
out["dropped_tp_quality"] = dict(n=len(dsc), n_score_ge_0_15=int((dsc >= 0.15).sum()),
                                 n_score_0_20_0_30=int(((dsc >= 0.2) & (dsc < 0.3)).sum()))

# 2) ownership drill-down: pre-ownership mask is TP-quality, post is not
own = [g for g in fn if g["cat"] == "drop_ownership"]
kind = Counter()
occ = Counter()
for g in own:
    r = byname[g["file"]]
    j = None
    ri = g["rec"]
    # locate GT index by matching features (rows keep order per record)
    recrows = [x for x in rows["gts"] if x["file"] == g["file"] and x["rec"] == ri]
    j = next(i for i, x in enumerate(recrows) if x is g)
    gi = int(np.argmax(r["I_pre"][ri][j]))
    alive_i = r["alive"].index(gi) if gi in r["alive"] else None
    kind["killed(<MIN_AREA after clip)" if alive_i is None else "clipped"] += 1
    if alive_i is not None and alive_i in base[g["file"]]:
        kind["clipped_and_kept"] += 1
    kb = base[g["file"]]
    best_k = float(r["I_post"][ri][j][kb].max()) if kb else 0.0
    occ["kept pred overlaps GT 0.05-0.5" if best_k >= 0.05 else "no kept overlap"] += 1
    occ["agree==0"] += g["agree"] == 0 and g["n_rec"] > 1
    occ["n_rec>1"] += g["n_rec"] > 1
out["ownership"] = dict(n=len(own), kind=dict(kind), occupier=dict(occ),
                        area_median=float(np.median([g["area"] for g in own])))

# 3) LB-gap checks
day = lambda n: n[:8]  # noqa: E731
fold = {r["name"]: r["fold"] for r in res}
dd = defaultdict(list)
for r in res:
    dd[day(r["name"])].append(r["name"])
iso = np.array([not any(fold[m] != fold[r["name"]] for m in dd[day(r["name"])] if m != r["name"]) for r in res])
out["pq_files_without_sameday_in_other_fold"] = dict(n=int(iso.sum()), pq=pq(C0[iso].sum(0)), pq_rest=pq(C0[~iso].sum(0)))
# annotator-cluster resampling: public subset drawn by annotator code (records grouped by code)
recC = defaultdict(lambda: np.zeros(4))
for g in rows["gts"]:
    k = (g["file"], g["rec"])
    recC[k] += (g["iou"], 1, 0, 0) if g["tp"] else (0, 0, 0, 1)
for f in rows["fps"]:
    recC[(f["file"], f["rec"])][2] += 1
code = {}
for r in res:
    for ri, i in enumerate(r["ids"]):
        code[(r["name"], ri)] = i.split("-")[0]
byc = defaultdict(lambda: np.zeros(4))
nrc = Counter()
for k, c in recC.items():
    byc[code[k]] += c
    nrc[code[k]] += 1
codes = [c for c in byc if nrc[c] >= 5]
rng = np.random.default_rng(85)
dr = []
for _ in range(20000):  # ~150 public records from ~6-10 annotator codes
    pick = rng.choice(codes, rng.integers(6, 11), replace=True)
    dr.append(pq(sum(byc[c] for c in pick)))
dr = np.asarray(dr)
out["annotator_cluster_resample"] = dict(std=float(dr.std()), p05=float(np.quantile(dr, 0.05)),
                                         p01=float(np.quantile(dr, 0.01)), min=float(dr.min()),
                                         p_le_0_40=float((dr <= 0.40).mean()))
out["annotator_pq_range"] = {c: pq(byc[c]) for c in sorted(codes, key=lambda c: pq(byc[c]))[:5]}
(OUT / "extra.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
