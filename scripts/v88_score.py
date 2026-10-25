"""V88 stage 2 (CPU): score every ownership arm with the V84 rule (cut 0.30, MIN200, demote refit cross-fitted on
the arm's own features), C's x cross-fitted by fold, paired file bootstrap vs control G.
Reads outputs/v88/extract.pkl, writes outputs/v88/result.json.
  python scripts/v88_score.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import v84_bundle as B  # noqa: E402
XS = (0.1, 0.2, 0.3, 0.5, 0.7)  # == v88_extract.XS

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v88"
GATE = dict(delta=0.005, p=0.95, folds=4)
PRIMARY = ["A_argmax", "B_pin", "B_yhat", "C_cf", "D_small"]

data = pickle.load(open(OUT / (sys.argv[1] if len(sys.argv) > 1 else "extract.pkl"), "rb"))
ffold = np.asarray([r["fold"] for r in data["G"]])
counts, out = {}, {"n_files": len(ffold), "arms": {}}
for arm, res in data.items():
    A = B.arrays(res)
    c, keep, lams = B.run_arm(res, A, B.crossfit(A), 1, 1)
    counts[arm] = c
    print(arm, round(B.V.pq(c.sum(0)), 5), flush=True)
pq = B.V.pq
assert len(ffold) < 707 or abs(pq(counts["G"].sum(0)) - 0.45898) < 5e-5, pq(counts["G"].sum(0))

cs = [f"C_{x}" for x in XS]
xk, cf = [], np.zeros_like(counts["G"])
for k in range(5):
    j = int(np.argmax([pq(counts[a][ffold != k].sum(0)) for a in cs]))
    xk.append(XS[j])
    cf[ffold == k] = counts[cs[j]][ffold == k]
counts["C_cf"] = cf
out["C_cf_x_per_fold"] = xk

b = counts["G"]
for arm, c in counts.items():
    t = c.sum(0)
    a = dict(pq=pq(t), sq=float(t[0] / max(t[1], 1)), tp=int(t[1]), fp=int(t[2]), fn=int(t[3]))
    if arm != "G":
        a["boot"] = B.V.boot(b, c, seed=88)
        a["fold_delta"] = [pq(c[ffold == k].sum(0)) - pq(b[ffold == k].sum(0)) for k in range(5)]
        a["pass"] = bool(a["boot"]["delta"] >= GATE["delta"] and a["boot"]["p_gt_0"] >= GATE["p"]
                         and sum(d > 0 for d in a["fold_delta"]) >= GATE["folds"])
    out["arms"][arm] = a
    print(arm, json.dumps(a), flush=True)
cand = max(PRIMARY, key=lambda a: out["arms"][a]["boot"]["delta"])
out["candidate"] = cand
out["verdict"] = "CONTINUE" if out["arms"][cand]["pass"] else "KEEP GREEDY"
(OUT / ("result.json" if len(sys.argv) == 1 else "result_smoke.json")).write_text(json.dumps(out, indent=1))
print("candidate", cand, "verdict", out["verdict"])
