"""V92: OOF PQ of the V84 soft/t0.50/MIN200 path at the probe operating points (conf 0.20 demote off,
conf 0.40 demote on, conf 0.30 both), to pair with the public-board probes. CPU; reads outputs/v84/extract_soft.pkl.
  python scripts/v92_conf_sweep.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import v83_rule as V  # noqa: E402
import v84_bundle as B  # noqa: E402

res = pickle.load(open(B.OUT / "extract_soft.pkl", "rb"))[0.5]
A = B.arrays(res)
pred = B.crossfit(A)
print("min candidate score", float(A["sc"].min()), "n", len(A["sc"]), flush=True)
out = {}
for cut, demote in [(0.20, 0), (0.30, 0), (0.30, 1), (0.40, 0), (0.40, 1)]:
    B.CUT = cut
    c, keep, lams = B.run_arm(res, A, pred, 1, demote)
    t = c.sum(0)
    k = f"c{cut:.2f}_demote{demote}"
    out[k] = dict(pq=V.pq(t), tp=int(t[1]), fp=int(t[2]), fn=int(t[3]), kept=int(keep.sum()))
    print(json.dumps({k: out[k]}), flush=True)
(B.OUT.parent / "v92/conf_sweep.json").write_text(json.dumps(out, indent=1) + "\n")
