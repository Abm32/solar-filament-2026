"""V89 5-fold (CPU): V84 extract + bundle for the SSL-init soft refiner family vs V84 soft (0.45898).
Checkpoints: outputs/v89/five/soft_fold{0..4}_refiner.pt (fold 0 = symlink to the screened ft_<arm>).
  CUDA_VISIBLE_DEVICES= python scripts/v89_fivefold.py extract   # -> outputs/v89/extract_ssl.pkl
  python scripts/v89_fivefold.py compare                          # -> outputs/v89/fivefold.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import v84_extract as X84  # noqa: E402
import v84_bundle as B  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v89"
GATE = dict(delta=0.010, p=0.90)


def compare():
    out, counts = {}, {}
    for fam, p in (("v84_soft", ROOT / "outputs/v84/extract_soft.pkl"), ("ssl", OUT / "extract_ssl.pkl")):
        res = pickle.load(open(p, "rb"))[0.5]
        A = B.arrays(res)
        c, keep, lams = B.run_arm(res, A, B.crossfit(A), 1, 1)
        counts[fam], tt = c, c.sum(0)
        out[fam] = dict(pq=B.V.pq(tt), tp=int(tt[1]), fp=int(tt[2]), fn=int(tt[3]), lambdas=lams,
                        names=[r["name"] for r in res])
    assert out["v84_soft"]["names"] == out["ssl"]["names"]
    assert abs(out["v84_soft"]["pq"] - 0.45898) < 5e-5, out["v84_soft"]["pq"]
    ffold = np.asarray([r["fold"] for r in pickle.load(open(OUT / "extract_ssl.pkl", "rb"))[0.5]])
    b = B.V.boot(counts["v84_soft"], counts["ssl"], seed=89)
    fd = [B.V.pq(counts["ssl"][ffold == k].sum(0)) - B.V.pq(counts["v84_soft"][ffold == k].sum(0)) for k in range(5)]
    for v in out.values():
        v.pop("names")
    rep = dict(files=len(ffold), arms=out, boot=b, fold_delta=fd, gate=GATE,
               verdict="PASS" if b["delta"] >= GATE["delta"] and b["p_gt_0"] >= GATE["p"] else "FAIL")
    (OUT / "fivefold.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        X84.CKPT["ssl"], X84.OUT = "outputs/v89/five/soft_fold{}", OUT
        sys.argv = [sys.argv[0], "ssl"]
        X84.main()
    else:
        compare()
