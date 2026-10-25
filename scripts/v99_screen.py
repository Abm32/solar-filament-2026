"""V99 fold-0 screen: V89's extract and gate, outputs under outputs/v99, with two additions in the gate:
  * veto cells  (`*_veto`): also drop candidates whose refined mask fell back to the seed (`fallback` feature);
  * cut sweep   (`t050_cXX`): cut in CUTS instead of only 0.30 (the d4 arm's consensus scores are on a /8 scale).
Control is always `v84_soft` = outputs/v84/extract_soft.pkl fold 0 (deployed refiner on p2first views).

Set SOLAR_RECIPE (e.g. p2first_1280, p2first_d4) BEFORE running so train_v73_refiner / v84_extract read that
detector's OOF views; FLOOR overrides the candidate score floor (v83_extract.FLOOR = 0.15). Never reads test.
  SOLAR_RECIPE=... PROCS=22 CUDA_VISIBLE_DEVICES= python scripts/v99_screen.py extract TAG CKPT
  python scripts/v99_screen.py gate TAG [TAG...]   -> outputs/v99/screen_TAG.json (and screen.json)
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v89_screen as V89  # noqa: E402

B, V, X84 = V89.B, V89.V, V89.X84
V89.OUT = OUT = V89.ROOT / "outputs/v99"
OUT.mkdir(parents=True, exist_ok=True)
CUTS = (0.15, 0.20, 0.25, 0.30)
FB = V.FEATS.index("fallback")
PRIMARY = os.environ.get("PRIMARY", "t050_c030")
if os.environ.get("FLOOR"):
    X84.X.FLOOR = float(os.environ["FLOOR"])


def counts(res, A, model, lam):
    base = V89.counts(res, A, model, lam)  # t050_c030 and full_v84, exactly V89
    keep = (A["sc"] >= B.CUT) & (A["area"] >= B.MINA)
    full = keep & (model.predict(A["X"]) > lam / 2) if len(A["X"]) else keep
    ok = A["X"][:, FB] == 0 if len(A["X"]) else np.ones(0, bool)
    base["t050_c030_veto"] = V.file_counts(res, (A["sc"] >= B.CUT) & ok, A["fi"])
    base["full_v84_veto"] = V.file_counts(res, full & ok, A["fi"])
    for c in CUTS:
        base[f"t050_c{c:.2f}".replace("0.", "0")] = V.file_counts(res, A["sc"] >= c, A["fi"])
    return base


def cmd_gate(tags):
    soft = pickle.load(open(V89.ROOT / "outputs/v84/extract_soft.pkl", "rb"))
    res5 = soft[0.5]
    A5 = B.arrays(res5)
    pred = B.crossfit(A5)
    ref, _, lams = B.run_arm(res5, A5, pred, 1, 1)
    ffold = np.asarray([r["fold"] for r in res5])
    tr = A5["fold"] != 0
    model, lam = V.fit(A5["X"][tr], A5["y"][tr], A5["w"][tr]), lams[0]
    ctrl = [r for r in res5 if r["fold"] == 0]
    arms = {"v84_soft": {0.5: ctrl, 0.45: [r for r in soft[0.45] if r["fold"] == 0]}}
    for t in tags:
        arms[t] = pickle.load(open(OUT / f"extract_{t}.pkl", "rb"))
    names = [r["name"] for r in ctrl]
    C = {}
    for a, by_t in arms.items():
        assert [r["name"] for r in by_t[0.5]] == names, a
        C[a] = counts(by_t[0.5], B.arrays(by_t[0.5]), model, lam)
    repro = float(np.abs(C["v84_soft"]["full_v84"] - ref[ffold == 0]).max())
    assert repro < 1e-9, repro
    rep = {"files": len(names), "lambda_fold0": lam, "control_repro_max_file_diff_vs_v84_bundle": repro,
           "primary": PRIMARY, "gate": V89.GATE, "arms": {}}
    for a, c in C.items():
        r = rep["arms"][a] = {k: V.summ(v) for k, v in c.items()}
        if a == "v84_soft":
            continue
        for k in c:
            r[f"d_{k}"] = V.boot(C["v84_soft"][k], c[k], seed=89)
        d = r[f"d_{PRIMARY}"]
        r["verdict"] = ("CONTINUE" if d["delta"] >= 0.008 and d["p_gt_0"] >= 0.90
                        else "INCONCLUSIVE" if d["delta"] >= 0.003 else "KILL")
    txt = json.dumps(rep, indent=1, default=float) + "\n"
    (OUT / f"screen_{tags[0]}.json").write_text(txt)
    (OUT / "screen.json").write_text(txt)
    for a, r in rep["arms"].items():
        line = f"{a:10s} " + " ".join(f"{k}={r[k]['pq']:.4f}" for k in ("t050_c030", "t050_c030_veto", "full_v84"))
        if "verdict" in r:
            d = r[f"d_{PRIMARY}"]
            line += f"  d_{PRIMARY}={d['delta']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] P={d['p_gt_0']:.2f} {r['verdict']}"
        print(line)


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        V89.cmd_extract(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "gate":
        cmd_gate(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
