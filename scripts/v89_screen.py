"""V89 fold-0 screen (CPU): the full V84 path on the 141 fold-0 files with a swapped fold-0 refiner.
p2first fold-0 OOF views -> flip consensus -> refiner (k7, t0.50) -> cut 0.30 -> MIN200 -> V83/V84 demote.
Demote for fold 0 = exactly V84's: LightGBM fit on V84-soft t0.50 candidates of folds 1-4, lambda from V84's
fold-0-free inner predictions. Only the fold-0 candidates (and so their features) change with the refiner.

  CUDA_VISIBLE_DEVICES= python scripts/v89_screen.py extract TAG CKPT    # -> outputs/v89/extract_TAG.pkl
  CUDA_VISIBLE_DEVICES= python scripts/v89_screen.py gate TAG [TAG ...]   # -> outputs/v89/screen.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from multiprocessing import Pool, set_start_method
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import v84_extract as X84  # noqa: E402
import v84_bundle as B  # noqa: E402

E, V73, V = X84.E, X84.V73, B.V
ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v89"
GATE = dict(delta=0.008, p=0.90)


def cmd_extract(tag, ckpt):
    import torch
    set_start_method("fork")
    V73.DEV = "cpu"
    E.RT = V73.runtime()
    m = V73.build_model(E.RT["base"], "imagenet", "resnet18")
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
    E.MODELS[0] = m.eval()
    fmap = V73.R.fold_of_file()
    names = sorted(n for n, k in fmap.items() if k == 0)
    assert len(names) == 141
    with torch.no_grad(), Pool(int(os.environ.get("PROCS", "22")), initializer=E.init) as p:
        res = p.map(X84.work, [(n, 0) for n in names], chunksize=1)
    by_t = {t: [r[t] for r in res] for t in X84.TS}
    tj = Path(ckpt).with_name(Path(ckpt).name.replace("_refiner.pt", ".json"))
    if tj.exists():  # cross-check vs the training job's own GPU scoring
        pf = json.loads(tj.read_text())["per_file"]
        for t, rs in by_t.items():
            d = max(max(abs(a - b) for a, b in zip(r["tot"], pf[r["name"]]["k7"][f"t{t:.2f}_c0.30"])) for r in rs)
            print(tag, t, "max file diff vs training json", d, flush=True)
            assert d <= 1e-3
    pickle.dump(by_t, open(OUT / f"extract_{tag}.pkl", "wb"))


def counts(res, A, model, lam):
    keep = (A["sc"] >= B.CUT) & (A["area"] >= B.MINA)
    full = keep & (model.predict(A["X"]) > lam / 2) if len(A["X"]) else keep
    return {"t050_c030": V.file_counts(res, A["sc"] >= B.CUT, A["fi"]),
            "full_v84": V.file_counts(res, full, A["fi"])}


def cmd_gate(tags):
    soft = pickle.load(open(ROOT / "outputs/v84/extract_soft.pkl", "rb"))
    res5 = soft[0.5]
    A5 = B.arrays(res5)
    pred = B.crossfit(A5)
    ref, _, lams = B.run_arm(res5, A5, pred, 1, 1)  # V84 bundle, all folds
    ffold = np.asarray([r["fold"] for r in res5])
    tr = A5["fold"] != 0
    model, lam = V.fit(A5["X"][tr], A5["y"][tr], A5["w"][tr]), lams[0]
    assert np.allclose(model.predict(A5["X"][~tr]), pred[0][~tr]), "demote refit not deterministic"
    ctrl = [r for r in res5 if r["fold"] == 0]
    arms = {"v84_soft": ({0.5: ctrl, 0.45: [r for r in soft[0.45] if r["fold"] == 0]})}
    for t in tags:
        arms[t] = pickle.load(open(OUT / f"extract_{t}.pkl", "rb"))
    names = [r["name"] for r in ctrl]
    C = {}
    for a, by_t in arms.items():
        assert [r["name"] for r in by_t[0.5]] == names
        A = B.arrays(by_t[0.5])
        C[a] = counts(by_t[0.5], A, model, lam)
        C[a]["t045_c030"] = np.asarray([r["tot"] for r in by_t[0.45]])
    repro = float(np.abs(C["v84_soft"]["full_v84"] - ref[ffold == 0]).max())
    rep = {"files": len(names), "lambda_fold0": lam, "control_repro_max_file_diff_vs_v84_bundle": repro,
           "gate": GATE, "arms": {}}
    assert repro < 1e-9, repro
    for a, c in C.items():
        r = rep["arms"][a] = {k: V.summ(v) for k, v in c.items()}
        if a == "v84_soft":
            continue
        for k in c:
            r[f"d_{k}"] = V.boot(C["v84_soft"][k], c[k], seed=89)
        d = r["d_full_v84"]
        r["verdict"] = "PASS" if d["delta"] >= GATE["delta"] and d["p_gt_0"] >= GATE["p"] else "FAIL"
    (OUT / "screen.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        cmd_extract(sys.argv[2], sys.argv[3])
    else:
        cmd_gate(sys.argv[2:])
