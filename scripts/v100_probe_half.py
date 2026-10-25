"""V100 arm C: split-half-by-density public probe. Prereg: docs/v100_preregistration_2026-10-25.md.

  python scripts/v100_probe_half.py predict   # OOF prediction from outputs/v84/extract_soft.pkl (write BEFORE submitting)
  python scripts/v100_probe_half.py build     # outputs/v100/probe_half.csv from outputs/v84/export/submission_v84.csv
Rule: rank images by number of kept masks (ties: total mask area), keep the top half, drop every row of the rest.
Reads no labels for `build`; `predict` uses train OOF counts only.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "outputs/v70/pylib"))
ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v100"
SCALE = 0.39 / 0.4590


def pq(t):
    d = t[1] + 0.5 * (t[2] + t[3])
    return float(t[0] / d) if d else 0.0


def predict():
    import v83_rule as V
    import v84_bundle as B
    res = pickle.load(open(ROOT / "outputs/v84/extract_soft.pkl", "rb"))[0.5]
    A = B.arrays(res)
    pred = B.crossfit(A)
    counts, keep, _ = B.run_arm(res, A, pred, 1, 1)  # V84 path per file: (iou, tp, fp, fn)
    n_file = len(res)
    kept = np.bincount(A["fi"][keep], minlength=n_file)
    area = np.bincount(A["fi"][keep], weights=A["area"][keep], minlength=n_file)
    n_gt = np.asarray([sum(r["n_gt"]) for r in res])
    order = np.lexsort((-area, -kept))
    dense = np.zeros(n_file, bool); dense[order[: n_file // 2]] = True
    full = counts.sum(0)
    pooled = counts.copy(); pooled[~dense] = 0; pooled[~dense, 3] = n_gt[~dense]  # blanked: every GT is FN
    per_img = np.array([pq(c) for c in counts])
    rep = dict(files=n_file, dense_files=int(dense.sum()), gt_share_dense=float(n_gt[dense].sum() / n_gt.sum()),
               kept_share_dense=float(kept[dense].sum() / kept.sum()),
               oof_full_pooled=pq(full), oof_half_pooled=pq(pooled.sum(0)),
               oof_full_mean=float(per_img.mean()), oof_half_mean=float(np.where(dense, per_img, 0).mean()),
               oof_dense_only_pooled=pq(counts[dense].sum(0)), oof_dense_only_mean=float(per_img[dense].mean()))
    rep["public_pred_pooled"] = rep["oof_half_pooled"] * SCALE
    rep["public_pred_mean"] = rep["oof_half_mean"] * SCALE
    rep["rule"] = "keep densest half by kept-mask count (ties: total area); other half has no rows"
    (OUT / "probe_half_prediction.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


def build():
    import pandas as pd
    from pycocotools import mask as mu
    src = ROOT / "outputs/v84/export/submission_v84.csv"
    df = pd.read_csv(src, dtype=str)
    df["stem"] = df.filament_id.str.rsplit("_", n=1).str[0]
    df["area"] = [int(mu.area({"size": [2048, 2048], "counts": c.encode()})) for c in df.segmentation_rle]
    test = sorted(p.stem for p in (ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images").glob("*.jpeg"))
    assert len(test) == 180
    g = df.groupby("stem").agg(n=("filament_id", "size"), area=("area", "sum")).reindex(test).fillna(0)
    g = g.sort_values(["n", "area"], ascending=False)
    dense = set(g.index[:90])
    out = df[df.stem.isin(dense)][["filament_id", "segmentation_rle"]]
    path = OUT / "probe_half.csv"
    out.to_csv(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    man = dict(source=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), rows_in=len(df),
               rows_out=len(out), images_kept=len(dense), images_dropped=180 - len(dense),
               kept_min_rows=int(g.n.iloc[89]), dropped_max_rows=int(g.n.iloc[90]), sha256=sha,
               kept_images=sorted(dense))
    (OUT / "probe_half_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k != "kept_images"}, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"predict": predict, "build": build}[sys.argv[1]]()
