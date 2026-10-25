"""V84 export feature parity (CPU): export_v74.refine_image with the fold's soft refiner + the fold's
cross-fitted demote model on fold-held-out TRAIN files (OOF cached views) must reproduce the V84 bundle:
same yhat per candidate, same kept masks (score, area) and same per-file PQ counts.
  CUDA_VISIBLE_DEVICES= python scripts/v84_export_parity.py [files_per_fold]
"""
from __future__ import annotations

import json
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import export_v74 as X  # noqa: E402
import train_v73_refiner as V73  # noqa: E402

ROOT = SCRIPTS.parent
P = ROOT / "outputs/v84/export/parity"


def main():
    import torch
    from pycocotools import mask as mu
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ref = pickle.load(open(P / "bundle_ref.pkl", "rb"))
    V73.DEV = "cpu"
    rt = V73.runtime()
    args = Namespace(mask_threshold=0.50, confidence=0.30, min_area=200)
    report, worst_y = [], 0.0
    for k in range(5):
        bare = Path(f"/tmp/v84_soft_fold{k}_state.pt")
        torch.save(torch.load(ROOT / f"outputs/v84/soft_fold{k}_refiner.pt", map_location="cpu",
                              weights_only=True)["model"], bare)
        core, tta, base, refiner = X.refiner_runtime(bare, "cpu")
        demote = X.Demote(P / f"demote_fold{k}.json", core)
        names = sorted(nm for nm, r in ref.items() if r["fold"] == k)
        names = names[:: max(1, len(names) // n)][:n]
        for name in names:
            r = ref[name]
            views = V73.R.K.read_cache_views(V73.R.O.cache_path(V73.RECIPE, k, name))
            with torch.no_grad():
                preds, kept, small, dem, yh = X.refine_image(
                    core, tta, base, refiner, views, rt["base"].CFG.train_dir / name, args, demote)
            # yhat parity over all candidates (score >= 0.15), as sorted (score, yhat) pairs
            got = sorted(yh.values())
            want = sorted(r["yhat"])
            assert len(got) == len(want), (name, len(got), len(want))
            dy = float(np.max(np.abs(np.subtract(got, want)))) if got else 0.0
            worst_y = max(worst_y, dy)
            # kept-set parity
            gk = sorted((round(s, 9), int(mu.area(m))) for m, s in zip(preds, kept))
            wk = sorted((round(s, 9), int(a)) for s, a, kp in zip(r["scores"], r["area"], r["keep"]) if kp)
            # count parity vs annotator records
            tot = np.zeros(4)
            for image_id in sorted(rt["by_file"][name], key=str):
                info = rt["images"][image_id]
                g = [rt["core"].annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
                g = [x for x in g if mu.area(x) > 0]
                I = np.asarray(mu.iou(g, preds, [0] * len(preds))) if g and preds else np.zeros((len(g), len(preds)))
                tot += rt["core"].pq_counts(I, len(g), len(preds))
            dc = float(np.max(np.abs(tot - np.asarray(r["counts"]))))
            ok = gk == wk and dc <= 1e-3 and dy <= 1e-9
            report.append(dict(name=name, fold=k, kept=len(preds), demoted=dem, small=small,
                               max_dyhat=dy, max_dcount=dc, kept_match=gk == wk, ok=ok))
            print(json.dumps(report[-1]), flush=True)
    out = dict(files=len(report), all_ok=all(x["ok"] for x in report), worst_dyhat=worst_y,
               demoted=sum(x["demoted"] for x in report), kept=sum(x["kept"] for x in report))
    (P / "parity.json").write_text(json.dumps(dict(summary=out, files=report), indent=1) + "\n")
    print(json.dumps(out))
    assert out["all_ok"], "export parity FAILED"


if __name__ == "__main__":
    main()
