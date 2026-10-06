"""V61b: quick parameter diagnostic for DenseCRF on V17 masks (first N tune files, cached V17).

The first full run (bilateral spatial scale 40px) erased masks (area ratio 0.92 -> ~0). The
paper's parameters imply spatial sigma ~2-3 px. This sweeps paper-scale settings on a SUBSET and
reports, per setting: median area ratio (pred/GT), reservoir pushed / good damaged, and median
IoU delta. The goal is only to find out whether ANY setting keeps masks intact and moves IoU up,
before paying for a full run. GT is used only to score.

  python scripts/diagnose_v61_crf_params.py --collected outputs/v61_densecrf.collected.pkl \
      --n-files 14 --out outputs/v61b_params.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import eval_v60a_refiner as E  # noqa: E402
import eval_v61_densecrf as V  # noqa: E402
import reproduce_v17_from_cache as v50  # noqa: E402

# (unary conf, bilateral weight, bilateral spatial sigma px, intensity sigma, ring px)
GRID = [(0.9, 3.0, 3.0, 10.0, 6), (0.9, 5.0, 3.0, 20.0, 6), (0.9, 5.0, 8.0, 20.0, 6),
        (0.9, 10.0, 8.0, 30.0, 10), (0.95, 5.0, 3.0, 20.0, 10), (0.95, 10.0, 8.0, 30.0, 10)]


def main():
    import pickle

    import cv2
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collected", type=Path, required=True)
    ap.add_argument("--n-files", type=int, default=14)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    per_image = pickle.loads(args.collected.read_bytes())
    rt = v50.runtime_state()
    names = sorted(rt["split"][1])[:args.n_files]
    base = rt["base"]

    rows = {}
    for cfg in GRID:
        conf, wb, sxy, srgb, ring = cfg
        area_b, area_a, res_d, good_d = [], [], [], []
        push = n_res = dmg = n_good = 0
        for name, d in zip(names, per_image[:args.n_files], strict=True):
            gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
            for v in d["v17"]:
                rf = V.crf_refine(v, gray, conf, wb, sxy_b=sxy, srgb=srgb, ring=ring)
                best, bg = 0.0, None
                for (_i, g) in d["gt_full"]:
                    iv = E._iou(v.astype(bool), g.astype(bool))
                    if iv > best:
                        best, bg = iv, g
                if bg is None:
                    continue
                ir = E._iou(rf.astype(bool), bg.astype(bool))
                area_b.append(v.sum() / max(bg.sum(), 1))
                area_a.append(rf.sum() / max(bg.sum(), 1))
                if 0.3 <= best < 0.5:
                    n_res += 1
                    res_d.append(ir - best)
                    push += int(ir >= 0.5)
                elif best >= 0.5:
                    n_good += 1
                    good_d.append(ir - best)
                    dmg += int(ir < 0.5)
        rows[str(cfg)] = {
            "area_before": float(np.median(area_b)), "area_after": float(np.median(area_a)),
            "reservoir_pushed": push, "reservoir_n": n_res,
            "good_damaged": dmg, "good_n": n_good,
            "median_iou_delta_reservoir": float(np.median(res_d)) if res_d else 0.0,
            "median_iou_delta_good": float(np.median(good_d)) if good_d else 0.0}
        print(cfg, rows[str(cfg)], flush=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
