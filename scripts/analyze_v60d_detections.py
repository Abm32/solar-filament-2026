"""Composition of the baseline tile detections: area distribution, whole-tile blobs, TP vs FP.

Uses the census cache (no GPU). Answers: does the baseline's "empty tile = full-tile filament"
training artefact show up as whole-tile predictions, and how much of the detection volume is
true-positive vs false-positive at each confidence.

  python scripts/analyze_v60d_detections.py --tag base_f0 --out outputs/v60d_dets_base_f0.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

SM = 512
TILE = 192          # 768px tile at 2048 -> 192px at 512


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache", type=Path, default=Path("outputs/census_cache"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dets = pickle.loads((args.cache / f"tiles_{args.tag}.pkl").read_bytes())
    gtv = pickle.loads((args.cache / "gt_v17.pkl").read_bytes())

    def un(p):
        return np.unpackbits(p)[:SM * SM].astype(np.float32)

    area, score, hit = [], [], []
    for name, ds in dets.items():
        gts = [un(g["m"]) for g in gtv[name]["gt"]]
        G = np.stack(gts) if gts else np.zeros((0, SM * SM), np.float32)
        gsum = G.sum(1)
        for d in ds:
            m = un(d["m"])
            a = float(m.sum())
            if len(G):
                inter = G @ m
                iou = inter / (gsum + a - inter + 1e-9)
                h = bool((iou >= 0.5).any())
            else:
                h = False
            area.append(a)
            score.append(d["s"])
            hit.append(h)
    area, score, hit = np.array(area), np.array(score), np.array(hit)
    whole = TILE * TILE
    rep = {"tag": args.tag, "n_dets": int(len(area)),
           "area_quantiles_p10_50_90_99": [int(np.percentile(area, q)) for q in (10, 50, 90, 99)],
           "whole_tile_area": whole, "overall_tp_frac": round(float(hit.mean()), 4)}
    for thr in (0.25, 0.5, 0.8):
        b = area >= thr * whole
        rep[f"area_ge_{thr}_tile"] = {"n": int(b.sum()), "frac": round(float(b.mean()), 5),
                                      "tp": int(hit[b].sum())}
    for lo, hi in ((0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
        b = (score >= lo) & (score < hi)
        rep[f"score_{lo}-{hi}"] = {"n": int(b.sum()),
                                   "tp_frac": round(float(hit[b].mean()), 4) if b.any() else None}
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
