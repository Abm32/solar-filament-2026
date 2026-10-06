"""Re-examine the V60B/V60C-0 "NMS collapse" attribution using the census cache.

V60B's funnel stage "nms" applied TWO rules together: the add-only V17-overlap rule
(reject a tile mask whose IoU with ANY V17 instance >= overlap_max) and tile-vs-tile NMS.
The reported "108 -> 1 tile-only GT" collapse was attributed to confidence-ranked NMS. But
tile-only GT (tile hits >=0.5, V17 pool misses <0.5) are largely filaments V17 PARTLY covers
(IoU 0.3-0.5 = the reservoir-1 near-misses). An add-only rule with overlap_max 0.10/0.25 rejects
exactly those by construction.

For each tile-only GT (union of the cached detector set) this reports, for the tile
proposals that match it at IoU>=0.5, the LOWEST IoU any of them has with a V17 instance
(best case for admission), and how many would pass overlap_max 0.10 / 0.25 / 0.50, split by
whether V17 already overlaps the GT at IoU>=0.3 (reservoir-1 near-miss) or not.

  python scripts/analyze_v60d_overlap.py --tag base_union4 --out outputs/v60d_overlap.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import census_v60d as C  # noqa: E402

GRID = (0.10, 0.25, 0.50)


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache", type=Path, default=Path("outputs/census_cache"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    dets = pickle.loads((args.cache / f"tiles_{args.tag}.pkl").read_bytes())
    gtv = pickle.loads((args.cache / "gt_v17.pkl").read_bytes())

    rows = []
    for name in sorted(gtv):
        gts, v17, tiles = gtv[name]["gt"], gtv[name]["v17"], dets[name]
        if not gts or not tiles:
            continue
        G = np.stack([C.unpack(g["m"]) for g in gts])
        T = np.stack([C.unpack(d["m"]) for d in tiles])
        V = (np.stack([C.unpack(d["m"]) for d in v17]) if v17
             else np.zeros((0, C.SMALL * C.SMALL), bool))
        tg = C.iou_matrix(torch, device, T, G)          # (nT, nG)
        vg = C.iou_matrix(torch, device, V, G)          # (nV, nG)
        tv = C.iou_matrix(torch, device, T, V)          # (nT, nV)
        for j in range(G.shape[0]):
            best_t = tg[:, j].max()
            best_v = vg[:, j].max() if vg.size else 0.0
            if not (best_t >= 0.5 and best_v < 0.5):
                continue
            match = np.where(tg[:, j] >= 0.5)[0]
            ov = tv[match].max(1) if tv.shape[1] else np.zeros(len(match))
            rows.append({"v17_gt_iou": float(best_v), "min_overlap": float(ov.min())})

    n = len(rows)
    rep = {"tag": args.tag, "n_tile_only_gt": n}
    for label, sel in (("all", lambda r: True),
                       ("v17_near_miss_iou>=0.3", lambda r: r["v17_gt_iou"] >= 0.3),
                       ("v17_iou<0.3", lambda r: r["v17_gt_iou"] < 0.3)):
        sub = [r for r in rows if sel(r)]
        rep[label] = {"n": len(sub)}
        for g in GRID:
            rep[label][f"admissible_overlap_lt_{g}"] = int(sum(r["min_overlap"] < g for r in sub))
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
