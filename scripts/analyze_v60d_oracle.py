"""Oracle recall by pool size: how much of the 0.86 tiling oracle is tiling vs more proposals?

Oracle recall (GT matched at IoU>=0.5 by ANY proposal) is monotone in the number of proposals,
so a 4-model union (~180 proposals/image) inflates it relative to one deployable model. This
reports, from the census cache, oracle recall for: V17 uncut, single tile model, 4-model tile
union, and V17 + each, with proposals/image, so recall can be read against pool size.

  CUDA_VISIBLE_DEVICES=N python scripts/analyze_v60d_oracle.py --out outputs/v60d_oracle.json
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


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("outputs/census_cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tags", nargs="+", default=["base_f0", "base_union4"])
    args = ap.parse_args()
    device = torch.device("cuda:0")
    gtv = pickle.loads((args.cache / "gt_v17.pkl").read_bytes())
    pools = {t: pickle.loads((args.cache / f"tiles_{t}.pkl").read_bytes()) for t in args.tags}

    acc = {}
    n_gt = 0
    n_img = 0
    for name in sorted(gtv):
        gts = gtv[name]["gt"]
        if not gts:
            continue
        n_img += 1
        G = np.stack([C.unpack(g["m"]) for g in gts])
        n_gt += len(gts)
        v17 = gtv[name]["v17"]
        srcs = {"v17_uncut": v17}
        for t, p in pools.items():
            srcs[t] = p[name]
            srcs[f"v17+{t}"] = v17 + p[name]
        for label, dets in srcs.items():
            best = np.zeros(len(gts))
            if dets:
                M = np.stack([C.unpack(d["m"]) for d in dets])
                best = C.iou_matrix(torch, device, M, G).max(0)
            a = acc.setdefault(label, {"cov": 0, "props": 0})
            a["cov"] += int((best >= 0.5).sum())
            a["props"] += len(dets)
    rep = {"n_images": n_img, "n_gt": n_gt,
           "pools": {k: {"oracle_recall_iou0.5": round(v["cov"] / n_gt, 4),
                         "proposals_per_image": round(v["props"] / n_img, 1)}
                     for k, v in acc.items()}}
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
