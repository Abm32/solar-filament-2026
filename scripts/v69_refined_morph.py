"""V69: does a 1-px erosion / min-area 200 also help the REFINED pipeline? CPU only.

Same population and path as scripts/eval_v62_refiner.py (141 canonical held-out files, OOF
proposals -> flip_consensus -> V11 refiner -> refined_constrained), with the refiner forced to CPU
(CUDA hidden; torch device strings remapped in-process). Variants are applied to the final
pixel-disjoint refined masks, then scored per annotator record with core.pq_counts.
Usage: CUDA_VISIBLE_DEVICES= python v69_refined_morph.py <repo> <out.json> [limit]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "must hide GPUs"
torch.cuda.is_available = lambda: True  # pass the guard; every device below is remapped to cpu
_mod_to, _ten_to, _load = torch.nn.Module.to, torch.Tensor.to, torch.load


def _cpu(args):
    return tuple("cpu" if isinstance(a, str) and a.startswith("cuda") else a for a in args)


torch.nn.Module.to = lambda self, *a, **k: _mod_to(self, *_cpu(a), **k)
torch.Tensor.to = lambda self, *a, **k: _ten_to(self, *_cpu(a), **k)
torch.load = lambda *a, **k: _load(*a, **{**k, "map_location": "cpu"})
torch.set_num_threads(16)

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "scripts"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import eval_v62_refiner as R  # noqa: E402
import eval_v61_oof as O  # noqa: E402
import reproduce_v17_from_cache as v50  # noqa: E402
import v61_toolkit as K  # noqa: E402

TS = (0.45, 0.50, 0.55)
CS = (0.30, 0.35, 0.40)
VARS = ("base", "erode1", "minarea200", "erode1_minarea200")
KER = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def variant(masks, name):
    out = []
    for m in masks:
        v = cv2.erode(m, KER) if "erode1" in name else m
        if "minarea200" in name and v.sum() < 200:
            v = np.zeros_like(v)
        out.append(v)
    return out


def main():
    from pycocotools import mask as mu

    out_path, limit = Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rt = v50.runtime_state()
    core, base, tta = rt["core"], rt["base"], rt["tta"]
    held, _, _, _, by_file = rt["split"]
    names = sorted(held)[: limit or None]
    fmap = R.fold_of_file()
    tot = {}
    per = {}
    for i, name in enumerate(names, 1):
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        gts = []
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            gt = [core.annotation_rle(info, e) for e in rt["annotations"].get(image_id, [])]
            gts.append([r for r in gt if mu.area(r) > 0])
        for recipe in ("baseline", "p2first"):
            views = K.read_cache_views(O.cache_path(recipe, fmap[name], name))
            props, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
            if not props:
                continue
            masks = [mu.decode(r) for r in props]
            bounds, _, probs = core.crop_probabilities(base, rt["model"], gray, masks, True)
            for t in TS:
                rles, fs = core.clean_refined_predictions(masks, scores, bounds, probs, t, v50.MODE)
                dec = [mu.decode(r) for r in rles]
                for vn in VARS:
                    vm = variant(dec, vn)
                    enc = [K.encode(m) for m in vm]
                    for c in CS:
                        kept = [e for e, s, m in zip(enc, fs, vm) if s >= c and m.any()]
                        acc = np.zeros(4)
                        for gt in gts:
                            iou = (mu.iou(kept, gt, [0] * len(gt)).T if kept and gt
                                   else np.zeros((len(gt), len(kept))))
                            acc += core.pq_counts(iou, len(gt), len(kept))
                        key = f"{recipe}|t{t:.2f}|c{c:.2f}|{vn}"
                        tot[key] = tot.get(key, 0) + acc
                        per.setdefault(key, []).append(acc.tolist())
        if i % 10 == 0:
            print(i, len(names), flush=True)
    rep = {k: {"pq": K.pq_value(v), "tp": int(v[1]), "fp": int(v[2]), "fn": int(v[3])}
           for k, v in sorted(tot.items())}
    # paired bootstrap erode1 vs base at each cell
    rng = np.random.default_rng(69)
    boot = {}
    for k in tot:
        if not k.endswith("|base"):
            continue
        for vn in VARS[1:]:
            a, b = np.asarray(per[k]), np.asarray(per[k[:-4] + vn])
            d = [K.pq_value(b[ix].sum(0)) - K.pq_value(a[ix].sum(0))
                 for ix in (rng.integers(0, len(a), len(a)) for _ in range(1000))]
            boot[k[:-5] + "|" + vn] = [K.pq_value(b.sum(0)) - K.pq_value(a.sum(0)),
                                       float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    out_path.write_text(json.dumps({"files": len(names), "grid": rep, "delta_vs_base": boot},
                                   indent=1))
    print(json.dumps(boot, indent=1))


if __name__ == "__main__":
    main()
