"""V99 arm `hardneg`: the V84 soft refiner recipe plus hard-negative stage-B crops.

Pre-registration: docs/v99_arms_preregistration_2026-10-25.md. Never reads test.
Positives: exactly V82 `soft` (train_v82_refiner_targets.fit_records / target_u8). Negatives: fold OOF proposals
with consensus score >= NEG_SCORE whose max IoU to every GT of every record of the file is < NEG_IOU, target =
all zeros, subsampled to NEG_FRAC x positives (seed NEG_SEED). Stage A, schedule, seed, loss, crops unchanged.
At inference nothing changes in this script; the veto (drop candidates whose refined mask fell back to the seed)
is a gate cell in v99_screen.py, read from the existing `fallback` feature.

  CUDA_VISIBLE_DEVICES=2 python scripts/v99_hardneg.py fold --arm hardneg --fold 0 --out outputs/v99
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v82_refiner_targets as V82  # noqa: E402

V73, R, v50 = V82.V73, V82.R, V82.v50
V73.ARMS["hardneg"] = V73.ARMS["base"]
V82.ARM = "soft"  # positives use the V84 soft target
NEG_SCORE, NEG_IOU, NEG_FRAC, NEG_SEED = 0.20, 0.10, 0.25, 99

_target_u8 = V82.target_u8


def target_u8(r, arm):
    if r.get("neg"):
        from pycocotools import mask as mu
        return np.zeros(mu.decode(r["seed"]).shape, np.uint8)
    return _target_u8(r, arm)


V82.target_u8 = target_u8  # CropDataset.__getitem__ looks this name up in V82's globals


def fit_records(rt, recipe, names, fmap):
    from pycocotools import mask as mu

    pos = V82.fit_records(rt, recipe, names, fmap)
    base, core, tta = rt["base"], rt["core"], rt["tta"]
    negs = []
    for name in names:
        views = R.K.read_cache_views(R.O.cache_path(recipe, fmap[name], name))
        proposals, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        if not proposals:
            continue
        targets = []
        for image_id in sorted(rt["by_file"].get(name, []), key=str):
            info = rt["images"][image_id]
            targets += [t for t in (core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, []))
                        if mu.area(t) > 0]
        mx = (np.asarray(mu.iou(targets, proposals, [0] * len(proposals))).max(0) if targets
              else np.zeros(len(proposals)))
        for pi, (p, s) in enumerate(zip(proposals, scores)):
            if s >= NEG_SCORE and mx[pi] < NEG_IOU:
                negs.append({"name": name, "seed": p, "target": p, "bounds": tuple(base.square_bounds(mu.decode(p))),
                             "pi": pi, "agree": [], "q": 0.0, "neg": True})
    k = min(len(negs), int(round(NEG_FRAC * len(pos))))
    pick = sorted(np.random.default_rng(NEG_SEED).choice(len(negs), k, replace=False).tolist()) if k else []
    print(f"hardneg: positives={len(pos)} negatives_available={len(negs)} used={k}", flush=True)
    return pos + [negs[i] for i in pick]


def demo():
    """CPU self-check: negative target is all zeros and positives are untouched."""
    from pycocotools import mask as mu
    a = np.zeros((8, 8), np.uint8, order="F"); a[2:6, 2:6] = 1
    ra = mu.encode(a)
    assert target_u8({"seed": ra, "neg": True}, "soft").sum() == 0
    assert (target_u8({"target": ra, "agree": [ra], "q": 1.0}, "soft") == a * 255).all()
    print("demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
        raise SystemExit
    V73.V63.fit_records, V73.CropDataset, V73.train = fit_records, V82.CropDataset, V82.train
    V73.main()
