"""V60-0b: characterise the 0.3-0.5 IoU "mask-quality reservoir".

Hypothesis (v60 reframing): ~109 GT are roughly localised by V17 (best IoU in 0.3-0.5) but
their masks are too POOR to cross 0.5 — eroded barbs/thin extensions from downscaling. If
V17's best-matching proposal for these is systematically SMALLER than the GT, that confirms
a mask-quality (segmentation-first high-res) lever with real headroom.

For each GT, find V17's best-overlap proposal (full pool, pre-c-cutoff) and bucket by best
IoU: [0,0.3), [0.3,0.5), [0.5,1]. In the 0.3-0.5 band report pred/GT area ratio (does V17
under-segment?), and class mix (thin/small/normal). Full-res masks for accurate area.

Firewall: V17 from caches, tune labels scoring only, no test/submission.
  CUDA_VISIBLE_DEVICES=N python scripts/diagnose_v60_maskquality.py --out outputs/v60_mq.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import reproduce_v17_from_cache as v50  # noqa: E402

SMALL = 768          # higher than 512 here so thin masks survive the diagnostic resize
THIN_ELONG = 4.0
SMALL_AREA_FULL = 1000


def to_small(cv2, m):
    if m.ndim == 3:
        m = m[..., 0]
    return cv2.resize(m, (SMALL, SMALL), interpolation=cv2.INTER_NEAREST).astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import cv2
    from pycocotools import mask as mask_utils

    rt = v50.runtime_state()
    core, tta, base = rt["core"], rt["tta"], rt["base"]
    _, tune, _, _, by_file = rt["split"]

    bands = {"lt0.3": 0, "0.3-0.5": 0, "ge0.5": 0}
    band_area_ratio = {"0.3-0.5": [], "ge0.5": []}
    band_class = {"lt0.3": {"thin": 0, "small": 0, "normal": 0},
                  "0.3-0.5": {"thin": 0, "small": 0, "normal": 0},
                  "ge0.5": {"thin": 0, "small": 0, "normal": 0}}
    for name in sorted(tune):
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        views = v50.read_signed_views(name, "train", "mr1_iou030", v50.DETECTORS["seed1"])
        props, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        pool = []
        if props:
            masks = [mask_utils.decode(r) for r in props]
            bounds, _, probs = core.crop_probabilities(base, rt["model"], gray, masks, True)
            rles, _sc = core.clean_refined_predictions(masks, scores, bounds, probs, 0.45, v50.MODE)
            pool = [to_small(cv2, mask_utils.decode(r)) for r in rles]
        parea = np.array([p.sum() for p in pool], float) if pool else np.zeros(0)
        pf = np.stack([p.ravel() for p in pool]).astype(np.float32) if pool else None
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            for e in rt["annotations"].get(image_id, []):
                r = core.annotation_rle(info, e)
                if mask_utils.area(r) <= 0:
                    continue
                gm = to_small(cv2, mask_utils.decode(r))
                ga = float(gm.sum())
                ys, xs = np.nonzero(gm)
                w = (xs.max() - xs.min() + 1) if len(xs) else 1
                h = (ys.max() - ys.min() + 1) if len(ys) else 1
                elong = max(w, h) / max(min(w, h), 1)
                area_full = ga * (2048 / SMALL) ** 2
                if elong >= THIN_ELONG:
                    kind = "thin"
                elif area_full < SMALL_AREA_FULL:
                    kind = "small"
                else:
                    kind = "normal"
                if pf is None:
                    bands["lt0.3"] += 1
                    band_class["lt0.3"][kind] += 1
                    continue
                gflat = gm.ravel().astype(np.float32)
                inter = pf @ gflat
                iou = inter / (parea + ga - inter + 1e-9)
                bi = int(iou.argmax())
                best = float(iou[bi])
                band = "lt0.3" if best < 0.3 else ("0.3-0.5" if best < 0.5 else "ge0.5")
                bands[band] += 1
                band_class[band][kind] += 1
                if band in band_area_ratio and ga > 0:
                    band_area_ratio[band].append(parea[bi] / ga)
        print(f"mq {name}", flush=True)

    def summ(a):
        a = np.array(a)
        return {"n": len(a), "mean_pred_over_gt_area": round(float(a.mean()), 3) if len(a) else 0,
                "median": round(float(np.median(a)), 3) if len(a) else 0,
                "frac_undersized<0.8": round(float((a < 0.8).mean()), 3) if len(a) else 0}
    report = {"iou_bands": bands,
              "band_0.3_0.5_area_ratio": summ(band_area_ratio["0.3-0.5"]),
              "band_ge0.5_area_ratio": summ(band_area_ratio["ge0.5"]),
              "class_by_band": band_class, "reads_test": False}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
