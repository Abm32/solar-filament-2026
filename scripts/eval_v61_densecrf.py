"""V61: DenseCRF edge refinement of V17 masks (MORDEN/AHAS idea), scored with the V17 harness.

MORDEN (arXiv 2607.24525) uses DenseCRF with the model probability as the unary term and
full-resolution intensity in the pairwise term to snap filament boundaries. Here the unary is the
V17 refined mask (softened), the pairwise kernels use the FULL 2048 image, and each instance is
refined in a padded local window so cost stays small.

Risks measured explicitly (not assumed):
  * V17 masks are already UNDER-segmented (pred/GT area ~0.69 in the near-miss band) and the
    annotators draw looser masks than the physical filament, so edge-snapping could SHRINK them.
    -> report median area ratio before/after, and reservoir (IoU 0.3-0.5) / good (>=0.5) IoU moves.
  * No GT and no area rule drive the refinement; GT is used only to score.

Sweep: unary confidence u in {0.7, 0.9}, bilateral weight w1 in {5, 10}, iterations 5.
Score = same 512px V17-identical per-annotator PQ as eval_v60a_refiner (baseline 0.3382).

Firewall: V17 from signed caches, no training, tune labels only to score. No test, no submission.

  python scripts/eval_v61_densecrf.py --annotations <ann> --out outputs/v61_densecrf.json
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
import reproduce_v17_from_cache as v50  # noqa: E402
import train_v52_cldice_semantic as v52  # noqa: E402
import train_v56_maskrcnn as v56  # noqa: E402

FULL = 2048
PAD = 64
GRID = [(0.7, 5.0), (0.7, 10.0), (0.9, 5.0), (0.9, 10.0)]   # (unary conf, bilateral weight)


def crf_refine(mask, gray, conf, w_bil, iters=5, sxy_g=3.0, srgb=12.0, sxy_b=40.0,
               ring=12, w_gauss=0.5):
    """DenseCRF on a padded window around `mask`. Returns a refined uint8 full-size mask.

    Unary: inside the mask p_fg=conf; in a `ring`-px band just outside p_fg=0.5 (uncertain, so
    image intensity decides whether to grow into it); farther out p_fg=1-conf.
    """
    import cv2
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import create_pairwise_bilateral, unary_from_softmax

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask
    y0, y1 = max(0, ys.min() - PAD), min(FULL, ys.max() + 1 + PAD)
    x0, x1 = max(0, xs.min() - PAD), min(FULL, xs.max() + 1 + PAD)
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    g = np.ascontiguousarray(gray[y0:y1, x0:x1])
    h, w = m.shape
    near = cv2.dilate(m, np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)) > 0
    p_fg = np.full(m.shape, 1.0 - conf, np.float32)
    p_fg[near] = 0.5
    p_fg[m > 0] = conf
    probs = np.stack([1.0 - p_fg, p_fg])                      # (2, h, w): bg, fg
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(np.ascontiguousarray(unary_from_softmax(probs)))
    d.addPairwiseGaussian(sxy=sxy_g, compat=w_gauss)
    img = np.ascontiguousarray(np.stack([g, g, g], -1))
    d.addPairwiseEnergy(create_pairwise_bilateral(sdims=(sxy_b, sxy_b), schan=(srgb,) * 3,
                                                  img=img, chdim=2), compat=w_bil)
    q = np.array(d.inference(iters)).reshape(2, h, w)
    out = np.zeros_like(mask, dtype=np.uint8)
    out[y0:y1, x0:x1] = (q[1] > q[0]).astype(np.uint8)
    return out


def collect(args):
    import cv2
    from pycocotools import mask as mask_utils

    rt = v50.runtime_state()
    core, tta, base = rt["core"], rt["tta"], rt["base"]
    _, tune, _, _, by_file = rt["split"]
    per_image = []
    for name in sorted(tune):
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        views = v50.read_signed_views(name, "train", E.V17_CELL, v50.DETECTORS[E.V17_SEED])
        proposals, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        v17 = []
        if proposals:
            masks = [mask_utils.decode(r) for r in proposals]
            bounds, _, probs = core.crop_probabilities(base, rt["model"], gray, masks, True)
            rles, _ = core.clean_refined_predictions(masks, scores, bounds, probs,
                                                     E.T_REFINE, v50.MODE)
            for r in rles:
                mm = mask_utils.decode(r)
                mm = mm[..., 0] if mm.ndim == 3 else mm
                if int(mm.sum()) >= 2:
                    v17.append(mm.astype(np.uint8))
        variants = {}
        for conf, wb in GRID:
            variants[(conf, wb)] = [crf_refine(m, gray, conf, wb) for m in v17]
        gt_full, gt_small = [], []
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            for e in rt["annotations"].get(image_id, []):
                rr = core.annotation_rle(info, e)
                if mask_utils.area(rr) > 0:
                    dd = mask_utils.decode(rr)
                    dd = dd[..., 0] if dd.ndim == 3 else dd
                    gt_full.append((image_id, dd.astype(np.uint8)))
                    gt_small.append((image_id, cv2.resize(
                        dd, (E.SMALL, E.SMALL), interpolation=cv2.INTER_NEAREST).astype(bool)))
        gt_sets = {iid: [m for (i2, m) in gt_small if i2 == iid] for iid in sorted(by_file[name])}
        per_image.append({"v17": v17, "variants": variants, "gt_full": gt_full,
                          "gt_sets": gt_sets})
        print(f"collected {name} v17={len(v17)}", flush=True)
    return per_image, cv2


def load_or_collect(args):
    """Cache the (slow) CRF collection next to the output so a scoring bug cannot discard it."""
    import pickle

    import cv2
    cache = args.out.with_suffix(".collected.pkl")
    if cache.exists():
        return pickle.loads(cache.read_bytes()), cv2
    per_image, cv2 = collect(args)
    cache.write_bytes(pickle.dumps(per_image))
    return per_image, cv2


def move_stats(per_image, key):
    """Area ratio + IoU movement by V17 population for one CRF variant."""
    res_d, good_d, area_b, area_a = [], [], [], []
    good_dmg = res_push = n_res = n_good = 0
    for d in per_image:
        for v, rf in zip(d["v17"], d["variants"][key], strict=True):
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
                res_push += int(ir >= 0.5)
            elif best >= 0.5:
                n_good += 1
                good_d.append(ir - best)
                good_dmg += int(ir < 0.5)
    return {"reservoir_n": n_res, "pushed_to_0.5": res_push,
            "median_iou_delta_reservoir": float(np.median(res_d)) if res_d else 0.0,
            "good_n": n_good, "good_damaged": good_dmg,
            "median_iou_delta_good": float(np.median(good_d)) if good_d else 0.0,
            "median_area_ratio_before": float(np.median(area_b)),
            "median_area_ratio_after": float(np.median(area_a))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if v52.sha256(args.annotations) != v56.ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")
    per_image, cv2 = load_or_collect(args)
    baseline_pi = [{"v17": d["v17"], "refined": d["v17"], "gt_sets": d["gt_sets"]}
                   for d in per_image]
    baseline = E.score_selective(baseline_pi, cv2, None, baseline=True)
    rows = {}
    for key in GRID:
        pi = [{"v17": d["variants"][key], "refined": d["variants"][key], "gt_sets": d["gt_sets"]}
              for d in per_image]
        r = E.score_selective(pi, cv2, None, baseline=True)
        r["delta"] = round(r["pq"] - baseline["pq"], 4)
        r.update(move_stats(per_image, key))
        rows[f"u{key[0]}_w{key[1]}"] = r
    best = max(rows, key=lambda k: rows[k]["pq"])
    rep = {"baseline_v17_pq": baseline, "variants": rows, "best": best,
           "best_delta": rows[best]["delta"], "reads_test": False}
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
