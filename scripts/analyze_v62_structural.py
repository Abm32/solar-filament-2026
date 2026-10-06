"""V62 structural analysis: oracle recall by category, proposals/image, FP/image.

Reads cached four-view predictions (outputs/v61_oof_cache/<recipe>/fold<K>) for any recipes on the
SAME held-out files and reports, per recipe:

  pool stage  (consensus of the 4 flip views at the detector floor conf 0.1, after pixel
               ownership, i.e. everything a confidence cut could still select):
      oracle recall @IoU>0.5 overall and by GT category, proposals/image
  cut stage   (same set thresholded at conf c, c in {0.35, recipe-own-best}):
      proposals/image, FP/image (a proposal matching NO GT of any record at IoU>0.5),
      recall by category

GT categories and the GT set are identical to eval_v61_compare / diagnose_v59_0_recall. Oracle
recall counts a GT as reachable if ANY pooled proposal has IoU>0.5 with it (per annotator record,
summed over records as in the PQ scorer, so denominators match the PQ GT count).

  python scripts/analyze_v62_structural.py --fold 0 --a baseline --noise baseline_seed2 \
      --b p2last p2first --out outputs/v62_structural_fold0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import eval_v61_compare as C  # noqa: E402
import eval_v61_oof as O  # noqa: E402
import v61_toolkit as K  # noqa: E402

CATS = C.CATS
CUTS = (0.35,)


def file_structural(recipe, fold, name, gts_full, cats_by_record, cuts):
    """-> dict: pool/cut stats for one file."""
    from pycocotools import mask as mask_utils

    views = K.read_cache_views(O.cache_path(recipe, fold, name))
    rles, scores = K.cluster_views(views)
    rles, scores = K.own_pixels(rles, scores)
    scores = np.asarray(scores)
    n_pool = len(rles)
    out = {"n_pool": n_pool, "stages": {}}

    def stage(sel):
        keep = [r for r, s in zip(rles, scores, strict=True) if sel(s)]
        hits = dict.fromkeys(CATS, 0)
        tot = dict.fromkeys(CATS, 0)
        matched_props = set()
        for gt, cats in zip(gts_full, cats_by_record, strict=True):
            iou = (mask_utils.iou(keep, gt, [0] * len(gt)).T if keep and gt
                   else np.zeros((len(gt), len(keep))))
            hit_g = (iou > 0.5).any(axis=1) if iou.size else np.zeros(len(gt), bool)
            if iou.size:
                matched_props |= set(np.where((iou > 0.5).any(axis=0))[0].tolist())
            for h, cat in zip(hit_g, cats, strict=True):
                tot[cat] += 1
                hits[cat] += int(h)
        return {"n": len(keep), "fp": len(keep) - len(matched_props), "hits": hits, "tot": tot}

    out["stages"]["pool"] = stage(lambda s: True)
    for c in cuts:
        out["stages"][f"cut{c:.2f}"] = stage(lambda s, c=c: s >= c)
    return out


def summarize(per_file, stage):
    hits = dict.fromkeys(CATS, 0)
    tot = dict.fromkeys(CATS, 0)
    n = fp = 0
    for f in per_file:
        s = f["stages"][stage]
        n += s["n"]
        fp += s["fp"]
        for c in CATS:
            hits[c] += s["hits"][c]
            tot[c] += s["tot"][c]
    fine_h = sum(hits[c] for c in ("thin", "small"))
    fine_t = sum(tot[c] for c in ("thin", "small"))
    nf = len(per_file)
    return {"proposals_per_image": n / nf, "fp_per_image": fp / nf,
            "recall": {c: (hits[c] / tot[c] if tot[c] else None) for c in CATS},
            "recall_all": sum(hits.values()) / max(sum(tot.values()), 1),
            "recall_small_thin": fine_h / fine_t if fine_t else None,
            "n_gt": tot}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--a", default="baseline")
    ap.add_argument("--noise", default="baseline_seed2")
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    gt = C.load_gt(args.fold)
    names = sorted(gt)
    recipes = [args.a, args.noise] + list(args.b)
    rep = {"fold": args.fold, "files": len(names), "recipes": {}}
    for r in recipes:
        pf = [file_structural(r, args.fold, n, gt[n][0], gt[n][1], CUTS) for n in names]
        rep["recipes"][r] = {st: summarize(pf, st) for st in pf[0]["stages"]}
    base, noise = rep["recipes"][args.a], rep["recipes"][args.noise]
    rep["seed_noise_small_thin_pool_gain"] = (noise["pool"]["recall_small_thin"]
                                              - base["pool"]["recall_small_thin"])
    rep["gate"] = {}
    for r in args.b:
        v = rep["recipes"][r]
        gain = v["pool"]["recall_small_thin"] - base["pool"]["recall_small_thin"]
        ratio_props = v["cut0.35"]["proposals_per_image"] / base["cut0.35"]["proposals_per_image"]
        ratio_fp = v["cut0.35"]["fp_per_image"] / base["cut0.35"]["fp_per_image"]
        n_gain = rep["seed_noise_small_thin_pool_gain"]
        checks = {"gain_small_thin_pool": gain,
                  "gain_ge_0.030": gain >= 0.030,
                  "gain_ge_2x_seed_noise": gain >= 2 * abs(n_gain),
                  "proposals_ratio_at_0.35": ratio_props,
                  "proposals_ratio_le_1.30": ratio_props <= 1.30,
                  "fp_ratio_at_0.35": ratio_fp, "fp_ratio_le_1.30": ratio_fp <= 1.30}
        checks["structural_pass"] = bool(checks["gain_ge_0.030"] and checks["gain_ge_2x_seed_noise"]
                                         and checks["proposals_ratio_le_1.30"]
                                         and checks["fp_ratio_le_1.30"])
        rep["gate"][r] = checks
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep["gate"], indent=2))


if __name__ == "__main__":
    main()
