"""V61 paired fold comparison: A (baseline) vs B (variant) [vs A' noise control] on ONE fold.

Everything is computed from the cached four-view predictions of each recipe on the SAME held-out
files, with the same consensus, pixel ownership and per-record scorer (v61_toolkit).

Reported per recipe at (i) its own best confidence and (ii) the BASELINE's best confidence
(to separate an objective effect from a threshold shift):
  direct PQ, TP/FP/FN, mean matched IoU (iou_sum/TP), recall by GT category.

GT categories reuse scripts/diagnose_v59_0_recall.py EXACTLY (512px GT, priority order):
  thin   : bbox elongation >= 4
  small  : else area_full < 1000 px      (area_full = 512px area * 16)
  faint  : else GT mean intensity <= 25th percentile of the image's disk pixels (gray_512 > 20)
  normal : otherwise

Paired statistics: per-file counts -> bootstrap over FILES of PQ(B)-PQ(A) (2000 resamples).
The same is done for A' vs A so the retraining-noise scale n is reported next to d.

  python scripts/eval_v61_compare.py --fold 0 --a baseline --b focal_dice --noise baseline_seed2 \
      --out outputs/v61_compare_fold0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import eval_v61_oof as O  # noqa: E402
import v61_toolkit as K  # noqa: E402

SMALL = 512
SMALL_AREA_FULL = 1000
THIN_ELONG = 4.0
FAINT_PCTL = 25
CATS = ("thin", "small", "faint", "normal")


def categorize(area_full, elong, mean_int, faint_cut):
    """Priority order identical to diagnose_v59_0_recall."""
    if elong >= THIN_ELONG:
        return "thin"
    if area_full < SMALL_AREA_FULL:
        return "small"
    if mean_int <= faint_cut:
        return "faint"
    return "normal"


def gt_category(mask_small, gray_small, faint_cut):
    """Category of one GT given its 512px bool mask and the 512px gray image."""
    ys, xs = np.nonzero(mask_small)
    area_full = float(mask_small.sum()) * (2048 / SMALL) ** 2
    if len(xs):
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        elong = max(w, h) / max(min(w, h), 1)
        mean_int = float(gray_small[mask_small].mean())
    else:
        elong, mean_int = 1.0, 255.0
    return categorize(area_full, elong, mean_int, faint_cut)


def faint_cut_of(gray_small):
    sel = gray_small[gray_small > 20]
    return float(np.percentile(sel, FAINT_PCTL)) if sel.size else 0.0


def file_detail(recipe, fold, name, gts_full, cats_by_record, confidences):
    """Per confidence: counts (4,), category hits/totals. gts_full: list (per record) of RLEs."""
    from pycocotools import mask as mask_utils

    views = K.read_cache_views(O.cache_path(recipe, fold, name))
    rles, scores = K.cluster_views(views)
    rles, scores = K.own_pixels(rles, scores)
    scores = np.asarray(scores)
    out = {}
    for c in confidences:
        keep = [r for r, s in zip(rles, scores, strict=True) if s >= c]
        counts = np.zeros(4)
        hits = dict.fromkeys(CATS, 0)
        tot = dict.fromkeys(CATS, 0)
        for gt, cats in zip(gts_full, cats_by_record, strict=True):
            iou = (mask_utils.iou(keep, gt, [0] * len(gt)).T if keep and gt
                   else np.zeros((len(gt), len(keep))))
            counts += K.pq_counts(iou, len(gt), len(keep))
            hit_g = (iou > 0.5).any(axis=1) if iou.size else np.zeros(len(gt), bool)
            for h, cat in zip(hit_g, cats, strict=True):
                tot[cat] += 1
                hits[cat] += int(h)
        out[c] = {"counts": counts, "hits": hits, "tot": tot}
    return out


def load_gt(fold):
    """-> {name: (gts_full per record, cats per record)} for the fold's held-out files."""
    import cv2
    from pycocotools import mask as mask_utils

    doc = json.loads(O.ANNOTATIONS.read_text())
    info_by_id = {i["id"]: i for i in doc["images"]}
    anns = {}
    for a in doc["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    by_file = {}
    for iid, info in info_by_id.items():
        by_file.setdefault(info["file_name"], []).append(iid)
    out = {}
    for name in O.load_folds()[fold]:
        gray = cv2.imread(str(O.IMAGE_DIR / name), cv2.IMREAD_GRAYSCALE)
        gs = cv2.resize(gray, (SMALL, SMALL), interpolation=cv2.INTER_AREA)
        cut = faint_cut_of(gs)
        gts, cats = [], []
        for iid in sorted(by_file[name]):
            info = info_by_id[iid]
            rec, rc = [], []
            for a in anns.get(iid, []):
                r = mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], info["height"],
                                                            info["width"]))
                if mask_utils.area(r) <= 0:
                    continue
                m = mask_utils.decode(r)
                ms = cv2.resize(m.astype(np.uint8), (SMALL, SMALL),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                rec.append(r)
                rc.append(gt_category(ms, gs, cut))
            gts.append(rec)
            cats.append(rc)
        out[name] = (gts, cats)
    return out


def summarize(per_file, conf):
    tot = sum(f[conf]["counts"] for f in per_file)
    hits = {c: sum(f[conf]["hits"][c] for f in per_file) for c in CATS}
    n = {c: sum(f[conf]["tot"][c] for f in per_file) for c in CATS}
    return {"conf": conf, "pq": K.pq_value(tot), "tp": int(tot[1]), "fp": int(tot[2]),
            "fn": int(tot[3]), "mean_matched_iou": float(tot[0] / tot[1]) if tot[1] else 0.0,
            "recall": {c: (hits[c] / n[c] if n[c] else None) for c in CATS},
            "n_gt": n}


def paired_bootstrap(counts_a, counts_b, n=2000, seed=61):
    """counts_*: (F,4). Returns delta PQ, 95% CI, P(delta>0)."""
    a, b = np.asarray(counts_a, float), np.asarray(counts_b, float)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(a), len(a))
        deltas.append(K.pq_value(b[idx].sum(0)) - K.pq_value(a[idx].sum(0)))
    d = K.pq_value(b.sum(0)) - K.pq_value(a.sum(0))
    return {"delta": d, "ci95": [float(np.percentile(deltas, 2.5)),
                                 float(np.percentile(deltas, 97.5))],
            "p_gt_0": float(np.mean(np.asarray(deltas) > 0))}


def best_conf(per_file):
    return max(K.CONFIDENCES, key=lambda c: K.pq_value(sum(f[c]["counts"] for f in per_file)))


def verdict(d_own, n_noise, p_gt0, d_at_base, gain_fine, gain_normal):
    """Pre-registered rules (docs/v61_phaseB_preregistration_2026-10-05.md)."""
    if d_own < 0.005 or d_own < n_noise:
        return "kill"
    ok = (d_own >= 0.010 and d_own >= 2 * n_noise and p_gt0 >= 0.90 and d_at_base > 0
          and gain_fine >= gain_normal)
    return "continue" if ok else "inconclusive"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--a", default="baseline")
    ap.add_argument("--b", required=True)
    ap.add_argument("--noise", default=None, help="recipe name of the A' seed-control run")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    gt = load_gt(args.fold)
    names = sorted(gt)
    detail = {}
    for r in [args.a, args.b] + ([args.noise] if args.noise else []):
        detail[r] = [file_detail(r, args.fold, n, gt[n][0], gt[n][1], K.CONFIDENCES)
                     for n in names]
    base_c = best_conf(detail[args.a])
    rep = {"fold": args.fold, "files": len(names), "baseline_best_conf": base_c, "recipes": {}}
    for r, pf in detail.items():
        own = best_conf(pf)
        rep["recipes"][r] = {"own_best": summarize(pf, own), "at_baseline_conf":
                             summarize(pf, base_c)}

    def counts(r, c):
        return np.asarray([f[c]["counts"] for f in detail[r]])

    own_b, own_a = rep["recipes"][args.b]["own_best"], rep["recipes"][args.a]["own_best"]
    d_own = own_b["pq"] - own_a["pq"]
    pb = paired_bootstrap(counts(args.a, own_a["conf"]), counts(args.b, own_b["conf"]))
    pb_base = paired_bootstrap(counts(args.a, base_c), counts(args.b, base_c))
    rep["paired_B_vs_A"] = {"own_best": pb, "at_baseline_conf": pb_base}
    n_noise = 0.0
    if args.noise:
        own_n = rep["recipes"][args.noise]["own_best"]
        pn = paired_bootstrap(counts(args.a, own_a["conf"]), counts(args.noise, own_n["conf"]))
        rep["paired_Aprime_vs_A"] = pn
        n_noise = abs(pn["delta"])
    fine = ("thin", "small", "faint")

    def pooled(summ):
        h = sum(summ["recall"][c] * summ["n_gt"][c] for c in fine if summ["n_gt"][c])
        t = sum(summ["n_gt"][c] for c in fine)
        return h / t if t else 0.0
    gain_fine = pooled(own_b) - pooled(own_a)
    gain_norm = ((own_b["recall"]["normal"] or 0) - (own_a["recall"]["normal"] or 0))
    rep["recall_gain_fine_pooled"], rep["recall_gain_normal"] = gain_fine, gain_norm
    rep["verdict"] = verdict(d_own, n_noise, pb["p_gt_0"], pb_base["delta"], gain_fine,
                             gain_norm)
    rep["noise_scale_n"] = n_noise
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
