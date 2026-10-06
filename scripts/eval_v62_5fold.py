"""V62 5-fold pooled paired comparison + mechanical verdict (docs/v62_5fold_preregistration).

Pools all five folds of a treatment recipe vs the baseline (same files) and applies the
pre-registered rule. Uses only cached four-view predictions (no GPU).

  python scripts/eval_v62_5fold.py --a baseline --b p2first --noise baseline_seed2 \
      --out outputs/v62_5fold_p2first.json

Primary cut = PRIMARY_CONF (0.35, the baseline's pooled best from Phase A, fixed in advance).
B's own best cut is reported but never gates. Noise n = |pooled PQ(A') - PQ(A)| at the primary cut.
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
import v61_toolkit as K  # noqa: E402

PRIMARY_CONF = 0.35
FOLDS = (0, 1, 2, 3, 4)


def pooled_detail(recipe, folds=FOLDS):
    """-> list of per-file dicts (ordered by fold, then sorted name) with the fold index."""
    out = []
    for fold in folds:
        gt = C.load_gt(fold)
        for name in sorted(gt):
            d = C.file_detail(recipe, fold, name, gt[name][0], gt[name][1], K.CONFIDENCES)
            out.append({"fold": fold, "name": name, "d": d})
    return out


def counts_at(detail, conf):
    return np.asarray([f["d"][conf]["counts"] for f in detail])


def cat_summary(detail, conf):
    hits = dict.fromkeys(C.CATS, 0)
    tot = dict.fromkeys(C.CATS, 0)
    for f in detail:
        for c in C.CATS:
            hits[c] += f["d"][conf]["hits"][c]
            tot[c] += f["d"][conf]["tot"][c]
    return {c: (hits[c] / tot[c] if tot[c] else None) for c in C.CATS}, tot


def structural(recipe, folds=FOLDS):
    """proposals/image and FP/image at PRIMARY_CONF, pooled."""
    import analyze_v62_structural as S

    n = fp = files = 0
    for fold in folds:
        gt = C.load_gt(fold)
        for name in sorted(gt):
            s = S.file_structural(recipe, fold, name, gt[name][0], gt[name][1], (PRIMARY_CONF,))
            st = s["stages"][f"cut{PRIMARY_CONF:.2f}"]
            n += st["n"]
            fp += st["fp"]
            files += 1
    return {"proposals_per_image": n / files, "fp_per_image": fp / files}


def verdict(d, n_noise, p_gt0, ci_low, folds_pos, small_gain, d_own, prop_ratio, fp_ratio):
    """Pre-registered rule, as a pure function (tested)."""
    if d < 0.005 or folds_pos < 3:
        return "kill"
    ok = (d >= max(0.010, 2 * n_noise) and p_gt0 >= 0.90 and ci_low > 0 and folds_pos >= 4
          and small_gain > 0 and d_own > 0 and prop_ratio <= 1.30 and fp_ratio <= 1.30)
    return "continue" if ok else "inconclusive"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default="baseline")
    ap.add_argument("--b", required=True)
    ap.add_argument("--noise", default="baseline_seed2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    args = ap.parse_args()
    folds = tuple(args.folds)
    da, db = pooled_detail(args.a, folds), pooled_detail(args.b, folds)
    if [f["name"] for f in da] != [f["name"] for f in db]:
        raise RuntimeError("recipes cover different files")
    dn = pooled_detail(args.noise, folds) if args.noise else None
    fold_of = np.asarray([f["fold"] for f in da])
    ca, cb = counts_at(da, PRIMARY_CONF), counts_at(db, PRIMARY_CONF)
    pb = C.paired_bootstrap(ca, cb)
    own_b = max(K.CONFIDENCES, key=lambda c: K.pq_value(counts_at(db, c).sum(0)))
    pb_own = C.paired_bootstrap(ca, counts_at(db, own_b))
    per_fold = {int(k): K.pq_value(cb[fold_of == k].sum(0)) - K.pq_value(ca[fold_of == k].sum(0))
                for k in sorted(set(fold_of.tolist()))}
    rec_a, n_gt = cat_summary(da, PRIMARY_CONF)
    rec_b, _ = cat_summary(db, PRIMARY_CONF)
    sa, sb = structural(args.a, folds), structural(args.b, folds)
    n_noise, noise_rep = 0.0, None
    if dn:
        cn = counts_at(dn, PRIMARY_CONF)
        n_noise = abs(K.pq_value(cn.sum(0)) - K.pq_value(ca.sum(0)))
        boot = C.paired_bootstrap(ca, cn)
        noise_rep = {"delta": K.pq_value(cn.sum(0)) - K.pq_value(ca.sum(0)),
                     "ci95": boot["ci95"], "p_gt_0": boot["p_gt_0"]}
    rep = {
        "folds": list(folds), "files": len(da), "primary_conf": PRIMARY_CONF,
        "pq_a": K.pq_value(ca.sum(0)), "pq_b": K.pq_value(cb.sum(0)),
        "tp_fp_fn_a": [int(x) for x in ca.sum(0)[1:]],
        "tp_fp_fn_b": [int(x) for x in cb.sum(0)[1:]],
        "mean_iou_a": float(ca[:, 0].sum() / ca[:, 1].sum()),
        "mean_iou_b": float(cb[:, 0].sum() / cb[:, 1].sum()),
        "paired_primary": pb, "b_own_best_conf": own_b, "paired_own_best": pb_own,
        "per_fold_delta": per_fold, "noise_aprime_minus_a": noise_rep, "noise_scale_n": n_noise,
        "recall_a": rec_a, "recall_b": rec_b, "n_gt": n_gt,
        "structural_a": sa, "structural_b": sb,
    }
    small_gain = (rec_b["small"] or 0) - (rec_a["small"] or 0)
    rep["verdict"] = verdict(
        pb["delta"], n_noise, pb["p_gt_0"], pb["ci95"][0],
        sum(1 for v in per_fold.values() if v > 0), small_gain, pb_own["delta"],
        sb["proposals_per_image"] / sa["proposals_per_image"],
        sb["fp_per_image"] / sa["fp_per_image"])
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    keys = ("files", "pq_a", "pq_b", "paired_primary", "per_fold_delta", "noise_scale_n",
            "verdict")
    print(json.dumps({k: rep[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
