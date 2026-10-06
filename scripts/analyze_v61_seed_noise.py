"""V61 seed-noise analysis: A' (baseline, different training seed) vs A (baseline) on fold 0.

READS ONLY the `baseline` and `baseline_seed2` caches. It cannot see any treatment recipe, so it
can be run and inspected before B exists without influencing the B verdict.

Reports, on the SAME held-out files:
  * aggregate PQ for A and A' at the baseline-best confidence and at each recipe's own best,
  * the per-file paired dPQ distribution (A' - A): mean, sd, quantiles, how many files move
    by more than 0.1 / 0.2, and the share of files where A' > A,
  * the aggregate paired delta with a file-bootstrap 95% CI,

Per-file PQ = iou_sum / (tp + 0.5 fp + 0.5 fn) for that file at the chosen confidence; files with
no GT and no predictions give 0/0, reported as NaN and excluded from the per-file distribution
(they remain in the aggregate counts).

  python scripts/analyze_v61_seed_noise.py --fold 0 --out outputs/v61_seed_noise_fold0.json
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

A, APRIME = "baseline", "baseline_seed2"


def per_file_pq(counts):
    """counts (F,4) -> per-file PQ, NaN where the denominator is 0."""
    den = counts[:, 1] + 0.5 * (counts[:, 2] + counts[:, 3])
    out = np.full(len(counts), np.nan)
    ok = den > 0
    out[ok] = counts[ok, 0] / den[ok]
    return out


def summarize_delta(d):
    d = d[~np.isnan(d)]
    return {"n_files": int(len(d)), "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "median": float(np.median(d)),
            "quantiles_5_25_75_95": [float(np.percentile(d, q)) for q in (5, 25, 75, 95)],
            "abs_gt_0.10": int((np.abs(d) > 0.10).sum()),
            "abs_gt_0.20": int((np.abs(d) > 0.20).sum()),
            "share_aprime_better": float((d > 0).mean()),
            "share_equal": float((d == 0).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    gt = C.load_gt(args.fold)
    names = sorted(gt)
    detail = {r: [C.file_detail(r, args.fold, n, gt[n][0], gt[n][1], K.CONFIDENCES)
                  for n in names] for r in (A, APRIME)}
    base_c = C.best_conf(detail[A])
    own = {r: C.best_conf(detail[r]) for r in detail}

    def counts(r, c):
        return np.asarray([f[c]["counts"] for f in detail[r]])

    rep = {"fold": args.fold, "files": len(names), "baseline_best_conf": base_c,
           "own_best_conf": own, "reads_recipes": [A, APRIME], "aggregate": {}}
    pairs = {"at_baseline_conf": (base_c, base_c), "own_best": (own[A], own[APRIME])}
    for tag, (ca, cb) in pairs.items():
        a, b = counts(A, ca), counts(APRIME, cb)
        pb = C.paired_bootstrap(a, b)
        rep["aggregate"][tag] = {
            "conf_A": ca, "conf_Aprime": cb, "pq_A": K.pq_value(a.sum(0)),
            "pq_Aprime": K.pq_value(b.sum(0)), "delta": pb["delta"], "ci95": pb["ci95"],
            "p_gt_0": pb["p_gt_0"],
            "tp_fp_fn_A": [int(x) for x in a.sum(0)[1:]],
            "tp_fp_fn_Aprime": [int(x) for x in b.sum(0)[1:]]}
    a, b = counts(A, base_c), counts(APRIME, base_c)
    rep["per_file_delta_Aprime_minus_A_at_baseline_conf"] = summarize_delta(
        per_file_pq(b) - per_file_pq(a))
    rep["noise_scale_n"] = abs(rep["aggregate"]["own_best"]["delta"])
    rep["note"] = ("noise_scale_n is |PQ(A')-PQ(A)| at each recipe's own best confidence, the "
                   "quantity used by the pre-registered rule d >= 2*n")
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
