"""V66: consensus-VOTING ensembles of out-of-fold detectors (docs/v66_preregistration_2026-10-07).

An ensemble of members M1..Mn is the existing flip_consensus applied to the CONCATENATION of all
members' four flip views (4n views): a detection's score = sum(matched view scores) / (4n), so a
detection only one member makes is capped at 1/n. Pixel ownership and the per-record PQ scorer are
exactly V61's. Invariant (tested): an ensemble of one model, or of a model repeated, reproduces that
model's single-model output.

Cut selection is CROSS-FITTED per fold (maximiser of pooled PQ on the other four folds), identically
for ensembles and single references, so no recipe gets a cut tuned on the files it is scored on.

  python scripts/eval_v66_ensemble.py --ref p2first --out outputs/v66_stage1.json \
      --ensembles E_all4=p2first,p2first_cp,baseline,baseline_seed2
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

FOLDS = (0, 1, 2, 3, 4)
FIXED_CONF = 0.35


def ensemble_views(members, fold, name):
    """Concatenate every member's four flip views for one file (4n views)."""
    views = []
    for m in members:
        views.extend(K.read_cache_views(O.cache_path(m, fold, name)))
    return views


def file_counts_ensemble(members, fold, name, gts_full):
    """Per-confidence counts for one file under an ensemble of `members`."""
    views = ensemble_views(members, fold, name)
    rles, scores = K.cluster_views(views)
    rles, scores = K.own_pixels(rles, scores)
    return K.score_file(rles, scores, gts_full), rles, scores


def pooled_ensemble(members, folds=FOLDS):
    """-> (counts {conf: (F,4)}, fold_of (F,), names list, n_props_at_fixed, n_fp placeholder)."""
    counts = {c: [] for c in K.CONFIDENCES}
    fold_of, names = [], []
    for fold in folds:
        gt = C.load_gt(fold)
        for name in sorted(gt):
            res, _r, _s = file_counts_ensemble(members, fold, name, gt[name][0])
            for c in K.CONFIDENCES:
                counts[c].append(res[c])
            fold_of.append(fold)
            names.append(name)
    return {c: np.asarray(v) for c, v in counts.items()}, np.asarray(fold_of), names


def crossfit_counts(counts, fold_of):
    """Per-file counts at the cross-fitted cut: cut for fold k = argmax pooled PQ on other folds."""
    out = np.zeros_like(counts[K.CONFIDENCES[0]])
    cuts = {}
    for k in sorted(set(fold_of.tolist())):
        other = fold_of != k
        best = max(K.CONFIDENCES, key=lambda c: K.pq_value(counts[c][other].sum(0)))
        cuts[int(k)] = best
        out[fold_of == k] = counts[best][fold_of == k]
    return out, cuts


def paired(ref_counts, ens_counts, fold_of):
    pb = C.paired_bootstrap(ref_counts, ens_counts)
    per_fold = {}
    for k in sorted(set(fold_of.tolist())):
        m = fold_of == k
        per_fold[int(k)] = K.pq_value(ens_counts[m].sum(0)) - K.pq_value(ref_counts[m].sum(0))
    return pb, per_fold


def verdict(d, p_gt0, ci_low, folds_pos, fp_ratio):
    """Pre-registered stage-1 rule (pure function, tested)."""
    if d < 0.005 or folds_pos < 3:
        return "kill"
    ok = d >= 0.010 and p_gt0 >= 0.90 and ci_low > 0 and folds_pos >= 4 and fp_ratio <= 1.30
    return "continue" if ok else "inconclusive"


def summarize(members, folds):
    counts, fold_of, names = pooled_ensemble(members, folds)
    cf, cuts = crossfit_counts(counts, fold_of)
    return {"members": members, "counts": counts, "fold_of": fold_of, "names": names,
            "crossfit": cf, "cuts": cuts}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="p2first", help="reference single model (same members list)")
    ap.add_argument("--ensembles", nargs="+", required=True,
                    help="NAME=m1,m2,... (equal weights)")
    ap.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    folds = tuple(args.folds)
    ref_members = [args.ref]
    ref = summarize(ref_members, folds)
    rep = {"ref": args.ref, "folds": list(folds), "files": len(ref["names"]),
           "ref_pq_crossfit": K.pq_value(ref["crossfit"].sum(0)),
           "ref_pq_fixed_0.35": K.pq_value(ref["counts"][FIXED_CONF].sum(0)),
           "ref_cuts": ref["cuts"], "ensembles": {}}
    for spec in args.ensembles:
        name, _, ml = spec.partition("=")
        members = ml.split(",")
        e = summarize(members, folds)
        if e["names"] != ref["names"]:
            raise RuntimeError("ensemble covers different files than the reference")
        pb, per_fold = paired(ref["crossfit"], e["crossfit"], ref["fold_of"])
        pb_fix, _ = paired(ref["counts"][FIXED_CONF], e["counts"][FIXED_CONF], ref["fold_of"])
        own = max(K.CONFIDENCES, key=lambda c: K.pq_value(e["counts"][c].sum(0)))
        fp_ratio = (e["crossfit"][:, 2].sum() / max(ref["crossfit"][:, 2].sum(), 1))
        v = verdict(pb["delta"], pb["p_gt_0"], pb["ci95"][0],
                    sum(1 for x in per_fold.values() if x > 0), fp_ratio)
        rep["ensembles"][name] = {
            "members": members, "pq_crossfit": K.pq_value(e["crossfit"].sum(0)),
            "tp_fp_fn_crossfit": [int(x) for x in e["crossfit"].sum(0)[1:]],
            "ref_tp_fp_fn_crossfit": [int(x) for x in ref["crossfit"].sum(0)[1:]],
            "delta_crossfit": pb, "per_fold_delta": per_fold, "fp_ratio": fp_ratio,
            "pq_fixed_0.35": K.pq_value(e["counts"][FIXED_CONF].sum(0)),
            "delta_fixed_0.35": pb_fix, "own_best_conf": own,
            "pq_own_best": K.pq_value(e["counts"][own].sum(0)), "cuts": e["cuts"],
            "verdict": v}
        print(f"{name:10s} crossfit PQ {rep['ensembles'][name]['pq_crossfit']:.4f} "
              f"vs {args.ref} {rep['ref_pq_crossfit']:.4f}  d {pb['delta']:+.4f} "
              f"CI [{pb['ci95'][0]:+.4f},{pb['ci95'][1]:+.4f}] P {pb['p_gt_0']:.3f}  "
              f"folds+ {sum(1 for x in per_fold.values() if x > 0)}/5  FPx{fp_ratio:.2f}  {v}",
              flush=True)
    args.out.write_text(json.dumps(rep, indent=2) + "\n")


if __name__ == "__main__":
    main()
