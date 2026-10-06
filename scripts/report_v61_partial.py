"""Partial baseline OOF report over the folds whose predictions exist (no comparison, A only).

Prints per-fold and pooled direct PQ across the declared confidence grid plus TP/FP/FN, so the
scoring path can be sanity-checked on real held-out data before the B comparison exists.
This looks at the BASELINE only; it cannot influence the B pre-registered gate.

  python scripts/report_v61_partial.py --recipe baseline --folds 0 1 2 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import eval_v61_oof as O  # noqa: E402
import v61_toolkit as K  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", default="baseline")
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    import json

    doc = json.loads(O.ANNOTATIONS.read_text())
    gts = O.gt_by_file(doc)
    folds = O.load_folds()
    per_fold = {}
    for k in args.folds:
        rows = {c: [] for c in K.CONFIDENCES}
        for name in folds[k]:
            res = O.file_counts(args.recipe, k, name, gts[name])
            for c in K.CONFIDENCES:
                rows[c].append(res[c])
        per_fold[k] = {c: np.asarray(v) for c, v in rows.items()}
    print("conf  " + "  ".join(f"fold{k}" for k in args.folds) + "   pooled    TP    FP    FN")
    best = (-1, None)
    for c in K.CONFIDENCES:
        pf = [K.pq_value(per_fold[k][c].sum(0)) for k in args.folds]
        tot = sum(per_fold[k][c].sum(0) for k in args.folds)
        pooled = K.pq_value(tot)
        if pooled > best[0]:
            best = (pooled, c)
        print(f"{c:.2f}  " + "  ".join(f"{p:.4f}" for p in pf)
              + f"   {pooled:.4f}  {int(tot[1]):5d} {int(tot[2]):5d} {int(tot[3]):5d}")
    allrows = np.concatenate([per_fold[k][best[1]] for k in args.folds])
    lo, hi = K.bootstrap_pq(allrows)
    print(f"best pooled conf={best[1]:.2f} PQ={best[0]:.4f} 95%CI=[{lo:.4f},{hi:.4f}] "
          f"files={len(allrows)}")


if __name__ == "__main__":
    main()
