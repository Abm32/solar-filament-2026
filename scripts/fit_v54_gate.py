"""V54 step 2: fit a proposal-quality logistic gate on FIT features (pure numpy).

Input: fit_features.npz (X=[N,F] features, y=[N] TP/FP labels) from
dump_v54_fit_features.py. Output: gate.json with standardization (mean/std),
logistic weights/bias, feature order, and a FIT held-out AUC.

The gate maps each V52 proposal to P(TP) = sigmoid(w.z + b), z = (x-mean)/std.
This replaces mean-probability as the fusion score so the V53 recall path (beta>0)
can become net-positive. FIT ONLY — no tune labels ever touch this (firewall).

No sklearn (not installed on the VM): standardization + L2-regularized logistic
regression via full-batch gradient descent, 80/20 FIT split for honest AUC.

  python scripts/fit_v54_gate.py --features outputs/v54_gate/fit_features.npz \
      --out outputs/v54_gate/gate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(x, y, l2=1e-3, lr=0.5, iters=4000, seed=0):
    """Full-batch GD logistic regression on standardized x. Returns (w, b)."""
    rng = np.random.default_rng(seed)
    n, f = x.shape
    w = rng.normal(0, 0.01, f)
    b = 0.0
    for _ in range(iters):
        p = sigmoid(x @ w + b)
        g = p - y
        gw = x.T @ g / n + l2 * w
        gb = float(g.mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def auc(y_true, scores):
    """ROC AUC via rank statistic (Mann-Whitney U)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    x = data["X"].astype(np.float64)
    y = data["y"].astype(np.float64)
    names = [str(s) for s in data["feature_names"]]

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    z = (x - mean) / std

    # honest 80/20 FIT split for AUC
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, va = perm[:cut], perm[cut:]

    w, b = fit_logistic(z[tr], y[tr], l2=args.l2, iters=args.iters, seed=args.seed)
    val_auc = auc(y[va], sigmoid(z[va] @ w + b))

    # mean-prob baseline AUC (the score V53 used) for comparison
    mp_idx = names.index("mean_prob")
    mp_auc = auc(y[va], x[va, mp_idx])

    # refit on all FIT data for the deployed gate
    w_full, b_full = fit_logistic(z, y, l2=args.l2, iters=args.iters, seed=args.seed)

    report = dict(
        n=len(y), n_tp=int(y.sum()), n_fp=int(len(y) - y.sum()),
        feature_names=names, mean=mean.tolist(), std=std.tolist(),
        weights=w_full.tolist(), bias=float(b_full),
        val_auc_gate=val_auc, val_auc_mean_prob=mp_auc,
        l2=args.l2, iters=args.iters, seed=args.seed,
        trained_on="FIT_only", reads_tune=False, reads_test=False,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("n", "n_tp", "n_fp", "val_auc_gate",
                               "val_auc_mean_prob")}, indent=2))
    print("weights:", dict(zip(names, (round(x, 3) for x in w_full), strict=True)))


if __name__ == "__main__":
    main()
