"""Decisive out-of-fold test for E-C: can a LEARNED combination of cheap features
beat the existing 0.40 consensus gate at a matched accepted-proposal budget?

Consumes `outputs/ec_feasibility_2026-09-06/proposal_pool.csv` produced by
`scripts/analyze_ec_feasibility.py`. CPU only; trains nothing on GPU; runs no
inference; exports no submission.

Why this is the decisive step
-----------------------------
The fixed-feature study showed the consensus score is already the strongest
single ranker (AUC ~0.726) and that no cheap geometric feature beats it. That
does NOT yet answer E-C's question, because E-C proposes a *learned combination*
(a mask-scoring head), which could add incremental signal on top of consensus.

This script answers it honestly with grouped K-fold cross-validation:

  * Folds are split by FILE, never within a file, so proposals from one image
    never appear in both train and validation. This is the leakage control that
    a naive row-wise split would violate.
  * Every reported number is OUT-OF-FOLD: the model never sees the validation
    rows' labels.
  * The comparison is at a MATCHED accepted-proposal budget, exactly as E-C
    predeclares -- same number of accepted proposals as the 0.40 gate, so a
    "gain" cannot come from simply admitting more proposals.

Logistic regression is implemented directly in numpy (sklearn is unavailable in
this environment). It is a linear probe: it establishes whether the FEATURES
carry incremental information, not the ceiling of a deep mask-scoring head. A
CNN head reading mask pixels could exceed a linear probe on these summary
features. So a negative result here bounds the value of *these cheap features*,
and is evidence about, but not proof against, the full E-C design.

Interpretation guide, fixed before running
------------------------------------------
  * OOF AUC materially above the consensus AUC AND a positive matched-budget
    delta  -> E-C's premise is supported; the pool is separable; fund E-C.
  * OOF AUC ~= consensus AUC and matched-budget delta ~<= 0 -> these features
    add nothing over the score the pipeline already uses. E-C would need
    genuinely new inputs (mask-pixel features from a trained head), so its
    cheap-feature version is dead and the budget should shift toward V12.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ec_feasibility_2026-09-06"
ACCEPT_CONF = 0.40
N_FOLDS = 5
SEED = 42


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact ROC AUC via the rank-sum identity, ties averaged."""
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2,
                 iters: int = 400) -> np.ndarray:
    """Newton-ish gradient descent logistic regression with L2, numpy only."""
    X = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (p - y) / len(X)
        grad[1:] += l2 * w[1:]
        # Diagonal Hessian approximation keeps this stable without scipy.
        h = (X ** 2).T @ (p * (1 - p)) / len(X)
        h[1:] += l2
        w -= grad / np.maximum(h, 1e-8)
    return w


def predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.hstack([np.ones((len(X), 1)), X])
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def main() -> None:
    frame = pd.read_csv(OUT / "proposal_pool.csv")
    y = frame["label"].to_numpy()

    feature_sets = {
        "consensus_only": ["consensus_score"],
        "geometry_only": ["log_area", "fill_ratio", "elongation", "compactness",
                          "solidity_proxy", "n_components"],
        "consensus_plus_geometry": ["consensus_score", "log_area", "fill_ratio",
                                    "elongation", "compactness",
                                    "solidity_proxy", "n_components"],
    }

    # Group folds by file so no image spans train and validation.
    files = frame["file_name"].to_numpy()
    unique_files = np.array(sorted(pd.unique(files)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_files)
    fold_of_file = {f: i % N_FOLDS for i, f in enumerate(unique_files)}
    fold = np.array([fold_of_file[f] for f in files])

    accepted = frame["consensus_score"].to_numpy() >= ACCEPT_CONF
    budget = int(accepted.sum())
    gate_positives = int(y[accepted].sum())

    results = {}
    for setname, cols in feature_sets.items():
        X = frame[cols].to_numpy(dtype=float)
        oof = np.full(len(frame), np.nan)
        for k in range(N_FOLDS):
            tr, va = fold != k, fold == k
            mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            w = fit_logistic((X[tr] - mu) / sd, y[tr])
            oof[va] = predict_logistic(w, (X[va] - mu) / sd)
        assert not np.isnan(oof).any()
        order = np.argsort(-oof, kind="mergesort")[:budget]
        admitted = int(y[order].sum())
        results[setname] = {
            "features": cols,
            "oof_auc": roc_auc(oof, y),
            "matched_budget_positives": admitted,
            "delta_vs_gate": admitted - gate_positives,
        }

    gate_auc = roc_auc(frame["consensus_score"].to_numpy(), y)
    best = max(results, key=lambda k: results[k]["oof_auc"])
    best_delta = max(r["delta_vs_gate"] for r in results.values())

    verdict = (
        "SUPPORTED" if (results[best]["oof_auc"] > gate_auc + 0.02 and best_delta > 0)
        else "NOT SUPPORTED (cheap-feature version)")

    summary = {
        "scope": ("94 tuning files; frozen V6 flip_consensus grouped proposals; "
                  f"{N_FOLDS}-fold cross-validation grouped BY FILE; seed {SEED}"),
        "warning": (
            "Out-of-fold linear probe on cheap summary features. Establishes "
            "whether THESE FEATURES carry incremental signal over the consensus "
            "score. A trained CNN mask-scoring head reading mask pixels could "
            "exceed a linear probe, so a negative result bounds the cheap-feature "
            "variant of E-C, not every possible E-C implementation. Labels use "
            "pre-refiner proposal geometry. Not PQ; not a public score forecast."),
        "n_proposals": int(len(frame)),
        "n_positive": int(y.sum()),
        "gate": {
            "accept_confidence": ACCEPT_CONF,
            "accepted_budget": budget,
            "accepted_positives": gate_positives,
            "auc": gate_auc,
        },
        "models": results,
        "best_model": {"name": best, **results[best]},
        "best_matched_budget_delta": best_delta,
        "verdict": verdict,
    }
    (OUT / "ec_oof_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps({k: v for k, v in summary.items() if k != "models"},
                     indent=2, sort_keys=True))
    print("\nOut-of-fold model comparison (budget "
          f"{budget}, gate admits {gate_positives} positives, "
          f"gate AUC {gate_auc:.4f}):")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["oof_auc"]):
        print(f"  {name:26s} oof_auc={r['oof_auc']:.4f}  "
              f"pos@budget={r['matched_budget_positives']:4d}  "
              f"delta={r['delta_vs_gate']:+d}")


if __name__ == "__main__":
    main()
