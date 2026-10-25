"""V67 magnetogram features per mask (numpy/scipy/cv2 only) and small stats helpers.

All thresholds are fixed by docs/v67_preregistration_2026-10-07.md. Inputs are on the 512x512
H-alpha grid (2048 / 4).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.stats import rankdata

SIGMA = 2.0
T_POL = 25.0
STRONG = 150.0
PIL_RADIUS = 6
D_CAP = 40.0
FEATURES = ("mean_abs_bz", "max_abs_bz", "frac_strong", "mean_dpil", "frac_dpil_le5",
            "mean_bz", "abs_mean_bz")


def smooth_bz(bz512: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(bz512.astype(np.float32), (0, 0), SIGMA)


def pil_distance(bz_smooth: np.ndarray) -> np.ndarray:
    """Distance (px) to the nearest polarity-inversion pixel, capped at D_CAP."""
    pos = bz_smooth > T_POL
    neg = bz_smooth < -T_POL
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PIL_RADIUS + 1, 2 * PIL_RADIUS + 1))
    near_pos = cv2.dilate(pos.astype(np.uint8), k).astype(bool)
    near_neg = cv2.dilate(neg.astype(np.uint8), k).astype(bool)
    pil = near_pos & near_neg
    if not pil.any():
        return np.full(bz_smooth.shape, D_CAP, np.float32)
    return np.minimum(ndi.distance_transform_edt(~pil), D_CAP).astype(np.float32)


def mask_features(bz_smooth: np.ndarray, dpil: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Feature vector (len FEATURES) for one boolean mask on the 512 grid."""
    v = bz_smooth[mask]
    d = dpil[mask]
    if v.size == 0:
        return np.zeros(len(FEATURES), np.float32)
    a = np.abs(v)
    return np.array([a.mean(), a.max(), (a > STRONG).mean(), d.mean(), (d <= 5).mean(),
                     v.mean(), abs(v.mean())], np.float32)


def geometry(mask: np.ndarray) -> tuple[float, float]:
    """(log area, elongation = sqrt(l1/l2) of the coordinate covariance, >= 1)."""
    ys, xs = np.nonzero(mask)
    if ys.size < 3:
        return float(np.log1p(ys.size)), 1.0
    c = np.cov(np.vstack([xs, ys]).astype(np.float64))
    w = np.clip(np.linalg.eigvalsh(c), 1e-6, None)
    return float(np.log(ys.size)), float(np.sqrt(w[1] / w[0]))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based ROC-AUC (ties averaged); NaN if one class is missing."""
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """L2 logistic regression via L-BFGS on standardised inputs; returns weights (incl. bias)."""
    xb = np.c_[x, np.ones(len(x))]
    y = np.asarray(y, float)

    def f(w):
        z = xb @ w
        loss = np.logaddexp(0, z) - y * z
        p = 1 / (1 + np.exp(-z))
        reg = 0.5 * l2 * (w[:-1] @ w[:-1])
        grad = (xb.T @ (p - y) + l2 * np.r_[w[:-1], 0]) / len(y)
        return loss.sum() / len(y) + reg / len(y), grad

    return minimize(f, np.zeros(xb.shape[1]), jac=True, method="L-BFGS-B").x


def crossfit_scores(x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Out-of-fold logistic scores: fit on the other folds (standardised), predict fold k."""
    out = np.zeros(len(y))
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        w = fit_logistic((x[tr] - mu) / sd, y[tr])
        out[te] = np.c_[(x[te] - mu) / sd, np.ones(te.sum())] @ w
    return out


def delta_auc_bootstrap(y, s_base, s_aug, file_ids, n=2000, seed=67):
    """Paired file-bootstrap of AUC(aug) - AUC(base). Returns (delta, lo, hi, p_gt0)."""
    y, s_base, s_aug = np.asarray(y, bool), np.asarray(s_base), np.asarray(s_aug)
    files = np.unique(file_ids)
    rows = {f: np.flatnonzero(file_ids == f) for f in files}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = np.concatenate([rows[f] for f in rng.choice(files, len(files))])
        a, b = auc(y[idx], s_aug[idx]), auc(y[idx], s_base[idx])
        if not (np.isnan(a) or np.isnan(b)):
            deltas.append(a - b)
    d = np.asarray(deltas)
    return (auc(y, s_aug) - auc(y, s_base), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float((d > 0).mean()))


def verdict(d1: dict, d2: dict) -> str:
    """Pre-registered gate on the two diagnostics. Each dict: delta, lo, folds_pos."""
    for r in (d1, d2):
        if r["delta"] >= 0.030 and r["lo"] > 0 and r["folds_pos"] >= 4:
            return "continue"
    if d1["delta"] < 0.015 and d2["delta"] < 0.015:
        return "kill"
    return "inconclusive"
