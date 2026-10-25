"""Tests for V67 magnetogram features, AUC/logistic helpers, registration geometry, and the gate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import v61_toolkit  # noqa: E402,F401  (ensures scripts dir importable like the other tests)
import v67_features as F  # noqa: E402
import v67_register as R  # noqa: E402


def test_auc_matches_known_values():
    y = np.array([0, 0, 1, 1])
    assert F.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert F.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert F.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert np.isnan(F.auc(np.array([1, 1]), np.array([0.2, 0.3])))


def test_logistic_recovers_signal_and_crossfit_is_out_of_fold():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 2))
    y = (x[:, 0] + 0.3 * rng.normal(size=600) > 0).astype(int)
    fold = np.arange(600) % 5
    s = F.crossfit_scores(x, y, fold)
    assert F.auc(y, s) > 0.9
    noise = F.crossfit_scores(rng.normal(size=(600, 2)), y, fold)
    assert abs(F.auc(y, noise) - 0.5) < 0.08          # no leakage: noise features stay ~0.5


def test_pil_distance_zero_on_inversion_line_and_capped_elsewhere():
    bz = np.zeros((128, 128), np.float32)
    bz[:, :64] = 200.0
    bz[:, 64:] = -200.0
    d = F.pil_distance(bz)
    assert d[:, 63:65].max() == 0.0
    assert d[64, 5] > 40 - 1e-6 or d[64, 5] >= 20     # far from the line
    assert d.max() <= F.D_CAP
    unipolar = np.full((128, 128), 300.0, np.float32)
    assert (F.pil_distance(unipolar) == F.D_CAP).all()


def test_mask_features_unipolar_vs_bipolar():
    bz = np.zeros((128, 128), np.float32)
    bz[:, :64], bz[:, 64:] = 200.0, -200.0
    dpil = F.pil_distance(bz)
    on_line = np.zeros_like(bz, bool)
    on_line[30:90, 60:68] = True
    off_line = np.zeros_like(bz, bool)
    off_line[30:90, 5:13] = True
    f_on, f_off = F.mask_features(bz, dpil, on_line), F.mask_features(bz, dpil, off_line)
    i = F.FEATURES.index("mean_dpil")
    assert f_on[i] < f_off[i]
    assert F.mask_features(bz, dpil, np.zeros_like(on_line)).sum() == 0
    assert abs(f_off[F.FEATURES.index("abs_mean_bz")] - 200.0) < 1e-3


def test_geometry_elongation():
    thin = np.zeros((100, 100), bool)
    thin[50:52, 10:90] = True
    blob = np.zeros((100, 100), bool)
    blob[40:60, 40:60] = True
    assert F.geometry(thin)[1] > 5 * F.geometry(blob)[1]


def test_verdict_rules_are_mechanical():
    ok = dict(delta=0.04, lo=0.01, folds_pos=5)
    flat = dict(delta=0.005, lo=-0.01, folds_pos=2)
    mid = dict(delta=0.02, lo=0.001, folds_pos=5)
    assert F.verdict(ok, flat) == "continue" and F.verdict(flat, ok) == "continue"
    assert F.verdict(flat, flat) == "kill"
    assert F.verdict(mid, flat) == "inconclusive"
    assert F.verdict(dict(delta=0.04, lo=-0.001, folds_pos=5), flat) == "inconclusive"
    assert F.verdict(dict(delta=0.04, lo=0.01, folds_pos=3), flat) == "inconclusive"


def test_warp_maps_disk_circle_to_disk_circle_and_flips_rows():
    n = 860
    yy, xx = np.mgrid[:n, :n]
    bz = np.where(yy < n // 2, 100.0, -100.0).astype(np.float32)      # top half positive (row 0)
    geom = dict(cx=429.0, cy=430.0, r=380.0, time="")
    disk = (1023.0, 1024.0, 900.0)
    w = R.warp_mag(bz, geom, disk, out_size=2048)
    assert w[1024, 1023] != 0 or w[1030, 1023] != 0
    assert (w[:50, :50] == 0).all() and (w[-50:, -50:] == 0).all()   # off-disk zeroed
    # FITS row 0 is the bottom: positive half (low rows) must map to the BOTTOM of the image
    assert w[1500, 1023] > 0 > w[550, 1023]
    inside = ((np.mgrid[:2048, :2048][1] - 1023) ** 2 + (np.mgrid[:2048, :2048][0] - 1024) ** 2
              < (0.97 * 900) ** 2)
    assert (w[inside] != 0).mean() > 0.99


def test_fit_disk_recovers_synthetic_limb():
    import cv2

    img = np.zeros((2048, 2048), np.uint8)
    cv2.circle(img, (1010, 1040), 880, 170, -1)
    cv2.circle(img, (1010, 1040), 960, 40, 1, lineType=cv2.LINE_AA)            # dim halo ring
    halo = cv2.GaussianBlur(img, (0, 0), 25)
    img = np.maximum(img, (halo * 0.6).astype(np.uint8))
    cx, cy, r = R.fit_disk(img)
    assert abs(cx - 1010) < 4 and abs(cy - 1040) < 4 and abs(r - 880) < 8
