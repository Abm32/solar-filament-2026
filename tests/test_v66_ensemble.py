"""Tests for the V66 voting-ensemble evaluator (synthetic masks + cache files, no GPU)."""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
pytest.importorskip("pycocotools")
from pycocotools import mask as mask_utils  # noqa: E402

spec = importlib.util.spec_from_file_location("eval_v66_ensemble", SCRIPTS / "eval_v66_ensemble.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
import v61_toolkit as K  # noqa: E402


def _rle(x0, y0, x1, y1, n=2048):
    m = np.zeros((n, n), np.uint8)
    m[y0:y1, x0:x1] = 1
    return mask_utils.encode(np.asfortranarray(m))


def _write_cache(path, views):
    """views: list of 4 lists of (rle, score)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = [dict(rles=[dict(size=list(r["size"]), counts=r["counts"].decode()) for r, _s in v],
                scores=[s for _r, s in v]) for v in views]
    with gzip.open(path, "wt") as f:
        json.dump(raw, f)


def _views(dets):
    """Same detections (list of (rle, score)) in all four flip views."""
    return [list(dets) for _ in range(4)]


def test_ensemble_of_model_with_itself_equals_single(tmp_path):
    a = tmp_path / "a.json.gz"
    _write_cache(a, _views([(_rle(100, 100, 200, 200), 0.9), (_rle(500, 500, 560, 560), 0.4)]))
    single = K.read_cache_views(a)
    r1, s1 = K.cluster_views(single)
    both = K.read_cache_views(a) + K.read_cache_views(a)
    r2, s2 = K.cluster_views(both)
    assert len(r1) == len(r2)
    assert np.allclose(sorted(s1), sorted(s2))


def test_detection_from_one_member_is_capped_at_one_over_n(tmp_path):
    a, b = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    shared = _rle(100, 100, 200, 200)          # area 10,000: both members detect it
    lonely = _rle(800, 800, 850, 850)          # area 2,500: only member A detects it
    _write_cache(a, _views([(shared, 0.9), (lonely, 0.9)]))
    _write_cache(b, _views([(shared, 0.9)]))
    rles, scores = K.cluster_views(K.read_cache_views(a) + K.read_cache_views(b))
    by_area = {int(mask_utils.area(r)): s for r, s in zip(rles, scores, strict=True)}
    assert set(by_area) == {10000, 2500}
    assert abs(by_area[10000] - 0.9) < 1e-9     # agreed by both members: keeps full score
    assert abs(by_area[2500] - 0.45) < 1e-9     # one member of two: capped at 0.9 * 1/2


def test_verdict_rules():
    v = E.verdict
    assert v(0.015, 0.95, 0.004, 5, 1.0) == "continue"
    assert v(0.004, 0.99, 0.001, 5, 1.0) == "kill"
    assert v(0.02, 0.99, 0.01, 2, 1.0) == "kill"
    assert v(0.012, 0.85, 0.004, 5, 1.0) == "inconclusive"     # P < 0.90
    assert v(0.012, 0.95, -0.001, 5, 1.0) == "inconclusive"    # CI crosses 0
    assert v(0.012, 0.95, 0.004, 3, 1.0) == "inconclusive"     # < 4 folds
    assert v(0.012, 0.95, 0.004, 5, 1.31) == "inconclusive"    # FP blow-up
    assert v(0.010, 0.90, 0.001, 4, 1.30) == "continue"        # boundaries inclusive


def test_crossfit_uses_only_other_folds():
    confs = K.CONFIDENCES
    # fold 0 prefers the LOWEST cut, folds 1-2 prefer the HIGHEST cut
    fold_of = np.array([0, 0, 1, 1, 2, 2])
    counts = {}
    for c in confs:
        rows = []
        for f in fold_of:
            tp = 10 if ((f == 0 and c == confs[0]) or (f != 0 and c == confs[-1])) else 2
            rows.append([tp * 0.7, tp, 1, 1])
        counts[c] = np.asarray(rows, float)
    cf, cuts = E.crossfit_counts(counts, fold_of)
    assert cuts[0] == confs[-1]            # fold 0's cut comes from folds 1-2 only
    assert cuts[1] == confs[-1] or cuts[1] == confs[0]
    assert cf.shape == counts[confs[0]].shape
    # fold 0's rows use the cut chosen WITHOUT fold 0 (the high cut), i.e. tp = 2 there
    assert cf[fold_of == 0][:, 1].sum() == 4
