"""Tests for V60B add-only fusion logic (synthetic masks, no torch model)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    # fuse_v60b imports heavy modules at module load; stub sys.path and import lazily.
    spec = importlib.util.spec_from_file_location("fuse_v60b", SCRIPTS / "fuse_v60b.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                       # pragma: no cover - env-dependent
        pytest.skip(f"fuse_v60b deps unavailable: {e}")
    return m


S = 512


def _box_mask(x0, y0, x1, y1):
    m = np.zeros((S, S), bool)
    m[y0:y1, x0:x1] = True
    return m


def _tile(mask, score=0.9, ver=1.0):
    ys, xs = np.nonzero(mask)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    return {"mask": mask, "score": score, "ver": ver,
            "area": int(mask.sum()), "elong": max(w, h) / max(min(w, h), 1)}


def test_v17_base_only_matches_gt():
    fz = _load()
    gt = _box_mask(10, 10, 60, 60)
    img = {"v17": [gt.copy()], "v17s": np.array([0.9]), "tiles": [],
           "gt_sets": [[gt]], "all_gt": [gt], "tile_only_gt": np.array([False])}
    r = fz.score_fusion([img], conf_thr=0.5, ver_thr=0.0, overlap_max=0.25)
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0


def test_tile_only_addition_recovers_missed_gt():
    fz = _load()
    g1 = _box_mask(10, 10, 60, 60)              # covered by V17
    g2 = _box_mask(300, 300, 350, 350)          # missed by V17, found by a tile
    tile = _tile(g2.copy(), score=0.9)
    img = {"v17": [g1.copy()], "v17s": np.array([0.9]), "tiles": [tile],
           "gt_sets": [[g1, g2]], "all_gt": [g1, g2],
           "tile_only_gt": np.array([False, True])}
    base = fz.score_fusion([img], conf_thr=1.01, ver_thr=0.0, overlap_max=0.25)  # no tiles
    fused = fz.score_fusion([img], conf_thr=0.5, ver_thr=0.0, overlap_max=0.25)
    assert base["tp"] == 1 and base["fn"] == 1
    assert fused["tp"] == 2 and fused["fn"] == 0   # tile-only GT recovered, no new FP


def test_add_only_rejects_overlapping_tile():
    fz = _load()
    gt = _box_mask(10, 10, 60, 60)
    dup = _tile(_box_mask(12, 12, 62, 62), score=0.95)   # heavy overlap with V17
    img = {"v17": [gt.copy()], "v17s": np.array([0.9]), "tiles": [dup],
           "gt_sets": [[gt]], "all_gt": [gt], "tile_only_gt": np.array([False])}
    r = fz.score_fusion([img], conf_thr=0.5, ver_thr=0.0, overlap_max=0.25)
    assert r["tp"] == 1 and r["fp"] == 0           # overlapping tile NOT added -> no FP


def test_cross_source_nms_dedups_tiles():
    fz = _load()
    g = _box_mask(300, 300, 350, 350)
    t1 = _tile(g.copy(), score=0.9)
    t2 = _tile(_box_mask(302, 302, 352, 352), score=0.8)   # near-dup of t1
    img = {"v17": [], "v17s": np.zeros(0), "tiles": [t1, t2],
           "gt_sets": [[g]], "all_gt": [g], "tile_only_gt": np.array([True])}
    r = fz.score_fusion([img], conf_thr=0.5, ver_thr=0.0, overlap_max=0.25)
    assert r["tp"] == 1 and r["fp"] == 0           # only one of the duplicate tiles kept


def test_verifier_threshold_blocks_low_prob_tile():
    fz = _load()
    g = _box_mask(300, 300, 350, 350)
    weak = _tile(g.copy(), score=0.9, ver=0.3)      # confident detector, low verifier
    img = {"v17": [], "v17s": np.zeros(0), "tiles": [weak],
           "gt_sets": [[g]], "all_gt": [g], "tile_only_gt": np.array([True])}
    off = fz.score_fusion([img], conf_thr=0.5, ver_thr=0.0, overlap_max=0.25)
    on = fz.score_fusion([img], conf_thr=0.5, ver_thr=0.7, overlap_max=0.25)
    assert off["tp"] == 1                           # verifier off -> added
    assert on["tp"] == 0 and on["fn"] == 1          # verifier gate blocks it
