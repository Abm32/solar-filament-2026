"""Tests for V60C-0 consensus-mask PQ logic (synthetic masks, no torch model)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"{name} deps unavailable: {e}")
    return m


S = 512


def _box(x0, y0, x1, y1):
    m = np.zeros((S, S), bool)
    m[y0:y1, x0:x1] = True
    return m


def _tile(mask, score=0.9, ver=1.0):
    ys, xs = np.nonzero(mask)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    return {"mask": mask, "score": score, "ver": ver,
            "area": int(mask.sum()), "elong": max(w, h) / max(min(w, h), 1)}


def test_cluster_collapses_to_one_consensus_tp():
    c0 = _load("fuse_v60c0")
    g = _box(300, 300, 350, 350)
    # 3 near-duplicate partial views of the same filament -> one cluster
    t1 = _tile(_box(300, 300, 350, 348), 0.9)
    t2 = _tile(_box(302, 300, 350, 350), 0.8)
    t3 = _tile(_box(300, 302, 348, 350), 0.7)
    d = {"v17": [], "tiles": [t1, t2, t3], "gt_sets": [[g]], "all_gt": [g],
         "tile_only_gt": np.array([True])}
    r = c0.score_c0([d], tau=0.25, conf_thr=0.0, overlap_max=0.25)
    # the cloud collapses to ONE object -> at most 1 added, and it should match the GT
    assert r["consensus_instances_added"] == 1
    assert r["tp"] == 1 and r["consensus_only_tp"] == 1
    assert r["fp"] == 0            # the key win: cloud -> one object, not 3 FP


def test_consensus_not_added_over_good_v17():
    c0 = _load("fuse_v60c0")
    g = _box(100, 100, 160, 160)
    v17 = g.copy()
    overlapping = _tile(_box(102, 102, 158, 158), 0.95)
    d = {"v17": [v17], "tiles": [overlapping], "gt_sets": [[g]], "all_gt": [g],
         "tile_only_gt": np.array([False])}
    r = c0.score_c0([d], tau=0.25, conf_thr=0.0, overlap_max=0.25)
    assert r["consensus_instances_added"] == 0   # weak/absent-region rule protects V17
    assert r["tp"] == 1 and r["fp"] == 0


def test_conf_floor_blocks_low_cluster():
    c0 = _load("fuse_v60c0")
    g = _box(300, 300, 350, 350)
    weak = _tile(g.copy(), score=0.2)
    d = {"v17": [], "tiles": [weak], "gt_sets": [[g]], "all_gt": [g],
         "tile_only_gt": np.array([True])}
    hi = c0.score_c0([d], tau=0.25, conf_thr=0.0, overlap_max=0.25)
    lo = c0.score_c0([d], tau=0.25, conf_thr=0.5, overlap_max=0.25)
    assert hi["consensus_instances_added"] == 1
    assert lo["consensus_instances_added"] == 0   # cluster max-score below floor
