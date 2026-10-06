"""Tests for V60D census pure metrics (no torch, no GPU)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("census_v60d", SCRIPTS / "census_v60d.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"deps unavailable: {e}")
    return m


C = _load()


def _file(score, gid, iou_pg, iou_pp=None):
    n = len(score)
    pp = np.eye(n, dtype=np.float32) if iou_pp is None else np.asarray(iou_pp, np.float32)
    return {"score": np.asarray(score, np.float32), "gid": np.asarray(gid, int),
            "iou_pg": np.asarray(iou_pg, np.float32).reshape(n, -1), "iou_pp": pp,
            "gt_ann": np.zeros(np.asarray(iou_pg).reshape(n, -1).shape[1], int)}


def test_select_conf_and_nms():
    score = np.array([0.9, 0.8, 0.3], np.float32)
    pp = np.array([[1, .7, 0], [.7, 1, 0], [0, 0, 1]], np.float32)
    assert list(C.select(score, pp, 0.5, None)) == [0, 1]
    assert list(C.select(score, pp, 0.5, 0.5)) == [0]           # dup of 0 suppressed
    assert list(C.select(score, pp, 0.0, 0.5)) == [0, 2]


def test_single_perfect_proposal_has_no_dups_no_fp():
    f = _file([0.9], [0], [[0.9]])
    m = C.census_metrics([f], 0.0, None)
    assert m["gt_covered"] == 1 and m["dups_per_covered_gt"] == 0
    assert m["fp_per_image"] == 0 and m["proposals_per_image"] == 1


def test_duplicates_counted_and_split_within_vs_cross_group():
    # 3 proposals all hit the one GT: two from group 0 (within-tile dup), one from group 1
    f = _file([0.9, 0.8, 0.7], [0, 0, 1], [[0.8], [0.7], [0.6]])
    m = C.census_metrics([f], 0.0, None)
    assert m["dups_per_covered_gt"] == 2.0
    assert m["within_group_per_covered_gt"] == 2.0
    assert m["groups_per_covered_gt"] == 2.0


def test_fp_is_proposal_matching_no_gt():
    f = _file([0.9, 0.8], [0, 0], [[0.8], [0.1]])
    m = C.census_metrics([f], 0.0, None)
    assert m["fp_per_image"] == 1.0 and m["gt_covered"] == 1


def test_nms_removes_duplicates_but_can_lose_recall():
    pp = [[1, .9, 0], [.9, 1, 0], [0, 0, 1]]
    f = _file([0.9, 0.8, 0.7], [0, 1, 2], [[0.8, 0], [0.7, 0], [0, 0.9]], pp)
    raw = C.census_metrics([f], 0.0, None)
    nms = C.census_metrics([f], 0.0, 0.5)
    assert raw["dups_per_covered_gt"] > 0 and nms["dups_per_covered_gt"] == 0
    assert nms["gt_covered"] == raw["gt_covered"] == 2


def test_uncovered_gt_lowers_covered_frac():
    f = _file([0.9], [0], [[0.9, 0.0]])          # 2 GT, only first matched
    m = C.census_metrics([f], 0.0, None)
    assert m["gt_total"] == 2 and m["gt_covered"] == 1 and m["covered_frac"] == 0.5
