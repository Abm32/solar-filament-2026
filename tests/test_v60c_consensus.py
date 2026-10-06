"""Tests for V60C-0 consensus diagnostic pure logic (clusters, ranks)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("diagnose_v60c_consensus",
                                                  SCRIPTS / "diagnose_v60c_consensus.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"deps unavailable: {e}")
    return m


def test_clusters_single_link():
    c = _load()
    # 0-1 overlap, 2 isolated, 3-4 overlap
    tt = np.array([[1, .5, 0, 0, 0],
                   [.5, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, .4],
                   [0, 0, 0, .4, 1]], np.float32)
    comps = sorted([sorted(x) for x in c._clusters(tt, 0.1)])
    assert comps == [[0, 1], [2], [3, 4]]


def test_clusters_chain_merges():
    c = _load()
    # chain 0-1-2 (0-2 don't directly overlap) must merge via single-link
    tt = np.array([[1, .3, 0],
                   [.3, 1, .3],
                   [0, .3, 1]], np.float32)
    comps = c._clusters(tt, 0.1)
    assert len(comps) == 1 and sorted(comps[0]) == [0, 1, 2]


def test_rank_of_correct_proposal():
    c = _load()
    cluster = [0, 1, 2]
    conf = np.array([0.9, 0.5, 0.7])     # idx0 highest
    # correct proposal is idx1 (lowest confidence) -> rank 3 under confidence
    assert c._rank_of(conf, 1, cluster) == 3
    # under a consensus score that favors idx1 -> rank 1
    cons = np.array([0.1, 0.9, 0.2])
    assert c._rank_of(cons, 1, cluster) == 1
