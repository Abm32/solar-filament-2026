"""Focused tests for v53_fusion pure logic (no GPU, no V17 caches).

Covers the key(), control_key(), fuse() matching semantics and read_v52() cache I/O
+ validation. The full fusion/eval/merge path runs on the VM against real caches.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_v53():
    # v53_fusion imports reproduce_v17_from_cache; make both importable by path.
    import sys

    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("v53_fusion", SCRIPTS / "v53_fusion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v53 = _load_v53()


def _box_rle(r0, r1, c0, c1):
    m = np.zeros((2048, 2048), dtype=np.uint8, order="F")
    m[r0:r1, c0:c1] = 1
    rle = mask_utils.encode(np.asfortranarray(m))
    return dict(size=list(rle["size"]), counts=rle["counts"])


def test_key_and_control_key():
    assert v53.key("w1_t060", 0.0, 0.0, 0.45, 0.30) == "w1_t060/a0.00_b0.00/t0.45_c0.30"
    assert v53.control_key("base32_t040", 0.45, 0.30) == "base32_t040/a0.00_b0.00/t0.45_c0.30"


def test_fuse_matches_overlapping_and_reports_unmatched():
    # V17 proposal A overlaps extra[0] fully; extra[1] is disjoint (unmatched).
    v17 = [_box_rle(0, 100, 0, 100)]
    extra = [_box_rle(0, 100, 0, 100), _box_rle(500, 600, 500, 600)]
    scores = [0.9, 0.4]
    matched, unmatched = v53.fuse(mask_utils, v17, extra, scores)
    assert matched.shape == (1,)
    assert matched[0] == pytest.approx(0.9)  # best matching extra score
    assert unmatched == [1]  # the disjoint extra is unmatched -> candidate for beta add


def test_fuse_no_overlap_all_unmatched():
    v17 = [_box_rle(0, 50, 0, 50)]
    extra = [_box_rle(500, 600, 500, 600)]
    matched, unmatched = v53.fuse(mask_utils, v17, extra, [0.7])
    assert matched[0] == 0.0
    assert unmatched == [0]


def test_fuse_empty_v17_keeps_all_extras_unmatched():
    extra = [_box_rle(0, 50, 0, 50), _box_rle(500, 600, 500, 600)]
    matched, unmatched = v53.fuse(mask_utils, [], extra, [0.3, 0.8])
    assert len(matched) == 0
    assert unmatched == [0, 1]


def test_read_v52_roundtrip_and_validation(tmp_path):
    name = "20200101000000Ch.jpeg"
    cache_set = tmp_path / "w1_t060"
    cache_set.mkdir(parents=True)
    rle = mask_utils.encode(np.asfortranarray(
        (lambda m: (m.__setitem__((slice(0, 100), slice(0, 100)), 1), m)[1])(
            np.zeros((2048, 2048), dtype=np.uint8, order="F"))))
    payload = dict(file=name, threshold=0.6,
                   rles=[dict(size=[2048, 2048], counts=rle["counts"].decode())],
                   scores=[0.87])
    with gzip.open(cache_set / (name + ".json.gz"), "wt") as stream:
        json.dump(payload, stream)

    old_cache = v53.V52_CACHE
    v53.V52_CACHE = tmp_path
    try:
        rles, scores = v53.read_v52("w1_t060", name)
    finally:
        v53.V52_CACHE = old_cache
    assert len(rles) == 1
    assert rles[0]["size"] == [2048, 2048]
    assert isinstance(rles[0]["counts"], bytes)  # re-encoded for pycocotools
    assert scores == [0.87]


def test_read_v52_rejects_wrong_file(tmp_path):
    cache_set = tmp_path / "w1_t060"
    cache_set.mkdir(parents=True)
    with gzip.open(cache_set / ("a.jpeg.json.gz"), "wt") as stream:
        json.dump(dict(file="DIFFERENT.jpeg", threshold=0.6, rles=[], scores=[]), stream)
    old_cache = v53.V52_CACHE
    v53.V52_CACHE = tmp_path
    try:
        with pytest.raises(RuntimeError, match="Stale/invalid"):
            v53.read_v52("w1_t060", "a.jpeg")
    finally:
        v53.V52_CACHE = old_cache


def test_predeclared_grid_and_gates():
    assert v53.ALPHAS == (0.0, 0.25, 0.5)
    assert v53.BETAS == (0.0, 0.5, 1.0)
    assert v53.PROMOTION_DELTA == 0.010
    assert v53.V17_TUNE_PQ == pytest.approx(0.44870271727943206)
    # control (a0 b0) must exist in the grid for every extra set (bit-identity anchor)
    for s in v53.EXTRA_SETS:
        assert 0.0 in v53.ALPHAS and 0.0 in v53.BETAS
        assert v53.control_key(s, 0.45, 0.30).endswith("a0.00_b0.00/t0.45_c0.30")
