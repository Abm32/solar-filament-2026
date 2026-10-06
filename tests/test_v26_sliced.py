import ast
import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from scripts import evaluate_v26_sliced as v26

ROOT = Path(__file__).resolve().parents[1]


def encode(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def test_frozen_geometry_and_grid():
    assert v26.TILE_SIZE == 1024
    assert v26.TILE_STARTS == (0, 512, 1024)
    assert v26.OWNER_BOUNDARIES == (768, 1280)
    assert len(v26.CELLS) == 2
    assert len(v26.THRESHOLDS) * len(v26.CONFIDENCES) == 42
    assert v26.ARMS == ("control", "sliced", "hybrid")
    assert v26.NMS_IOU == 0.5


def test_owner_boundaries_are_deterministic():
    assert v26.owner_axis(0) == v26.owner_axis(767.999) == 0
    assert v26.owner_axis(768) == v26.owner_axis(1279.999) == 1
    assert v26.owner_axis(1280) == v26.owner_axis(2047.999) == 2
    assert v26.owner_origin(600, 1400) == (0, 1024)
    with pytest.raises(ValueError):
        v26.owner_axis(2048)


def test_tile_embedding_and_owner_filter():
    local = np.zeros((1024, 1024), dtype=np.uint8)
    local[480:520, 480:520] = 1
    full = v26.embed_tile_mask(local, 512, 512)
    assert full.shape == (2048, 2048)
    assert full.sum() == local.sum()
    assert v26.mask_centroid(full) == pytest.approx((1011.5, 1011.5))
    assert v26.owns_mask(full, 512, 512)
    assert not v26.owns_mask(full, 0, 0)


def test_tile_flip_roundtrip_preserves_global_geometry():
    tile = np.zeros((1024, 1024), dtype=np.uint8)
    tile[20:40, 700:730] = 1
    for axes in ((), (1,), (0,), (0, 1)):
        restored = np.ascontiguousarray(np.flip(np.flip(tile, axis=axes), axis=axes)) if axes else tile.copy()
        np.testing.assert_array_equal(restored, tile)
        full = v26.embed_tile_mask(restored, 1024, 0)
        assert full[20:40, 1724:1754].all()


def test_mask_nms_prefers_score_then_stable_label():
    first = np.zeros((64, 64), bool); first[5:25, 5:25] = 1
    duplicate = first.copy()
    separate = np.zeros((64, 64), bool); separate[35:55, 35:55] = 1
    rles = [encode(first), encode(duplicate), encode(separate)]
    kept, scores, labels = v26.mask_nms(rles, [0.8, 0.9, 0.7], ["full", "tile", "other"])
    assert scores == [0.9, 0.7]
    assert labels == ["tile", "other"]
    tied = v26.mask_nms(rles[:2], [0.9, 0.9], ["tile", "full"])
    assert tied[2] == ["full"]


def test_cache_identity_separates_branch_cell_origin_and_detector(monkeypatch, tmp_path):
    monkeypatch.setattr(v26, "OUTPUT", tmp_path)
    cell0, cell1 = v26.CELLS
    values = {
        v26.cache_path("full", "a", "x.jpeg", cell0),
        v26.cache_path("full", "a", "x.jpeg", cell1),
        v26.cache_path("tile", "a", "x.jpeg", cell0, (0, 0)),
        v26.cache_path("tile", "a", "x.jpeg", cell0, (512, 0)),
        v26.cache_path("tile", "b", "x.jpeg", cell0, (0, 0)),
    }
    assert len(values) == 5


def test_shards_cover_each_name_once():
    names = [f"{i:03}.jpeg" for i in range(94)]
    shards = [v26.shard_names(names, shard) for shard in range(4)]
    combined = [name for shard in shards for name in shard]
    assert sorted(combined) == names
    assert len(combined) == len(set(combined)) == 94
    assert [len(shard) for shard in shards] == [24, 24, 23, 23]


def test_candidate_grid_and_parse_are_frozen():
    values = {
        v26.candidate_key(arm, cell["name"], threshold, confidence)
        for arm in v26.ARMS for cell in v26.CELLS
        for threshold in v26.THRESHOLDS for confidence in v26.CONFIDENCES
    }
    assert len(values) == 252
    for value in values:
        arm, cell, threshold, confidence = v26.parse_candidate(value)
        assert arm in v26.ARMS and cell in v26.CELLS
        assert threshold in v26.THRESHOLDS and confidence in v26.CONFIDENCES
    with pytest.raises(ValueError):
        v26.parse_candidate("hybrid/iou030/t0.99_c0.30")


def test_tune_gate_constants_are_strict():
    assert v26.TUNE_REQUIRED == pytest.approx(v26.V17_TUNE_PQ + 0.01)
    assert v26.RAW_RECALL_DELTA == 0.02
    assert v26.HISTORICAL_RESCUE_REQUIRED == 15
    assert v26.BOTTOM_RECALL_DELTA == 0.03
    assert v26.BOTTOM_AREA_MAX == 800.0


def test_audit_path_is_gated_and_no_test_export_exists():
    source = (ROOT / "scripts/evaluate_v26_sliced.py").read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "submit" not in calls and "competitions_submit" not in calls
    assert "test/test_images" not in source
    assert "submission.csv" not in source
    assert "tune gates did not authorize audit" in source


def test_metric_record_matches_pq_definition():
    record = v26.metric_record(np.asarray([2.0, 4.0, 2.0, 2.0]))
    assert record["pq"] == pytest.approx(1 / 3)
    assert record["sq"] == pytest.approx(0.5)
    assert record["rq"] == pytest.approx(2 / 3)
