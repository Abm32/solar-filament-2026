from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from scripts import analyze_v39_fragment_graph_opportunity as v39

ROOT = Path(__file__).resolve().parents[1]


def encode(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def test_frozen_contract_matches_design():
    assert v39.MIN_GT_COVERAGE == 0.05
    assert v39.MIN_PROPOSAL_PURITY == 0.50
    assert v39.MIN_FRAGMENT_MASKS == 2
    assert v39.MIN_FRAGMENT_SOURCES == 2
    assert v39.MIN_UNION_AREA_COVERAGE == 0.65
    assert v39.MIN_UNION_SPINE_COVERAGE == 0.65
    assert v39.MISS_COVERAGE_GATE == 0.65
    assert v39.RECOVERY_IOU == 0.65
    assert v39.PROJECTED_PQ_GATE == 0.50
    assert v39.MEDIAN_COVERAGE_GATE == 0.75
    assert v39.EXPECTED_SOURCE_MATCHES == (841, 794, 826, 884, 843, 893)
    assert v39.EXPECTED_UNION_MATCHES == 922
    assert v39.EXPECTED_UNION_MISSES == 212


def test_fragment_union_is_addressable_and_exact_masks_are_deduplicated():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5:15, 4:16] = 1
    left = np.zeros_like(gt)
    left[5:15, 4:11] = 1
    right = np.zeros_like(gt)
    right[5:15, 9:16] = 1
    spine = np.zeros_like(gt)
    spine[10, 4:16] = 1

    result = v39.analyze_fragments(
        {"a": [encode(left)], "b": [encode(right), encode(left)]}, encode(gt), spine
    )

    assert result["admissible_fragment_masks"] == 2
    assert result["admissible_sources"] == 2
    assert result["graph_addressable"] is True
    assert result["union_area_coverage"] == 1.0
    assert result["union_spine_coverage"] == 1.0


def test_impure_or_incidental_proposals_are_rejected():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5:15, 5:15] = 1
    incidental = np.zeros_like(gt)
    incidental[5, 5] = 1
    impure = np.ones_like(gt)
    spine = np.zeros_like(gt)
    spine[10, 5:15] = 1

    result = v39.analyze_fragments(
        {"a": [encode(incidental)], "b": [encode(impure)]}, encode(gt), spine
    )

    assert result["admissible_fragment_masks"] == 0
    assert result["graph_addressable"] is False
    assert result["union_area_coverage"] == 0.0


def test_two_sources_without_two_distinct_masks_are_not_addressable():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5:15, 5:15] = 1
    fragment = gt.copy()
    spine = np.zeros_like(gt)
    spine[10, 5:15] = 1

    result = v39.analyze_fragments(
        {"a": [encode(fragment)], "b": [encode(fragment)]}, encode(gt), spine
    )

    assert result["admissible_fragment_masks"] == 1
    assert result["admissible_sources"] == 2
    assert result["graph_addressable"] is False


def test_projected_pq_requires_138_zero_fp_recoveries_to_clear_half():
    below = v39.projected_v17(137)
    passing = v39.projected_v17(138)
    assert below["pq"] < 0.50
    assert passing["pq"] >= 0.50
    assert passing["tp"] == 842
    assert passing["fn"] == 292


def test_runner_has_no_training_image_audit_test_export_or_submission_calls():
    source = (ROOT / "scripts/analyze_v39_fragment_graph_opportunity.py").read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute | ast.Name)
    }
    assert not ({"train", "fit", "predict", "to_csv", "submit", "imread"} & calls)
    lowered = source.lower()
    assert "test_images" not in lowered
    assert "kaggle competitions submit" not in lowered


def test_real_population_is_tune_only_and_hash_pinned():
    runtime = v39.canonical.load_population(v39.canonical.CANONICAL_ANNOTATIONS)
    assert runtime["populations"]["tune"] == {
        "files": 94,
        "records": 150,
        "annotations": 1134,
    }
    assert v39.canonical.sha256(runtime["path"]) == v39.canonical.ANNOTATION_SHA256
