from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from scripts import analyze_v40_multiscale_ridge_signal as v40

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_contract_matches_design():
    assert v40.FRANGI_SIGMAS == (1, 2, 4, 8)
    assert v40.FRANGI_ALPHA == 0.5
    assert v40.FRANGI_BETA == 0.5
    assert v40.FRANGI_GAMMA is None
    assert v40.BLACK_RIDGES is True
    assert v40.CROP_MARGIN == 48
    assert v40.SPINE_DILATION_RADIUS == 2
    assert (v40.BACKGROUND_INNER, v40.BACKGROUND_OUTER) == (8.0, 24.0)
    assert v40.AUC_GATE == 0.70
    assert v40.Q90_COVERAGE_GATE == 0.25
    assert v40.MISS_ADDRESSABLE_GATE == 0.65
    assert v40.ZERO_FRAGMENT_ADDRESSABLE_GATE == 0.65
    assert v40.FOUND_ADDRESSABLE_GATE == 0.80
    assert v40.RECOVERY_IOU == 0.65
    assert v40.PROJECTED_PQ_GATE == 0.50
    assert (v40.EXPECTED_FOUND, v40.EXPECTED_MISSED, v40.EXPECTED_ZERO_FRAGMENT) == (
        922, 212, 113
    )


def test_average_rank_auc_handles_order_and_ties():
    assert v40.average_rank_auc(np.array([3, 4]), np.array([0, 1])) == 1.0
    assert v40.average_rank_auc(np.ones(3), np.ones(4)) == 0.5
    assert v40.average_rank_auc(np.array([0, 1]), np.array([3, 4])) == 0.0


def test_dark_ridge_response_exceeds_bright_and_flat_controls():
    checks = v40.synthetic_checks()
    assert checks["passed"] is True
    assert checks["dark_score"] > checks["bright_score"]
    assert checks["dark_score"] > checks["flat_score"]


def test_measure_signal_detects_synthetic_dark_line():
    size = 160
    image = np.full((size, size), 180, dtype=np.uint8)
    gt = np.zeros((size, size), dtype=bool)
    gt[76:85, 30:130] = True
    spine = np.zeros_like(gt)
    spine[80, 32:128] = True
    cv = image.copy()
    cv[76:85, 30:130] = 50
    all_annotations = gt.copy()

    result = v40.measure_signal(cv, gt, spine, all_annotations)

    assert result["positive_pixels"] >= v40.MIN_POSITIVE_PIXELS
    assert result["clean_background_pixels"] >= v40.MIN_BACKGROUND_PIXELS
    assert result["auc"] >= v40.AUC_GATE
    assert result["ridge_addressable"] is True


def test_projected_pq_uses_frozen_global_arithmetic():
    below = v40.projected_v17(137)
    passing = v40.projected_v17(138)
    assert below["pq"] < 0.50
    assert passing["pq"] >= 0.50


def test_runner_has_only_tune_image_read_and_no_training_export_paths():
    source = (ROOT / "scripts/analyze_v40_multiscale_ridge_signal.py").read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute | ast.Name)
    }
    assert not ({"train", "fit", "predict", "to_csv", "submit"} & calls)
    lowered = source.lower()
    assert "test_images" not in lowered
    assert "audit_files" not in lowered
    assert "kaggle competitions submit" not in lowered


def test_real_population_and_v39_are_hash_pinned():
    runtime = v40.canonical.load_population(v40.canonical.CANONICAL_ANNOTATIONS)
    assert runtime["populations"]["tune"] == {
        "files": 94,
        "records": 150,
        "annotations": 1134,
    }
    assert v40.canonical.sha256(runtime["path"]) == v40.canonical.ANNOTATION_SHA256
    assert v40.sha256(v40.V39_RESULT) == v40.V39_SHA256
