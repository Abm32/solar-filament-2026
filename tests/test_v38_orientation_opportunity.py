from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from scripts import analyze_v38_orientation_opportunity as v38

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_contract_matches_design():
    assert v38.PROXIMITY_RADIUS == 16.0
    assert v38.ORIENTATION_BIN_DEGREES == 30.0
    assert v38.RECOVERY_IOU == 0.65
    assert v38.MISS_COVERAGE_GATE == 0.65
    assert v38.ENRICHMENT_GATE == 1.25
    assert v38.PROJECTED_PQ_GATE == 0.50
    assert v38.EXPECTED_SOURCE_MATCHES == (841, 794, 826, 884, 843, 893)
    assert v38.EXPECTED_UNION_MATCHES == 922
    assert v38.EXPECTED_UNION_MISSES == 212


def test_local_tangent_and_unoriented_angle_are_direction_invariant():
    horizontal = np.asarray([[x, 10] for x in range(10)], dtype=np.float64)
    vertical = np.asarray([[5, y] for y in range(10)], dtype=np.float64)
    diagonal = np.asarray([[x, x] for x in range(10)], dtype=np.float64)
    h = v38.local_tangent(horizontal, 5)
    assert v38.unoriented_angle_degrees(h, v38.local_tangent(horizontal[::-1], 4)) == 0.0
    assert v38.unoriented_angle_degrees(h, v38.local_tangent(vertical, 5)) == 90.0
    assert np.isclose(v38.unoriented_angle_degrees(h, v38.local_tangent(diagonal, 5)), 45.0)


def test_pair_geometry_distinguishes_parallel_crossing_and_endpoints():
    horizontal = np.asarray([[x, 10] for x in range(21)], dtype=np.float64)
    parallel = np.asarray([[x, 14] for x in range(21)], dtype=np.float64)
    vertical = np.asarray([[10, y] for y in range(21)], dtype=np.float64)
    far = np.asarray([[x + 100, 10] for x in range(21)], dtype=np.float64)

    same = v38.pair_geometry(horizontal, parallel)
    crossing = v38.pair_geometry(horizontal, vertical)
    distant = v38.pair_geometry(horizontal, far)

    assert same["proximity"] and same["parallel"] and not same["crossing_like"]
    assert crossing["proximity"] and crossing["crossing_like"] and not crossing["parallel"]
    assert not distant["addressable"]
    assert same["nearest_distance"] == 4.0
    assert crossing["nearest_distance"] == 0.0


def test_projected_pq_uses_global_count_arithmetic():
    projected = v38.projected_v17(138)
    expected = (v38.V17_COUNTS[0] + 138 * 0.65) / (
        v38.V17_COUNTS[1] + 138
        + 0.5 * v38.V17_COUNTS[2]
        + 0.5 * (v38.V17_COUNTS[3] - 138)
    )
    assert np.isclose(projected["pq"], expected)
    assert projected["tp"] == 842
    assert projected["fp"] == 306
    assert projected["fn"] == 292


def test_runner_has_no_training_audit_test_export_or_submission_path():
    source = (ROOT / "scripts/analyze_v38_orientation_opportunity.py").read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute | ast.Name)
    }
    assert not ({"train", "fit", "predict", "to_csv", "submit"} & calls)
    lowered = source.lower()
    assert "audit_files" not in lowered
    assert "test_images" not in lowered
    assert "kaggle competitions submit" not in lowered


def test_real_population_is_tune_only_and_hash_pinned():
    runtime = v38.canonical.load_population(v38.canonical.CANONICAL_ANNOTATIONS)
    assert runtime["populations"]["tune"] == {"files": 94, "records": 150, "annotations": 1134}
    assert v38.canonical.sha256(runtime["path"]) == v38.canonical.ANNOTATION_SHA256
