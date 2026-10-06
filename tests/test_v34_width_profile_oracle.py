import ast
from pathlib import Path

import numpy as np
import pytest

from scripts import v34_width_profile_oracle as v34

ROOT = Path(__file__).resolve().parents[1]


def test_v34_constants_match_frozen_design():
    assert v34.RAY_STEP == 0.25
    assert v34.RAY_CAP == 128.0
    assert v34.TANGENT_RADIUS == 3
    assert v34.VALIDITY_GATE == 0.995
    assert v34.COMBINED_PQ_GATE == 0.95
    assert v34.SPLIT_PQ_GATE == 0.94
    assert v34.RECALL_GATE == 0.99
    assert v34.MEDIAN_IOU_GATE == 0.90
    assert v34.BILATERAL_DELTA_GATE == 0.005


def test_synthetic_parity_and_invariance_pass():
    checks = v34.synthetic_checks()
    assert checks == {
        "asymmetric_exact": True,
        "reverse_invariant": True,
        "symmetric_parity": True,
        "deterministic": True,
        "passed": True,
    }


def test_ordered_projection_is_deterministic_and_inside_mask():
    mask = np.zeros((40, 60), dtype=bool)
    mask[12:29, 5:55] = True
    points, report = v34.ordered_projected_points(mask, [3, 10, 30, 20, 57, 30])
    repeated, repeated_report = v34.ordered_projected_points(mask, [3, 10, 30, 20, 57, 30])
    np.testing.assert_array_equal(points, repeated)
    assert report == repeated_report and report["valid"]
    assert np.all(mask[points[:, 0], points[:, 1]])


def test_width_profile_captures_asymmetry():
    mask = np.zeros((64, 96), dtype=bool)
    mask[20:32, 10:86] = True
    points = np.asarray([[26, x] for x in range(10, 86)], dtype=np.int32)
    profile = v34.measure_profile(mask, points)
    assert profile["valid"]
    assert np.median(profile["left"]) != np.median(profile["right"])
    rebuilt = v34.reconstruct_ribbon(
        mask.shape, points, profile["normals"], profile["left"], profile["right"]
    )
    np.testing.assert_array_equal(rebuilt, mask)


def test_strict_metric_counts_half_iou_as_miss():
    target = np.zeros((32, 32), dtype=bool); target[5:15, 5:15] = True
    prediction = np.zeros_like(target); prediction[5:15, 5:10] = True
    counts, matched = v34.counts_and_ious([v34.rle(prediction)], [v34.rle(target)])
    result = v34.metric(counts)
    assert not matched
    assert result["tp"] == 0 and result["fp"] == result["fn"] == 1


def test_real_population_uses_only_canonical_fit_tune():
    runtime = v34.canonical.load_population()
    assert runtime["populations"]["fit"] == v34.canonical.EXPECTED_FIT
    assert runtime["populations"]["tune"] == v34.canonical.EXPECTED_TUNE
    assert runtime["populations"]["audit"] == v34.canonical.EXPECTED_AUDIT


def test_oracle_has_no_images_gpu_audit_test_export_or_submission_path():
    source = (ROOT / "scripts/v34_width_profile_oracle.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "torch" not in imports and "torchvision" not in imports
    assert "imread" not in calls and "YOLO" not in calls and "train" not in calls
    assert "submit" not in calls and "to_csv" not in calls
    assert '"reads_images": False' in source
    assert '"reads_audit": False' in source
    assert '"reads_test": False' in source


def test_noncanonical_annotation_path_is_rejected(tmp_path):
    alternate = tmp_path / "train.json"
    alternate.write_bytes(v34.canonical.CANONICAL_ANNOTATIONS.read_bytes())
    with pytest.raises(RuntimeError, match="only canonical"):
        v34.run_oracle(alternate, tmp_path / "out.json", limit=1)


def test_overlapping_predictions_are_one_to_one_matched():
    target_a = np.zeros((32, 32), dtype=bool); target_a[5:20, 5:20] = True
    target_b = np.zeros((32, 32), dtype=bool); target_b[5:20, 10:25] = True
    prediction = np.logical_or(target_a, target_b)
    counts, matched = v34.counts_and_ious(
        [v34.rle(prediction)], [v34.rle(target_a), v34.rle(target_b)]
    )
    result = v34.metric(counts)
    assert len(matched) == 1
    assert result["tp"] == 1 and result["fp"] == 0 and result["fn"] == 1
