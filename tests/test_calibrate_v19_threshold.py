"""Tests for canonical all-annotator V19 threshold calibration."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cal = load("calibrate_v19_under_test", ROOT / "scripts/calibrate_v19_threshold.py")
DATA_ROOT = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
requires_dataset = pytest.mark.skipif(
    not ANNOTATIONS.exists(), reason="competition dataset not staged locally"
)


def test_threshold_grid_is_bounded_and_stricter_than_the_failed_default():
    assert cal.THRESHOLDS == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def test_tune_selection_ignores_audit_and_prefers_stricter_tie():
    rows = [
        {"threshold": 0.5, "pq": 0.30},
        {"threshold": 0.6, "pq": 0.35},
        {"threshold": 0.7, "pq": 0.35},
    ]
    assert cal.select_tune_threshold(rows)["threshold"] == 0.7


def test_population_guard_rejects_the_original_v19_single_annotator_shape():
    records = [
        {"targets": [{"annotations": [{}]}]}
        for _ in range(cal.EXPECTED_POPULATION["tune"]["files"])
    ]
    with pytest.raises(ValueError, match="Canonical tune population mismatch"):
        cal.assert_population(records, "tune")


def test_pq_value_uses_global_project_arithmetic():
    assert cal.pq_value([8.0, 10, 2, 4]) == pytest.approx(8.0 / 13.0)


@requires_dataset
def test_real_canonical_population_matches_v11_v18_counts():
    image_dir = DATA_ROOT / "train/train_images"
    split = cal.load_split(ANNOTATIONS)
    tune = cal.build_canonical_records(ANNOTATIONS, image_dir, split.tune_files)
    audit = cal.build_canonical_records(ANNOTATIONS, image_dir, split.audit_files)
    assert cal.assert_population(tune, "tune") == {
        "files": 94,
        "annotator_records": 150,
        "gt_instances": 1134,
    }
    assert cal.assert_population(audit, "audit") == {
        "files": 47,
        "annotator_records": 67,
        "gt_instances": 459,
    }


def test_calibration_module_never_trains_exports_or_submits():
    source = (ROOT / "scripts/calibrate_v19_threshold.py").read_text()
    assert ".backward(" not in source
    assert "optimizer" not in source
    assert "to_csv(" not in source
    assert "competitions submit" not in source
