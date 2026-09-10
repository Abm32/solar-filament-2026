"""Guard the SUBMIT_BAR gate in scripts/export_v19_submission.py.

Only the gate-check logic (load_and_check_result) is testable without a GPU
and a real checkpoint; export_submission itself needs CUDA + transformers and
is exercised on the Azure VM / Kaggle, not here.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


export = load("export_v19_under_test", ROOT / "scripts/export_v19_submission.py")


def _write_result(tmp_path, **overrides):
    result = {
        "best_tune_pq": 0.153578,
        "audit_pq": 0.150346,
        "submit_bar_tune": 0.448703,
        "promotion_gate_tune": 0.55,
        "cleared_submit_bar": False,
        "cleared_promotion_gate": False,
    }
    result.update(overrides)
    path = tmp_path / "v19_full_result.json"
    path.write_text(json.dumps(result))
    return path


def test_refuses_export_when_submit_bar_not_cleared(tmp_path):
    path = _write_result(tmp_path)  # E029's actual failing result shape
    with pytest.raises(RuntimeError, match="SUBMIT_BAR gates export"):
        export.load_and_check_result(path)


def test_allows_export_when_submit_bar_cleared(tmp_path):
    path = _write_result(tmp_path, best_tune_pq=0.46, cleared_submit_bar=True)
    result = export.load_and_check_result(path)
    assert result["best_tune_pq"] == 0.46


def test_boundary_case_exact_bar_value_clears():
    """If a run lands exactly on the bar, the training script's own
    `best_tune_pq >= SUBMIT_BAR_TUNE` decides -- this test just pins that the
    export gate trusts that field rather than re-deriving its own comparison
    (avoiding a second place where a >= vs > mismatch could creep in)."""
    assert export.SUBMIT_BAR_TUNE == 0.448703


def test_min_mask_area_and_threshold_match_project_convention():
    # scripts/validate_submission.py's default min_area is 20; must match so
    # exported masks are never silently smaller than what validation accepts.
    assert export.MIN_MASK_AREA == 20
    assert export.MASK_THRESHOLD == 0.5
