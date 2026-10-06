import ast
from pathlib import Path

import pytest

from scripts import continue_v29_yolov8l_native2048 as v29


def test_continuation_attempt_is_hash_pinned_and_recorded():
    assert v29.ATTEMPTED_TOTAL_EPOCHS == 40
    assert v29.CHECKPOINT_SHA256 == "d26cdb664b62cae949701805b9771864ab4208c43d7c5649e2f59fe7469842c8"


def test_continuation_is_retired_instead_of_silently_repeating_epoch_20():
    source = Path("scripts/continue_v29_yolov8l_native2048.py").read_text()
    assert '"imgsz": 2048' in source
    assert '"mask_ratio": 4' in source
    assert "No 40-epoch V29 result exists" in source
    assert "model.train" not in source
    with pytest.raises(SystemExit):
        v29.main()


def test_continuation_has_no_audit_test_export_or_submit():
    source = Path("scripts/continue_v29_yolov8l_native2048.py").read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "test/test_images" not in source
    assert "submission.csv" not in source
    assert "submit" not in calls and "to_csv" not in calls
