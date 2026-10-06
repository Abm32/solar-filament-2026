import ast
from pathlib import Path

from scripts import prepare_v30_representative as prep
from scripts import train_v30_representative as train


def test_representative_population_and_selection_are_frozen():
    assert prep.EXPECTED_TRAIN_FILES == 566
    assert prep.EXPECTED_VAL_FILES == 141
    source = Path("scripts/prepare_v30_representative.py").read_text()
    assert "line_count(pair[1])" in source
    assert "source_name(image)" in source


def test_training_combines_public_model_resolution_and_target_policy():
    assert train.EPOCHS == 40
    source = Path("scripts/train_v30_representative.py").read_text()
    assert "imgsz=2048" in source
    assert "mask_ratio=4" in source
    assert 'manifest["train_files"] != 566' in source


def test_v30_has_no_audit_test_or_submission_path():
    source = Path("scripts/train_v30_representative.py").read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "test/test_images" not in source
    assert "submission.csv" not in source
    assert "submit" not in calls and "to_csv" not in calls
