import ast
from pathlib import Path

from scripts import train_v32_omnipose_overfit as g2

ROOT = Path(__file__).resolve().parents[1]


def test_g2_contract_is_bounded_and_hash_pinned():
    assert g2.SELECTION_ID == "e561c523b60993e93f82ce03116737ce512f54f4103b4557c29e409c88c92653"
    assert g2.SAMPLES == 144
    assert g2.MAX_STEPS == 2000
    assert g2.EVAL_STEPS == (0, 250, 500, 1000, 1500, 2000)
    assert g2.TARGET_PQ == 0.85
    assert g2.MEMORY_GATE_MIB == 14336
    assert g2.BACKBONE_SHA256 == "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"


def test_g2_selection_is_exact_and_uses_only_fit():
    import json
    document = json.loads(g2.Path(g2.__file__).resolve().parents[1].joinpath(
        "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/"
        "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    ).read_text())
    records = g2.selection_records(document)
    assert len(records) == 16
    assert len({row["file_name"] for row in records}) == 9
    assert sum(row["annotations"] for row in records) == 57


def test_g2_has_no_tune_audit_test_export_or_submission_path():
    source = (ROOT / "scripts/train_v32_omnipose_overfit.py").read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "test_images" not in source and "submission.csv" not in source
    assert "submit" not in calls and "to_csv" not in calls
    assert '"reads_tune": False' in source
    assert '"reads_audit": False' in source
    assert '"reads_test": False' in source


def test_classifier_initialization_and_loss_are_frozen():
    source = (ROOT / "scripts/train_v32_omnipose_overfit.py").read_text()
    assert "head.bias[2] = -5.0" in source
    assert "head.bias[3] = -5.0" in source
    assert "5.0 * flow_loss + distance_loss + boundary_loss" in source
    assert "set_batchnorm_eval(torch, model)" in source
    assert "(step - 1) % SAMPLES" in source


def test_tile_normalization_stays_float32():
    import numpy as np
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    output = g2.normalize_tile(np, image)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert output.min() >= 0 and output.max() <= 1


def test_g2_sets_cublas_determinism_before_torch_import():
    source = (ROOT / "scripts/train_v32_omnipose_overfit.py").read_text()
    assert source.index('CUBLAS_WORKSPACE_CONFIG') < source.index('    import torch')
