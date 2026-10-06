import ast

from scripts import evaluate_v29_refiner as v29


def test_refiner_grid_reuses_v17_geometry_and_bounded_confidence_extension():
    assert v29.IMAGE_SIZE == 2048
    assert v29.NMS_IOU == 0.0
    assert v29.THRESHOLDS == (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    assert v29.CONFIDENCES == (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)


def test_refiner_evaluator_has_no_audit_test_or_export_path():
    source = open("scripts/evaluate_v29_refiner.py").read()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "test/test_images" not in source
    assert "submission.csv" not in source
    assert "submit" not in calls and "to_csv" not in calls


def test_dense_detector_masks_are_encoded_before_v26_refiner():
    source = open("scripts/evaluate_v29_refiner.py").read()
    assert 'proposal_rles = [runtime["core"].encode(mask) for mask in masks]' in source
    assert 'refine(runtime, proposal_rles, scores, image_path)' in source


def test_refiner_requires_material_promotion_before_audit():
    assert v29.PROMOTION_PQ == 0.50
    assert v29.EXPECTED_TUNE == {"files": 94, "records": 150, "gt": 1134}
