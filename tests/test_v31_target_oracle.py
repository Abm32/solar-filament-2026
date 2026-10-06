import ast
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from scripts import v31_target_oracle as v31

ROOT = Path(__file__).resolve().parents[1]


def rectangle(y0, y1, x0, x1, shape=(96, 96)):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def rle(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def rectangle_annotation(annotation_id, y0, y1, x0, x1, spine):
    return {
        "id": str(annotation_id),
        "image_id": "record",
        "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
        "spine": spine,
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "area": float((y1 - y0) * (x1 - x0)),
        "category_id": 1,
    }


def polygon_annotation(annotation_id="1"):
    return {
        "id": annotation_id,
        "image_id": "record",
        "segmentation": [
            [10, 10, 40, 10, 40, 30, 10, 30],
            [50, 50, 70, 50, 70, 70, 50, 70],
        ],
        "spine": [11, 20, 39, 20],
        "bbox": [10, 10, 60, 60],
        "area": 1000.0,
        "category_id": 1,
    }


def test_frozen_constants_match_design():
    assert v31.ANNOTATION_SHA256 == "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
    assert v31.EXPECTED_FIT == {"files": 566, "records": 937, "annotations": 6606}
    assert v31.EXPECTED_TUNE == {"files": 94, "records": 150, "annotations": 1134}
    assert v31.EXPECTED_AUDIT == {"files": 47, "records": 67, "annotations": 459}
    assert v31.SEED_THRESHOLDS == (0.65, 0.75, 0.85)
    assert v31.ENERGY_BINS == 10
    assert v31.SDT_ALPHA == 0.8
    assert v31.SPINE_PROJECTION_RADIUS == 3
    assert v31.TARGET_PQ_GATE == 0.97
    assert v31.TARGET_RECALL_GATE == 0.99


def test_data_firewall_accepts_only_hash_pinned_canonical_path(tmp_path):
    assert v31.assert_canonical_annotation_path(v31.CANONICAL_ANNOTATIONS) == v31.CANONICAL_ANNOTATIONS.resolve()
    alternate = tmp_path / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    alternate.write_bytes(v31.CANONICAL_ANNOTATIONS.read_bytes())
    with pytest.raises(RuntimeError, match="only canonical"):
        v31.assert_canonical_annotation_path(alternate)
    public = tmp_path / "public_magfilo" / "MAGFiLO_1.0_Annotations.json"
    public.parent.mkdir()
    public.write_text("{}")
    with pytest.raises(RuntimeError, match="firewall"):
        v31.assert_canonical_annotation_path(public)


def test_real_population_is_exact_and_physically_disjoint():
    runtime = v31.load_population()
    assert runtime["populations"] == {
        "all": v31.EXPECTED_ALL,
        "fit": v31.EXPECTED_FIT,
        "tune": v31.EXPECTED_TUNE,
        "audit": v31.EXPECTED_AUDIT,
    }
    assert not set(runtime["fit"]) & set(runtime["tune"])
    assert not set(runtime["fit"]) & set(runtime["audit"])
    assert not set(runtime["tune"]) & set(runtime["audit"])
    assert all(entry.get("spine") for entries in runtime["annotations"].values() for entry in entries)


def test_polygon_rasterization_preserves_all_components():
    info = {"height": 96, "width": 96}
    mask = v31.annotation_mask(info, polygon_annotation())
    assert mask[20, 20]
    assert mask[60, 60]
    assert not mask[45, 45]
    assert mask_utils.area(v31.annotation_rle(info, polygon_annotation())) == mask.sum()


def test_densify_and_projection_are_stable_and_inside_mask():
    mask = rectangle(20, 31, 10, 51)
    dense = v31.densify_polyline([8.2, 18.2, 30.0, 25.0, 53.0, 32.0])
    assert len(dense) > 40
    projected, report = v31.project_spine(mask, [8.2, 18.2, 30.0, 25.0, 53.0, 32.0])
    assert report["valid"]
    assert report["projected_points"] >= 2
    assert report["max_projection_distance"] <= v31.SPINE_PROJECTION_RADIUS
    assert np.all(projected <= mask)
    repeated, repeated_report = v31.project_spine(mask, [8.2, 18.2, 30.0, 25.0, 53.0, 32.0])
    np.testing.assert_array_equal(repeated, projected)
    assert repeated_report == report


def test_projection_tie_breaks_by_row_then_column():
    mask = np.zeros((9, 9), dtype=bool)
    mask[3, 4] = True
    mask[4, 3] = True
    projected, report = v31.project_spine(mask, [4, 4, 4, 4], radius=1)
    assert not report["valid"]
    assert projected[3, 4]
    assert not projected[4, 3]


def test_dt_and_sdt_energy_have_frozen_geometry():
    mask = rectangle(20, 40, 10, 70)
    spine = np.zeros_like(mask)
    spine[29, 12:68] = True
    dt = v31.instance_energy(mask, "dt")
    sdt = v31.instance_energy(mask, "sdt", spine)
    assert dt.shape == sdt.shape == mask.shape
    assert dt.dtype == sdt.dtype == np.float32
    assert dt[~mask].max() == sdt[~mask].max() == 0
    assert dt.max() == pytest.approx(1.0)
    assert sdt[29, 30] > 0.99
    assert sdt[20, 30] < sdt[29, 30]
    with pytest.raises(ValueError, match="projected expert spine"):
        v31.instance_energy(mask, "sdt", np.zeros_like(mask))


def test_quantization_roundtrip_and_threshold_semantics():
    energy = np.asarray([[0.0, 0.01, 0.10, 0.649, 0.65, 0.999, 1.0]], dtype=np.float32)
    foreground = np.asarray([[False, True, True, True, True, True, True]])
    classes = v31.quantize_energy(energy, foreground)
    assert classes.tolist() == [[0, 1, 2, 7, 7, 10, 10]]
    restored = v31.dequantize_energy(classes)
    np.testing.assert_allclose(
        restored,
        np.asarray([[0.0, 0.05, 0.15, 0.65, 0.65, 0.95, 0.95]], dtype=np.float32),
    )


def test_sdt_target_decodes_two_instances_exactly_at_frozen_low_seed():
    entries = [
        rectangle_annotation(1, 10, 22, 8, 52, [10, 16, 49, 16]),
        rectangle_annotation(2, 58, 74, 35, 86, [38, 66, 83, 66]),
    ]
    info = {"height": 96, "width": 96}
    target = v31.build_record_target(info, entries, "sdt")
    predictions = v31.decode_target(target["classes"], 0.65)
    metrics = v31.metric_record(v31.pq_counts(predictions, target["targets"]))
    assert metrics["tp"] == 2
    assert metrics["fp"] == metrics["fn"] == 0
    assert metrics["pq"] > 0.99
    assert not target["ignore"].any()


def test_overlap_pixels_are_flagged_without_erasing_foreground():
    entries = [
        rectangle_annotation(1, 10, 30, 10, 40, [12, 20, 38, 20]),
        rectangle_annotation(2, 20, 40, 30, 60, [32, 30, 58, 30]),
    ]
    target = v31.build_record_target({"height": 96, "width": 96}, entries, "sdt")
    assert target["ignore"][20:30, 30:40].all()
    assert target["foreground"][20:30, 30:40].all()


def test_strict_pq_counts_below_half_as_fp_and_fn():
    target = rectangle(10, 30, 10, 30)
    half = rectangle(10, 30, 10, 20)
    perfect = v31.metric_record(v31.pq_counts([rle(target)], [rle(target)]))
    strict = v31.metric_record(v31.pq_counts([rle(half)], [rle(target)]))
    assert perfect["pq"] == 1.0
    assert strict["tp"] == 0
    assert strict["fp"] == strict["fn"] == 1
    assert strict["pq"] == 0.0


def test_oracle_source_has_no_gpu_model_export_or_submission_path():
    source = (ROOT / "scripts/v31_target_oracle.py").read_text()
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
    assert "YOLO" not in calls and "train" not in calls
    assert "submit" not in calls and "to_csv" not in calls
    assert "submission.csv" not in source
    assert '"reads_audit": False' in source
    assert '"reads_test": False' in source
