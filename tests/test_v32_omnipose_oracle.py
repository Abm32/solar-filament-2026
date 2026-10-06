import ast
import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from scripts import run_v32_omnipose_oracle as launch
from scripts import v32_omnipose_worker as worker

ROOT = Path(__file__).resolve().parents[1]


def rectangle_annotation(annotation_id, y0, y1, x0, x1):
    return {
        "id": str(annotation_id), "image_id": "record",
        "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "area": float((y1 - y0) * (x1 - x0)), "category_id": 1,
    }


def test_frozen_reference_and_gates_match_design():
    assert launch.UPSTREAM_COMMIT == "21e4d0199cc03662669c150f6151e3365dc201fd"
    assert launch.UPSTREAM_HASHES["omnipose/core.py"] == "300f4cf24ad5256c35726442e024efa9e98cb27b200a3f36a6da01e3839ab332"
    assert launch.ANNOTATION_SHA256 == worker.ANNOTATION_SHA256
    assert worker.CROP_MARGIN == 16
    assert worker.MIN_AREA == 20
    assert worker.DIAM_THRESHOLD == 12.0
    assert worker.EPS == pytest.approx(2 ** 0.5)
    assert worker.PQ_GATE == 0.90
    assert worker.SPLIT_PQ_GATE == 0.89
    assert worker.RECALL_GATE == 0.99
    assert worker.FP_RATIO_GATE == 0.25
    assert worker.ZERO_RECORD_RATE_GATE == 0.01


def test_launcher_has_no_torch_or_omnipose_import():
    source = (ROOT / "scripts/run_v32_omnipose_oracle.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "torch" not in imports
    assert "omnipose" not in imports


def test_firewall_accepts_only_canonical_annotation(tmp_path):
    assert launch.assert_annotation(launch.CANONICAL_ANNOTATIONS) == launch.CANONICAL_ANNOTATIONS.resolve()
    alternate = tmp_path / launch.CANONICAL_ANNOTATIONS.name
    alternate.write_bytes(launch.CANONICAL_ANNOTATIONS.read_bytes())
    with pytest.raises(RuntimeError, match="only canonical"):
        launch.assert_annotation(alternate)
    public = tmp_path / "public_magfilo" / "MAGFiLO_1.0_Annotations.json"
    public.parent.mkdir(); public.write_text("{}")
    with pytest.raises(RuntimeError, match="firewall"):
        launch.assert_annotation(public)


def test_canonical_population_is_exact():
    populations = launch.split_populations(launch.CANONICAL_ANNOTATIONS)
    assert populations == launch.EXPECTED


def test_evidence_is_hash_pinned_and_selects_auto_adaptive():
    report = launch.verify_evidence()
    assert report["selected_mode"] == "auto_adaptive"
    assert report["hashes"] == launch.EVIDENCE_HASHES


def test_upstream_verification_rejects_changed_source(tmp_path):
    root = tmp_path / "upstream"
    for relative in launch.UPSTREAM_HASHES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed")
    with pytest.raises(RuntimeError, match="pinned upstream file changed"):
        launch.verify_upstream(root)


def test_crop_record_excludes_overlap_and_preserves_scoring_targets():
    info = {"height": 96, "width": 96}
    entries = [
        rectangle_annotation(1, 10, 35, 10, 45),
        rectangle_annotation(2, 25, 55, 35, 70),
    ]
    cropped = worker.crop_record(info, entries)
    labels = cropped["labels"]
    assert cropped["overlap_pixels"] > 0
    x0, y0, x1, y1 = cropped["bounds"]
    decoded = [worker.annotation_mask(info, entry)[y0:y1, x0:x1] for entry in entries]
    overlap = decoded[0] & decoded[1]
    assert int(overlap.sum()) == cropped["overlap_pixels"]
    assert not np.any(labels[overlap] > 0)
    assert len(cropped["targets"]) == 2
    assert all(mask_utils.area(target) > 0 for target in cropped["targets"])


def test_metric_is_strictly_greater_than_half():
    full = np.zeros((20, 20), np.uint8); full[2:12, 2:12] = 1
    half = np.zeros_like(full); half[2:12, 2:7] = 1
    encode = lambda value: mask_utils.encode(np.asfortranarray(value))
    perfect = worker.metric_record(worker.metric_counts([encode(full)], [encode(full)]))
    strict = worker.metric_record(worker.metric_counts([encode(half)], [encode(full)]))
    assert perfect["pq"] == 1.0
    assert strict["pq"] == 0.0
    assert strict["tp"] == 0 and strict["fp"] == strict["fn"] == 1


def test_worker_has_no_image_audit_test_export_or_submission_path():
    source = (ROOT / "scripts/v32_omnipose_worker.py").read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "cv2" not in source and "imread" not in calls
    assert "test_images" not in source and "submission.csv" not in source
    assert "submit" not in calls and "to_csv" not in calls
    assert '"reads_audit": False' in source
    assert '"reads_test": False' in source


def test_mode_selection_payload_is_the_frozen_probe():
    payload = json.loads((launch.EVIDENCE_ROOT / "mode_selection.json").read_text())
    assert payload["records"] == 26
    assert payload["summary"]["auto_adaptive"]["tp"] == 234
    assert payload["summary"]["auto_adaptive"]["fp"] == 44
    assert payload["summary"]["auto_adaptive"]["fn"] == 2
    assert payload["summary"]["auto_adaptive"]["mean_record_pq"] == pytest.approx(0.8953761538298969)
