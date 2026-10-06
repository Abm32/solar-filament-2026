import ast
import json
from pathlib import Path

import pytest

from scripts import evaluate_v25_missing_label as evaluate
from scripts import train_v25_missing_label as train

ROOT = Path(__file__).resolve().parents[1]


def population(monkeypatch):
    annotation = next((ROOT / "data/raw").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    provenance = ROOT / "outputs/v6_inference_source/v6_provenance.json"
    monkeypatch.setattr(train, "ANNOTATIONS", annotation)
    train.configure_v24_paths()
    _, images, annotations = train.v24.load_population()
    split = train.v24.dataset_partitions(images, annotations, json.loads(provenance.read_text()))
    return images, annotations, split


def test_uncertainty_mapping_is_exact_and_deterministic(monkeypatch):
    images, annotations, split = population(monkeypatch)
    first, summary = train.omitted_regions(images, annotations, split)
    second, second_summary = train.omitted_regions(images, annotations, split)
    assert first == second and summary == second_summary
    assert summary["uncertainty_instances"] == 2659
    assert summary["records_with_uncertainty"] == 541
    assert summary["max_per_record"] == 28
    assert sum(len(v) for v in first.values()) == 2659
    assert all(len(entry["xywhn"]) == 4 for entries in first.values() for entry in entries)
    assert all(0 <= value <= 1 for entries in first.values() for entry in entries for value in entry["xywhn"])


def test_uncertainty_never_replaces_original_positives(monkeypatch):
    images, annotations, split = population(monkeypatch)
    mapping, _ = train.omitted_regions(images, annotations, split)
    assert sum(len(annotations[i]) for i in split["train_ids"]) == 6606
    assert len(split["train_ids"]) == 937
    for image_id in split["train_ids"]:
        stem = f"{Path(images[image_id]['file_name']).stem}_ann{image_id}"
        own = {str(a["id"]) for a in annotations[image_id]}
        assert own.isdisjoint({entry["annotation_id"] for entry in mapping.get(stem, [])})


def test_matched_training_differs_only_by_uncertainty_weight():
    assert train.CONTROL_WEIGHT == 1.0 and train.ARM_WEIGHT == 0.1
    primary = train.training_arguments(seed=42)
    replica = train.training_arguments(seed=20260912)
    assert {k for k in primary if primary[k] != replica[k]} == {"seed"}
    assert primary["epochs"] == 20 and primary["imgsz"] == 896
    assert primary["batch"] == 1 and primary["mask_ratio"] == 1


def test_tune_gates_are_strict(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "OUTPUT", tmp_path)
    candidate = "iou030/t0.45_c0.30"
    def report(arm, score, recall):
        return {"arm":arm,"group":"tune","results":{candidate:{"pq":score}},"raw_proposals":{cell:{"recall_gt_0_5":recall,"mean_best_iou":.6,"max_best_iou":.9} for cell in ("iou030","iou070")}}
    reports={"control":report("control",.445,.70),"uncertain":report("uncertain",evaluate.TUNE_REQUIRED+.001,.711),"control-replica":report("control-replica",.44,.69),"uncertain-replica":report("uncertain-replica",.441,.70)}
    for arm,value in reports.items(): (tmp_path/f"{arm}_tune.json").write_text(json.dumps(value))
    assert evaluate.select_tune()["audit_authorized"]
    reports["uncertain"]["raw_proposals"]["iou030"]["recall_gt_0_5"] = .705
    (tmp_path/"uncertain_tune.json").write_text(json.dumps(reports["uncertain"]))
    assert not evaluate.select_tune()["audit_authorized"]


def test_no_test_export_or_submission_path():
    for relative in ("scripts/train_v25_missing_label.py","scripts/v25_loss_runtime.py","scripts/evaluate_v25_missing_label.py"):
        source=(ROOT/relative).read_text(); tree=ast.parse(source)
        calls={node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,(ast.Attribute,ast.Name))}
        assert "submit" not in calls and "competitions_submit" not in calls
        assert "test/test_images" not in source and "submission.csv" not in source


def test_gate_constants():
    assert evaluate.TUNE_REQUIRED == pytest.approx(evaluate.V17_TUNE_PQ + .01)
    assert evaluate.MATCHED_CONTROL_DELTA == .01
    assert evaluate.RAW_RECALL_DELTA == .01


def test_runtime_source_casts_numeric_foreground_masks():
    source = (ROOT / "scripts/v25_loss_runtime.py").read_text()
    assert "~fg_mask[batch_index].bool()" in source
    assert "(~fg_mask.bool()).sum()" in source
    assert "(~fg_mask).sum()" not in source
