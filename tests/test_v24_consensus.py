import ast
import json
from pathlib import Path

import pytest

from scripts import evaluate_v24_consensus as evaluate
from scripts import train_v24_consensus as train

ROOT = Path(__file__).resolve().parents[1]


def real_population(monkeypatch):
    annotation = next((ROOT / "data/raw").rglob(
        "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    ))
    provenance = ROOT / "outputs/v6_inference_source/v6_provenance.json"
    monkeypatch.setattr(train, "ANNOTATIONS", annotation)
    _, images, annotations = train.load_population()
    split = train.dataset_partitions(images, annotations, json.loads(provenance.read_text()))
    return images, annotations, split


def test_real_consensus_population_is_deterministic_and_nonduplicating(monkeypatch):
    images, annotations, split = real_population(monkeypatch)
    consensus, diagnostics = train.build_consensus(images, annotations, split)
    second, second_diagnostics = train.build_consensus(images, annotations, split)
    assert {
        name: [entry["id"] for entry in entries] for name, entries in consensus.items()
    } == {
        name: [entry["id"] for entry in entries] for name, entries in second.items()
    }
    assert diagnostics == second_diagnostics
    assert len(consensus) == 566
    assert sum(len(split["by_file"][name]) > 1 for name in split["train_files"]) == 248
    assert sum(len(entries) for entries in consensus.values()) == 5127
    assert sum(len(entries) * len(split["by_file"][name])
               for name, entries in consensus.items()) == 9265
    assert all(
        sum(row["support_histogram"].values()) == row["clusters"]
        for row in diagnostics.values()
    )


def test_constrained_clustering_prevents_known_transitive_bridge(monkeypatch):
    images, annotations, split = real_population(monkeypatch)
    name = "20160810234034Lh.jpeg"
    clusters = train.consensus_clusters(split["by_file"][name], images, annotations)
    assert len(clusters) == 13
    assert all(cluster["support"] <= cluster["rater_count"] for cluster in clusters)
    flattened = [member for cluster in clusters for member in cluster["member_annotation_ids"]]
    expected = [str(entry["id"]) for image_id in split["by_file"][name]
                for entry in annotations[image_id]]
    assert sorted(flattened) == sorted(expected)


def test_arm_changes_targets_only_not_sampling_weight(monkeypatch):
    images, annotations, split = real_population(monkeypatch)
    consensus, _ = train.build_consensus(images, annotations, split)
    assert len(split["train_ids"]) == 937
    assert sum(len(annotations[image_id]) for image_id in split["train_ids"]) == 6606
    assert sum(len(consensus[images[image_id]["file_name"]])
               for image_id in split["train_ids"]) == 9265
    assert all(images[image_id]["file_name"] in consensus for image_id in split["train_ids"])


def test_training_recipe_is_frozen_and_matched():
    primary = train.training_arguments(seed=train.PRIMARY_SEED)
    replica = train.training_arguments(seed=train.REPLICATION_SEED)
    assert primary["epochs"] == 20
    assert primary["imgsz"] == 896
    assert primary["batch"] == 1
    assert primary["mask_ratio"] == 1
    assert primary["resume"] is False
    differing = {key for key in primary if primary[key] != replica[key]}
    assert differing == {"seed"}


def test_tune_selection_requires_absolute_control_recall_and_replication(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "OUTPUT", tmp_path)
    candidate = "iou030/t0.45_c0.30"

    def report(arm, pq, recall):
        return {
            "arm": arm, "group": "tune", "results": {candidate: {"pq": pq}},
            "raw_proposals": {
                "iou030": {"recall_gt_0_5": recall, "mean_best_iou": 0.6,
                           "max_best_iou": 0.9},
                "iou070": {"recall_gt_0_5": recall, "mean_best_iou": 0.6,
                           "max_best_iou": 0.9},
            },
        }

    reports = {
        "control": report("control", 0.445, 0.70),
        "consensus": report("consensus", evaluate.TUNE_REQUIRED + 0.001, 0.711),
        "control-replica": report("control-replica", 0.443, 0.69),
        "consensus-replica": report("consensus-replica", 0.445, 0.70),
    }
    for arm, value in reports.items():
        (tmp_path / f"{arm}_tune.json").write_text(json.dumps(value))
    selected = evaluate.select_tune()
    assert selected["audit_authorized"]
    reports["consensus"]["raw_proposals"]["iou030"]["recall_gt_0_5"] = 0.705
    (tmp_path / "consensus_tune.json").write_text(json.dumps(reports["consensus"]))
    assert not evaluate.select_tune()["audit_authorized"]


def test_v24_has_no_test_export_or_submission_path():
    for relative in ("scripts/train_v24_consensus.py", "scripts/evaluate_v24_consensus.py"):
        source = (ROOT / relative).read_text()
        tree = ast.parse(source)
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        assert "competitions_submit" not in called
        assert "submit" not in called
        assert "test/test_images" not in source
        assert "submission.csv" not in source


def test_gate_constants_are_pinned():
    assert evaluate.TUNE_REQUIRED == pytest.approx(evaluate.V17_TUNE_PQ + 0.01)
    assert evaluate.MATCHED_CONTROL_DELTA == 0.01
    assert evaluate.RAW_RECALL_DELTA == 0.01
    assert evaluate.V17_AUDIT_PQ == pytest.approx(0.4707634285309831)
