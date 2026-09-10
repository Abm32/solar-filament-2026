"""Guard V19's split reproduction, dataset conversion, and staging bundle.

These run on CPU without torch/transformers -- run_smoke_test itself needs
CUDA and is exercised on Kaggle, not here. What IS testable locally and must
stay correct: the tune/audit split matches every V6-V18 run's convention
exactly (same 707/566/94/47 shape), the dataset conversion carries real
annotation counts through, and the staged Kaggle bundle self-verifies and
compiles.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v19 = load("v19_mask2former_under_test", ROOT / "scripts/train_v19_mask2former.py")
prepare_v19 = load("prepare_v19_under_test", ROOT / "scripts/prepare_v19_mask2former.py")

DATA_ROOT = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

requires_dataset = pytest.mark.skipif(
    not ANNOTATIONS.exists(), reason="competition dataset not staged locally"
)


class _StubModel:
    """Minimal stand-in for evaluate_pq's model.eval() call in tests that
    mock out predict_instances entirely and never touch the model otherwise."""

    def eval(self):
        pass


# --- protocol constants match the design doc exactly ---

def test_protocol_constants_match_v19_design_doc():
    assert v19.SUBMIT_BAR_TUNE == 0.448703  # V17's accepted tune PQ
    assert v19.PROMOTION_GATE_TUNE == 0.55
    assert v19.IMG_SIZE_TRAIN == 1024
    assert v19.IMG_SIZE_INFER == 2048
    assert v19.BATCH_SIZE == 2
    assert v19.GRAD_ACCUM == 4
    assert v19.NUM_QUERIES == 100
    assert v19.SEED == 42


# --- the split must reproduce optimize_inference.validation_split() exactly ---

def test_split_algorithm_matches_optimize_inference_validation_split_shape():
    """Same shuffle seed, same 20%-then-94/rest partition as every V6-V18 run
    (kaggle_inference/optimize_inference.py's validation_split): sanity-check
    the shape on a synthetic file list before trusting it on the real one."""
    import random

    files = [f"img_{i:04d}" for i in range(707)]
    random.Random(v19.SEED).shuffle(files)
    held_out = files[:round(len(files) * 0.2)]
    assert len(held_out) == 141
    assert len(held_out[:94]) == 94
    assert len(held_out[94:]) == 47


@requires_dataset
def test_split_reproduces_the_known_707_566_94_47_shape_on_real_data():
    """V18's manifest.json (outputs/v18_smoke_timing_kernel/v18_manifest.json)
    recorded training_files=566, validation_files=141 for this exact dataset
    snapshot; tune=94/audit=47 is the fixed 94-vs-rest split of that 141."""
    split = v19.load_split(ANNOTATIONS)
    assert len(split.train_files) == 566
    assert len(split.tune_files) == 94
    assert len(split.audit_files) == 47
    assert set(split.train_files).isdisjoint(split.tune_files)
    assert set(split.train_files).isdisjoint(split.audit_files)
    assert set(split.tune_files).isdisjoint(split.audit_files)


@requires_dataset
def test_build_hf_dataset_carries_real_annotation_counts():
    image_dir = DATA_ROOT / "train" / "train_images"
    split = v19.load_split(ANNOTATIONS)
    dataset = v19.build_hf_dataset(ANNOTATIONS, image_dir, split)
    assert len(dataset["train"]) == len(split.train_files)
    assert len(dataset["tune"]) == len(split.tune_files)
    assert len(dataset["audit"]) == len(split.audit_files)
    total_annotations = sum(len(r["annotations"]) for part in dataset.values() for r in part)
    assert total_annotations > 0
    # No record should carry another record's annotations (a stale file_to_id
    # bug would silently duplicate counts across splits).
    densest = max(dataset["train"], key=lambda r: len(r["annotations"]))
    assert densest["file_name"] in split.train_files
    assert len(densest["annotations"]) > 0


# --- run_smoke_test / main() refuse to proceed without required inputs ---

def test_run_smoke_test_refuses_without_cuda(monkeypatch):
    torch_stub = type(sys)("torch")
    torch_stub.cuda = type(sys)("cuda")
    torch_stub.cuda.is_available = lambda: False
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    with pytest.raises(RuntimeError, match="CUDA not available"):
        v19.run_smoke_test({"train": [{"file_name": "x", "annotations": [{}]}]}, Path("/tmp"))


def test_run_smoke_test_refuses_on_empty_train_split():
    with pytest.raises(RuntimeError, match="empty train split"):
        v19.run_smoke_test({"train": []}, Path("/tmp"))


def test_main_requires_smoke_or_full(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_v19_mask2former.py"])
    with pytest.raises(SystemExit):
        v19.main()


def test_full_mode_calls_run_full_training_after_smoke_has_cleared(monkeypatch):
    """The NotImplementedError guard is gone now that run_full_training exists
    (the smoke test cleared 11.39 GiB on Kaggle, 2026-09-10) -- pin that --full
    now reaches run_full_training rather than silently no-op'ing or reverting
    to the old placeholder behavior."""
    calls = []
    monkeypatch.setattr(v19, "ensure_environment", lambda: None)
    monkeypatch.setattr(v19, "ensure_data", lambda root: root)
    monkeypatch.setattr(v19, "load_split", lambda coco_path: v19.Split([], [], []))
    monkeypatch.setattr(v19, "build_hf_dataset", lambda *a, **k: {"train": [], "tune": [], "audit": []})
    monkeypatch.setattr(v19, "run_full_training", lambda dataset, out_dir: calls.append((dataset, out_dir)))
    monkeypatch.setattr(sys, "argv", ["train_v19_mask2former.py", "--full"])
    v19.main()
    assert len(calls) == 1


# --- staging bundle: self-verifying, compiles, and carries real provenance ---

@requires_dataset
def test_staged_bundle_self_verifies_and_compiles(tmp_path):
    output = prepare_v19.stage(ROOT, tmp_path / "v19_smoke_kernel")
    bundled_path = output / "train_v19_mask2former.py"
    source = bundled_path.read_text()
    compile(source, "train_v19_mask2former.py", "exec")
    # The bundle embeds its own SHA-256 self-check; verify the hash actually
    # matches what got embedded, i.e. staging did not silently drift.
    manifest = json.loads((output / "staging_manifest.json").read_text())
    assert manifest["module_sha256"]["train_v19_mask2former"] in source
    assert manifest["script_sha256"] == hashlib.sha256(bundled_path.read_bytes()).hexdigest()


@requires_dataset
def test_staged_bundle_pins_annotation_hash_to_the_real_dataset(tmp_path):
    output = prepare_v19.stage(ROOT, tmp_path / "v19_smoke_kernel")
    manifest = json.loads((output / "staging_manifest.json").read_text())
    assert manifest["annotation_sha256"] == hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest()


@requires_dataset
def test_staged_kernel_metadata_matches_project_convention(tmp_path):
    output = prepare_v19.stage(ROOT, tmp_path / "v19_smoke_kernel")
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert metadata["id"] == "abhimanyu122/solar-filament-v19-mask2former-smoke"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["competition_sources"] == ["filament-segmentation-2026"]


@requires_dataset
def test_staged_manifest_declares_submits_false_and_no_prior_checkpoint(tmp_path):
    """V19 has no warm-start checkpoint to pin (new detector family), unlike
    every prepare_v11-v18 script; the manifest must say so honestly rather
    than silently omitting the field prior manifests always had."""
    output = prepare_v19.stage(ROOT, tmp_path / "v19_smoke_kernel")
    manifest = json.loads((output / "staging_manifest.json").read_text())
    assert manifest["submits"] is False
    assert "warm_start_sha256" not in manifest
    assert manifest["submit_bar_tune"] == 0.448703
    assert manifest["promotion_gate_tune"] == 0.55


@requires_dataset
def test_full_mode_stages_the_full_flag_and_a_distinct_slug(tmp_path):
    output = prepare_v19.stage(ROOT, tmp_path / "v19_full_kernel", mode="full")
    source = (output / "train_v19_mask2former.py").read_text()
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    manifest = json.loads((output / "staging_manifest.json").read_text())
    assert "'--full'" in source
    assert "'--smoke'" not in source
    assert metadata["id"] == "abhimanyu122/solar-filament-v19-mask2former-full"
    assert manifest["mode"] == "full"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown staging mode"):
        prepare_v19.stage(ROOT, Path("/tmp/whatever"), mode="bogus")


# --- full training loop: PQ accounting must match optimize_inference.py's ---
# --- pq_counts/pq_value exactly (same IoU>0.5 hit rule, same tp/fp/fn) ------

def test_evaluate_pq_empty_gt_all_counted_as_fp(monkeypatch):
    """Mirrors optimize_inference.pq_counts: if n_gt == 0, every prediction
    is a false positive and nothing contributes IoU."""
    monkeypatch.setattr(v19, "_load_instance_map", lambda record: (
        None, __import__("numpy").zeros((4, 4), dtype="int32"), {}
    ))
    monkeypatch.setattr(v19, "predict_instances", lambda *a, **k: (
        [__import__("numpy").ones((4, 4), dtype=bool)] * 3, [0.9, 0.8, 0.7]
    ))
    result = v19.evaluate_pq(_StubModel(), None, [{"file_name": "x"}], "cpu", 4)
    assert result["pq"] == 0.0
    assert result["fp"] == 3
    assert result["tp"] == 0
    assert result["fn"] == 0


def test_evaluate_pq_empty_pred_all_counted_as_fn(monkeypatch):
    import numpy as np

    gt_map = np.zeros((4, 4), dtype="int32")
    gt_map[:2, :2] = 1
    gt_map[2:, 2:] = 2
    monkeypatch.setattr(v19, "_load_instance_map", lambda record: (None, gt_map, {}))
    monkeypatch.setattr(v19, "predict_instances", lambda *a, **k: ([], []))
    result = v19.evaluate_pq(_StubModel(), None, [{"file_name": "x"}], "cpu", 4)
    assert result["pq"] == 0.0
    assert result["fn"] == 2
    assert result["tp"] == 0
    assert result["fp"] == 0


def test_evaluate_pq_perfect_match_scores_pq_one(monkeypatch):
    """Two GT instances, two predictions with identical masks: every match
    has IoU 1.0, no fp/fn -- PQ must be exactly 1.0 (tp_iou=2, tp=2, denom=2)."""
    import numpy as np

    gt_map = np.zeros((4, 4), dtype="int32")
    gt_map[:2, :2] = 1
    gt_map[2:, 2:] = 2
    pred_masks = [gt_map == 1, gt_map == 2]
    monkeypatch.setattr(v19, "_load_instance_map", lambda record: (None, gt_map, {}))
    monkeypatch.setattr(v19, "predict_instances", lambda *a, **k: (pred_masks, [0.9, 0.9]))
    result = v19.evaluate_pq(_StubModel(), None, [{"file_name": "x"}], "cpu", 4)
    assert result["pq"] == pytest.approx(1.0)
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_evaluate_pq_below_threshold_iou_counted_as_both_fp_and_fn(monkeypatch):
    """A low-overlap pair (IoU <= 0.5) must NOT count as a hit: it should
    show up as one fp AND one fn, exactly like optimize_inference.pq_counts'
    (~hits.any(axis=0)).sum() / (~hits.any(axis=1)).sum() convention."""
    import numpy as np

    gt_map = np.zeros((10, 10), dtype="int32")
    gt_map[:4, :4] = 1  # 16 px
    pred_mask = np.zeros((10, 10), dtype=bool)
    pred_mask[:2, :4] = True  # 8 px, overlap 8px -> IoU = 8/16 = 0.5, not > 0.5
    monkeypatch.setattr(v19, "_load_instance_map", lambda record: (None, gt_map, {}))
    monkeypatch.setattr(v19, "predict_instances", lambda *a, **k: ([pred_mask], [0.9]))
    result = v19.evaluate_pq(_StubModel(), None, [{"file_name": "x"}], "cpu", 10)
    assert result["pq"] == 0.0
    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1


def test_evaluate_pq_aggregates_across_multiple_records(monkeypatch):
    """PQ must accumulate tp/fp/fn/tp_iou across every record before dividing
    -- a single global PQ over the whole split, not an average of per-image
    PQs (matching every prior version's accounting)."""
    import numpy as np

    call_state = {"i": 0}
    gt_map_a = np.zeros((4, 4), dtype="int32")
    gt_map_a[:2, :2] = 1
    gt_map_b = np.zeros((4, 4), dtype="int32")
    gt_map_b[2:, 2:] = 1

    def fake_load(record):
        maps = [gt_map_a, gt_map_b]
        return None, maps[call_state["i"]], {}

    def fake_predict(*a, **k):
        maps = [gt_map_a == 1, gt_map_b == 1]
        mask = maps[call_state["i"]]
        call_state["i"] += 1
        return [mask], [0.9]

    monkeypatch.setattr(v19, "_load_instance_map", fake_load)
    monkeypatch.setattr(v19, "predict_instances", fake_predict)
    result = v19.evaluate_pq(_StubModel(), None, [{"file_name": "a"}, {"file_name": "b"}], "cpu", 4)
    assert result["tp"] == 2
    assert result["pq"] == pytest.approx(1.0)


def test_pq_threshold_constant_matches_optimize_inference():
    """optimize_inference.py imports torch at module level (unavailable in
    this CPU-only test env), so read its source text directly rather than
    importing the module -- confirms v19's threshold and hit rule stay in
    sync with the file's actual pq_counts implementation without requiring
    torch just to run this one assertion."""
    source = (ROOT / "kaggle_inference/optimize_inference.py").read_text()
    assert v19.PQ_IOU_THRESHOLD == 0.5
    assert "iou > 0.5" in source


def test_run_full_training_refuses_without_cuda(monkeypatch):
    torch_stub = type(sys)("torch")
    torch_stub.cuda = type(sys)("cuda")
    torch_stub.cuda.is_available = lambda: False
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    with pytest.raises(RuntimeError, match="CUDA not available"):
        v19.run_full_training({"train": [], "tune": [], "audit": []}, Path("/tmp"))
