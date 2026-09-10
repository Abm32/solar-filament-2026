"""Guard the export-only V11 path: no training, no re-selection, pinned identity."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


export = load("v11_export_under_test", ROOT / "kaggle_inference/v11_export.py")
prepare = load("prepare_v11_export_under_test", ROOT / "scripts/prepare_v11_export.py")

RUN_IDENTITY = "7c45acf26f210acce00f1bff581288bc2eb6121485237c7c028588d1de69164c"
CANDIDATE = "v11_proposal_centered_e4_refined_constrained"
GOOD_CONFIG = dict(arm="proposal_centered", epoch=4, conditioned=False,
                   mode="refined_constrained", priority=2)


def selection(**overrides):
    base = dict(candidate=CANDIDATE, config=dict(GOOD_CONFIG), run_identity=RUN_IDENTITY,
                frozen_before_audit=True, smoke=False,
                tune_counts={CANDIDATE: [431.2, 629, 206, 505]})
    base.update(overrides)
    return base


def test_export_never_constructs_a_training_loop():
    """A measurement export must not train, optimize, or step a scheduler."""
    source = (ROOT / "kaggle_inference/v11_export.py").read_text()
    # Strip the module docstring: it describes the guarantee in prose.
    body = source.split('"""', 2)[2]
    for forbidden in ("AdamW", "backward()", "optimizer", "scheduler", "GradScaler",
                      "train_arm", "build_training_cohort", "ProposalDataset", ".train()"):
        assert forbidden not in body, forbidden


def test_export_does_not_reimplement_selection_or_gates():
    """Selection and the gate verdict are read from the frozen run, never recomputed."""
    source = (ROOT / "kaggle_inference/v11_export.py").read_text()
    for forbidden in ("select_candidate", "promotion_gate", "declared_candidates"):
        assert forbidden not in source, forbidden
    assert export.GATE["promotable"] is False


def test_frozen_selection_accepts_only_the_measured_candidate():
    assert export.verify_frozen_selection(selection()) == GOOD_CONFIG


def test_ablation_arm_is_accepted_only_if_the_run_evaluated_it():
    """gt_centered_e4 was evaluated by the same run, so it may be exported."""
    payload = selection(tune_counts={
        "v11_proposal_centered_e4_refined_constrained": [431.2, 629, 206, 505],
        "v11_gt_centered_e4_refined_constrained": [430.5, 628, 207, 506]})
    config = export.verify_frozen_selection(payload, "gt_centered")
    assert config["arm"] == "gt_centered" and config["epoch"] == 4
    assert config["conditioned"] is False
    with pytest.raises(ValueError):
        export.verify_frozen_selection(payload, "proposal_conditioned")


def test_unevaluated_candidate_cannot_be_exported():
    payload = selection(tune_counts={"v11_proposal_centered_e4_refined_constrained": [1, 1, 1, 1]})
    with pytest.raises(ValueError):
        export.verify_frozen_selection(payload, "gt_centered")


def test_expected_candidate_names_match_the_run_convention():
    assert export.expected_candidate("gt_centered") == "v11_gt_centered_e4_refined_constrained"
    assert export.expected_candidate() == CANDIDATE
    assert set(export.ARM_TUNE_PQ) == {"proposal_centered", "gt_centered"}
    # The ablation must be recorded as scoring lower locally than the winner.
    assert export.ARM_TUNE_PQ["gt_centered"] < export.ARM_TUNE_PQ["proposal_centered"]


@pytest.mark.parametrize("overrides", [
    {"candidate": "v11_gt_centered_e4_refined_constrained"},
    {"run_identity": "0" * 64},
    {"frozen_before_audit": False},
    {"smoke": True},
    {"config": dict(GOOD_CONFIG, epoch=1)},
    {"config": dict(GOOD_CONFIG, conditioned=True)},
    {"config": dict(GOOD_CONFIG, mode="refined_free")},
    {"config": dict(GOOD_CONFIG, arm="gt_centered")},
])
def test_other_candidates_and_foreign_runs_are_rejected(overrides):
    with pytest.raises(ValueError):
        export.verify_frozen_selection(selection(**overrides))


def test_gate_record_preserves_the_failure_and_its_threshold():
    """The exported artifacts must keep saying the gate failed."""
    assert export.GATE["reason"] == "predeclared improvement gates failed"
    assert export.GATE["required_min_tune_delta_v6"] == 0.01
    assert export.GATE["deltas"]["tune_v6"] < 0.01
    assert export.GATE["deltas"]["tune_v10"] > 0
    assert export.GATE["deltas"]["audit_v6"] > 0
    assert export.GATE["deltas"]["audit_v10"] > 0


def test_checkpoint_lookup_requires_exactly_one_hash_match(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a/proposal_centered_e4.pt").write_bytes(b"real")
    (tmp_path / "b/proposal_centered_e4.pt").write_bytes(b"other")
    monkeypatch.setattr(export, "v11", type("stub", (), {
        "sha256": staticmethod(lambda path: "good" if Path(path).read_bytes() == b"real" else "bad")}),
        raising=False)
    assert export.locate_checkpoint(tmp_path, "good").read_bytes() == b"real"
    with pytest.raises(ValueError):
        export.locate_checkpoint(tmp_path, "absent")


def test_completed_source_run_must_be_the_gate_failed_run(tmp_path):
    manifest = dict(run_identity=RUN_IDENTITY, complete=True, selected_candidate=CANDIDATE,
                    promotable=False, detector_test_inference=False, gate={"promotable": False})
    (tmp_path / "inference_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "selected_config.json").write_text(json.dumps(selection()))
    assert prepare.verify_completed_run(tmp_path)[1]["candidate"] == CANDIDATE
    for bad in ({"complete": False}, {"promotable": True}, {"run_identity": "0" * 64},
                {"selected_candidate": "other"}, {"detector_test_inference": True}):
        (tmp_path / "inference_manifest.json").write_text(json.dumps({**manifest, **bad}))
        with pytest.raises(ValueError):
            prepare.verify_completed_run(tmp_path)


def test_staged_export_uses_a_distinct_slug_and_pins_the_source_run():
    assert prepare.SOURCE_RUN == "abhimanyu122/solar-filament-v11-refiner"
    assert prepare.CANDIDATE == CANDIDATE
    assert prepare.RUN_IDENTITY == RUN_IDENTITY


@pytest.mark.skipif(not (ROOT / "outputs/kaggle_v11_full/checkpoints/proposal_centered_e4.pt").exists(),
                    reason="requires the downloaded V11 run outputs")
def test_real_staging_pins_checkpoint_and_control_hash(tmp_path):
    staged = prepare.stage(ROOT, tmp_path / "stage")
    manifest = json.loads((staged / "staging_manifest.json").read_text())
    metadata = json.loads((staged / "kernel-metadata.json").read_text())
    assert manifest["promotable"] is False and manifest["trains"] is False
    assert manifest["candidate"] == CANDIDATE
    assert manifest["cached_images"] == 321
    assert len(manifest["checkpoint_sha256"]) == 64
    assert manifest["expected_control_sha256"] == (
        "bf11690e9bcce73b0c627096cc1198d9fb0916650f7fa4b3f08f1167bb76bde9")
    assert metadata["id"] == "abhimanyu122/solar-filament-v11-export"
    assert metadata["is_private"] == "true"
    assert prepare.SOURCE_RUN in metadata["kernel_sources"]
    bundle = (staged / "v11_export.py").read_text()
    # v11_refiner is bundled as a shared module, so run_v11 exists as a definition.
    # What matters is that the entry point invokes only the export path.
    entry = bundle.split("if __name__ == '__main__':")[1]
    assert "exporter.run_export()" in entry
    assert "run_v11()" not in entry


@pytest.mark.skipif(not (ROOT / "outputs/kaggle_v11_full/checkpoints/gt_centered_e4.pt").exists(),
                    reason="requires the downloaded V11 run outputs")
def test_ablation_arm_stages_a_distinct_slug_and_checkpoint(tmp_path):
    """The gt_centered ablation must not collide with the winner's notebook."""
    winner = prepare.stage(ROOT, tmp_path / "win", arm="proposal_centered")
    ablation = prepare.stage(ROOT, tmp_path / "abl", arm="gt_centered")
    win_manifest = json.loads((winner / "staging_manifest.json").read_text())
    abl_manifest = json.loads((ablation / "staging_manifest.json").read_text())
    assert abl_manifest["arm"] == "gt_centered"
    assert abl_manifest["candidate"] == "v11_gt_centered_e4_refined_constrained"
    assert abl_manifest["checkpoint_sha256"] != win_manifest["checkpoint_sha256"]
    win_id = json.loads((winner / "kernel-metadata.json").read_text())["id"]
    abl_id = json.loads((ablation / "kernel-metadata.json").read_text())["id"]
    assert win_id != abl_id
    assert abl_id == "abhimanyu122/solar-filament-v11-export-gt-centered"
    # Kaggle slugifies the title to resolve the id; a mismatch silently pushes
    # to a different ref, so assert they agree for every arm.
    for staged_dir in (winner, ablation):
        meta = json.loads((staged_dir / "kernel-metadata.json").read_text())
        slug = meta["title"].lower().replace(" ", "-")
        assert meta["id"].split("/", 1)[1] == slug, meta["id"]
    # Both must still verify against the same accepted control CSV.
    assert abl_manifest["expected_control_sha256"] == win_manifest["expected_control_sha256"]
    assert "ARM = 'gt_centered'" in (ablation / "v11_export.py").read_text()
