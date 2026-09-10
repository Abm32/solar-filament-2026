"""Guard V14 staging: pinned V11 warm-start, frozen inputs, gates, distinct slugs.

Mirrors test_v11_export.py's pattern (static module assertions plus a
skip-gated real-staging test) since prepare_v14_dualhead.py follows the same
prepare_vNN staging shape as prepare_v11_export.py / prepare_v13_compose.py,
just for a training (not export-only) run.
"""
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


prepare = load("prepare_v14_dualhead_under_test", ROOT / "scripts/prepare_v14_dualhead.py")
v14 = load("v14_dualhead_under_test", ROOT / "kaggle_inference/v14_dualhead.py")

RUN_IDENTITY = "7c45acf26f210acce00f1bff581288bc2eb6121485237c7c028588d1de69164c"
SOURCE_RUN = "abhimanyu122/solar-filament-v11-refiner"


# --------------------------------------------------------------- static wiring

def test_stager_pins_the_same_v11_parent_run_as_v13_and_ea():
    assert prepare.RUN_IDENTITY == RUN_IDENTITY
    assert prepare.SOURCE_RUN == SOURCE_RUN
    assert prepare.ARM == "proposal_centered"


def test_bundled_module_set_covers_v14_and_its_full_dependency_chain():
    assert set(prepare.MODULES) == {
        "optimize_inference", "detector_tta", "v10_quality", "v11_refiner",
        "v14_dualhead", "v14_runtime"}


def test_slugs_and_titles_are_distinct_between_full_and_smoke():
    assert prepare.SLUGS["full"] != prepare.SLUGS["smoke"]
    assert prepare.SLUGS["smoke"].endswith("-smoke")
    assert prepare.SLUGS["full"].startswith("solar-filament-v14")
    for mode in prepare.SLUGS:
        assert prepare.TITLES[mode].lower().replace(" ", "-") == prepare.SLUGS[mode]


def test_smoke_limit_is_bounded_so_it_cannot_become_a_full_run():
    assert 0 < prepare.SMOKE_LIMIT_CAP < 20


# ------------------------------------------------------------- mode discipline

def test_full_mode_rejects_any_limit(tmp_path):
    with pytest.raises(ValueError, match="Full mode must not set --limit"):
        prepare.stage(ROOT, tmp_path / "stage", mode="full", limit=4)


@pytest.mark.parametrize("limit", [0, -1, prepare.SMOKE_LIMIT_CAP + 1])
def test_smoke_mode_requires_a_bounded_positive_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="Smoke mode requires --limit"):
        prepare.stage(ROOT, tmp_path / "stage", mode="smoke", limit=limit)


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown staging mode"):
        prepare.stage(ROOT, tmp_path / "stage", mode="bogus")


def test_revision_must_be_alphanumeric_with_dashes(tmp_path):
    with pytest.raises(ValueError, match="Revision must be alphanumeric"):
        prepare.stage(ROOT, tmp_path / "stage", mode="full", revision="not ok!")


# ---------------------------------------------------- decision-logic assertions

def test_stager_checks_v14_references_the_pinned_v11_parent():
    assert v14.V11_RUN_IDENTITY == RUN_IDENTITY
    assert v14.V11_SOURCE_RUN == SOURCE_RUN


def test_stager_checks_v14_grid_and_gates_match_the_tested_values():
    assert v14.verify_grid() is True
    assert len(v14.declared_candidates()) == 21
    assert v14.SUBMIT_BAR["min_tune_pq"] == v14.BASELINES["v13_compose"]["tune_pq"]
    assert v14.PROMOTION_GATE["required_tune_delta_vs_v13"] == pytest.approx(0.010)
    assert v14.CONTROL_REFERENCE["tune"][1:] == [626, 211, 508]


def test_drifted_promotion_gate_margin_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(v14, "PROMOTION_GATE",
                        dict(v14.PROMOTION_GATE, required_tune_delta_vs_v13=0.005))
    monkeypatch.setattr(prepare, "load_helpers",
                        lambda root, name: v14 if name == "v14_dualhead" else
                        _real_load_helpers(root, name))
    with pytest.raises(ValueError, match="PROMOTION_GATE margin changed"):
        prepare.stage(ROOT, tmp_path / "stage", mode="full")


def _real_load_helpers(root, name):
    from prepare_v11_refiner import load_helpers
    return load_helpers(root, name)


# -------------------------------------------------------- real-staging (skip-gated)

HAS_LOCAL_INPUTS = (
    (ROOT / "outputs/kaggle_v11_full/checkpoints/proposal_centered_e4.pt").exists()
    and (ROOT / "outputs/kaggle_detector_tta/detector_proposal_cache").exists())


@pytest.mark.skipif(not HAS_LOCAL_INPUTS, reason="requires the downloaded V6/V11 run outputs")
def test_real_full_staging_bundles_both_v14_modules_and_pins_every_hash(tmp_path):
    staged = prepare.stage(ROOT, tmp_path / "full", mode="full")
    manifest = json.loads((staged / "staging_manifest.json").read_text())
    metadata = json.loads((staged / "kernel-metadata.json").read_text())
    assert manifest["trains"] is True and manifest["submits"] is False
    assert manifest["cached_images"] == 321
    assert manifest["declared_candidates"] == 21
    assert set(manifest["module_sha256"]) == {
        "optimize_inference", "detector_tta", "v10_quality", "v11_refiner",
        "v14_dualhead", "v14_runtime"}
    assert len(manifest["v11_checkpoint_sha256"]) == 64
    assert manifest["expected_control_sha256"] == (
        "bf11690e9bcce73b0c627096cc1198d9fb0916650f7fa4b3f08f1167bb76bde9")
    assert metadata["id"] == "abhimanyu122/solar-filament-v14-dualhead"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert prepare.SOURCE_RUN in metadata["kernel_sources"]
    bundle = (staged / "v14_dualhead.py").read_text()
    entry = bundle.split("if __name__ == '__main__':")[1]
    assert "runner.run_experiment()" in entry
    assert "'--limit'" not in entry


@pytest.mark.skipif(not HAS_LOCAL_INPUTS, reason="requires the downloaded V6/V11 run outputs")
def test_real_smoke_staging_injects_the_limit_argv_and_a_distinct_slug(tmp_path):
    staged = prepare.stage(ROOT, tmp_path / "smoke", mode="smoke", limit=4)
    manifest = json.loads((staged / "staging_manifest.json").read_text())
    metadata = json.loads((staged / "kernel-metadata.json").read_text())
    assert manifest["limit"] == 4
    assert metadata["id"] == "abhimanyu122/solar-filament-v14-dualhead-smoke"
    bundle = (staged / "v14_dualhead.py").read_text()
    entry = bundle.split("if __name__ == '__main__':")[1]
    assert "sys.argv = ['v14_dualhead'] + ['--limit', '4']" in entry


@pytest.mark.skipif(not HAS_LOCAL_INPUTS, reason="requires the downloaded V6/V11 run outputs")
def test_real_staging_full_and_smoke_never_share_a_slug(tmp_path):
    full = prepare.stage(ROOT, tmp_path / "full", mode="full")
    smoke = prepare.stage(ROOT, tmp_path / "smoke", mode="smoke", limit=4)
    full_id = json.loads((full / "kernel-metadata.json").read_text())["id"]
    smoke_id = json.loads((smoke / "kernel-metadata.json").read_text())["id"]
    assert full_id != smoke_id


@pytest.mark.skipif(not HAS_LOCAL_INPUTS, reason="requires the downloaded V6/V11 run outputs")
def test_real_staging_revision_suffix_changes_slug_and_output_default():
    default = ROOT / "outputs/v14_full_revA_kernel"
    staged = prepare.stage(ROOT, default, mode="full", revision="revA")
    metadata = json.loads((staged / "kernel-metadata.json").read_text())
    assert metadata["id"] == "abhimanyu122/solar-filament-v14-dualhead-revA"
    assert metadata["title"] == "Solar Filament V14 Dualhead REVA"
