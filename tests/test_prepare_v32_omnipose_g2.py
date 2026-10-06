import json
from pathlib import Path

from scripts import prepare_v32_omnipose_g2 as prep

ROOT = Path(__file__).resolve().parents[1]


def test_v32_g2_stage_is_private_bounded_and_hash_pinned(tmp_path):
    stage = prep.stage(ROOT, tmp_path / "stage")
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    manifest = json.loads((stage / "staging_manifest.json").read_text())
    assert metadata["id"] == "abhimanyu122/solar-filament-v32-omnipose-g2-fix1"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["competition_sources"] == ["filament-segmentation-2026"]
    assert manifest["max_steps"] == 2000
    assert manifest["records"] == 16 and manifest["samples"] == 144
    assert manifest["target_pq"] == 0.85 and manifest["memory_gate_mib"] == 14336
    assert manifest["reads_tune"] is False
    assert manifest["reads_audit"] is False
    assert manifest["reads_test"] is False
    assert manifest["exports"] is False and manifest["submits"] is False
    assert manifest["upstream_commit"] == prep.UPSTREAM_COMMIT
    assert manifest["backbone_sha256"] == prep.BACKBONE_SHA256


def test_v32_g2_wrapper_self_checks_and_has_no_submission_action(tmp_path):
    stage = prep.stage(ROOT, tmp_path / "stage")
    source = (stage / "train_v32_omnipose_overfit_kernel.py").read_text()
    compile(source, "v32_g2_kernel.py", "exec")
    assert "bundled V32 G2 source changed" in source
    assert "pinned Omnipose core changed" in source
    assert "public ImageNet backbone identity changed" in source
    assert '"--steps", str(MAX_STEPS)' in source
    assert "kaggle competitions submit" not in source
    assert "submission.csv" not in source
    assert "test/test_images" not in source


def test_decoder_dependencies_are_exact_and_ncolor_stays_v1():
    assert "ncolor==1.2.1" in prep.PACKAGES
    assert all("==" in package for package in prep.PACKAGES)
