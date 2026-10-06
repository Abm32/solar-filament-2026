import json
from pathlib import Path

from scripts import prepare_v33_balanced_omnipose_g2 as prepare

ROOT = Path(__file__).resolve().parents[1]


def test_v33_stage_is_private_bounded_and_hash_pinned(tmp_path):
    output = prepare.stage(ROOT, tmp_path / "stage")
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    manifest = json.loads((output / "staging_manifest.json").read_text())
    code = (output / metadata["code_file"]).read_text()
    assert metadata["id"] == "abhimanyu122/solar-filament-v33-balanced-omnipose-g2-fix1"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["competition_sources"] == ["filament-segmentation-2026"]
    assert manifest["max_steps"] == 2000
    assert manifest["records"] == 16 and manifest["samples"] == 144
    assert manifest["target_pq"] == 0.85
    assert not manifest["reads_tune"] and not manifest["reads_audit"]
    assert not manifest["reads_test"] and not manifest["exports"] and not manifest["submits"]
    assert "V33_SOURCE" in code and "V32_SOURCE" in code
    assert '"--steps", str(MAX_STEPS)' in code
    assert "test_images" not in code and "submission.csv" not in code
