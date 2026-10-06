import json
from pathlib import Path

from scripts import prepare_v41_context_topology_g2 as prepare

ROOT = Path(__file__).resolve().parents[1]


def test_v41_stage_is_private_offline_bounded_and_hash_pinned(tmp_path):
    output = prepare.stage(ROOT, tmp_path / "stage")
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    manifest = json.loads((output / "staging_manifest.json").read_text())
    code = (output / metadata["code_file"]).read_text()
    assert metadata["id"] == "abhimanyu122/solar-filament-v41-context-topology-g2"
    assert metadata["is_private"] == "true" and metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == "false"
    assert metadata["competition_sources"] == ["filament-segmentation-2026"]
    assert manifest["selection_id"] == prepare.SELECTION_ID
    assert manifest["max_steps"] == 2000 and manifest["records"] == 16
    assert manifest["images"] == 16 and manifest["annotations"] == 308
    assert manifest["small_thin"] == 244 and manifest["target_pq"] == 0.85
    assert not manifest["reads_tune"] and not manifest["reads_audit"]
    assert not manifest["reads_test"] and not manifest["exports"] and not manifest["submits"]
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in code
    assert "pip" not in code and "urlopen" not in code and "git clone" not in code
    assert "test_images" not in code and "submission.csv" not in code
