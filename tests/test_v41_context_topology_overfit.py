import ast
from pathlib import Path

from scripts import train_v41_context_topology_overfit as v41

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/train_v41_context_topology_overfit.py").read_text()


def test_v41_contract_is_frozen_and_bounded():
    assert v41.SELECTION_ID == "741d8124fd834c465743cb4ddbee265298354b4e59b5ab9650379d162673ca2c"
    assert v41.SAMPLES == 144 and v41.MAX_STEPS == 2000
    assert v41.EVAL_STEPS == (0, 250, 500, 1000, 1500, 2000)
    assert v41.THRESHOLDS == (0.35, 0.50, 0.65)
    assert v41.TARGET_PQ == 0.85 and v41.TARGET_RECALL == 0.90
    assert v41.TARGET_MEDIAN_IOU == 0.80 and v41.MAX_FP == 40
    assert v41.MEMORY_GATE_MIB == 14336 and v41.RUNTIME_GATE_SECONDS == 7200


def test_v41_architecture_contains_context_texture_and_topology_mechanisms():
    assert "class ResidualSE" in SOURCE and "class SE" in SOURCE
    assert "for dilation in (1, 2, 4, 8)" in SOURCE
    assert "self.texture = torch.nn.Conv2d(24, 2, 1)" in SOURCE
    assert "self.topology = torch.nn.Sequential" in SOURCE
    assert "texture[:, :1] + residual" in SOURCE
    assert "value[:, 2:3]" in SOURCE and "value[:, 3:4]" in SOURCE


def test_v41_losses_and_decoder_are_frozen():
    assert '2 * terms["final_bce"]' in SOURCE
    assert 'terms["spine_bce"]' in SOURCE and 'terms["spine_dice"]' in SOURCE
    assert "watershed(-foreground_probability" in SOURCE
    assert "MIN_SEED_AREA = 5" in SOURCE and "MIN_AREA = 20" in SOURCE


def test_v41_gpu_preflight_precedes_image_and_model_work():
    main_source = SOURCE[SOURCE.index("def main():") :]
    assert main_source.index("provenance = gpu_preflight(torch)") < main_source.index(
        "records = selection_records"
    )
    assert main_source.index("provenance = gpu_preflight(torch)") < main_source.index(
        "model = build_model"
    )
    assert "torch.cuda.device_count() != 1" in SOURCE
    assert '"T4" not in name.upper()' in SOURCE


def test_v41_has_no_tune_audit_test_export_or_submission_path():
    tree = ast.parse(SOURCE)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute | ast.Name)
    }
    assert "test_images" not in SOURCE and "submission.csv" not in SOURCE
    assert not ({"submit", "to_csv"} & calls)
    assert '"reads_tune": False' in SOURCE and '"reads_audit": False' in SOURCE
    assert '"reads_test": False' in SOURCE and '"exports": False' in SOURCE


def test_v41_design_is_frozen_before_implementation():
    design = (ROOT / "docs/v41_design_2026-09-14.md").read_text()
    assert "before V41 implementation, staging, or model execution" in design
    assert v41.SELECTION_ID in design
    assert "244 both small and thin" in design
