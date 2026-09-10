"""Stage the pinned V11 export-only notebook; never launches or submits anything.

The export reuses the completed V11 run's trained checkpoint and frozen
selection, delivered as Kaggle kernel-output inputs. This stager pins the
checkpoint hash, the frozen selection identity, all 321 raw detector caches, and
every bundled module source, so the notebook cannot silently export a different
candidate or a retrained model.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_v11_refiner import load_helpers, without_main


CANDIDATE = "v11_proposal_centered_e4_refined_constrained"
RUN_IDENTITY = "7c45acf26f210acce00f1bff581288bc2eb6121485237c7c028588d1de69164c"
SOURCE_RUN = "abhimanyu122/solar-filament-v11-refiner"
ARMS = ("proposal_centered", "gt_centered")
# Kaggle derives the notebook slug from the title, so the id must equal the
# title's slug or the push lands on an unexpected ref.
SLUGS = {"proposal_centered": "solar-filament-v11-export",
         "gt_centered": "solar-filament-v11-export-gt-centered"}
TITLES = {"proposal_centered": "Solar Filament V11 Export",
          "gt_centered": "Solar Filament V11 Export Gt Centered"}


def verify_completed_run(run_output, candidate=CANDIDATE, identity=RUN_IDENTITY):
    """The local V11 outputs must be the completed, gate-failed run we measured."""
    manifest = json.loads((run_output / "inference_manifest.json").read_text())
    if manifest.get("run_identity") != identity or not manifest.get("complete"):
        raise ValueError("Local V11 outputs are not the completed pinned run")
    if manifest.get("selected_candidate") != candidate:
        raise ValueError("Completed run selected a different candidate")
    if manifest.get("promotable") is not False:
        raise ValueError("Expected the gate-failed run; a promotable run exports itself")
    if manifest.get("detector_test_inference"):
        raise ValueError("Source run already exported test predictions")
    selection = json.loads((run_output / "selected_config.json").read_text())
    if selection.get("run_identity") != identity or selection.get("candidate") != candidate:
        raise ValueError("Frozen selection does not match the completed run")
    if not selection.get("frozen_before_audit") or selection.get("smoke"):
        raise ValueError("Selection must be the non-smoke, pre-audit frozen choice")
    return manifest, selection


def stage(root, output, run_output=None, arm="proposal_centered"):
    root, output = Path(root), Path(output)
    if arm not in ARMS:
        raise ValueError(f"Unknown export arm {arm}")
    run_output = Path(run_output) if run_output else root / "outputs/kaggle_v11_full"
    modules = {name: without_main((root / "kaggle_inference" / f"{name}.py").read_text())
               for name in ("optimize_inference", "detector_tta", "v10_quality",
                            "v11_refiner", "v11_export")}
    for name, source in modules.items():
        compile(source, name + ".py", "exec")
    v10, v11 = load_helpers(root, "v10_quality"), load_helpers(root, "v11_refiner")
    manifest_source, selection = verify_completed_run(run_output)
    if f"v11_{arm}_e4_refined_constrained" not in selection["tune_counts"]:
        raise ValueError(f"The completed run did not evaluate arm {arm}")
    checkpoint = run_output / "checkpoints" / f"{arm}_e4.pt"
    checkpoint_sha256 = v11.sha256(checkpoint)
    # Reuse the exact input pins the training run verified, so the export cannot
    # drift onto different sources, caches, or annotations.
    source_root = root / "outputs/v6_inference_source"
    if v11.sha256(source_root / "v6_provenance.json") != v10.PROVENANCE_SHA256:
        raise ValueError("Changed V6 provenance before staging")
    provenance = json.loads((source_root / "v6_provenance.json").read_text())
    for name, digest in provenance["sha256"].items():
        if v11.sha256(source_root / name) != digest:
            raise ValueError(f"Changed V6 artifact before staging: {name}")
    cache_source = root / "outputs/kaggle_detector_tta"
    if v11.sha256(cache_source / "inference_manifest.json") != v10.SOURCE_PINS["v6"]["manifest"]:
        raise ValueError("Changed frozen V6 source manifest")
    if v11.sha256(cache_source / "inference_image_diagnostics.csv") != v10.BASELINE_DIAGNOSTICS_SHA256:
        raise ValueError("Changed frozen V6 diagnostics")
    paths = sorted((cache_source / "detector_proposal_cache").glob("*.json.gz"))
    if len(paths) != 321:
        raise ValueError("Download all 321 V6 validation/test raw caches before staging")
    inventories = {path.name: v11.sha256(path) for path in paths}
    data_root = root / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    tests = sorted(path.name for path in (data_root / "test/test_images").glob("*.jpeg"))
    expected = {v10.raw_cache_id(name, "train") for name in provenance["validation_files"]}
    expected.update(v10.raw_cache_id(name, "test") for name in tests)
    if len(tests) != 180 or set(inventories) != expected:
        raise ValueError("Raw-cache names do not cover the exact frozen validation/test split")
    if v11.sha256(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json") != provenance["annotation_sha256"]:
        raise ValueError("Canonical annotation identity changed")
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in modules.items()}
    bundled = "import sys\nimport types\nimport hashlib\n\n"
    bundled += (f"SOURCES = {modules!r}\nMODULE_SHA256 = {hashes!r}\n"
                f"RAW_CACHE_SHA256 = {inventories!r}\nCHECKPOINT_SHA256 = {checkpoint_sha256!r}\n"
                f"ARM = {arm!r}\n\n")
    bundled += (
        "if __name__ == '__main__':\n"
        "    for name, source in SOURCES.items():\n"
        "        if hashlib.sha256(source.encode()).hexdigest() != MODULE_SHA256[name]:\n"
        "            raise ValueError('Bundled source hash mismatch: ' + name)\n"
        "        module = types.ModuleType(name)\n"
        "        module.__file__ = __file__\n"
        "        sys.modules[name] = module\n"
        "        exec(compile(source, name + '.py', 'exec'), module.__dict__)\n"
        "    exporter = sys.modules['v11_export']\n"
        "    exporter.MODULE_SHA256 = MODULE_SHA256\n"
        "    exporter.RAW_CACHE_SHA256 = RAW_CACHE_SHA256\n"
        "    exporter.CHECKPOINT_SHA256 = CHECKPOINT_SHA256\n"
        "    exporter.ARM = ARM\n"
        "    exporter.run_export()\n")
    compile(bundled, "v11_export.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "v11_export.py").write_text(bundled)
    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    metadata.update(id=f"abhimanyu122/{SLUGS[arm]}",
                    title=TITLES[arm],
                    code_file="v11_export.py", language="python", kernel_type="script",
                    is_private="true", enable_gpu="true", enable_internet="true",
                    machine_shape="NvidiaTeslaT4",
                    docker_image="gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
                    docker_image_pinning_type="original",
                    dataset_sources=["abhimanyu122/solar-filament-v6-checkpoints"],
                    competition_sources=["filament-segmentation-2026"],
                    kernel_sources=["abhimanyu122/solar-filament-coco-refiner",
                                    "abhimanyu122/solar-filament-detector-tta",
                                    SOURCE_RUN])
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = dict(experiment="v11_export_only", source_run=SOURCE_RUN, arm=arm,
                    candidate=f"v11_{arm}_e4_refined_constrained",
                    source_run_identity=RUN_IDENTITY,
                    checkpoint_sha256=checkpoint_sha256,
                    frozen_selection=selection["config"],
                    source_gate=manifest_source["gate"], promotable=False, trains=False,
                    module_sha256=hashes, script_sha256=v11.sha256(output / "v11_export.py"),
                    cached_images=321, cache_inventory_sha256=v11.digest_json(inventories),
                    source_provenance_sha256=v10.PROVENANCE_SHA256,
                    expected_control_sha256=v10.CONTROL_SHA256,
                    note="Measurement-only export of a gate-failed candidate; promotion still requires the original gate")
    (output / "staging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--arm", choices=ARMS, default="proposal_centered")
    args = parser.parse_args()
    default = root / ("outputs/v11_export_kernel" if args.arm == "proposal_centered"
                      else f"outputs/v11_export_{args.arm}_kernel")
    print(stage(root, args.output or default, args.run_output, args.arm))


if __name__ == "__main__":
    main()
