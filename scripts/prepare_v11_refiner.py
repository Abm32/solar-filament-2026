"""Stage isolated V11 modules and exact inputs; never launch or submit anything."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def without_main(source):
    tree = ast.parse(source)
    guards = [node for node in tree.body if isinstance(node, ast.If)
              and ast.unparse(node.test) in ("__name__ == '__main__'", "'__main__' == __name__")]
    if len(guards) != 1:
        raise ValueError("Expected exactly one module entry-point guard")
    guard, lines = guards[0], source.splitlines(keepends=True)
    return "".join(lines[:guard.lineno - 1] + lines[guard.end_lineno:])


def load_helpers(root, name):
    spec = importlib.util.spec_from_file_location(name + "_stage", root / "kaggle_inference" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage(root, output, smoke=False, train_limit=3, tune_limit=3, sample_limit=32):
    root, output = Path(root), Path(output)
    if min(train_limit, tune_limit, sample_limit) < 1:
        raise ValueError("Smoke limits must be positive")
    if not smoke and (train_limit, tune_limit, sample_limit) != (3, 3, 32):
        raise ValueError("Smoke limits cannot change the full protocol")
    modules = {name: without_main((root / "kaggle_inference" / f"{name}.py").read_text())
               for name in ("optimize_inference", "detector_tta", "v10_quality", "v11_refiner")}
    for name, source in modules.items():
        compile(source, name + ".py", "exec")
    v10, v11 = load_helpers(root, "v10_quality"), load_helpers(root, "v11_refiner")
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
    source_manifest = json.loads((cache_source / "inference_manifest.json").read_text())
    v10.verify_manifest(source_manifest, "v6", provenance,
                         set(source_manifest["tune_files"]), set(source_manifest["audit_files"]))
    data_root = root / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    tests = sorted(path.name for path in (data_root / "test/test_images").glob("*.jpeg"))
    expected = {v10.raw_cache_id(name, "train") for name in provenance["validation_files"]}
    expected.update(v10.raw_cache_id(name, "test") for name in tests)
    if len(tests) != 180 or set(inventories) != expected:
        raise ValueError("Raw-cache names do not cover the exact frozen validation/test split")
    if v11.sha256(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json") != provenance["annotation_sha256"]:
        raise ValueError("Canonical annotation identity changed")
    reference_path = root / "outputs/kaggle_v10_quality/inference_image_diagnostics.csv"
    if v11.sha256(reference_path) != v11.V10_REFERENCE["diagnostics_sha256"]:
        raise ValueError("Changed V10 reference diagnostics")
    import pandas as pd
    import numpy as np
    reference = pd.read_csv(reference_path)
    reference = reference[(reference.candidate == v11.V10_REFERENCE["candidate"]) & (reference.target == "coco")]
    for group in ("tune", "audit"):
        rows = reference[reference.group == group]
        if set(rows.file_name) != set(source_manifest[f"{group}_files"]):
            raise ValueError("Incomplete V10 reference split")
        np.testing.assert_allclose(rows[list(v11.COUNTS)].sum().to_numpy(float),
                                   v11.V10_REFERENCE[group], rtol=0, atol=1e-10)
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in modules.items()}
    defaults = (["--smoke", "--train-limit", str(train_limit), "--tune-limit", str(tune_limit),
                 "--sample-limit", str(sample_limit)] if smoke else [])
    bundled = "import sys\nimport types\nimport hashlib\n\n"
    bundled += f"SOURCES = {modules!r}\nMODULE_SHA256 = {hashes!r}\nRAW_CACHE_SHA256 = {inventories!r}\n\n"
    bundled += (
        "if __name__ == '__main__':\n"
        f"    if not sys.argv[1:]:\n        sys.argv.extend({defaults!r})\n"
        f"    if {smoke!r} and '--smoke' not in sys.argv:\n"
        "        raise ValueError('A smoke bundle cannot execute a full experiment')\n"
        f"    if {not smoke!r} and '--smoke' in sys.argv:\n"
        "        raise ValueError('Stage a separate smoke bundle and notebook ID')\n"
        "    for name, source in SOURCES.items():\n"
        "        if hashlib.sha256(source.encode()).hexdigest() != MODULE_SHA256[name]:\n"
        "            raise ValueError('Bundled source hash mismatch: ' + name)\n"
        "        module = types.ModuleType(name)\n"
        "        module.__file__ = __file__\n"
        "        sys.modules[name] = module\n"
        "        exec(compile(source, name + '.py', 'exec'), module.__dict__)\n"
        "    experiment = sys.modules['v11_refiner']\n"
        "    experiment.RAW_CACHE_SHA256 = RAW_CACHE_SHA256\n"
        "    experiment.MODULE_SHA256 = MODULE_SHA256\n"
        "    experiment.run_v11()\n")
    compile(bundled, "v11_refiner.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "v11_refiner.py").write_text(bundled)
    slug = "solar-filament-v11-refiner-smoke" if smoke else "solar-filament-v11-refiner"
    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    metadata.update(id=f"abhimanyu122/{slug}", title="Solar Filament V11 Refiner" + (" Smoke" if smoke else ""),
                    code_file="v11_refiner.py", is_private="true", enable_gpu="true",
                    docker_image="gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
                    docker_image_pinning_type="original",
                    kernel_sources=["abhimanyu122/solar-filament-coco-refiner",
                                    "abhimanyu122/solar-filament-detector-tta"])
    # Runtime timeout is a kernels_push(timeout=...) API argument, not metadata.
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = dict(module_sha256=hashes, script_sha256=v11.sha256(output / "v11_refiner.py"),
                    protocol=v11.protocol(smoke, train_limit, tune_limit, sample_limit),
                    v10_reference=v11.V10_REFERENCE, cached_images=321,
                    intended_kernel_versions={"solar-filament-detector-tta": 1},
                    cache_inventory_sha256=v11.digest_json(inventories),
                    source_provenance_sha256=v10.PROVENANCE_SHA256,
                    note="Unversioned Kaggle inputs are accepted only when exact source/cache hashes match")
    (output / "staging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-limit", type=int, default=3)
    parser.add_argument("--tune-limit", type=int, default=3)
    parser.add_argument("--sample-limit", type=int, default=32)
    args = parser.parse_args()
    output = args.output or root / ("outputs/v11_refiner_smoke_kernel" if args.smoke else "outputs/v11_refiner_kernel")
    print(stage(root, output, args.smoke, args.train_limit, args.tune_limit, args.sample_limit))


if __name__ == "__main__":
    main()
