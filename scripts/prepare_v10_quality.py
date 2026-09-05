"""Stage a private, standalone V10 script; never launch or submit anything."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def without_main(source):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    guards = [node for node in tree.body if isinstance(node, ast.If)
              and ast.unparse(node.test) in ("__name__ == '__main__'", "'__main__' == __name__")]
    if len(guards) != 1:
        raise ValueError("Expected exactly one module entry-point guard")
    guard = guards[0]
    return "".join(lines[:guard.lineno - 1] + lines[guard.end_lineno:])


def stage(root, output, limit=0):
    if limit < 0:
        raise ValueError("Smoke limit must be nonnegative")
    modules = {name: without_main((root / "kaggle_inference" / f"{name}.py").read_text())
               for name in ("optimize_inference", "detector_tta", "v10_quality")}
    for name, source in modules.items():
        compile(source, name + ".py", "exec")
    spec = importlib.util.spec_from_file_location("v10_stage_helpers", root / "kaggle_inference/v10_quality.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)  # CPU-only declarations; no torch import or runtime entry point.
    inventories = {}
    for detector, directory in (("v6", "kaggle_detector_tta"), ("v8", "kaggle_v8_coco_tta")):
        source = root / "outputs" / directory
        if helpers.sha256(source / "inference_manifest.json") != helpers.SOURCE_PINS[detector]["manifest"]:
            raise ValueError(f"Changed {detector} source manifest before staging")
        cache_root = source / "detector_proposal_cache"
        if detector == "v8":
            cache_root /= helpers.SOURCE_PINS[detector]["detector"][:16]
        paths = sorted(cache_root.glob("*.json.gz"))
        if len(paths) != 321:
            raise ValueError(f"Download all 321 {detector} validation/test raw caches before staging")
        inventories[detector] = {path.name: helpers.sha256(path) for path in paths}
    # Isolated modules avoid the name/global collisions of plain source concatenation.
    bundled = "import sys\nimport types\n\n"
    bundled += f"SOURCES = {modules!r}\nRAW_CACHE_SHA256 = {inventories!r}\n\n"
    bundled += (
        "if __name__ == '__main__':\n"
        f"    if {limit} and '--limit' not in sys.argv:\n"
        f"        sys.argv.extend(['--limit', '{limit}'])\n"
        "    for name, source in SOURCES.items():\n"
        "        module = types.ModuleType(name)\n"
        "        module.__file__ = __file__\n"
        "        sys.modules[name] = module\n"
        "        exec(compile(source, name + '.py', 'exec'), module.__dict__)\n"
        "    sys.modules['v10_quality'].RAW_CACHE_SHA256 = RAW_CACHE_SHA256\n"
        "    sys.modules['v10_quality'].run_v10()\n")
    compile(bundled, "v10_quality.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "v10_quality.py").write_text(bundled)
    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    metadata.update(id="abhimanyu122/solar-filament-v10-quality-rescue", title="Solar Filament V10 Quality Rescue",
                    code_file="v10_quality.py", is_private="true", enable_gpu="true",
                    kernel_sources=["abhimanyu122/solar-filament-coco-refiner",
                                    "abhimanyu122/solar-filament-detector-tta",
                                    "abhimanyu122/solar-filament-v8-coco-tta"])
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "staging_manifest.json").write_text(json.dumps(dict(
        module_sha256={name: hashlib.sha256(source.encode()).hexdigest() for name, source in modules.items()},
        script_sha256=helpers.sha256(output / "v10_quality.py"), cached_images_per_detector=321,
        smoke_limit=limit,
        intended_kernel_versions={"solar-filament-detector-tta": 1, "solar-filament-v8-coco-tta": 1},
        note="Kaggle source slugs are unversioned; exact manifest and all raw-cache hashes enforce identity",
        declared_candidates=helpers.declared_candidates()), indent=2) + "\n")
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "outputs/v10_quality_kernel")
    parser.add_argument("--limit", type=int, default=0, help="Bake a tuning-only smoke limit into the staged script")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    print(stage(root, args.output, args.limit))


if __name__ == "__main__":
    main()
