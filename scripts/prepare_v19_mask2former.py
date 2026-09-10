"""Stage the V19 Mask2Former smoke test as a private Kaggle kernel; never launches.

Unlike prepare_v11-v18 (which pin a warm-start checkpoint from a prior YOLO-seg
run), V19 has no prior artifact to pin -- it is a new detector family trained
from the public `facebook/mask2former-swin-tiny-coco-instance` checkpoint via
`transformers`. What IS pinned, following the same discipline: the bundled
module source (self-checked by SHA-256, same pattern as prepare_v18), and the
canonical competition annotations (by SHA-256), so the smoke test measures
memory against the real dataset, not a hypothetical one.

Only stages `scripts/train_v19_mask2former.py --smoke`. The full training loop
does not exist yet (train_v19_mask2former.py raises NotImplementedError for
--full by design, per docs/v19_design_2026-09-09.md) -- there is nothing to
stage for a full run until the smoke test reports a safe memory margin.
"""
import argparse
import hashlib
import json
from pathlib import Path


MODULE = "train_v19_mask2former"
MODES = {
    "smoke": dict(slug="solar-filament-v19-mask2former-smoke",
                  title="Solar Filament V19 Mask2Former Smoke", flag="--smoke", epochs=None),
    "timing": dict(slug="solar-filament-v19-mask2former-timing",
                   title="Solar Filament V19 Mask2Former Timing", flag="--full", epochs=2),
    "full": dict(slug="solar-filament-v19-mask2former-full",
                 title="Solar Filament V19 Mask2Former Full", flag="--full", epochs=None),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_main(source: str) -> str:
    """Strip the `if __name__ == "__main__":` guard so main() can be called
    directly from the bundling wrapper instead (same convention as
    prepare_v11_refiner.load_helpers's sibling `without_main` helper)."""
    marker = '\nif __name__ == "__main__":'
    index = source.find(marker)
    return source[:index] if index != -1 else source


def stage(root: Path, output: Path, mode: str = "smoke") -> Path:
    root, output = Path(root), Path(output)
    if mode not in MODES:
        raise ValueError(f"Unknown staging mode {mode!r}; expected one of {sorted(MODES)}")
    config = MODES[mode]
    script_path = root / "scripts" / f"{MODULE}.py"
    module_source = without_main(script_path.read_text())
    compile(module_source, f"{MODULE}.py", "exec")

    # Canonical annotations this smoke test's split and densest-image search
    # must run against -- pin by hash so staging fails loudly if the dataset
    # snapshot changes under it, same discipline as every prepare_v*.py.
    data_root = root / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    annotations = data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    annotation_sha256 = sha256(annotations)

    coco_module_source = (root / "src" / "coco.py").read_text()
    compile(coco_module_source, "coco.py", "exec")

    module_hash = hashlib.sha256(module_source.encode()).hexdigest()
    coco_hash = hashlib.sha256(coco_module_source.encode()).hexdigest()

    bundled = "import sys\nimport types\nimport hashlib\n\n"
    bundled += (
        f"SOURCES = {{'train_v19_mask2former': {module_source!r}, 'coco': {coco_module_source!r}}}\n"
        f"MODULE_SHA256 = {{'train_v19_mask2former': {module_hash!r}, 'coco': {coco_hash!r}}}\n\n"
    )
    bundled += (
        "if __name__ == '__main__':\n"
        "    for name, source in SOURCES.items():\n"
        "        if hashlib.sha256(source.encode()).hexdigest() != MODULE_SHA256[name]:\n"
        "            raise ValueError('Bundled source hash mismatch: ' + name)\n"
        "    src_pkg = types.ModuleType('src')\n"
        "    src_pkg.__path__ = []\n"
        "    sys.modules['src'] = src_pkg\n"
        "    coco_module = types.ModuleType('src.coco')\n"
        "    coco_module.__file__ = __file__\n"
        "    exec(compile(SOURCES['coco'], 'coco.py', 'exec'), coco_module.__dict__)\n"
        "    sys.modules['src.coco'] = coco_module\n"
        "    runner = types.ModuleType('train_v19_mask2former')\n"
        "    runner.__file__ = __file__\n"
        "    sys.modules['train_v19_mask2former'] = runner\n"
        "    exec(compile(SOURCES['train_v19_mask2former'], 'train_v19_mask2former.py', 'exec'),\n"
        "         runner.__dict__)\n"
        "    sys.argv = ['train_v19_mask2former', %r,\n"
        "                '--data-root', '/kaggle/input/competitions',\n"
        "                '--out-dir', '/kaggle/working/v19_mask2former'%s]\n"
        "    runner.main()\n" % (
            config["flag"],
            f", '--epochs', '{config['epochs']}'" if config["epochs"] is not None else "",
        )
    )
    compile(bundled, f"{MODULE}.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{MODULE}.py").write_text(bundled)

    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    metadata.update(
        id=f"abhimanyu122/{config['slug']}", title=config["title"], code_file=f"{MODULE}.py",
        language="python", kernel_type="script", is_private="true",
        enable_gpu="true", enable_internet="true", machine_shape="NvidiaTeslaT4",
        docker_image="gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        docker_image_pinning_type="original",
        dataset_sources=[], competition_sources=["filament-segmentation-2026"],
        kernel_sources=[], model_sources=[],
    )
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    manifest = dict(
        experiment="v19_mask2former", mode=mode, trains=True,
        backbone="facebook/mask2former-swin-tiny-coco-instance",
        img_size_train=1024, img_size_infer=2048,
        batch_size=2, grad_accum=4, num_queries=100, seed=42,
        epochs=config["epochs"],  # None means "use the script's default (20)"
        submit_bar_tune=0.448703,
        promotion_gate_tune=0.55,
        annotation_sha256=annotation_sha256,
        module_sha256={"train_v19_mask2former": module_hash, "coco": coco_hash},
        script_sha256=sha256(output / f"{MODULE}.py"),
        submits=False,
        note=(
            "Memory/shape probe on the worst-case (max-instance-density) training "
            "image, decoding REAL annotation masks (not placeholders), per the V18 "
            "OOM post-mortem: never conclude a per-batch memory ceiling from a "
            "random subsample. No prior checkpoint to pin -- this is a new "
            "detector family trained from the public HF Mask2Former checkpoint, "
            "not a continuation of any V6-V18 artifact. Full training intentionally "
            "not staged; train_v19_mask2former.py --full raises NotImplementedError "
            "until this smoke test confirms a safe memory margin on this VM."
            if mode == "smoke" else
            "Short timing probe (2 epochs) to measure real per-epoch wall time "
            "before committing GPU quota to a full 20-epoch run, same discipline "
            "V18's smoke_timing mode used. Not a submission candidate; its "
            "training_history.json's per-epoch timestamps are the only output "
            "that matters here."
            if mode == "timing" else
            "Full training run: 20 epochs, AdamW lr=1e-4, batch=2/grad_accum=4, "
            "evaluated every epoch on the SAME tune/audit split every V6-V18 run "
            "uses, via a direct IoU-matrix PQ (matching "
            "kaggle_inference/optimize_inference.py's pq_counts/pq_value exactly: "
            "IoU>0.5 hit rule, tp/fp/fn accumulated across the whole split before "
            "dividing). Authorized by the smoke test's cleared 11.39 GiB peak "
            "(2026-09-10, well under the 13 GiB safety threshold and the T4's "
            "14.56 GiB budget). Best-tune-PQ checkpoint is reloaded before the "
            "final audit evaluation. Gated on SUBMIT_BAR (tune >= V17's 0.448703) "
            "and PROMOTION_GATE (tune >= 0.55) per docs/v19_design_2026-09-09.md; "
            "export/submission is a separate, not-yet-written step."
        ),
    )
    (output / "staging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=sorted(MODES), default="smoke")
    args = parser.parse_args()
    default = root / f"outputs/v19_{args.mode}_kernel"
    print(stage(root, args.output or default, args.mode))


if __name__ == "__main__":
    main()
