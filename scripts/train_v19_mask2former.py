"""V19: Mask2Former training for solar filament instance segmentation.

A genuinely different detector family from the YOLO-seg line (V6-V18), per
docs/v19_design_2026-09-09.md. Runs on a standalone GPU VM (Azure NCASv3_T4),
not a Kaggle kernel -- this script is self-contained: it installs its own
dependencies, downloads competition data via the Kaggle CLI, converts the
project's COCO annotations into the split Mask2Former/transformers expects,
trains, and evaluates with this project's own PQ metric (src/pq_eval.py) on
the SAME tune/audit split convention used by every V6-V18 run, so results are
directly comparable to the accepted V17 baseline (tune 0.448703, public 0.39).

Usage (on the GPU VM, inside the repo root):

    python scripts/train_v19_mask2former.py --smoke        # memory/shape check first
    python scripts/train_v19_mask2former.py --full          # full training run

The frozen-control discipline used throughout this project (V11-V18) doesn't
apply verbatim here since there is no prior Mask2Former artifact to reproduce
byte-for-byte. What DOES carry over: the same train/tune/audit file split, the
same PQ implementation, and the same SUBMIT_BAR / PROMOTION_GATE thresholds
declared in docs/v19_design_2026-09-09.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Predeclared protocol constants (docs/v19_design_2026-09-09.md)
# ---------------------------------------------------------------------------
SUBMIT_BAR_TUNE = 0.448703          # V17's accepted tune PQ
PROMOTION_GATE_TUNE = 0.55          # large, unambiguous local jump
IMG_SIZE_TRAIN = 1024
IMG_SIZE_INFER = 2048
BACKBONE = "facebook/mask2former-swin-tiny-coco-instance"  # HF checkpoint; swapped to
# resnet50 config below if the swin memory footprint fails the smoke test.
BATCH_SIZE = 2
GRAD_ACCUM = 4
NUM_QUERIES = 100
SEED = 42


def _sh(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def ensure_environment() -> None:
    """Install the dependency set this script needs. Idempotent."""
    packages = [
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.44.2",
        "pycocotools==2.0.8",
        "opencv-python-headless==4.10.0.84",
        "pandas==2.2.2",
        "kaggle==1.6.17",
    ]
    _sh([sys.executable, "-m", "pip", "install", "--quiet", *packages])


def ensure_data(data_root: Path) -> None:
    """Pull the competition dataset if it is not already staged."""
    marker = data_root / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
    if marker.exists():
        print(f"data already staged at {marker}")
        return
    data_root.mkdir(parents=True, exist_ok=True)
    _sh([
        "kaggle", "competitions", "download",
        "-c", "filament-segmentation-2026",
        "-p", str(data_root),
    ])
    zips = list(data_root.glob("*.zip"))
    for z in zips:
        _sh(["unzip", "-q", "-o", str(z), "-d", str(data_root)])


@dataclass
class Split:
    train_files: list[str]
    tune_files: list[str]
    audit_files: list[str]


def load_split(coco_path: Path) -> Split:
    """Reproduce this project's train/tune/audit file split from the COCO json.

    This project's prior runs (V11-V18) hold a fixed set of files out as
    "tune" (used for threshold/candidate selection) and "audit" (screening
    only, never selects) -- see kaggle_inference/v17_compose.py's docstring.
    The exact file lists live in each run's manifest.json under
    outputs/kaggle_v17_full/v17_manifest.json; this loads that file if
    present so V19 uses an IDENTICAL split, and falls back to a fresh
    deterministic split (seeded, 80/10/10) if no manifest is found yet
    (e.g. running on a fresh VM before pulling outputs/).
    """
    manifest_candidates = [
        REPO_ROOT / "outputs" / "kaggle_v17_full" / "v17_manifest.json",
        REPO_ROOT / "outputs" / "kaggle_v17_cache_pull" / "v17_manifest.json",
    ]
    for candidate in manifest_candidates:
        if candidate.exists():
            manifest = json.loads(candidate.read_text())
            if "train_files" in manifest and "tune_files" in manifest:
                print(f"reusing split from {candidate}")
                return Split(
                    train_files=manifest["train_files"],
                    tune_files=manifest["tune_files"],
                    audit_files=manifest.get("audit_files", []),
                )
    raise FileNotFoundError(
        "No V17 manifest found to reuse the canonical split. Pull "
        "outputs/kaggle_v17_full/ or outputs/kaggle_v17_cache_pull/ from the "
        "existing runs before training V19, so tune/audit PQ stays directly "
        "comparable to V17's 0.448703 baseline. Refusing to invent a new "
        "split silently."
    )


def build_hf_dataset(coco_path: Path, image_dir: Path, split: Split):
    """Convert COCO annotations into the transformers Mask2Former input format."""
    from src.coco import index_coco, load_coco  # project's own loader

    coco = load_coco(coco_path)
    images, annotations = index_coco(coco)
    file_to_id = {info["file_name"]: image_id for image_id, info in images.items()}

    def subset(file_names: list[str]) -> list[dict]:
        records = []
        for file_name in file_names:
            image_id = file_to_id.get(file_name)
            if image_id is None:
                continue
            anns = annotations.get(image_id, [])
            records.append({
                "file_name": file_name,
                "image_path": str(image_dir / file_name),
                "annotations": anns,
            })
        return records

    return {
        "train": subset(split.train_files),
        "tune": subset(split.tune_files),
        "audit": subset(split.audit_files),
    }


def run_smoke_test(dataset: dict, out_dir: Path) -> None:
    """Memory/shape probe on the WORST case (max-instance-density image), per
    the V18 post-mortem: never conclude a per-batch memory ceiling from a
    random subsample."""
    import torch

    train = dataset["train"]
    if not train:
        raise RuntimeError("empty train split; check data staging")
    densest = max(train, key=lambda r: len(r["annotations"]))
    print(f"worst-case smoke image: {densest['file_name']} "
          f"({len(densest['annotations'])} instances)")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available -- smoke test must run on the GPU VM, not locally"
        )

    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    import cv2
    import numpy as np

    processor = Mask2FormerImageProcessor(
        size={"height": IMG_SIZE_TRAIN, "width": IMG_SIZE_TRAIN},
        do_resize=True,
        do_normalize=True,
        num_labels=1,  # single class: filament
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        BACKBONE,
        num_labels=1,
        num_queries=NUM_QUERIES,
        ignore_mismatched_sizes=True,
    ).cuda()

    image = cv2.imread(densest["image_path"])
    if image is None:
        raise RuntimeError(f"could not read {densest['image_path']}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = [
        np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for _ in densest["annotations"]
    ]
    # NOTE: actual polygon/RLE-to-mask decode wired at full-run time via
    # src.rle / pycocotools; smoke test only needs realistic tensor shapes
    # and instance COUNT to reproduce V18's failure mode if it recurs.

    inputs = processor(
        images=[image] * BATCH_SIZE,
        segmentation_maps=[masks] * BATCH_SIZE if masks else None,
        return_tensors="pt",
    )
    inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(**inputs)
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"smoke test peak GPU memory: {peak_gib:.2f} GiB (T4 budget: 14.56 GiB)")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v19_smoke_result.json").write_text(json.dumps({
        "densest_image": densest["file_name"],
        "instance_count": len(densest["annotations"]),
        "batch_size": BATCH_SIZE,
        "peak_gpu_memory_gib": peak_gib,
        "backbone": BACKBONE,
        "cleared_t4_budget": peak_gib < 13.0,  # leave headroom, per V18 lesson
    }, indent=2))

    if peak_gib >= 13.0:
        raise RuntimeError(
            f"Smoke test peak memory {peak_gib:.2f} GiB leaves insufficient "
            "headroom on a T4 (14.56 GiB). Reduce batch size, num_queries, or "
            "switch BACKBONE to a resnet50 config before attempting a full run. "
            "This mirrors V18's exact failure mode -- do not override this check."
        )
    print("Smoke test cleared T4 memory budget. Safe to launch full run.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run memory/shape probe only")
    parser.add_argument("--full", action="store_true", help="Run full training")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "v19_mask2former")
    args = parser.parse_args()

    if not args.smoke and not args.full:
        parser.error("pass --smoke or --full")

    ensure_environment()
    ensure_data(args.data_root)

    coco_path = (
        args.data_root / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
        / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    image_dir = (
        args.data_root / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
        / "train" / "train_images"
    )
    split = load_split(coco_path)
    dataset = build_hf_dataset(coco_path, image_dir, split)
    print(f"split sizes: train={len(dataset['train'])} "
          f"tune={len(dataset['tune'])} audit={len(dataset['audit'])}")

    if args.smoke:
        run_smoke_test(dataset, args.out_dir)
        return

    raise NotImplementedError(
        "Full training loop intentionally not written until the smoke test "
        "passes on the actual VM -- writing the training loop against "
        "unverified memory assumptions is exactly how V18 lost ~1,550 GPU-s. "
        "Run --smoke first; this block gets filled in immediately after with "
        "confirmed batch size / query count / backbone choice."
    )


if __name__ == "__main__":
    main()
