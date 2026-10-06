"""Export one deterministic V21 test shard using the frozen V17 recipe.

Launch four processes with --shard-index 0..3 and --num-shards 4, each bound to
one visible GPU. The script emits a CSV and signed manifest for its disjoint set
of test files. It never submits to Kaggle.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

BASE = Path(os.environ.get("V21_WORK", "/mnt/solar-v21-full-refit"))
REPO = Path(os.environ.get("V21_REPO", "/home/azureuser/solar-filament-2026"))
DATA = Path(os.environ.get(
    "V21_DATA_ROOT",
    "/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
))
PERSIST = REPO / "outputs/v21_full_refit_persistent"
DETECTOR = Path(os.environ.get(
    "V21_DETECTOR", str(PERSIST / "checkpoints/detector_mask_ratio1_final.pt")
))
REFINER = Path(os.environ.get(
    "V21_REFINER", str(PERSIST / "checkpoints/refiner_proposal_centered_final.pt")
))
SOLAR_BASE = BASE / "inputs/solar_base.py"
RUNTIME = BASE / "runtime"
TEST_DIR = DATA / "test/test_images"
EXPECTED_DETECTOR_SHA = os.environ.get(
    "V21_EXPECTED_DETECTOR_SHA",
    "5cc803e4f3ca334c471fc1012ef1e27a142b59b085831fc09d5e35d621ca5629",
)
EXPECTED_REFINER_SHA = os.environ.get(
    "V21_EXPECTED_REFINER_SHA",
    "26013cec76b7c329f9891fdbf90def19abc60a696d134273837cf5dc3f5f0792",
)
BOX_IOU = 0.3
DETECTOR_FLOOR = 0.1
MASK_THRESHOLD = 0.45
CONFIDENCE = 0.30
MIN_AREA = 20

sys.path.insert(0, str(RUNTIME))
import optimize_inference as core  # noqa: E402
import detector_tta as tta  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def encode(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def proposal_views(detector, image_path: Path, cache_root: Path, detector_sha: str):
    token = (
        f"{detector_sha}|mr1|v21_iou030|test|{image_path.name}|"
        f"imgsz1024|conf{DETECTOR_FLOOR}"
    )
    cache = cache_root / (hashlib.sha256(token.encode()).hexdigest()[:16] + ".json.gz")
    marker = cache.with_suffix(".signature.json")
    signature = {
        "file_name": image_path.name, "detector_sha256": detector_sha,
        "mask_ratio": 1, "box_iou": BOX_IOU, "resolution": 1024,
        "confidence_floor": DETECTOR_FLOOR, "views": 4,
    }
    if cache.exists():
        saved = json.loads(marker.read_text())
        if saved["signature"] != signature or saved["sha256"] != sha256(cache):
            raise RuntimeError(f"stale or changed cache {cache}")
        with gzip.open(cache, "rt") as stream:
            raw = json.load(stream)
        return [
            ([{"size": item["size"], "counts": item["counts"].encode()} for item in view["rles"]],
             view["scores"])
            for view in raw
        ]

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (2048, 2048):
        raise RuntimeError(f"invalid test image {image_path}")
    views = []
    for axes in ((), (1,), (0,), (0, 1)):
        source = str(image_path) if not axes else tta.flip_array(image, axes)
        result = detector.predict(
            source, imgsz=1024, conf=DETECTOR_FLOOR, iou=BOX_IOU,
            agnostic_nms=True, retina_masks=True, verbose=False,
        )[0]
        rles, scores = [], []
        if result.masks is not None:
            scores = result.boxes.conf.detach().cpu().numpy().tolist()
            for tensor in result.masks.data:
                restored = tta.flip_array(tensor.cpu().numpy().astype(np.uint8), axes)
                if restored.shape != (2048, 2048):
                    raise RuntimeError(f"mask not restored to native geometry: {restored.shape}")
                rle = encode(restored)
                if mask_utils.area(rle) > 0:
                    rles.append(rle)
                else:
                    scores = scores[:len(rles)]
        if len(rles) != len(scores):
            # Preserve exact paired ordering while discarding only zero-area masks.
            paired = []
            if result.masks is not None:
                raw_scores = result.boxes.conf.detach().cpu().numpy().tolist()
                for tensor, score in zip(result.masks.data, raw_scores):
                    restored = tta.flip_array(tensor.cpu().numpy().astype(np.uint8), axes)
                    rle = encode(restored)
                    if mask_utils.area(rle) > 0:
                        paired.append((rle, score))
            rles = [item[0] for item in paired]
            scores = [float(item[1]) for item in paired]
        views.append((rles, scores))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt") as stream:
        json.dump([
            {"rles": [{"size": list(rle["size"]), "counts": rle["counts"].decode()}
                      for rle in rles], "scores": scores}
            for rles, scores in views
        ], stream)
    write_json(marker, {"signature": signature, "sha256": sha256(cache)})
    return views


def refine(base, model, views, image_path: Path):
    proposals, scores = tta.cluster_views(views, "flip_consensus", match_iou=0.5)
    if not proposals:
        return [], np.asarray([], dtype=float)
    masks = [mask_utils.decode(rle) for rle in proposals]
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (2048, 2048):
        raise RuntimeError(f"invalid grayscale image {image_path}")
    bounds, _, probabilities = core.crop_probabilities(base, model, gray, masks, True)
    rles, kept_scores = core.clean_refined_predictions(
        masks, scores, bounds, probabilities, MASK_THRESHOLD, "refined_constrained",
    )
    return rles, np.asarray(kept_scores, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard index")
    if sha256(DETECTOR) != EXPECTED_DETECTOR_SHA:
        raise RuntimeError("detector checkpoint identity changed")
    if sha256(REFINER) != EXPECTED_REFINER_SHA:
        raise RuntimeError("refiner checkpoint identity changed")

    base = core.load_base(SOLAR_BASE)
    base.CFG.root = DATA
    base.CFG.train_dir = DATA / "train/train_images"
    base.CFG.test_dir = TEST_DIR
    base.CFG.train_json = DATA / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    base.CFG.crop_size = 256
    base.CFG.crop_context = 1.8
    base.CFG.min_crop = 96
    base.CFG.imgsz = 1024
    base.seed_everything(42)
    cv2.setNumThreads(2)
    torch.set_num_threads(2)

    from ultralytics import YOLO
    detector = YOLO(str(DETECTOR))
    model = base.CropUNet(pretrained=False).to("cuda:0")
    model.load_state_dict(torch.load(REFINER, map_location="cuda:0", weights_only=True))
    model.eval()

    all_images = sorted(TEST_DIR.glob("*.jpeg"))
    if len(all_images) != 180:
        raise RuntimeError(f"expected 180 test images, found {len(all_images)}")
    selected = [path for index, path in enumerate(all_images) if index % args.num_shards == args.shard_index]
    if len(selected) != 180 // args.num_shards:
        raise RuntimeError(f"unbalanced shard: {len(selected)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.output_dir / "proposal_cache"
    rows, coverage, started = [], {}, time.monotonic()
    for index, image_path in enumerate(selected, 1):
        views = proposal_views(detector, image_path, cache_root, EXPECTED_DETECTOR_SHA)
        rles, scores = refine(base, model, views, image_path)
        predictions = [rle for rle, score in zip(rles, scores) if score >= CONFIDENCE]
        occupied = np.zeros((2048, 2048), dtype=bool)
        for number, rle in enumerate(predictions, 1):
            mask = mask_utils.decode(rle).astype(bool)
            if int(mask.sum()) < MIN_AREA or np.any(occupied & mask):
                raise RuntimeError(f"invalid area/overlap for {image_path.name}")
            occupied |= mask
            rows.append({
                "filament_id": f"{image_path.stem}_{number}",
                "segmentation_rle": rle["counts"].decode(),
            })
        coverage[image_path.name] = len(predictions)
        if index % 10 == 0 or index == len(selected):
            print(
                f"shard={args.shard_index}/{args.num_shards} images={index}/{len(selected)} "
                f"rows={len(rows)} elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )

    csv_path = args.output_dir / f"shard_{args.shard_index}.csv"
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(csv_path, index=False)
    manifest = {
        "status": "complete", "shard_index": args.shard_index,
        "num_shards": args.num_shards, "files": [path.name for path in selected],
        "evaluated_images": len(selected), "rows": len(rows),
        "empty_images": [name for name, count in coverage.items() if count == 0],
        "coverage": coverage, "detector_sha256": EXPECTED_DETECTOR_SHA,
        "refiner_sha256": EXPECTED_REFINER_SHA,
        "recipe": {
            "box_iou": BOX_IOU, "detector_floor": DETECTOR_FLOOR,
            "detector_views": 4, "policy": "flip_consensus",
            "mask_threshold": MASK_THRESHOLD, "confidence": CONFIDENCE,
            "mode": "refined_constrained", "crop_tta_views": 4,
            "min_area": MIN_AREA,
        },
        "csv_sha256": sha256(csv_path), "seconds": time.monotonic() - started,
    }
    write_json(args.output_dir / f"shard_{args.shard_index}.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "coverage"}, indent=2))


if __name__ == "__main__":
    main()
