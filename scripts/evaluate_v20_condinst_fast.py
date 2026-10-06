"""Fast, parallel-safe canonical evaluator for V20 CondInst.

Each invocation evaluates one checkpoint/group on one visible GPU. Ground-truth
polygons are converted to COCO RLE once; predictions are encoded once per
threshold. This is exactly equivalent to the original mask-array scorer but
avoids repeated 2048x2048 decode/encode work.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

BASE = Path(os.environ.get("V20_BASE", "/mnt/solar-v20-condinst"))
sys.path.insert(0, str(BASE))
import train_v20_condinst as v20  # noqa: E402


def target_rles(path: Path) -> tuple[dict[str, list[list[dict]]], dict[str, dict]]:
    data = v20.read_json(path)
    annotations = defaultdict(list)
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    targets, info = defaultdict(list), {}
    for image in data["images"]:
        rles = []
        for annotation in annotations[image["id"]]:
            rle = mask_utils.merge(mask_utils.frPyObjects(
                annotation["segmentation"], image["height"], image["width"],
            ))
            rles.append(rle)
        targets[image["file_name"]].append(rles)
        info[image["file_name"]] = image
    return dict(targets), info


def encode_predictions(predictions: list[np.ndarray]) -> list[dict]:
    if not predictions:
        return []
    encoded = mask_utils.encode(
        np.asfortranarray(np.stack(predictions, axis=-1).astype(np.uint8))
    )
    return encoded if isinstance(encoded, list) else [encoded]


def pq_counts_rles(predictions: list[dict], targets: list[dict]) -> list[float]:
    if not predictions or not targets:
        return [0.0, 0, len(predictions), len(targets)]
    iou = np.asarray(mask_utils.iou(predictions, targets, [0] * len(targets))).T
    hits = iou > 0.5
    return [
        float(iou[hits].sum()), int(hits.sum()),
        int((~hits.any(axis=0)).sum()), int((~hits.any(axis=1)).sum()),
    ]


def parity_self_test() -> None:
    rng = np.random.default_rng(42)
    for n_pred, n_gt in ((0, 0), (0, 3), (4, 0), (3, 4), (8, 7)):
        predictions = [rng.random((64, 64)) > 0.93 for _ in range(n_pred)]
        targets = [rng.random((64, 64)) > 0.94 for _ in range(n_gt)]
        old = v20.pq_counts(predictions, targets)
        new = pq_counts_rles(encode_predictions(predictions), encode_predictions(targets))
        if old != new:
            raise AssertionError(f"PQ parity failed: {old} != {new}")
    print("PQ_RLE_PARITY_OK", flush=True)


def evaluate_checkpoint(checkpoint: Path, split_path: Path, thresholds: tuple[float, ...],
                        out_path: Path) -> dict:
    from detectron2.engine import DefaultPredictor

    v20.assert_inputs()
    v20.register_datasets()
    cfg = v20.build_cfg()
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    targets_by_file, _ = target_rles(split_path)
    totals = {threshold: [0.0, 0, 0, 0] for threshold in thresholds}
    started = time.monotonic()
    for index, name in enumerate(sorted(targets_by_file), 1):
        image = cv2.imread(str(v20.IMAGE_ROOT / name))
        if image is None:
            raise RuntimeError(f"could not read {name}")
        outputs = predictor(image)["instances"].to("cpu")
        masks = outputs.pred_masks.numpy() if len(outputs) else np.zeros(
            (0, image.shape[0], image.shape[1]), dtype=bool,
        )
        scores = outputs.scores.numpy() if len(outputs) else np.zeros(0, np.float32)
        for threshold in thresholds:
            pred_rles = encode_predictions(v20.ownership_masks(masks, scores, threshold))
            for gt_rles in targets_by_file[name]:
                counts = pq_counts_rles(pred_rles, gt_rles)
                for position, value in enumerate(counts):
                    totals[threshold][position] += value
        if index % 20 == 0 or index == len(targets_by_file):
            print(f"{checkpoint.name}: inferred {index}/{len(targets_by_file)}", flush=True)
    result = {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": v20.sha256(checkpoint),
        "split": split_path.name,
        "thresholds": list(thresholds),
        "seconds": time.monotonic() - started,
        "results": {
            str(threshold): {
                "pq": v20.pq(total), "tp_iou": total[0], "tp": int(total[1]),
                "fp": int(total[2]), "fn": int(total[3]),
            }
            for threshold, total in totals.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print("FAST_EVALUATION", json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("tune", "audit"))
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    parity_self_test()
    if args.self_test and args.checkpoint is None:
        return
    if args.checkpoint is None or args.split is None or args.out is None:
        parser.error("--checkpoint, --split and --out are required outside --self-test")
    split_path = v20.TUNE_JSON if args.split == "tune" else v20.AUDIT_JSON
    thresholds = tuple(float(item) for item in args.thresholds.split(","))
    evaluate_checkpoint(args.checkpoint, split_path, thresholds, args.out)


if __name__ == "__main__":
    main()
