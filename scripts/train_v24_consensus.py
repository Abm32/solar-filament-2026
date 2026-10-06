"""V24: matched all-annotator versus physical-image consensus-union training.

The control reproduces the current V17-style incomplete-target loader.  The arm
keeps the same 937 sample records and image order/weighting, but every repeated
record for a physical image receives the same cross-rater consensus-union target.
Clusters use constrained union-find at IoU > 0.5 and retain one original medoid
annotation; masks are never geometrically merged.

No mode evaluates audit/test data, exports a CSV, or submits to Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("V24_BASE", "/mnt/solar-v24-consensus"))
DATA_ROOT = Path(os.environ.get(
    "V24_DATA_ROOT",
    "/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/"
    "MAGFiLO_1.0_Kaggle_2026",
))
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
RAW_IMAGES = DATA_ROOT / "train/train_images"
INPUTS = BASE / "inputs"
SOURCE_DETECTOR = INPUTS / "v17_detector.pt"
PROVENANCE = INPUTS / "v6_source/v6_provenance.json"
CONTROL_DATASET = BASE / "datasets/all_annotators"
CONSENSUS_DATASET = BASE / "datasets/consensus_union"
OUTPUT = BASE / "outputs"

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
DETECTOR_SHA256 = "ae476e663039fbbbfda77ec519635ca15e2d50d1fe9f1af54735f84dbdc251ef"
PROVENANCE_SHA256 = "ceb09f980d5be195db35f2954c26fd19ea7ffc0f318b2bae5b7c803504b17122"
EXPECTED = {"files": 707, "records": 1154, "annotations": 8199}
TRAIN_EXPECTED = {"files": 566, "records": 937, "annotations": 6606}
SPLIT_SEED = 42
PRIMARY_SEED = 42
REPLICATION_SEED = 20260912
MATCH_IOU = 0.5
TRAIN_RESOLUTION = 896
BATCH = 1
EPOCHS = 20
MEMORY_GATE_MIB = 15872  # 15.5 GiB; full-population smoke must remain below this.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def annotation_rle(info: dict, annotation: dict) -> dict:
    encoded = mask_utils.frPyObjects(
        annotation["segmentation"], int(info["height"]), int(info["width"])
    )
    return mask_utils.merge(encoded) if isinstance(encoded, list) else encoded


def load_population() -> tuple[dict, dict, dict]:
    if sha256(ANNOTATIONS) != ANNOTATION_SHA256:
        raise RuntimeError("canonical annotation identity changed")
    coco = json.loads(ANNOTATIONS.read_text())
    images = {entry["id"]: entry for entry in coco["images"]}
    annotations: dict = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    actual = {
        "files": len({entry["file_name"] for entry in images.values()}),
        "records": len(images),
        "annotations": len(coco["annotations"]),
    }
    if actual != EXPECTED:
        raise RuntimeError(f"population mismatch: {actual} != {EXPECTED}")
    return coco, images, annotations


def best_annotator_ids(images: dict, annotations: dict) -> dict[str, object]:
    by_file: dict[str, list] = defaultdict(list)
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)

    def completeness(image_id):
        entries = annotations.get(image_id, [])
        return len(entries), sum(entry.get("area", 0) for entry in entries)

    return {name: max(ids, key=completeness) for name, ids in by_file.items()}


def dataset_partitions(images: dict, annotations: dict, provenance: dict) -> dict:
    by_file: dict[str, list] = defaultdict(list)
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    best = best_annotator_ids(images, annotations)
    files = sorted(best)
    random.Random(SPLIT_SEED).shuffle(files)
    held_count = round(len(files) * 0.2)
    held = sorted(files[:held_count])
    train_files = sorted(files[held_count:])
    if train_files != sorted(provenance["training_files"]):
        raise ValueError("reconstructed train split differs from V6 provenance")
    if held != sorted(provenance["validation_files"]):
        raise ValueError("reconstructed held-out split differs from V6 provenance")
    train_ids = [image_id for name in train_files for image_id in sorted(by_file[name])]
    valid_ids = [best[name] for name in held]
    instances = sum(len(annotations.get(image_id, [])) for image_id in train_ids)
    actual = {"files": len(train_files), "records": len(train_ids), "annotations": instances}
    if actual != TRAIN_EXPECTED:
        raise ValueError(f"train population mismatch: {actual}")
    return {
        "by_file": {name: sorted(ids) for name, ids in by_file.items()},
        "train_files": train_files,
        "valid_files": held,
        "train_ids": train_ids,
        "valid_ids": valid_ids,
    }


def consensus_clusters(image_ids: list, images: dict, annotations: dict) -> list[dict]:
    """Cluster cross-rater instances without allowing bridge-merges.

    Edges are processed by descending IoU. Components merge only if their rater
    sets are disjoint, guaranteeing at most one annotation from each rater in a
    cluster. The chosen representative is an original annotation maximizing
    mean IoU to the other members (stable annotation-id tie break).
    """
    nodes = []
    for rater_index, image_id in enumerate(sorted(image_ids)):
        info = images[image_id]
        for annotation in sorted(annotations.get(image_id, []), key=lambda item: str(item["id"])):
            nodes.append({
                "rater": rater_index,
                "image_id": image_id,
                "annotation": annotation,
                "rle": annotation_rle(info, annotation),
            })
    parent = list(range(len(nodes)))
    raters = [{node["rater"]} for node in nodes]

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if nodes[left]["rater"] == nodes[right]["rater"]:
                continue
            iou = float(mask_utils.iou([nodes[left]["rle"]], [nodes[right]["rle"]], [0])[0, 0])
            if iou > MATCH_IOU:
                edges.append((-iou, str(nodes[left]["annotation"]["id"]),
                              str(nodes[right]["annotation"]["id"]), left, right))
    for _, _, _, left, right in sorted(edges):
        root_left, root_right = find(left), find(right)
        if root_left == root_right or raters[root_left] & raters[root_right]:
            continue
        parent[root_right] = root_left
        raters[root_left] |= raters[root_right]

    grouped: dict[int, list] = defaultdict(list)
    for index, node in enumerate(nodes):
        grouped[find(index)].append(node)
    clusters = []
    for members in grouped.values():
        if len({member["rater"] for member in members}) != len(members):
            raise RuntimeError("consensus cluster contains duplicate rater")
        candidates = []
        for member in members:
            similarities = [
                float(mask_utils.iou([member["rle"]], [other["rle"]], [0])[0, 0])
                for other in members if other is not member
            ]
            mean_iou = float(np.mean(similarities)) if similarities else 1.0
            candidates.append((-mean_iou, str(member["annotation"]["id"]), member))
        medoid = min(candidates)[2]
        clusters.append({
            "annotation": medoid["annotation"],
            "support": len(members),
            "rater_count": len(image_ids),
            "member_annotation_ids": sorted(str(member["annotation"]["id"]) for member in members),
        })
    return sorted(clusters, key=lambda cluster: str(cluster["annotation"]["id"]))


def polygon_lines(annotation: dict, width: int, height: int) -> list[str]:
    lines = []
    for polygon in annotation["segmentation"]:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
        points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
        lines.append("0 " + " ".join(f"{value:.6f}" for value in points.ravel()))
    return lines


def build_consensus(images: dict, annotations: dict, partitions: dict) -> tuple[dict, dict]:
    consensus = {}
    diagnostics = {}
    for name in partitions["train_files"]:
        image_ids = partitions["by_file"][name]
        clusters = consensus_clusters(image_ids, images, annotations)
        consensus[name] = [cluster["annotation"] for cluster in clusters]
        diagnostics[name] = {
            "raters": len(image_ids),
            "raw_annotations": sum(len(annotations.get(image_id, [])) for image_id in image_ids),
            "clusters": len(clusters),
            "support_histogram": {
                str(support): sum(cluster["support"] == support for cluster in clusters)
                for support in range(1, len(image_ids) + 1)
            },
            "clusters_detail": [{key: cluster[key] for key in
                                 ("support", "rater_count", "member_annotation_ids")}
                                for cluster in clusters],
        }
    return consensus, diagnostics


def build_dataset(root: Path, target_policy: str, images: dict, annotations: dict,
                  partitions: dict, consensus: dict, target_id: str) -> dict:
    if root.exists():
        raise ValueError(f"refusing to overwrite dataset {root}")
    counts = {}
    for split, image_ids, unique in (
        ("train", partitions["train_ids"], True),
        ("val", partitions["valid_ids"], False),
    ):
        image_root, label_root = root / "images" / split, root / "labels" / split
        image_root.mkdir(parents=True); label_root.mkdir(parents=True)
        labels = 0
        for image_id in image_ids:
            info = images[image_id]
            source = RAW_IMAGES / info["file_name"]
            if not source.is_file():
                raise ValueError(f"missing source image {source}")
            original = Path(info["file_name"])
            name = f"{original.stem}_ann{image_id}{original.suffix}" if unique else original.name
            (image_root / name).symlink_to(source)
            targets = (consensus[info["file_name"]]
                       if split == "train" and target_policy == "consensus_union"
                       else annotations.get(image_id, []))
            lines = [line for annotation in targets
                     for line in polygon_lines(annotation, info["width"], info["height"])]
            labels += len(lines)
            (label_root / f"{Path(name).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
        counts[split] = {"records": len(image_ids), "polygon_lines": labels}
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n"
    )
    manifest = {
        "target_policy": target_policy,
        "target_id": target_id,
        "annotation_sha256": ANNOTATION_SHA256,
        "counts": counts,
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def prepare() -> None:
    _, images, annotations = load_population()
    if sha256(PROVENANCE) != PROVENANCE_SHA256:
        raise RuntimeError("V6 provenance identity changed")
    provenance = json.loads(PROVENANCE.read_text())
    partitions = dataset_partitions(images, annotations, provenance)
    consensus, diagnostics = build_consensus(images, annotations, partitions)
    summary = {
        "train_files": len(partitions["train_files"]),
        "train_records": len(partitions["train_ids"]),
        "raw_training_annotations": TRAIN_EXPECTED["annotations"],
        "unique_consensus_instances": sum(len(value) for value in consensus.values()),
        "repeated_consensus_instances": sum(
            len(consensus[name]) * len(partitions["by_file"][name])
            for name in partitions["train_files"]
        ),
        "multi_annotator_files": sum(len(partitions["by_file"][name]) > 1
                                     for name in partitions["train_files"]),
        "diagnostics": diagnostics,
    }
    target_id = digest_json(summary)
    if CONTROL_DATASET.exists() or CONSENSUS_DATASET.exists():
        raise ValueError("dataset already exists; use verify instead of rebuilding")
    control_manifest = build_dataset(
        CONTROL_DATASET, "all_annotators", images, annotations, partitions, consensus, "control"
    )
    consensus_manifest = build_dataset(
        CONSENSUS_DATASET, "consensus_union", images, annotations, partitions, consensus, target_id
    )
    payload = {
        "status": "complete", "target_id": target_id, "summary": summary,
        "control_manifest": control_manifest, "consensus_manifest": consensus_manifest,
    }
    write_json(BASE / "consensus_manifest.json", payload)
    print(json.dumps({key: payload[key] for key in ("status", "target_id", "summary")
                      if key != "summary"} | {"summary": {k: summary[k] for k in summary
                                                          if k != "diagnostics"}}, indent=2))


def training_arguments(epochs: int = EPOCHS, seed: int = PRIMARY_SEED) -> dict:
    return dict(
        epochs=epochs, patience=0, batch=BATCH, imgsz=TRAIN_RESOLUTION, workers=4,
        resume=False, optimizer="AdamW", lr0=1e-4, lrf=0.1, momentum=0.9,
        weight_decay=5e-4, cos_lr=False, warmup_epochs=1.0, warmup_bias_lr=0.0,
        nbs=64, seed=seed, deterministic=True, single_cls=True, mask_ratio=1,
        overlap_mask=False, amp=True, rect=False, multi_scale=0.0, fraction=1.0,
        freeze=None, hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=10, translate=0.1,
        scale=0.3, shear=0.0, perspective=0.0, fliplr=0.0, flipud=0.0, bgr=0.0,
        mosaic=0.3, close_mosaic=5 if epochs > 1 else 0, mixup=0.0, cutmix=0.0,
        copy_paste=0.0, save_period=5, val=False, plots=False,
    )


def verify() -> dict:
    if sha256(SOURCE_DETECTOR) != DETECTOR_SHA256:
        raise RuntimeError("V17 detector identity changed")
    if sha256(PROVENANCE) != PROVENANCE_SHA256:
        raise RuntimeError("V6 provenance identity changed")
    manifest = json.loads((BASE / "consensus_manifest.json").read_text())
    expected_records = len(json.loads(PROVENANCE.read_text())["training_files"])
    if expected_records != 566 or manifest["summary"]["train_records"] != 937:
        raise RuntimeError("training population changed")
    for dataset in (CONTROL_DATASET, CONSENSUS_DATASET):
        for split, expected in (("train", 937), ("val", 141)):
            images = list((dataset / "images" / split).glob("*.jpeg"))
            labels = list((dataset / "labels" / split).glob("*.txt"))
            if len(images) != expected or len(labels) != expected:
                raise RuntimeError(f"{dataset}/{split} population mismatch")
            if {path.stem for path in images} != {path.stem for path in labels}:
                raise RuntimeError(f"{dataset}/{split} stems mismatch")
    payload = {
        "status": "verified", "source_detector_sha256": sha256(SOURCE_DETECTOR),
        "annotation_sha256": sha256(ANNOTATIONS), "target_id": manifest["target_id"],
        "training_arguments": training_arguments(), "memory_gate_mib": MEMORY_GATE_MIB,
        "submits": False,
    }
    write_json(BASE / "verification.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def train(policy: str, smoke: bool, seed: int) -> None:
    import pandas as pd
    import torch
    from ultralytics import YOLO

    verification = verify()
    if policy not in ("control", "consensus"):
        raise ValueError(f"unknown policy {policy}")
    dataset = CONTROL_DATASET if policy == "control" else CONSENSUS_DATASET
    seed_label = "seed1" if seed == PRIMARY_SEED else "seed2"
    stage = "smoke" if smoke else "train"
    output = OUTPUT / stage / policy / seed_label
    marker, final = output / "result.json", output / "final.pt"
    arguments = training_arguments(epochs=1 if smoke else EPOCHS, seed=seed)
    signature = {
        "policy": policy, "seed": seed, "smoke": smoke,
        "source_sha256": sha256(SOURCE_DETECTOR),
        "dataset_manifest": json.loads((dataset / "manifest.json").read_text()),
        "arguments": arguments,
    }
    if marker.is_file() and final.is_file():
        saved = json.loads(marker.read_text())
        if saved["signature"] != signature or saved["checkpoint_sha256"] != sha256(final):
            raise RuntimeError("saved stage identity changed")
        print(json.dumps(saved, indent=2)); return
    if output.exists():
        raise RuntimeError(f"refusing to overwrite partial stage {output}")
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = YOLO(str(SOURCE_DETECTOR))
    model.train(data=str(dataset / "data.yaml"), device="0", project=str(output),
                name="run", exist_ok=False, **arguments)
    elapsed = time.monotonic() - started
    shutil.copy2(Path(model.trainer.last), final)
    results_path = Path(model.trainer.save_dir) / "results.csv"
    results = pd.read_csv(results_path)
    peak_bytes = int(torch.cuda.max_memory_allocated(0))
    record = {
        "signature": signature, "verification_target_id": verification["target_id"],
        "elapsed_seconds": elapsed, "epochs_completed": len(results),
        "checkpoint_sha256": sha256(final), "results_sha256": sha256(results_path),
        "peak_allocated_bytes": peak_bytes,
        "peak_allocated_mib": peak_bytes / 2**20,
        "projected_hours": elapsed * (EPOCHS if smoke else 1) / 3600,
        "finite_results": bool(np.isfinite(results.select_dtypes(include=[np.number]).to_numpy()).all()),
    }
    if not record["finite_results"]:
        raise RuntimeError("non-finite training result")
    write_json(marker, record)
    print(json.dumps(record, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify", "smoke-control",
                                         "smoke-consensus", "train-control",
                                         "train-consensus"))
    parser.add_argument("--seed", type=int, choices=(PRIMARY_SEED, REPLICATION_SEED),
                        default=PRIMARY_SEED)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); cv2.setNumThreads(2)
    if args.mode == "prepare": prepare()
    elif args.mode == "verify": verify()
    elif args.mode == "smoke-control": train("control", True, args.seed)
    elif args.mode == "smoke-consensus": train("consensus", True, args.seed)
    elif args.mode == "train-control": train("control", False, args.seed)
    else: train("consensus", False, args.seed)


if __name__ == "__main__":
    main()
