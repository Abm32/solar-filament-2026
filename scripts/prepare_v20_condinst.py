"""Prepare split-safe, one-class COCO files for the proposed CondInst experiment.

Every annotator image record remains independent, while all records sharing one
physical filename stay in the same frozen train/tune/audit group. Chirality is
irrelevant to the competition metric, so categories 1-3 are remapped to one
`filament` class. No image, label, or external asset is copied or downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_v19_mask2former import load_split  # noqa: E402

EXPECTED = {
    "train": {"files": 566, "records": 937, "annotations": 6606},
    "tune": {"files": 94, "records": 150, "annotations": 1134},
    "audit": {"files": 47, "records": 67, "annotations": 459},
}
SUBMIT_BAR_TUNE = 0.448703
SUBMIT_BAR_AUDIT = 0.470763
PROMOTION_GATE_TUNE = 0.55


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_split(source: dict, file_names: set[str], group: str) -> tuple[dict, dict]:
    source_images = [image for image in source["images"] if image["file_name"] in file_names]
    source_ids = {image["id"] for image in source_images}
    source_annotations = [ann for ann in source["annotations"] if ann["image_id"] in source_ids]

    image_id_map: dict[str, int] = {}
    images = []
    for new_id, image in enumerate(source_images, start=1):
        image_id_map[image["id"]] = new_id
        converted = dict(image)
        converted["source_image_id"] = image["id"]
        converted["id"] = new_id
        images.append(converted)

    annotations = []
    original_categories = Counter()
    for new_id, annotation in enumerate(source_annotations, start=1):
        original_categories[int(annotation["category_id"])] += 1
        converted = dict(annotation)
        converted["source_annotation_id"] = annotation["id"]
        converted["id"] = new_id
        converted["image_id"] = image_id_map[annotation["image_id"]]
        converted["source_category_id"] = annotation["category_id"]
        converted["category_id"] = 1
        converted["iscrowd"] = int(annotation.get("iscrowd", 0))
        annotations.append(converted)

    output = {
        "info": {
            **source.get("info", {}),
            "description": f"Competition-only CondInst {group} split; all annotators; one class",
        },
        "licenses": source.get("licenses", []),
        "categories": [{"id": 1, "name": "filament", "supercategory": "filament"}],
        "images": images,
        "annotations": annotations,
    }
    summary = {
        "files": len({image["file_name"] for image in images}),
        "records": len(images),
        "annotations": len(annotations),
        "original_category_counts": dict(sorted(original_categories.items())),
        "max_instances_per_record": max(
            Counter(a["image_id"] for a in annotations).values(), default=0,
        ),
    }
    if {key: summary[key] for key in ("files", "records", "annotations")} != EXPECTED[group]:
        raise ValueError(f"{group} population mismatch: {summary} != {EXPECTED[group]}")
    return output, summary


def validate_partition(source: dict, converted: dict[str, dict]) -> None:
    groups = {
        group: {image["file_name"] for image in data["images"]}
        for group, data in converted.items()
    }
    if groups["train"] & groups["tune"] or groups["train"] & groups["audit"] or groups["tune"] & groups["audit"]:
        raise ValueError("Physical filenames overlap across CondInst splits")
    if set.union(*groups.values()) != {image["file_name"] for image in source["images"]}:
        raise ValueError("Converted splits do not cover every approved physical filename")
    if sum(len(data["images"]) for data in converted.values()) != len(source["images"]):
        raise ValueError("Annotator records were lost or duplicated")
    if sum(len(data["annotations"]) for data in converted.values()) != len(source["annotations"]):
        raise ValueError("Annotations were lost or duplicated")
    for data in converted.values():
        image_ids = [image["id"] for image in data["images"]]
        annotation_ids = [ann["id"] for ann in data["annotations"]]
        if len(image_ids) != len(set(image_ids)) or len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("Converted COCO integer IDs are not unique")
        if any(ann["image_id"] not in set(image_ids) for ann in data["annotations"]):
            raise ValueError("Annotation references an unknown converted image")
        if any(ann["category_id"] != 1 for ann in data["annotations"]):
            raise ValueError("Category remapping failed")


def prepare(coco_path: Path, out_dir: Path) -> dict:
    source = json.loads(coco_path.read_text())
    split = load_split(coco_path)
    file_groups = {
        "train": set(split.train_files),
        "tune": set(split.tune_files),
        "audit": set(split.audit_files),
    }
    converted, summaries = {}, {}
    for group, names in file_groups.items():
        converted[group], summaries[group] = convert_split(source, names, group)
    validate_partition(source, converted)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for group, data in converted.items():
        path = out_dir / f"condinst_{group}.json"
        path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        output_hashes[path.name] = sha256(path)

    contract = {
        "experiment": "V20 CondInst rule-safe first run",
        "status": "PREPARED_NOT_AUTHORIZED_FOR_GPU_OR_SUBMISSION",
        "source_annotations": str(coco_path),
        "source_sha256": sha256(coco_path),
        "data_policy": {
            "allowed": "competition training images and labels; standard public COCO/ImageNet initialization",
            "excluded": [
                "full public MAGFiLO archive (contains competition test labels)",
                "AHAS/MHAS or solar-specific labels/weights without organizer permission",
                "test-time temporal label transfer",
            ],
        },
        "architecture": {
            "family": "CondInst R50-FPN",
            "class_count": 1,
            "mask_branch": "P2 stride-4",
            "relative_coordinates": True,
            "auxiliary_semantic_loss": True,
            "input_size": 1024,
            "normalization": "per-image robust disk normalization; no site/date prediction feature",
            "initialization": "standard public COCO CondInst or ImageNet backbone only",
        },
        "protocol": {
            "smoke_first": "densest 26-instance record; batch=1; require >=2 GiB GPU headroom",
            "tune_selection": "confidence/NMS grid fixed before audit; tune only",
            "audit_once": True,
            "submit_bar_tune": SUBMIT_BAR_TUNE,
            "submit_bar_audit": SUBMIT_BAR_AUDIT,
            "promotion_gate_tune": PROMOTION_GATE_TUNE,
            "export_requires_both_bars": True,
        },
        "populations": summaries,
        "output_sha256": output_hashes,
    }
    contract_path = out_dir / "experiment_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path,
        default=ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/v20_condinst_prep")
    args = parser.parse_args()
    print(json.dumps(prepare(args.annotations, args.out), indent=2))


if __name__ == "__main__":
    main()
