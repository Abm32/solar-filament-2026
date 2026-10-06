"""Build leakage-safe one-record-per-physical-image YOLO data for V30.

Selects the max-instance annotator record within each canonical training file,
matching the representative-target policy used by multiple public pipelines.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

EXPECTED_TRAIN_FILES = 566
EXPECTED_VAL_FILES = 141


def line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text().splitlines())


def source_name(image_link: Path) -> str:
    return image_link.resolve().name


def prepare(source: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    groups: dict[str, list[tuple[Path, Path]]] = {}
    for image in sorted((source / "images/train").glob("*.jpeg")):
        label = source / "labels/train" / f"{image.stem}.txt"
        groups.setdefault(source_name(image), []).append((image, label))
    if len(groups) != EXPECTED_TRAIN_FILES:
        raise RuntimeError(f"expected {EXPECTED_TRAIN_FILES} physical train files, got {len(groups)}")
    selected_instances = 0
    for name, candidates in sorted(groups.items()):
        selected = max(candidates, key=lambda pair: (line_count(pair[1]), pair[1].name))
        target_image = output / "images/train" / name
        target_label = output / "labels/train" / f"{Path(name).stem}.txt"
        target_image.symlink_to(selected[0].resolve())
        target_label.symlink_to(selected[1].resolve())
        selected_instances += line_count(selected[1])
    val_images = sorted((source / "images/val").glob("*.jpeg"))
    if len(val_images) != EXPECTED_VAL_FILES:
        raise RuntimeError(f"expected {EXPECTED_VAL_FILES} validation files")
    for image in val_images:
        (output / "images/val" / image.name).symlink_to(image.resolve())
        label = source / "labels/val" / f"{image.stem}.txt"
        (output / "labels/val" / label.name).symlink_to(label.resolve())
    (output / "data.yaml").write_text(
        f"path: {output}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n"
    )
    result = {"train_files": len(groups), "train_instances": selected_instances,
              "val_files": len(val_images), "selection": "max_instance_count_then_label_name"}
    (output / "manifest.json").write_text(__import__("json").dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.output))


if __name__ == "__main__":
    main()
