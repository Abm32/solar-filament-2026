from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.coco import best_annotator_image_ids, index_coco, load_coco


def polygon_to_yolo_line(polygon: list[float], width: int, height: int, class_id: int) -> str:
    points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    coords = " ".join(f"{value:.6f}" for value in points.flatten())
    return f"{class_id} {coords}"


def symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def write_split(
    split_name: str,
    image_ids: list[int],
    images: dict[int, dict],
    annotations: dict[int, list[dict]],
    train_dir: Path,
    yolo_dir: Path,
    cat_id_to_yolo: dict[int, int] | None,
) -> int:
    image_output = yolo_dir / "images" / split_name
    label_output = yolo_dir / "labels" / split_name
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for image_id in image_ids:
        info = images[image_id]
        height, width = info["height"], info["width"]
        file_name = info["file_name"]
        stem = Path(file_name).stem

        symlink_or_copy(train_dir / file_name, image_output / file_name)

        lines: list[str] = []
        for annotation in annotations.get(image_id, []):
            if annotation.get("iscrowd", 0):
                skipped += 1
                continue
            class_id = cat_id_to_yolo[annotation["category_id"]] if cat_id_to_yolo else 0
            for polygon in annotation["segmentation"]:
                if len(polygon) < 6:
                    skipped += 1
                    continue
                lines.append(polygon_to_yolo_line(polygon, width, height, class_id))

        (label_output / f"{stem}.txt").write_text("\n".join(lines))
    return skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/yolo_seg.yaml"))
    parser.add_argument("--valid-size", type=float, default=0.2)
    args = parser.parse_args()

    config = load_config(args.config)
    coco = load_coco(Path(config["train_json"]))
    images, annotations = index_coco(coco)
    file_to_best_id = best_annotator_image_ids(images, annotations)

    unique_files = sorted(file_to_best_id)
    rng = random.Random(config["seed"])
    rng.shuffle(unique_files)
    n_valid = max(1, int(round(len(unique_files) * args.valid_size)))
    valid_files = unique_files[:n_valid]
    train_files = unique_files[n_valid:]
    train_ids = [file_to_best_id[file_name] for file_name in train_files]
    valid_ids = [file_to_best_id[file_name] for file_name in valid_files]

    if config["use_categories"] and coco.get("categories"):
        categories = sorted(coco["categories"], key=lambda category: category["id"])
        cat_id_to_yolo = {category["id"]: index for index, category in enumerate(categories)}
        names = [category["name"] for category in categories]
    else:
        cat_id_to_yolo = None
        names = ["filament"]

    yolo_dir = Path(config["yolo_dir"])
    skipped_train = write_split(
        "train", train_ids, images, annotations, Path(config["train_dir"]), yolo_dir, cat_id_to_yolo
    )
    skipped_valid = write_split(
        "val", valid_ids, images, annotations, Path(config["train_dir"]), yolo_dir, cat_id_to_yolo
    )

    data_yaml = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(names)},
    }
    yaml_text = [
        f"path: {data_yaml['path']}",
        f"train: {data_yaml['train']}",
        f"val: {data_yaml['val']}",
        "names:",
    ]
    yaml_text.extend(f"  {index}: {name}" for index, name in data_yaml["names"].items())
    (yolo_dir / "data.yaml").write_text("\n".join(yaml_text) + "\n")
    print(f"train={len(train_ids)} valid={len(valid_ids)} skipped={skipped_train + skipped_valid}")
    print(f"wrote {yolo_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
