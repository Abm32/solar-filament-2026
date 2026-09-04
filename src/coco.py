from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_coco(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def index_coco(coco: dict[str, Any]) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    images = {image["id"]: image for image in coco["images"]}
    annotations: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)
    return images, annotations


def best_annotator_image_ids(
    images: dict[int, dict],
    annotations: dict[int, list[dict]],
) -> dict[str, int]:
    """Choose one image id per file, preferring the richest annotation set."""
    file_to_ids: dict[str, list[int]] = {}
    for image_id, image in images.items():
        file_to_ids.setdefault(image["file_name"], []).append(image_id)

    def completeness(image_id: int) -> tuple[int, float]:
        anns = annotations.get(image_id, [])
        return len(anns), float(sum(ann.get("area", 0.0) for ann in anns))

    return {file_name: max(ids, key=completeness) for file_name, ids in file_to_ids.items()}

