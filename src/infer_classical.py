from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.postprocess import clean_binary_mask, connected_instance_masks
from src.preprocess import dark_structure_mask, read_grayscale
from src.rle import encode_binary_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".fits"}


def infer_image(path: Path, min_area: int) -> list[str]:
    image = read_grayscale(str(path))
    mask = dark_structure_mask(image)
    mask = clean_binary_mask(mask, min_area=min_area)
    instances = connected_instance_masks(mask, min_area=min_area)
    return [encode_binary_mask(instance) for instance in instances]


def image_id(path: Path) -> str:
    return path.stem


def build_submission(input_dir: Path, output_path: Path, min_area: int) -> None:
    rows: list[dict[str, str]] = []
    image_paths = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")

    for path in tqdm(image_paths, desc="segmenting"):
        encoded_instances = infer_image(path, min_area=min_area)
        for index, counts in enumerate(encoded_instances, start=1):
            rows.append(
                {
                    "filament_id": f"{image_id(path)}_{index}",
                    "segmentation_rle": counts,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_submission(args.input_dir, args.output, args.min_area)


if __name__ == "__main__":
    main()

