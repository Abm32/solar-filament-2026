from __future__ import annotations

import cv2
import numpy as np


def read_grayscale(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def normalize_halpha(image: np.ndarray) -> np.ndarray:
    """Normalize contrast while reducing large-scale illumination variation."""
    image_f = image.astype(np.float32) / 255.0
    background = cv2.GaussianBlur(image_f, (0, 0), sigmaX=45, sigmaY=45)
    flattened = image_f / np.clip(background, 1e-3, None)
    flattened = cv2.normalize(flattened, None, 0, 1, cv2.NORM_MINMAX)
    return (flattened * 255).astype(np.uint8)


def dark_structure_mask(image: np.ndarray, block_size: int = 151, offset: int = 9) -> np.ndarray:
    normalized = normalize_halpha(image)
    threshold = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        offset,
    )
    return threshold > 0

