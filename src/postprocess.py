from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import label, regionprops


def clean_binary_mask(mask: np.ndarray, min_area: int = 64, close_radius: int = 3) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)

    if close_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * close_radius + 1, 2 * close_radius + 1),
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    labels = label(mask_u8, connectivity=2)
    cleaned = np.zeros_like(mask_u8)
    for region in regionprops(labels):
        if region.area >= min_area:
            cleaned[labels == region.label] = 1
    return cleaned.astype(bool)


def connected_instance_masks(mask: np.ndarray, min_area: int = 64) -> list[np.ndarray]:
    labels = label(mask > 0, connectivity=2)
    instances: list[np.ndarray] = []
    for region in regionprops(labels):
        if region.area < min_area:
            continue
        instances.append(labels == region.label)
    return instances

