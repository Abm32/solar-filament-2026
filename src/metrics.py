from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    intersection = np.logical_and(pred_b, target_b).sum()
    return float((2 * intersection + eps) / (pred_b.sum() + target_b.sum() + eps))


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    intersection = np.logical_and(pred_b, target_b).sum()
    union = np.logical_or(pred_b, target_b).sum()
    return float((intersection + eps) / (union + eps))

