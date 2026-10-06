"""Focused tests for V44 helpers (no GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_v44_full_population.py"
_spec = importlib.util.spec_from_file_location("v44", _PATH)
v44 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v44)


class _CpuArr:
    """Mimics a torch tensor enough for .cpu().numpy()."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeMasks:
    def __init__(self, arr):
        self.data = _CpuArr(arr)


class _FakeBoxes:
    def __init__(self, confs):
        self.conf = _CpuArr(np.asarray(confs, dtype=np.float32))


class _FakeResult:
    def __init__(self, masks, confs):
        self.masks = _FakeMasks(masks) if masks is not None else None
        self.boxes = _FakeBoxes(confs) if masks is not None else None


def test_annotation_hash_pinned():
    assert v44.ANNOTATION_SHA256 == (
        "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
    )
    assert v44.IMGSZ == 2048 and v44.MIN_AREA == 20


def test_disjoint_enforces_non_overlap():
    import cv2
    from pycocotools import mask as mask_utils

    # two overlapping masks; higher-conf must win the shared pixels
    a = np.zeros((2048, 2048), dtype=np.float32)
    a[100:200, 100:200] = 1.0  # conf 0.9
    b = np.zeros((2048, 2048), dtype=np.float32)
    b[150:250, 150:250] = 1.0  # conf 0.5, overlaps a
    res = _FakeResult(np.stack([a, b]), [0.9, 0.5])
    preds = v44.disjoint_instances(np, cv2, mask_utils, res, confidence=0.3)
    assert len(preds) == 2
    m0 = mask_utils.decode(preds[0]).astype(bool)
    m1 = mask_utils.decode(preds[1]).astype(bool)
    assert not np.any(m0 & m1), "instances must be disjoint"


def test_disjoint_confidence_floor():
    import cv2
    from pycocotools import mask as mask_utils

    a = np.zeros((2048, 2048), dtype=np.float32)
    a[100:200, 100:200] = 1.0
    res = _FakeResult(np.stack([a]), [0.1])  # below floor
    assert v44.disjoint_instances(np, cv2, mask_utils, res, confidence=0.3) == []


def test_disjoint_drops_tiny():
    import cv2
    from pycocotools import mask as mask_utils

    a = np.zeros((2048, 2048), dtype=np.float32)
    a[0:3, 0:3] = 1.0  # area 9 < 20
    res = _FakeResult(np.stack([a]), [0.9])
    assert v44.disjoint_instances(np, cv2, mask_utils, res, confidence=0.3) == []
