"""Tests for V58 verifier pure logic (load_candidates, auc). No GPU/torch needed."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("train_v58_verifier",
                                                  SCRIPTS / "train_v58_verifier.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v = _load()


def test_auc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert v.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert v.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_feature_keys_present():
    for k in ["score", "disk_r", "disk_theta", "contrast"]:
        assert k in v.FEATURE_KEYS


def test_load_candidates_roundtrip(tmp_path):
    d = tmp_path / "fold0"
    d.mkdir()
    crops = np.zeros((3, 128, 128), np.uint8)
    np.save(d / "crops.npy", crops)
    rows = []
    for i, lab in enumerate(["tp", "fp", "nearmiss"]):
        rows.append({"file": f"img{i}.jpeg", "iou": 0.6, "label": lab, "crop_idx": i,
                     "score": 0.5, "box_area": 100.0, "mask_area": 50.0, "log_area": 3.9,
                     "elongation": 2.0, "disk_r": 0.5, "disk_theta": 0.1, "mean_int": 0.3,
                     "contrast": 0.1})
    (d / "candidates.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cr, ft, labels, files = v.load_candidates([d])
    assert cr.shape == (3, 128, 128)
    assert ft.shape == (3, len(v.FEATURE_KEYS))
    assert labels == ["tp", "fp", "nearmiss"]
    assert files[0] == "img0.jpeg"
