"""Verify analyze_v16_proposals against V16's own reported raw-IoU summaries.

The point of these tests is that the offline re-analysis must reproduce numbers
the Kaggle run already published from the same caches. If it cannot, the offline
conclusions are not trustworthy and must not be acted on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_v16_proposals as mod  # noqa: E402

MANIFEST = ROOT / "outputs/kaggle_v16_full/v16_manifest.json"
CACHE = ROOT / "outputs/kaggle_v16_full/v16_proposal_cache"

pytestmark = pytest.mark.skipif(not MANIFEST.exists() or not CACHE.exists(),
                                reason="V16 artifacts not present")


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def test_cache_id_matches_the_kernel_key_construction():
    # Reproduces v16_maskratio.cache_id exactly; a drift here silently reads the
    # wrong arm's proposals, which is the one error that would invert results.
    import hashlib
    name, split, mask_ratio, sha = "20120402105814Th.jpeg", "train", 1, "deadbeef"
    key = f"{sha}|mr{mask_ratio}|1024|{split}|{name}"
    assert mod.cache_id(name, split, mask_ratio, sha) == \
        hashlib.sha256(key.encode()).hexdigest()[:16] + ".json.gz"


def test_every_arm_cache_is_present_and_distinct(manifest):
    arms = {"v16_frozen_control": (2, mod.V6_DETECTOR_SHA256)}
    for arm, training in manifest["training"].items():
        arms[f"v16_{arm}"] = (training["mask_ratio"], training["checkpoint_sha256"])
    assert len(arms) == 3
    names = sorted(set(manifest["tune_files"]) | set(manifest["audit_files"]))
    assert len(names) == 141
    seen = {}
    for candidate, (mask_ratio, sha) in arms.items():
        ids = {mod.cache_id(name, "train", mask_ratio, sha) for name in names}
        assert len(ids) == 141
        for path in ids:
            assert (CACHE / path).exists(), f"missing {candidate} cache {path}"
        seen[candidate] = ids
    # No two arms may share a cache file, or the comparison is vacuous.
    for a in seen:
        for b in seen:
            if a < b:
                assert not seen[a] & seen[b], f"{a} and {b} share caches"


def test_greedy_match_is_one_to_one_and_respects_the_threshold():
    iou = np.array([[0.9, 0.6], [0.55, 0.8]])
    pairs = mod.greedy_match(iou)
    assert sorted(pairs) == [(0, 0, 0.9), (1, 1, 0.8)]
    assert mod.greedy_match(np.array([[0.5, 0.4]])) == []  # strict > 0.5


def test_summarise_pq_formulas():
    rows = [dict(n_gt=2, n_pred=3, matched=1, iou_sum=0.8)]
    out = mod.summarise(rows, "x")
    assert out["recall"] == 0.5
    assert out["mean_matched_iou"] == pytest.approx(0.8)
    # oracle: FP dropped -> 0.8 / (1 + 0.5*1)
    assert out["oracle_pq"] == pytest.approx(0.8 / 1.5)
    # keep-all: 2 unmatched preds become FP -> 0.8 / (1 + 0.5*(2+1))
    assert out["keep_all_pq"] == pytest.approx(0.8 / 2.5)


def test_cluster_views_matches_detector_tta_exactly():
    """The grouping must be the shipped policy, not a lookalike.

    Compares against kaggle_inference/detector_tta.cluster_views on synthetic
    multi-view input. A drift here changes every 'grouped' number reported.
    """
    import importlib.util
    import types
    from pycocotools import mask as mask_utils

    # detector_tta imports torch and its sibling module at scope for the GPU paths;
    # this test only exercises cluster_views, which is pure numpy/pycocotools. Add
    # the package dir to the path and stub torch so the REAL shipped function is
    # compared rather than a copy of it.
    if "torch" not in sys.modules:
        stub = types.ModuleType("torch")

        def _no_grad(fn=None):
            return fn if fn is not None else (lambda f: f)

        stub.no_grad = _no_grad
        sys.modules["torch"] = stub
    inference_dir = str(ROOT / "kaggle_inference")
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    spec = importlib.util.spec_from_file_location(
        "detector_tta_ref", ROOT / "kaggle_inference/detector_tta.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    rng = np.random.default_rng(20260908)

    def blob(cx, cy, r):
        mask = np.zeros((64, 64), dtype=np.uint8)
        yy, xx = np.mgrid[0:64, 0:64]
        mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 1
        return mask_utils.encode(np.asfortranarray(mask))

    for _ in range(8):
        views = []
        for _ in range(4):
            count = int(rng.integers(0, 4))
            rles = [blob(int(rng.integers(12, 52)), int(rng.integers(12, 52)),
                         int(rng.integers(6, 16))) for _ in range(count)]
            scores = [float(rng.uniform(0.1, 1.0)) for _ in range(count)]
            views.append((rles, scores))
        mine_rles, mine_scores = mod.cluster_views(views)
        ref_rles, ref_scores = reference.cluster_views(views, "flip_consensus", 0.5)
        assert [r["counts"] for r in mine_rles] == [r["counts"] for r in ref_rles]
        assert mine_scores == pytest.approx(ref_scores)


def test_reproduces_reported_raw_max_and_mean(manifest, tmp_path):
    """The strongest available check: V16's kernel reported per-arm raw-proposal
    mean/max IoU from these same caches. Recompute with independent code."""
    reported = manifest["raw_proposal_summaries"]
    arms = {"v16_frozen_control": (2, mod.V6_DETECTOR_SHA256)}
    for arm, training in manifest["training"].items():
        arms[f"v16_{arm}"] = (training["mask_ratio"], training["checkpoint_sha256"])

    data_root = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    coco = json.loads((data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
    images = {image["id"]: image for image in coco["images"]}
    annotations = {}
    for entry in coco["annotations"]:
        annotations.setdefault(entry["image_id"], []).append(entry)
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)
    names = sorted(set(manifest["tune_files"]) | set(manifest["audit_files"]))

    from pycocotools import mask as mask_utils
    for candidate, (mask_ratio, sha) in arms.items():
        values = []
        for name in names:
            views = mod.read_views(CACHE / mod.cache_id(name, "train", mask_ratio, sha))
            rles = views[0][0] if views else []
            for image_id in sorted(by_file[name]):
                info = images[image_id]
                gt = [mod.annotation_rle(info, e) for e in annotations.get(image_id, [])]
                gt = [r for r in gt if mask_utils.area(r) > 0]
                if not rles or not gt:
                    values.extend([0.0] * len(gt))
                    continue
                iou = mask_utils.iou(rles, gt, [0] * len(gt))
                values.extend(iou.max(axis=0).tolist() if iou.size else [0.0] * len(gt))
        expected = reported[candidate]
        assert len(values) == expected["n"]
        assert float(np.mean(values)) == pytest.approx(expected["mean"], abs=1e-9)
        assert float(np.max(values)) == pytest.approx(expected["max"], abs=1e-9)
