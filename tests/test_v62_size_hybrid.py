from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_v62_size_hybrid as H  # noqa: E402


def _case():
    rng = np.random.default_rng(0)
    masks, bounds, probs = [], [], []
    for x0, y0, side in ((10, 10, 64), (40, 30, 64), (150, 150, 32)):
        m = np.zeros((256, 256), np.uint8)
        m[y0 + 8:y0 + side - 8, x0 + 8:x0 + side - 8] = 1
        masks.append(m)
        bounds.append((x0, y0, x0 + side, y0 + side))
        probs.append(rng.random((32, 32)).astype(np.float32))
    return masks, [0.9, 0.8, 0.5], bounds, probs


def test_no_direct_matches_core_refined_constrained():
    masks, scores, bounds, probs = _case()
    ours = H.clean_hybrid(masks, scores, bounds, probs, 0.45, [False] * 3)
    core = H.core.clean_refined_predictions(masks, scores, bounds, probs, 0.45,
                                            "refined_constrained")
    assert ours[0] == core[0]
    assert np.array_equal(ours[1], core[1])


def test_direct_keeps_proposal_mask():
    from pycocotools import mask as mask_utils

    masks, scores, bounds, probs = _case()
    rles, _ = H.clean_hybrid(masks, scores, bounds, probs, 0.45, [True] * 3)
    assert np.array_equal(mask_utils.decode(rles[0]), masks[0])  # highest score, no overlap loss
    assert np.array_equal(mask_utils.decode(rles[2]), masks[2])  # disjoint from the others


def test_hybrid_flags_by_area():
    small = np.zeros((64, 64), np.uint8)
    small[:10, :10] = 1
    big = np.ones((64, 64), np.uint8)
    assert H.direct_flags([small, big], "hybrid") == [True, False]
    assert H.direct_flags([small, big], "refined") == [False, False]
    assert H.direct_flags([small, big], "unrefined") == [True, True]
