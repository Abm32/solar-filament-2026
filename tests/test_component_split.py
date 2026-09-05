import numpy as np
from pycocotools import mask as mask_utils

from scripts.analyze_component_split import CONTROL, component_variants


def test_split_preserves_positions_and_nonoverlap_and_removes_only_small_parts():
    mask = np.zeros((48, 60), dtype=np.uint8, order="F")
    mask[4:14, 3:13] = 1  # 100 pixels
    mask[25:30, 40:50] = 1  # 50 pixels
    mask[20, 30] = 1  # below area cutoff
    rle = mask_utils.encode(mask)
    variants = component_variants([rle])
    assert variants[CONTROL] == [rle]
    for policy in ("split_min20", "split_10pct", "split_25pct"):
        assert len(variants[policy]) == 2
        decoded = mask_utils.decode(variants[policy])
        assert decoded.sum(axis=2).max() == 1
        expected = mask.copy()
        expected[20, 30] = 0
        np.testing.assert_array_equal(decoded.sum(axis=2), expected)


def test_relative_split_requires_two_substantial_components():
    mask = np.zeros((48, 60), dtype=np.uint8, order="F")
    mask[4:24, 3:23] = 1  # 400 pixels
    mask[30:35, 40:45] = 1  # 25 pixels
    rle = mask_utils.encode(mask)
    variants = component_variants([rle])
    assert len(variants["split_min20"]) == 2
    assert variants["split_10pct"] == [rle]
    assert variants["split_25pct"] == [rle]
    assert component_variants([])[CONTROL] == []
