import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from scripts.compare_submissions import compare_submissions, greedy_match


def encode(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")


def write_submission(path, rows):
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(path, index=False)


def test_greedy_match_is_one_to_one_and_strictly_above_half():
    iou = np.array([[0.9, 0.8], [0.7, 0.5]])
    assert greedy_match(iou) == [(0, 0, 0.9)]


def test_compare_submissions_reports_matches_and_unique_instances(tmp_path):
    shape = (8, 8)
    shared = np.zeros(shape, dtype=bool)
    shared[1:4, 1:4] = True
    left_unique = np.zeros(shape, dtype=bool)
    left_unique[5:7, 1:3] = True
    right_unique = np.zeros(shape, dtype=bool)
    right_unique[5:7, 5:7] = True

    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    write_submission(
        left,
        [
            {"filament_id": "image_1", "segmentation_rle": encode(shared)},
            {"filament_id": "image_2", "segmentation_rle": encode(left_unique)},
        ],
    )
    write_submission(
        right,
        [
            {"filament_id": "image_1", "segmentation_rle": encode(shared)},
            {"filament_id": "image_2", "segmentation_rle": encode(right_unique)},
        ],
    )

    summary, matched, left_only, right_only, per_image = compare_submissions(left, right, shape)

    assert summary["left_rows"] == summary["right_rows"] == 2
    assert summary["matched_iou_gt_0_5"] == 1
    assert summary["left_only"] == summary["right_only"] == 1
    assert summary["images_equal"] == 1
    assert matched.iloc[0].iou == 1.0
    assert left_only.iloc[0].best_iou_right == 0.0
    assert right_only.iloc[0].best_iou_left == 0.0
    assert per_image.iloc[0].matched == 1
