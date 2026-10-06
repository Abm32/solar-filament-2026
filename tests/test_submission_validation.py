import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_utils

from scripts.validate_submission import validate_submission


def test_submission_validation_checks_overlap_and_unknown_images(tmp_path):
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))["counts"].decode("utf-8")
    path = tmp_path / "submission.csv"
    rows = [{"filament_id": "test_1", "segmentation_rle": rle}]
    pd.DataFrame(rows).to_csv(path, index=False)
    assert validate_submission(path, {"test"}, shape=mask.shape)["rows"] == 1
    with pytest.raises(ValueError, match="Unknown test"):
        validate_submission(path, {"other"}, shape=mask.shape)
    rows.append({"filament_id": "test_2", "segmentation_rle": rle})
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Overlapping"):
        validate_submission(path, {"test"}, shape=mask.shape)


def test_submission_validation_rejects_non_sequential_ids(tmp_path):
    first = np.zeros((16, 16), dtype=np.uint8)
    first[2:6, 2:6] = 1
    second = np.zeros((16, 16), dtype=np.uint8)
    second[10:14, 10:14] = 1
    rows = [
        {
            "filament_id": "test_1",
            "segmentation_rle": mask_utils.encode(np.asfortranarray(first))["counts"].decode(
                "utf-8"
            ),
        },
        {
            "filament_id": "test_3",
            "segmentation_rle": mask_utils.encode(np.asfortranarray(second))["counts"].decode(
                "utf-8"
            ),
        },
    ]
    path = tmp_path / "submission.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Non-sequential filament IDs"):
        validate_submission(path, {"test"}, shape=first.shape)
