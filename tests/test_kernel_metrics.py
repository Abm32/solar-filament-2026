"""Exercise kernel helpers without triggering Kaggle's installation or training."""

import ast
from pathlib import Path
import unittest

import numpy as np
from pycocotools import mask as mask_utils


source = Path(__file__).resolve().parents[1] / "kaggle_kernel" / "train_and_submit.py"
tree = ast.parse(source.read_text())
names = {"pairwise_iou", "pq_for_image", "all_validation_annotator_ids", "annotation_mask"}
helpers = ast.Module(
    body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
    type_ignores=[],
)
namespace = {"np": np, "mask_utils": mask_utils}
exec(compile(helpers, str(source), "exec"), namespace)


class KernelMetricsTest(unittest.TestCase):
    def test_polygon_rasterization_preserves_coco_area_and_multiple_components(self):
        annotation = {"segmentation": [[2, 2, 12, 2, 12, 7, 2, 7],
                                        [16, 16, 20, 16, 20, 20, 16, 20]]}
        result = namespace["annotation_mask"]((32, 32), annotation)
        self.assertEqual(int(result.sum()), 66)
        self.assertEqual(result[7, 12], 0)
        self.assertEqual(namespace["annotation_mask"]((32, 32), {"segmentation": []}).sum(), 0)

    def test_rle_iou_matches_boolean_reference(self):
        rng = np.random.default_rng(42)
        gt = [rng.random((48, 64)) > 0.4 for _ in range(3)]
        pred = [gt[0].copy()] + [rng.random((48, 64)) > 0.6 for _ in range(3)]
        expected = np.array([
            [np.logical_and(g, p).sum() / np.logical_or(g, p).sum() for p in pred]
            for g in gt
        ])
        np.testing.assert_allclose(namespace["pairwise_iou"](gt, pred), expected)

    def test_empty_inputs_and_large_perfect_intersection(self):
        mask = np.ones((64, 64), dtype=np.uint8)
        self.assertEqual(namespace["pq_for_image"]([mask], [mask]), (1.0, 1, 0, 0))
        self.assertEqual(namespace["pq_for_image"]([], [mask]), (0.0, 0, 1, 0))
        self.assertEqual(namespace["pq_for_image"]([mask], []), (0.0, 0, 0, 1))
        self.assertEqual(namespace["pq_for_image"]([], []), (0.0, 0, 0, 0))

    def test_iou_threshold_is_strict(self):
        gt = np.ones((8, 8), dtype=np.uint8)
        pred = gt.copy()
        pred[:4] = 0
        self.assertEqual(namespace["pq_for_image"]([gt], [pred]), (0.0, 0, 1, 1))

    def test_validation_expands_only_held_out_files(self):
        images = {
            "a": {"file_name": "train.jpeg"},
            "b": {"file_name": "valid.jpeg"},
            "c": {"file_name": "valid.jpeg"},
            "d": {"file_name": "train.jpeg"},
        }
        self.assertEqual(namespace["all_validation_annotator_ids"](["b"], images), ["b", "c"])


if __name__ == "__main__":
    unittest.main()
