import json
from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_v27_sliced as evaluate
from scripts import train_v27_sliced as train

ROOT = Path(__file__).resolve().parents[1]


def real_population(monkeypatch):
    annotation = next((ROOT / "data/raw").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    provenance = ROOT / "outputs/v6_inference_source/v6_provenance.json"
    monkeypatch.setattr(train, "ANNOTATIONS", annotation)
    images, annotations = train.load_population()
    split = train.partitions(images, annotations, json.loads(provenance.read_text()))
    return images, annotations, split


def test_real_owner_population_is_pinned(monkeypatch):
    images, annotations, split = real_population(monkeypatch)
    positive = empty = retained = excluded = 0
    for image_id in split["train_ids"]:
        per_tile = {(x, y): 0 for y in train.TILE_STARTS for x in train.TILE_STARTS}
        for annotation in annotations[image_id]:
            x0, y0 = train.owner_origin(annotation)
            if train.owner_contains(annotation, x0, y0):
                per_tile[(x0, y0)] += 1; retained += 1
            else:
                excluded += 1
        positive += sum(count > 0 for count in per_tile.values())
        empty += sum(count == 0 for count in per_tile.values())
    assert len(split["train_ids"]) == 937
    assert positive == 4145 and empty == 4288
    assert retained == 6593 and excluded == 13


def test_matched_record_and_step_budget():
    assert train.RECORDS_PER_EPOCH == 5622
    assert train.CONTROL_REPETITIONS == 6
    assert train.EMPTY_TILES == 540
    assert train.EPOCHS == 2
    assert train.RECORDS_PER_EPOCH * train.EPOCHS == 11244
    args = train.training_arguments()
    assert args["imgsz"] == 1024 and args["batch"] == 1 and args["mask_ratio"] == 1
    assert args["warmup_epochs"] == 0.2


def test_owner_tile_policy_matches_v26_boundaries():
    annotation = {"bbox": [700, 700, 100, 100]}
    assert train.owner_origin(annotation) == (0, 0)
    annotation = {"bbox": [730, 730, 100, 100]}
    assert train.owner_origin(annotation) == (512, 512)
    assert train.owner_contains(annotation, 512, 512)


def test_polygon_translation_is_native_and_clipped():
    annotation = {"segmentation": [[512, 512, 612, 512, 612, 612, 512, 612]]}
    line = train.polygon_lines(annotation, 1024, 1024, (512, 512))[0]
    values = np.asarray([float(value) for value in line.split()[1:]])
    assert values.min() == 0
    assert values.max() == pytest.approx(100 / 1024, abs=5e-7)


def test_tiles_are_lossless_png_and_source_regression():
    source = (ROOT / "scripts/train_v27_sliced.py").read_text()
    assert 'tile_name = f"{Path(file_name).stem}_x{x0}_y{y0}.png"' in source
    assert 'images/train/{stem}.png' in source
    assert "cv2.IMWRITE_JPEG_QUALITY" not in source


def test_v27_runs_and_cache_roots_are_distinct(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "BASE", tmp_path)
    monkeypatch.setattr(evaluate, "OUTPUT", tmp_path / "evaluation")
    paths = {run: tmp_path / f"outputs/train/{run.split('-')[0]}/{run.split('-')[1]}/final.pt"
             for run in evaluate.RUNS}
    assert len(set(map(str, paths.values()))) == 4
    assert set(evaluate.RUNS) == {"control-seed1", "control-seed2", "sliced-seed1", "sliced-seed2"}


def test_frozen_v27_gates():
    assert evaluate.TUNE_REQUIRED == pytest.approx(evaluate.V17_TUNE + 0.01)
    assert evaluate.CONTROL_DELTA == 0.01
    assert evaluate.V26_DELTA == 0.02
    assert evaluate.RAW_MIN == 0.80
    assert evaluate.RESCUE_MIN == 70
    assert evaluate.FP_MAX == pytest.approx(367.2)


def test_no_test_export_or_submission_paths():
    for relative in ("scripts/train_v27_sliced.py", "scripts/evaluate_v27_sliced.py"):
        source = (ROOT / relative).read_text()
        assert "test/test_images" not in source
        assert "submission.csv" not in source
        assert "competitions_submit" not in source
        assert "tune gates did not authorize audit" in source if "evaluate_" in relative else True
