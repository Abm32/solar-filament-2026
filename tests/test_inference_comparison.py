import numpy as np
import pandas as pd
import pytest
import json
import sys

from scripts.analyze_inference_sweep import CONTROL, compare_group, main


def test_paired_comparison_groups_annotators_by_image():
    rows = []
    for name, annotators in (("a", 2), ("b", 1)):
        for _ in range(annotators):
            for candidate, overlap in ((CONTROL, 0.6), ("improved", 0.8)):
                rows.append(dict(file_name=name, candidate=candidate, group="audit",
                                 iou_sum=overlap, tp=1, fp=0, fn=0))
    result = compare_group(pd.DataFrame(rows), "improved", "audit", samples=100)
    assert result["images"] == 2
    assert result["delta_pq"] == pytest.approx(0.2)
    np.testing.assert_allclose(result["delta_95pct_interval"], [0.2, 0.2])


def test_paired_comparison_rejects_missing_images():
    rows = pd.DataFrame([
        dict(file_name="a", candidate=CONTROL, group="audit", iou_sum=0.6, tp=1, fp=0, fn=0),
        dict(file_name="b", candidate="improved", group="audit", iou_sum=0.8, tp=1, fp=0, fn=0),
    ])
    with pytest.raises(ValueError, match="Incomplete paired evaluation"):
        compare_group(rows, "improved", "audit")


def test_external_baseline_does_not_require_old_control_in_new_run(tmp_path, monkeypatch):
    candidate_dir, baseline_dir = tmp_path / "candidate", tmp_path / "baseline"
    manifest = dict(smoke_limit=0, refiner_epochs=0, tune_files=["a"], audit_files=["b"],
                    target_rasterization="coco", audit_caveat="previous checkpoint selection")
    for path, name, policy, overlap in ((candidate_dir, "flip_new", "flip_max", 0.8),
                                        (baseline_dir, "old_baseline", "single", 0.7)):
        path.mkdir()
        (path / "inference_manifest.json").write_text(json.dumps(manifest))
        pd.DataFrame([dict(candidate=name, resolution=1024, mode="refined_constrained",
                           threshold=0.5, tta=True, confidence=0.4, detector_policy=policy,
                           tune_pq=overlap, audit_pq=overlap, all_pq=overlap,
                           legacy_pq=overlap)]).to_csv(path / "inference_candidates.csv", index=False)
        pd.DataFrame([dict(candidate=name, file_name=filename, group=group,
                           iou_sum=overlap, tp=1, fp=0, fn=0)
                      for filename, group in (("a", "tune"), ("b", "audit"))]).to_csv(
                          path / "inference_image_diagnostics.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["analysis", str(candidate_dir), "--baseline-dir", str(baseline_dir)])
    main()
    report = json.loads((candidate_dir / "inference_comparison.json").read_text())
    assert report["config"]["detector_policy"] == "flip_max"
    assert report["comparison"]["audit"]["delta_pq"] == pytest.approx(0.1)
