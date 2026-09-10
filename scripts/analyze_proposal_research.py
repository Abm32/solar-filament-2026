"""CPU-only, tuning-only proposal coverage diagnosis; never exports predictions.

Support/crop coverage is an optimistic per-instance geometric opportunity, not
PQ or achievable recall. Different GT instances may use the same proposal.
"""

import ast
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research_2026-09-06"


def source_function(path, name, namespace):
    """Use the exact local pure helper without importing its GPU dependencies."""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def max_iou(proposals, gt):
    if not proposals:
        return np.zeros(len(gt))
    return mask_utils.iou(proposals, gt, [0] * len(gt)).max(axis=0)


def max_coverage(supports, gt):
    if not supports:
        return np.zeros(len(gt))
    # COCO crowd IoU divides intersection by detection area. GT is deliberately
    # the detection argument, so the result is intersection / GT area.
    return mask_utils.iou(gt, supports, [1] * len(supports)).max(axis=1)


def main():
    manifest = json.loads((ROOT / "outputs/kaggle_v10_quality/inference_manifest.json").read_text())
    names = manifest["tune_files"]
    assert len(names) == 94 and set(names).isdisjoint(manifest["audit_files"])
    annotation_path = next((ROOT / "data/raw").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    assert hashlib.sha256(annotation_path.read_bytes()).hexdigest() == manifest["annotation_sha256"]
    data = json.loads(annotation_path.read_text())
    infos, annotations = defaultdict(list), defaultdict(list)
    for info in data["images"]:
        infos[info["file_name"]].append(info)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)

    ns = {"np": np, "mask_utils": mask_utils,
          "POLICIES": ("single", "flip_max", "flip_consensus"),
          "CFG": SimpleNamespace(crop_context=1.8, min_crop=96)}
    cluster = source_function(ROOT / "kaggle_inference/detector_tta.py", "cluster_views", ns)
    bounds = source_function(ROOT / "kaggle_kernel/train_and_submit.py", "square_bounds", ns)
    cache_dirs = {
        "v6": ROOT / "outputs/kaggle_detector_tta/detector_proposal_cache",
        "v8": ROOT / "outputs/kaggle_v8_coco_tta/detector_proposal_cache/19c713e327e7bf39",
    }
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rows, cache_hashes = [], {}
    for number, name in enumerate(names, 1):
        remote_path = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images/" + name
        key = hashlib.sha256(remote_path.encode()).hexdigest()[:16] + ".json.gz"
        stages = {}
        for detector, directory in cache_dirs.items():
            path = directory / key
            cache_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            views = json.loads(gzip.decompress(path.read_bytes()))
            pairs = [(v["rles"], v["scores"]) for v in views]
            stages[detector + "_raw"] = [r for rs, _ in pairs for r in rs]
            grouped, scores = cluster(pairs, "flip_consensus")
            stages[detector + "_grouped"] = grouped
            stages[detector + "_accepted"] = [r for r, s in zip(grouped, scores) if s >= 0.4]
        supports, crops = [], []
        for rle in stages["v6_accepted"]:
            mask = mask_utils.decode(rle)
            x0, y0, x1, y1 = bounds(mask)
            support = np.zeros_like(mask)
            support[y0:y1, x0:x1] = cv2.dilate(mask[y0:y1, x0:x1], kernel)
            supports.append(mask_utils.encode(np.asfortranarray(support)))
            crop = np.zeros_like(mask)
            crop[y0:y1, x0:x1] = 1
            crops.append(mask_utils.encode(np.asfortranarray(crop)))

        for info in infos[name]:
            anns = annotations[info["id"]]
            gt = [mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], info["height"], info["width"])) for a in anns]
            if not gt:
                continue
            metrics = {stage + "_iou": max_iou(rles, gt) for stage, rles in stages.items()}
            metrics["v6_support_coverage"] = max_coverage(supports, gt)
            metrics["v6_crop_coverage"] = max_coverage(crops, gt)
            for index, (ann, rle) in enumerate(zip(anns, gt)):
                area = int(mask_utils.area(rle))
                assert area == ann["area"] and area > 0
                rows.append(dict(file_name=name, image_id=info["id"], gt_index=index,
                                 gt_area=area, category_id=ann["category_id"],
                                 **{k: float(v[index]) for k, v in metrics.items()}))
        if number % 20 == 0:
            print(f"Diagnosed {number}/{len(names)} tuning files", flush=True)

    frame = pd.DataFrame(rows)
    assert len(frame) == 1134
    raw6, raw8 = frame.v6_raw_iou > 0.5, frame.v8_raw_iou > 0.5
    summary = {
        "scope": "94 tuning files, separate annotator instances; no new audit evaluation",
        "warning": "Oracle geometric coverage, not PQ, one-to-one recall, or a model prediction",
        "n_files": len(names), "n_gt": len(frame),
        "raw_union": {"both": int((raw6 & raw8).sum()), "v6_only": int((raw6 & ~raw8).sum()),
                      "v8_only": int((~raw6 & raw8).sum()), "neither": int((~raw6 & ~raw8).sum())},
        "stage_gt_above_05": {k: int((frame[k] > 0.5).sum()) for k in metrics},
        "support_gt_at_least_09": int((frame.v6_support_coverage >= 0.9).sum()),
        "crop_gt_at_least_09": int((frame.v6_crop_coverage >= 0.9).sum()),
        "raw_match_lost_by_grouping": int(((frame.v6_raw_iou > 0.5) & (frame.v6_grouped_iou <= 0.5)).sum()),
        "grouped_match_lost_by_cutoff": int(((frame.v6_grouped_iou > 0.5) & (frame.v6_accepted_iou <= 0.5)).sum()),
        "crop_only_gt_opportunities": int(((frame.v6_crop_coverage > 0.5) & (frame.v6_support_coverage <= 0.5)).sum()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "annotation_sha256": manifest["annotation_sha256"],
        "raw_cache_sha256": cache_hashes,
    }
    # Explicitly check the crowd-IoU interpretation on real masks, independently.
    sample_gt, sample_support = gt[0], supports[0]
    expected = float(mask_utils.area(mask_utils.merge([sample_gt, sample_support], intersect=True))) / float(mask_utils.area(sample_gt))
    actual = float(mask_utils.iou([sample_gt], [sample_support], [1])[0, 0])
    assert abs(expected - actual) < 1e-12
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "proposal_diagnostics.csv", index=False)
    (OUT / "proposal_diagnostics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "raw_cache_sha256"}, indent=2))


if __name__ == "__main__":
    main()
