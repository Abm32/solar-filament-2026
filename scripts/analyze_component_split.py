"""Evaluate disconnected-component splitting from a complete validation cache.

Run as: python -m scripts.analyze_component_split CACHE_DIRECTORY
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from scripts.analyze_inference_sweep import compare_group, pq

CONTROL = "accepted_coco_refiner"
POLICIES = {"split_min20": 0.0, "split_10pct": 0.1, "split_25pct": 0.25}


def component_variants(rles):
    result = {CONTROL: rles, **{policy: [] for policy in POLICIES}}
    for rle in rles:
        mask = mask_utils.decode(rle)
        x, y, width, height = mask_utils.toBbox(rle).astype(int)
        if not width or not height:
            raise ValueError("Empty predictions are invalid")
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.ascontiguousarray(mask[y:y + height, x:x + width]), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        area = int(areas.sum())
        encoded = {}
        for policy, fraction in POLICIES.items():
            keep = np.flatnonzero(areas >= max(20, fraction * area)) + 1
            if len(keep) < 2:
                result[policy].append(rle)
                continue
            for index in keep:
                if index not in encoded:
                    full = np.zeros(mask.shape, dtype=np.uint8, order="F")
                    full[y:y + height, x:x + width] = labels == index
                    encoded[index] = mask_utils.encode(full)
                result[policy].append(encoded[index])
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--baseline", type=Path, default=root / "outputs/kaggle_coco_refiner")
    args = parser.parse_args()
    manifest = json.loads((args.baseline / "inference_manifest.json").read_text())
    selected = json.loads((args.baseline / "submission_candidates.json").read_text())[0]
    annotation_path = next((root / "data/raw").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if hashlib.sha256(annotation_path.read_bytes()).hexdigest() != manifest["source"]["annotation_sha256"]:
        raise ValueError("Annotation source differs from baseline")
    coco = json.loads(annotation_path.read_text())
    images, annotations = defaultdict(list), defaultdict(list)
    for info in coco["images"]:
        images[info["file_name"]].append(info)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    expected = set(manifest["tune_files"]) | set(manifest["audit_files"])
    if len(list(args.cache_dir.glob("*.json.gz"))) != len(expected):
        raise ValueError("A complete validation cache is required")
    cv2.setNumThreads(2)
    rows, seen = [], set()
    for path in sorted(args.cache_dir.glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            entry = json.load(stream)
        name = entry["file_name"]
        if name in seen or name not in expected or entry["resolution"] != selected["resolution"]:
            raise ValueError(f"Duplicate, unexpected, or wrong-resolution cache: {name}")
        seen.add(name)
        variant = next(v for v in entry["variants"] if all(
            v[key] == selected[key] for key in ("mode", "threshold", "tta")))
        rles = [dict(size=r["size"], counts=r["counts"].encode())
                for r, score in zip(variant["rles"], variant["scores"])
                if score >= selected["confidence"]]
        variants = component_variants(rles)
        for info in images[name]:
            gt = [mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], info["height"], info["width"]))
                  for a in annotations[info["id"]]]
            gt = [r for r in gt if mask_utils.area(r) > 0]
            for policy, predictions in variants.items():
                iou = (mask_utils.iou(predictions, gt, [0] * len(gt)).T if predictions and gt
                       else np.zeros((len(gt), len(predictions))))
                hits = iou > 0.5
                rows.append(dict(candidate=policy, file_name=name, image_id=info["id"],
                                 group="tune" if name in manifest["tune_files"] else "audit",
                                 iou_sum=float(iou[hits].sum()), tp=int(hits.sum()),
                                 fp=int((~hits.any(axis=0)).sum()), fn=int((~hits.any(axis=1)).sum())))
        if len(seen) % 20 == 0:
            print(f"Component analysis {len(seen)}/{len(expected)}", flush=True)
    if seen != expected:
        raise ValueError("Validation cache is incomplete")
    diagnostics = pd.DataFrame(rows)
    baseline = pd.read_csv(args.baseline / "inference_image_diagnostics.csv")
    baseline = baseline[(baseline.candidate == selected["candidate"]) & baseline.group.isin(["tune", "audit"])]
    fields = ["file_name", "image_id", "group", "iou_sum", "tp", "fp", "fn"]
    a = diagnostics[diagnostics.candidate == CONTROL][fields].sort_values(fields[:3]).reset_index(drop=True)
    b = baseline[fields].sort_values(fields[:3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-10)
    summaries = []
    for policy in (CONTROL, *POLICIES):
        subset = diagnostics[diagnostics.candidate == policy]
        row = dict(candidate=policy)
        for group in ("tune", "audit", "all"):
            frame = subset if group == "all" else subset[subset.group == group]
            row[f"{group}_pq"] = float(pq(frame[["iou_sum", "tp", "fp", "fn"]].sum().to_numpy(float)))
        summaries.append(row)
    candidates = pd.DataFrame(summaries).sort_values(["tune_pq", "candidate"], ascending=[False, True])
    selected_policy = candidates.iloc[0].candidate
    report = dict(selected_on="tune_pq only", candidate=selected_policy,
                  comparison={g: compare_group(diagnostics, selected_policy, g, control_candidate=CONTROL)
                              for g in ("tune", "audit", "all")},
                  baseline_submission=56025722, baseline_control_verified=True,
                  audit_caveat=manifest["audit_caveat"])
    candidates.to_csv(args.cache_dir.parent / "component_candidates.csv", index=False)
    diagnostics.to_csv(args.cache_dir.parent / "component_image_diagnostics.csv", index=False)
    (args.cache_dir.parent / "component_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(candidates.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
