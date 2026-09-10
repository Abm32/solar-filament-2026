"""Is V16 arm_r1's recall gain redundant with V13's box `iou=0.3` lever?

V16 measured `mask_ratio=1` against a control that used the Ultralytics DEFAULT box
`iou=0.7`, while V13 -- the 0.441777 candidate V16 was gated against -- uses box
`iou=0.3`. Both levers raise recall, so the open question before spending GPU on a
V17 is whether they are additive or the same gain twice.

V13's real `iou=0.3` proposal caches are on disk (`outputs/kaggle_v13_full/
proposal_cache`, signed, 321 files). This compares, at the grouped pre-refiner
stage on the same 141 validation files and the same GT:

    A. frozen V6 detector @ box iou=0.7   (V16's control)
    B. frozen V6 detector @ box iou=0.3   (V13's parent A -- the shipped winner)
    C. arm_r1 mask_ratio=1 detector @ box iou=0.7  (V16's winning arm)

If C > B, `mask_ratio=1` beats the lever already in the shipped pipeline and a V17
is worth running. If C <= B, arm_r1's advantage over A is just recovering what
`iou=0.3` already provides, and a V17 that stacks them buys much less than the
naive arm-vs-control delta suggests. Zero GPU: reads caches only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_v16_proposals import (  # noqa: E402
    CONFIDENCES, V6_DETECTOR_SHA256, annotation_rle, cache_id, cluster_views,
    greedy_match, read_views, summarise,
)

V13_CELL = "ea_iou030"


def v13_cache_id(name, split, cell_name=V13_CELL, resolution=1024, floor=0.1):
    """Reproduces v13_compose.cache_id exactly."""
    import hashlib
    token = f"{cell_name}|{split}|{name}|imgsz{resolution}|conf{floor}"
    return hashlib.sha256(token.encode()).hexdigest()[:16] + ".json.gz"


def load_gt(root):
    data_root = Path(root) / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    coco = json.loads((data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
    images = {image["id"]: image for image in coco["images"]}
    annotations = {}
    for entry in coco["annotations"]:
        annotations.setdefault(entry["image_id"], []).append(entry)
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)
    return images, annotations, by_file


def measure(view_paths, images, annotations, by_file, tune, label):
    """Grouped pre-refiner stage plus the confidence sweep, for one proposal source."""
    grouped_rows = []
    gate_rows = {c: [] for c in CONFIDENCES}
    for name, path in sorted(view_paths.items()):
        views = read_views(path)
        rles, scores = cluster_views(views)
        for image_id in sorted(by_file[name]):
            info = images[image_id]
            gt = [annotation_rle(info, entry) for entry in annotations.get(image_id, [])]
            gt = [rle for rle in gt if mask_utils.area(rle) > 0]
            group = "tune" if name in tune else "audit"
            iou = (mask_utils.iou(rles, gt, [0] * len(gt)).T if rles and gt
                   else np.zeros((len(gt), len(rles))))
            pairs = greedy_match(np.asarray(iou, dtype=np.float64))
            grouped_rows.append(dict(group=group, n_gt=len(gt), n_pred=len(rles),
                                     matched=len(pairs),
                                     iou_sum=sum(v for _, _, v in pairs)))
            for confidence in CONFIDENCES:
                keep = [r for r, s in zip(rles, scores) if s >= confidence]
                iou = (mask_utils.iou(keep, gt, [0] * len(gt)).T if keep and gt
                       else np.zeros((len(gt), len(keep))))
                pairs = greedy_match(np.asarray(iou, dtype=np.float64))
                gate_rows[confidence].append(
                    dict(group=group, n_gt=len(gt), n_pred=len(keep), matched=len(pairs),
                         iou_sum=sum(v for _, _, v in pairs)))
    out = {"grouped_tune": summarise([r for r in grouped_rows if r["group"] == "tune"],
                                    f"{label}/grouped/tune"),
           "grouped_audit": summarise([r for r in grouped_rows if r["group"] == "audit"],
                                      f"{label}/grouped/audit")}
    for confidence, rows in gate_rows.items():
        for group in ("tune", "audit"):
            out[f"c{confidence:.2f}_{group}"] = summarise(
                [r for r in rows if r["group"] == group], f"{label}/c{confidence:.2f}/{group}")
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16-manifest", type=Path,
                        default=root / "outputs/kaggle_v16_full/v16_manifest.json")
    parser.add_argument("--v16-cache", type=Path,
                        default=root / "outputs/kaggle_v16_full/v16_proposal_cache")
    parser.add_argument("--v13-cache", type=Path,
                        default=root / "outputs/kaggle_v13_full/proposal_cache")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/v16_proposal_analysis/lever_overlap.json")
    args = parser.parse_args()

    manifest = json.loads(args.v16_manifest.read_text())
    tune = set(manifest["tune_files"])
    names = sorted(tune | set(manifest["audit_files"]))
    arm_r1_sha = manifest["training"]["arm_r1"]["checkpoint_sha256"]

    images, annotations, by_file = load_gt(root)

    sources = {
        "A_iou070_mr2_control": {n: args.v16_cache / cache_id(n, "train", 2, V6_DETECTOR_SHA256)
                                 for n in names},
        "B_iou030_mr2_v13": {n: args.v13_cache / v13_cache_id(n, "train") for n in names},
        "C_iou070_mr1_arm_r1": {n: args.v16_cache / cache_id(n, "train", 1, arm_r1_sha)
                                for n in names},
    }
    results = {}
    for label, paths in sources.items():
        missing = [n for n, p in paths.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{label}: missing {len(missing)} caches, e.g. {missing[:2]}")
        results[label] = measure(paths, images, annotations, by_file, tune, label)
        print(f"done {label}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    header = f"{'source':24} {'recall':>8} {'meanIoU':>8} {'oraclePQ':>9} {'nPred':>7}"
    print("\nGrouped pre-refiner, tune group (all 3 sources, same GT and files):")
    print(header)
    print("-" * len(header))
    for label, entries in results.items():
        row = entries["grouped_tune"]
        print(f"{label:24} {row['recall']:8.4f} {row['mean_matched_iou']:8.4f} "
              f"{row['oracle_pq']:9.4f} {row['n_pred']:7d}")

    print("\nRecall by confidence (tune):")
    head = f"{'conf':>6} " + " ".join(f"{l.split('_')[0]+'_'+l.split('_')[1]:>22}" for l in results)
    print(head)
    print("-" * len(head))
    for confidence in CONFIDENCES:
        cells = [f"r{results[l][f'c{confidence:.2f}_tune']['recall']:.3f} "
                 f"PQ{results[l][f'c{confidence:.2f}_tune']['keep_all_pq']:.4f}"
                 for l in results]
        print(f"{confidence:6.2f} " + " ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
