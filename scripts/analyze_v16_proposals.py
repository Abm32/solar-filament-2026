"""Offline re-analysis of V16's returned proposal caches. No GPU, no torch.

V16 evaluated each arm at a SINGLE frozen operating point (box iou=0.7, t0.40,
c0.40 -- E015's settings) and then judged the result against V13's 0.441777,
which was obtained with a different pipeline (box iou=0.3, V11 refiner, best of
a 42-candidate threshold x confidence sweep). This script measures what the
caches can actually say about the detector alone, on equal footing across arms:

  * raw proposal recall at IoU > 0.5 and mean matched IoU (ceiling_analysis §4)
  * the oracle PQ implied by each arm's raw proposals
  * the same numbers after flip_consensus grouping, before any confidence gate

Everything here is detector-side, so it is unaffected by which refiner or
threshold a downstream run would pick.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

INFERENCE_RESOLUTION = 1024
MATCH_IOU = 0.5
V6_DETECTOR_SHA256 = "0abf568f5ffa504979a44c6ac4997c231adc38cb3c6d998461295c4dad3a38b8"


def cache_id(name, split, mask_ratio, detector_sha):
    key = f"{detector_sha}|mr{mask_ratio}|{INFERENCE_RESOLUTION}|{split}|{name}"
    return hashlib.sha256(key.encode()).hexdigest()[:16] + ".json.gz"


def read_views(path):
    with gzip.open(path, "rt") as stream:
        payload = json.load(stream)
    views = []
    for view in payload:
        rles = [{"size": tuple(r["size"]), "counts": r["counts"].encode()} for r in view["rles"]]
        views.append((rles, view["scores"]))
    return views


def annotation_rle(info, entry):
    """Canonical rasterisation: whatever COCO form the entry carries."""
    segmentation = entry["segmentation"]
    height, width = info["height"], info["width"]
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        return mask_utils.merge(rles)
    if isinstance(segmentation["counts"], list):
        return mask_utils.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode()
    return rle


def best_annotator_ids(images, annotations):
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)

    def completeness(image_id):
        entries = annotations.get(image_id, [])
        return len(entries), sum(entry.get("area", 0) for entry in entries)

    return {name: max(ids, key=completeness) for name, ids in by_file.items()}


def greedy_match(iou):
    """One-to-one greedy matching at IoU > 0.5, highest IoU first (PQ semantics)."""
    if iou.size == 0:
        return []
    pairs = []
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    used_gt, used_pred = set(), set()
    for gt_index, pred_index in order:
        value = iou[gt_index, pred_index]
        if value <= MATCH_IOU:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(int(gt_index))
        used_pred.add(int(pred_index))
        pairs.append((int(gt_index), int(pred_index), float(value)))
    return pairs


def cluster_views(views, match_iou=MATCH_IOU):
    """flip_consensus, reimplemented to match detector_tta.cluster_views exactly.

    Three details matter and were wrong in a naive version: proposals are visited
    in DESCENDING confidence (fixed highest-confidence anchors, so a broad
    intermediate mask cannot chain two neighbours); a group may absorb at most one
    proposal per view (`view_id not in group["views"]`); and the consensus score
    divides by the number of views, so missing views vote zero. Matching uses the
    best eligible overlap with `>=`, not the first over threshold.
    """
    proposals = []
    for view_id, (rles, scores) in enumerate(views):
        if len(rles) != len(scores):
            raise ValueError("Every proposal needs a score")
        proposals.extend((rle, float(score), view_id) for rle, score in zip(rles, scores))
    proposals.sort(key=lambda item: -item[1])
    groups = []
    for rle, score, view_id in proposals:
        eligible = [i for i, group in enumerate(groups) if view_id not in group["views"]]
        match = None
        if eligible:
            overlaps = mask_utils.iou([rle], [groups[i]["rle"] for i in eligible],
                                      [0] * len(eligible))[0]
            best = int(np.argmax(overlaps))
            if overlaps[best] >= match_iou:
                match = eligible[best]
        if match is None:
            groups.append(dict(rle=rle, scores=[score], views={view_id}))
        else:
            groups[match]["scores"].append(score)
            groups[match]["views"].add(view_id)
    scores = [sum(group["scores"]) / len(views) for group in groups]
    order = sorted(range(len(groups)), key=lambda i: -scores[i])
    return [groups[i]["rle"] for i in order], [scores[i] for i in order]


CONFIDENCES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)


def summarise(rows, label):
    n_gt = sum(row["n_gt"] for row in rows)
    matched = sum(row["matched"] for row in rows)
    iou_sum = sum(row["iou_sum"] for row in rows)
    n_pred = sum(row["n_pred"] for row in rows)
    recall = matched / n_gt if n_gt else 0.0
    mean_iou = iou_sum / matched if matched else 0.0
    # Oracle PQ: keep only matched predictions, so FP = 0.
    oracle_pq = iou_sum / (matched + 0.5 * (n_gt - matched)) if n_gt else 0.0
    # Keep-all PQ: every proposal kept, unmatched ones become FP.
    keep_all_pq = iou_sum / (matched + 0.5 * ((n_pred - matched) + (n_gt - matched))) if n_gt else 0.0
    return dict(label=label, n_gt=n_gt, matched=matched, recall=recall,
                mean_matched_iou=mean_iou, n_pred=n_pred,
                oracle_pq=oracle_pq, keep_all_pq=keep_all_pq)


def analyse(root, manifest_path, cache_root, output):
    manifest = json.loads(Path(manifest_path).read_text())
    arms = {"v16_frozen_control": (2, V6_DETECTOR_SHA256)}
    for arm, training in manifest["training"].items():
        arms[f"v16_{arm}"] = (training["mask_ratio"], training["checkpoint_sha256"])

    data_root = Path(root) / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    coco = json.loads((data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
    images = {image["id"]: image for image in coco["images"]}
    annotations = {}
    for entry in coco["annotations"]:
        annotations.setdefault(entry["image_id"], []).append(entry)
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)

    tune = set(manifest["tune_files"])
    audit = set(manifest["audit_files"])
    names = sorted(tune | audit)
    if len(names) != 141:
        raise ValueError(f"Expected 141 validation files, got {len(names)}")

    results = {}
    gates = {}
    for candidate, (mask_ratio, detector_sha) in arms.items():
        raw_rows, grouped_rows = [], []
        gate_rows = {c: [] for c in CONFIDENCES}
        for name in names:
            path = Path(cache_root) / cache_id(name, "train", mask_ratio, detector_sha)
            if not path.exists():
                raise FileNotFoundError(f"Missing cache for {candidate}/{name}: {path}")
            views = read_views(path)
            raw_rles = views[0][0] if views else []
            grouped_rles, grouped_scores = cluster_views(views)
            for image_id in sorted(by_file[name]):
                info = images[image_id]
                gt = [annotation_rle(info, entry) for entry in annotations.get(image_id, [])]
                gt = [rle for rle in gt if mask_utils.area(rle) > 0]
                group = "tune" if name in tune else "audit"
                for rles, rows in ((raw_rles, raw_rows), (grouped_rles, grouped_rows)):
                    iou = (mask_utils.iou(rles, gt, [0] * len(gt)).T if rles and gt
                           else np.zeros((len(gt), len(rles))))
                    pairs = greedy_match(np.asarray(iou, dtype=np.float64))
                    rows.append(dict(candidate=candidate, file_name=name, group=group,
                                     n_gt=len(gt), n_pred=len(rles), matched=len(pairs),
                                     iou_sum=sum(value for _, _, value in pairs)))
                # Confidence sweep on the GROUPED population: this is the axis
                # V16 froze at 0.40 and the axis V13 was free to tune.
                for confidence in CONFIDENCES:
                    keep = [rle for rle, score in zip(grouped_rles, grouped_scores)
                            if score >= confidence]
                    iou = (mask_utils.iou(keep, gt, [0] * len(gt)).T if keep and gt
                           else np.zeros((len(gt), len(keep))))
                    pairs = greedy_match(np.asarray(iou, dtype=np.float64))
                    gate_rows[confidence].append(
                        dict(candidate=candidate, file_name=name, group=group,
                             n_gt=len(gt), n_pred=len(keep), matched=len(pairs),
                             iou_sum=sum(value for _, _, value in pairs)))
        results[candidate] = {}
        for stage, rows in (("raw", raw_rows), ("grouped", grouped_rows)):
            for group in ("tune", "audit", "all"):
                subset = [r for r in rows if group == "all" or r["group"] == group]
                results[candidate][f"{stage}_{group}"] = summarise(subset, f"{candidate}/{stage}/{group}")
        gates[candidate] = {}
        for confidence, rows in gate_rows.items():
            for group in ("tune", "audit"):
                subset = [r for r in rows if r["group"] == group]
                gates[candidate][f"c{confidence:.2f}_{group}"] = summarise(
                    subset, f"{candidate}/c{confidence:.2f}/{group}")
        print(f"done {candidate}", flush=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(stages=results, confidence_sweep=gates), indent=2) + "\n")
    return results, gates


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=root / "outputs/kaggle_v16_full/v16_manifest.json")
    parser.add_argument("--cache-root", type=Path,
                        default=root / "outputs/kaggle_v16_full/v16_proposal_cache")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/v16_proposal_analysis/summary.json")
    args = parser.parse_args()
    results, gates = analyse(root, args.manifest, args.cache_root, args.output)
    header = f"{'candidate':22} {'stage':8} {'recall':>8} {'meanIoU':>8} {'oraclePQ':>9} {'keepAll':>8} {'nPred':>6}"
    print("\n" + header)
    print("-" * len(header))
    for candidate, entries in results.items():
        for stage in ("raw", "grouped"):
            row = entries[f"{stage}_tune"]
            print(f"{candidate:22} {stage:8} {row['recall']:8.4f} {row['mean_matched_iou']:8.4f} "
                  f"{row['oracle_pq']:9.4f} {row['keep_all_pq']:8.4f} {row['n_pred']:6d}")

    print("\nGrouped-proposal confidence sweep (tune group, pre-refiner):")
    header = f"{'conf':>6} " + " ".join(f"{c.replace('v16_',''):>26}" for c in gates)
    print(header)
    print("-" * len(header))
    for confidence in CONFIDENCES:
        cells = []
        for candidate in gates:
            row = gates[candidate][f"c{confidence:.2f}_tune"]
            cells.append(f"PQ{row['keep_all_pq']:.4f} r{row['recall']:.3f} n{row['n_pred']:5d}")
        print(f"{confidence:6.2f} " + " ".join(f"{cell:>26}" for cell in cells))


if __name__ == "__main__":
    main()
