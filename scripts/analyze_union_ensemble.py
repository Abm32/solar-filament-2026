"""Would a proposal-union detector ensemble actually beat one detector? (zero GPU)

`analyze_detector_complementarity.py` established that our six proposal sources miss
*different* instances: union recall 0.8131 vs 0.7875 for the best single source, with
212 of 1134 tune instances (18.7%) missed by every source. That is an ORACLE result --
it assumes we could keep the right proposal and pay no penalty for the wrong ones.

A real ensemble cannot do that. Merging six sources multiplies proposals, and PQ
penalises false positives at 0.5 each. This script simulates the honest version:

  1. cluster each source's flip views (as the shipped pipeline does)
  2. concatenate all sources' proposals into one pool
  3. cross-source NMS: sort by score, greedily keep a proposal only if its IoU with
     every already-kept proposal is below --nms-iou (default 0.5)
  4. apply the confidence gate, and measure recall / meanIoU / keep-all PQ

and compares that, at every confidence on the shipped grid, against the best single
source measured the same way. The question it answers is narrow and decisive:

    does the ensemble's extra recall survive the false positives it brings with it?

If yes, a V19 inference-only ensemble run is justified (no training: every checkpoint
and cache already exists). If no, the complementarity headroom is unreachable by
proposal union and the honest conclusion is that this pipeline is finished.

Usage:
    .venv/bin/python scripts/analyze_union_ensemble.py
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
    annotation_rle, cluster_views, greedy_match, read_views,
)
from analyze_lever_overlap import load_gt  # noqa: E402
from analyze_detector_complementarity import build_sources  # noqa: E402

MATCH_IOU = 0.5
CONFIDENCES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)


def cross_source_nms(rles, scores, nms_iou):
    """Greedy NMS across the concatenated multi-source proposal pool."""
    if not rles:
        return [], []
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    kept_rles, kept_scores = [], []
    for index in order:
        candidate = rles[index]
        if kept_rles:
            overlap = np.asarray(
                mask_utils.iou([candidate], kept_rles, [0] * len(kept_rles)),
                dtype=np.float64).ravel()
            if overlap.size and overlap.max() > nms_iou:
                continue
        kept_rles.append(candidate)
        kept_scores.append(float(scores[index]))
    return kept_rles, kept_scores


def score_pool(rles, scores, gt, confidence):
    """Recall / meanIoU / keep-all PQ for one proposal pool at one confidence."""
    keep = [r for r, s in zip(rles, scores) if s >= confidence]
    if keep and gt:
        iou = np.asarray(mask_utils.iou(keep, gt, [0] * len(gt)).T, dtype=np.float64)
        pairs = greedy_match(iou)
    else:
        pairs = []
    matched = len(pairs)
    iou_sum = sum(v for _, _, v in pairs)
    return {"n_gt": len(gt), "n_pred": len(keep), "matched": matched, "iou_sum": iou_sum}


def aggregate(rows, label):
    n_gt = sum(r["n_gt"] for r in rows)
    n_pred = sum(r["n_pred"] for r in rows)
    matched = sum(r["matched"] for r in rows)
    iou_sum = sum(r["iou_sum"] for r in rows)
    fp = n_pred - matched
    fn = n_gt - matched
    denom = matched + 0.5 * fp + 0.5 * fn
    return {"label": label, "n_gt": n_gt, "n_pred": n_pred, "matched": matched,
            "recall": matched / n_gt if n_gt else 0.0,
            "mean_matched_iou": iou_sum / matched if matched else 0.0,
            "fp": fp, "fn": fn,
            "keep_all_pq": iou_sum / denom if denom else 0.0}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16-manifest", type=Path,
                        default=root / "outputs/kaggle_v16_full/v16_manifest.json")
    parser.add_argument("--v16-cache", type=Path,
                        default=root / "outputs/kaggle_v16_full/v16_proposal_cache")
    parser.add_argument("--v13-cache", type=Path,
                        default=root / "outputs/kaggle_v13_cache_pull/proposal_cache")
    parser.add_argument("--v17-cache", type=Path,
                        default=root / "outputs/kaggle_v17_cache_pull/proposal_cache")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/v16_proposal_analysis/union_ensemble.json")
    args = parser.parse_args()

    manifest = json.loads(args.v16_manifest.read_text())
    tune = set(manifest["tune_files"])
    names = sorted(tune | set(manifest["audit_files"]))
    images, annotations, by_file = load_gt(root)
    sources = build_sources(root, args.v16_cache, args.v13_cache, args.v17_cache, names)
    labels = list(sources)

    for label, paths in sources.items():
        missing = [n for n, p in paths.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{label}: missing {len(missing)} caches")

    # rows[group][variant][confidence] -> list of per-image dicts
    rows = {g: {v: {c: [] for c in CONFIDENCES} for v in list(labels) + ["UNION_all", "UNION_pair"]}
            for g in ("tune", "audit")}
    # the best pair from the complementarity analysis
    PAIR = ["V6_iou070", "MR1s2_iou070"]

    for position, name in enumerate(sorted(names), start=1):
        per_source = {}
        for label in labels:
            rles, scores = cluster_views(read_views(sources[label][name]))
            per_source[label] = (rles, scores)

        pool_rles = [r for label in labels for r in per_source[label][0]]
        pool_scores = [s for label in labels for s in per_source[label][1]]
        union_rles, union_scores = cross_source_nms(pool_rles, pool_scores, args.nms_iou)

        pair_rles = [r for label in PAIR for r in per_source[label][0]]
        pair_scores = [s for label in PAIR for s in per_source[label][1]]
        pair_nms_rles, pair_nms_scores = cross_source_nms(pair_rles, pair_scores, args.nms_iou)

        for image_id in sorted(by_file[name]):
            info = images[image_id]
            gt = [annotation_rle(info, e) for e in annotations.get(image_id, [])]
            gt = [rle for rle in gt if mask_utils.area(rle) > 0]
            if not gt:
                continue
            group = "tune" if name in tune else "audit"
            for confidence in CONFIDENCES:
                for label in labels:
                    r, s = per_source[label]
                    rows[group][label][confidence].append(score_pool(r, s, gt, confidence))
                rows[group]["UNION_all"][confidence].append(
                    score_pool(union_rles, union_scores, gt, confidence))
                rows[group]["UNION_pair"][confidence].append(
                    score_pool(pair_nms_rles, pair_nms_scores, gt, confidence))
        if position % 40 == 0:
            print(f"  processed {position}/{len(names)} files", flush=True)

    report = {"nms_iou": args.nms_iou, "confidences": list(CONFIDENCES),
              "sources": labels, "pair": PAIR, "groups": {}}
    for group in ("tune", "audit"):
        report["groups"][group] = {}
        for variant, per_conf in rows[group].items():
            report["groups"][group][variant] = {
                f"c{c:.2f}": aggregate(per_conf[c], f"{variant}/c{c:.2f}/{group}")
                for c in CONFIDENCES}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    for group in ("tune", "audit"):
        print(f"\n{'=' * 96}")
        print(f"{group.upper()} group -- pre-refiner, cross-source NMS at IoU {args.nms_iou}")
        print(f"{'=' * 96}")
        header = f"{'variant':16} " + " ".join(f"{'c' + f'{c:.2f}':>13}" for c in CONFIDENCES)
        print(header)
        print("-" * len(header))
        for variant in list(labels) + ["UNION_pair", "UNION_all"]:
            cells = []
            for c in CONFIDENCES:
                a = report["groups"][group][variant][f"c{c:.2f}"]
                cells.append(f"{a['recall']:.3f}/{a['keep_all_pq']:.4f}")
            print(f"{variant:16} " + " ".join(f"{x:>13}" for x in cells))
        print("  (each cell is recall / keep-all PQ)")

        # the decisive comparison, at each confidence
        print(f"\n  best single source vs UNION_all, {group}:")
        print(f"    {'conf':>6} {'best single':>14} {'singlePQ':>10} "
              f"{'unionPQ':>10} {'delta':>10} {'unionRecall':>12} {'unionFP':>9}")
        verdict_rows = []
        for c in CONFIDENCES:
            key = f"c{c:.2f}"
            best = max(labels, key=lambda label: report["groups"][group][label][key]["keep_all_pq"])
            sp = report["groups"][group][best][key]["keep_all_pq"]
            u = report["groups"][group]["UNION_all"][key]
            verdict_rows.append(u["keep_all_pq"] - sp)
            print(f"    {c:6.2f} {best:>14} {sp:10.4f} {u['keep_all_pq']:10.4f} "
                  f"{u['keep_all_pq'] - sp:+10.4f} {u['recall']:12.4f} {u['fp']:9d}")
        wins = sum(1 for d in verdict_rows if d > 0)
        print(f"    UNION beats the best single source at {wins} of {len(CONFIDENCES)} "
              f"confidences (max gain {max(verdict_rows):+.4f})")

    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
