"""Do our detectors miss the SAME instances? (zero GPU, caches only)

Every experiment E015-E027 optimised a *single* detector's proposals. `ceiling_analysis`
§8 item "E-H fold ensemble" was never run, and its value hinges on one unmeasured
quantity: **do different detectors fail on the same GT instances, or different ones?**

If they fail on the same instances, the miss set is a property of the data/architecture
and no ensemble, threshold or refiner change reaches it -- the 0.39 plateau is
structural and the honest move is to stop. If they fail on *different* instances,
the union recall is materially above any single detector's, and a proposal-union
ensemble is a real, untested, inference-only lever.

Sources compared, all already on disk, same 141 files and same GT:

    V6_iou070   frozen V6 detector (mask_ratio=2) @ box iou=0.7   [V16 cache]
    V6_iou030   frozen V6 detector (mask_ratio=2) @ box iou=0.3   [V13 cache]
    MR1s1_iou030  V16 arm_r1 seed1 (mask_ratio=1) @ box iou=0.3   [V17 cache]
    MR1s1_iou070  V16 arm_r1 seed1 (mask_ratio=1) @ box iou=0.7   [V17 cache]
    MR1s2_iou030  V16 arm_r1 seed2 (mask_ratio=1) @ box iou=0.3   [V17 cache]
    MR1s2_iou070  V16 arm_r1 seed2 (mask_ratio=1) @ box iou=0.7   [V17 cache]

For every GT instance we record which sources match it at IoU>0.5 (grouped,
pre-refiner, pre-confidence-gate -- the same stage `analyze_lever_overlap` uses).
Then:

  1. per-source recall (reproduces lever_overlap's A/B/C as a correctness check)
  2. UNION recall over all sources, and over the two most complementary
  3. how many GT instances are missed by ALL sources  -> the structural floor
  4. best achievable oracle PQ from the union proposal pool
  5. pairwise complementarity: instances that A misses but B finds

Usage:
    .venv/bin/python scripts/analyze_detector_complementarity.py
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_v16_proposals import (  # noqa: E402
    V6_DETECTOR_SHA256, annotation_rle, cache_id, cluster_views, greedy_match, read_views,
)
from analyze_lever_overlap import load_gt, v13_cache_id  # noqa: E402

MATCH_IOU = 0.5

ARM_R1_SEED1 = "ae476e663039fbbbfda77ec519635ca15e2d50d1fe9f1af54735f84dbdc251ef"
ARM_R1_SEED2 = "097a57ed7b5fe56909393ab05897ba5c41b96f7c343cb88d2ba782dd9dfabbe7"


def v17_cache_id(name, split, cell_name, detector_sha, mask_ratio=1,
                 resolution=1024, floor=0.1):
    """Reproduces v17_compose.cache_id exactly (detector | mask_ratio | cell | split | name)."""
    token = (f"{detector_sha}|mr{mask_ratio}|{cell_name}|{split}|{name}"
             f"|imgsz{resolution}|conf{floor}")
    return hashlib.sha256(token.encode()).hexdigest()[:16] + ".json.gz"


def build_sources(root, v16_cache, v13_cache, v17_cache, names):
    return {
        "V6_iou070": {n: v16_cache / cache_id(n, "train", 2, V6_DETECTOR_SHA256) for n in names},
        "V6_iou030": {n: v13_cache / v13_cache_id(n, "train") for n in names},
        "MR1s1_iou030": {n: v17_cache / v17_cache_id(n, "train", "mr1_iou030", ARM_R1_SEED1)
                         for n in names},
        "MR1s1_iou070": {n: v17_cache / v17_cache_id(n, "train", "mr1_iou070", ARM_R1_SEED1)
                         for n in names},
        "MR1s2_iou030": {n: v17_cache / v17_cache_id(n, "train", "mr1_iou030", ARM_R1_SEED2)
                         for n in names},
        "MR1s2_iou070": {n: v17_cache / v17_cache_id(n, "train", "mr1_iou070", ARM_R1_SEED2)
                         for n in names},
    }


def collect(sources, images, annotations, by_file, tune):
    """Per-GT-instance match record across every source.

    Returns rows: one per (image_id, gt_index) with the best IoU from each source.
    """
    rows = []
    labels = list(sources)
    for name in sorted(next(iter(sources.values()))):
        # proposals per source for this file
        per_source = {}
        for label in labels:
            views = read_views(sources[label][name])
            rles, scores = cluster_views(views)
            per_source[label] = (rles, scores)
        for image_id in sorted(by_file[name]):
            info = images[image_id]
            gt = [annotation_rle(info, entry) for entry in annotations.get(image_id, [])]
            gt = [rle for rle in gt if mask_utils.area(rle) > 0]
            if not gt:
                continue
            group = "tune" if name in tune else "audit"
            best = {}
            for label in labels:
                rles, _ = per_source[label]
                if rles:
                    iou = np.asarray(
                        mask_utils.iou(rles, gt, [0] * len(gt)).T, dtype=np.float64)
                    pairs = greedy_match(iou)
                    matched = {g: v for g, _p, v in pairs}
                else:
                    matched = {}
                best[label] = matched
            for gt_index in range(len(gt)):
                row = {"file_name": name, "image_id": image_id,
                       "gt_index": gt_index, "group": group}
                for label in labels:
                    row[label] = best[label].get(gt_index, 0.0)
                rows.append(row)
    return rows, labels


def summarise(rows, labels, group):
    subset = [r for r in rows if r["group"] == group]
    n_gt = len(subset)
    out = {"group": group, "n_gt": n_gt, "per_source": {}, "union": {}, "pairwise": {}}
    if not n_gt:
        return out

    for label in labels:
        hits = [r for r in subset if r[label] > MATCH_IOU]
        out["per_source"][label] = {
            "matched": len(hits),
            "recall": len(hits) / n_gt,
            "mean_matched_iou": (sum(r[label] for r in hits) / len(hits)) if hits else 0.0,
        }

    # union over ALL sources: best IoU any source achieves
    union_best = [max(r[label] for label in labels) for r in subset]
    union_hits = [v for v in union_best if v > MATCH_IOU]
    out["union"]["all_sources"] = {
        "matched": len(union_hits),
        "recall": len(union_hits) / n_gt,
        "mean_matched_iou": (sum(union_hits) / len(union_hits)) if union_hits else 0.0,
        "missed_by_every_source": n_gt - len(union_hits),
    }

    best_single = max(out["per_source"], key=lambda k: out["per_source"][k]["recall"])
    out["union"]["best_single_source"] = best_single
    out["union"]["recall_gain_over_best_single"] = (
        out["union"]["all_sources"]["recall"] - out["per_source"][best_single]["recall"])

    # best PAIR by union recall
    best_pair, best_pair_recall = None, -1.0
    for a, b in itertools.combinations(labels, 2):
        hits = sum(1 for r in subset if max(r[a], r[b]) > MATCH_IOU)
        if hits / n_gt > best_pair_recall:
            best_pair, best_pair_recall = (a, b), hits / n_gt
    pair_best = [max(r[best_pair[0]], r[best_pair[1]]) for r in subset]
    pair_hits = [v for v in pair_best if v > MATCH_IOU]
    out["union"]["best_pair"] = {
        "sources": list(best_pair),
        "recall": best_pair_recall,
        "mean_matched_iou": (sum(pair_hits) / len(pair_hits)) if pair_hits else 0.0,
    }

    # pairwise complementarity: found by B, missed by A
    for a, b in itertools.permutations(labels, 2):
        only_b = sum(1 for r in subset if r[a] <= MATCH_IOU < r[b])
        out["pairwise"][f"{b}_rescues_{a}"] = only_b

    return out


def oracle_pq(matched, mean_iou, n_gt, fp):
    """PQ if we perfectly selected the matched set and incurred `fp` false positives."""
    tp = matched
    fn = n_gt - matched
    denom = tp + 0.5 * fp + 0.5 * fn
    return (tp * mean_iou) / denom if denom else 0.0


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16-manifest", type=Path,
                        default=root / "outputs/kaggle_v16_full/v16_manifest.json")
    parser.add_argument("--v16-cache", type=Path,
                        default=root / "outputs/kaggle_v16_full/v16_proposal_cache")
    # NOTE: outputs/kaggle_v13_full/proposal_cache holds only .signature.json files
    # (the ops CLI's default --file-pattern excluded .gz); the actual payloads are in
    # the dedicated re-pull directory. This is the same trap E026 hit.
    parser.add_argument("--v13-cache", type=Path,
                        default=root / "outputs/kaggle_v13_cache_pull/proposal_cache")
    # NOTE: outputs/kaggle_v17_full/proposal_cache is an INCOMPLETE pull (323 of 744
    # signatures, partial per cell). The complete re-pull is kaggle_v17_cache_pull.
    # Same trap E026 hit with the V13 caches; the missing-cache assertion below is
    # what makes it loud instead of silent.
    parser.add_argument("--v17-cache", type=Path,
                        default=root / "outputs/kaggle_v17_cache_pull/proposal_cache")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/v16_proposal_analysis/detector_complementarity.json")
    args = parser.parse_args()

    manifest = json.loads(args.v16_manifest.read_text())
    tune = set(manifest["tune_files"])
    names = sorted(tune | set(manifest["audit_files"]))

    images, annotations, by_file = load_gt(root)
    sources = build_sources(root, args.v16_cache, args.v13_cache, args.v17_cache, names)

    for label, paths in sources.items():
        missing = [n for n, p in paths.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{label}: missing {len(missing)} of {len(paths)} caches, e.g. {missing[:2]}")
    print(f"all {len(sources)} sources complete for {len(names)} files", flush=True)

    rows, labels = collect(sources, images, annotations, by_file, tune)
    report = {"match_iou": MATCH_IOU, "sources": labels,
              "groups": {g: summarise(rows, labels, g) for g in ("tune", "audit")}}

    # V17's deployed tune numbers, for an apples-to-apples oracle comparison
    v17 = json.loads((root / "outputs/kaggle_v17_full/v17_report.json").read_text())
    report["v17_deployed"] = {
        "tune_pq": v17["selected_tune_pq"], "audit_pq": v17["selected_audit_pq"],
        "candidate": v17["selected"]}

    for group in ("tune", "audit"):
        g = report["groups"][group]
        n_gt = g["n_gt"]
        u = g["union"]["all_sources"]
        # oracle PQ from the union pool at V17's actual FP, and at zero FP
        for fp_label, fp in (("at_v17_fp", 306 if group == "tune" else 124), ("at_zero_fp", 0)):
            g["union"].setdefault("oracle_pq", {})[fp_label] = oracle_pq(
                u["matched"], u["mean_matched_iou"], n_gt, fp)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    for group in ("tune", "audit"):
        g = report["groups"][group]
        print(f"\n{'=' * 74}\n{group.upper()} group: {g['n_gt']} GT instances"
              f"  (grouped, pre-refiner, pre-confidence-gate)\n{'=' * 74}")
        print(f"{'source':16} {'matched':>8} {'recall':>8} {'meanIoU':>9}")
        print("-" * 44)
        for label in labels:
            s = g["per_source"][label]
            print(f"{label:16} {s['matched']:8d} {s['recall']:8.4f} {s['mean_matched_iou']:9.4f}")
        u = g["union"]["all_sources"]
        bp = g["union"]["best_pair"]
        print("-" * 44)
        print(f"{'UNION (all 6)':16} {u['matched']:8d} {u['recall']:8.4f} "
              f"{u['mean_matched_iou']:9.4f}")
        print(f"  best single source      : {g['union']['best_single_source']} "
              f"(recall {g['per_source'][g['union']['best_single_source']]['recall']:.4f})")
        print(f"  union gain over it      : {g['union']['recall_gain_over_best_single']:+.4f}")
        print(f"  best PAIR               : {bp['sources'][0]} + {bp['sources'][1]} "
              f"-> recall {bp['recall']:.4f}")
        print(f"  missed by EVERY source  : {u['missed_by_every_source']} of {g['n_gt']} "
              f"({u['missed_by_every_source'] / g['n_gt']:.1%})  <- structural floor")
        print(f"  union oracle PQ @ V17 FP: {g['union']['oracle_pq']['at_v17_fp']:.4f}"
              f"   @ zero FP: {g['union']['oracle_pq']['at_zero_fp']:.4f}")
        if group == "tune":
            print(f"  V17 actually deployed   : {report['v17_deployed']['tune_pq']:.4f}")
        print("\n  top rescues (B finds what A misses):")
        top = sorted(g["pairwise"].items(), key=lambda kv: -kv[1])[:6]
        for key, count in top:
            print(f"    {key:34} {count:5d}")

    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
