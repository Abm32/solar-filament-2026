"""V53: fuse the V52 clDice *semantic* U-Net proposals into the exact V17 pipeline.

Why this and not V51: V51 fused V44 (YOLO11-seg, same family as V17) and failed —
V44 was nested inside V17 (FP correlation high, nothing orthogonal to add). The
V53 pre-flight diagnostic (scripts/diagnose_v52_v17_overlap.py) measured V52's FP
correlation with V17 at only 19-24% and showed V52 recovers up to 93 of V17's 317
missed GT instances on the 94-file fusion-tune split. V52 is therefore the first
genuinely orthogonal source — the one condition (E028) under which fusion can beat
V17's 0.448703. This harness runs the actual gated grid to find out.

Base pipeline (frozen, bit-identical control): V16 arm_r1 seed1 @ box iou 0.3,
4-view flip_consensus -> V11 proposal_centered_e4 refiner -> t/c grid. Imported from
reproduce_v17_from_cache (= v50); V50 reproduced V17 exactly from signed caches.

Extra source: V52 instance proposals pre-dumped per tune file at
  EXTRA_SETS[set] = /mnt/v52/cache/<set>/<file>.json.gz  {file,threshold,rles,scores}
V52 components are disjoint by construction (connected components), so NO clustering
is needed — the dump IS the extra-cluster set. extra score = mean filament probability.

Fusion grid (selection by TUNE PQ only, exactly V51's semantics):
  extra_set in EXTRA_SETS keys (V52 variants/thresholds)
  alpha in {0, 0.25, 0.5}: V17 score -> (1-a)*s17 + a*e,  e = best matching
      (IoU>=0.5) V52 score, else 0        [FP suppression / agreement up-weight]
  beta  in {0, 0.5, 1.0}: unmatched V52 proposals added with score beta*e
      (beta=0 adds none)                  [recall]
  threshold x confidence = V17's frozen 6 x 7 grid.
alpha=0, beta=0 (per extra_set) is the V17 control and MUST reproduce V50 exactly.

Gates (predeclared, firewall): PROMOTE only if tune PQ >= 0.448703 + 0.010 (E024).
Audit (47 files) opens ONCE for the single selected candidate, only if the tune gate
passes, and must be >= V17 audit 0.470763. No test inference / CSV / submission here.

  CUDA_VISIBLE_DEVICES=N python scripts/v53_fusion.py eval  --shard N
  python scripts/v53_fusion.py merge
  CUDA_VISIBLE_DEVICES=N python scripts/v53_fusion.py audit --shard N   (if authorized)
  python scripts/v53_fusion.py audit-merge
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import cv2
import numpy as np
import reproduce_v17_from_cache as v50

ROOT = v50.ROOT
OUTPUT = ROOT / "outputs/v53_fusion"
V52_CACHE = Path("/mnt/v52/cache")
# V52 dump sets available (produced by train_v52_cldice_semantic.py --dump-dir).
# w1_t060   = clDice-w1.0 variant @ prob 0.60 (highest precision, standalone PQ 0.3766)
# base32_t040 = base variant @ prob 0.40 (highest recall, recovers 93/317 V17 FNs)
EXTRA_SETS = ("w1_t060", "base32_t040")
ALPHAS = (0.0, 0.25, 0.5)
BETAS = (0.0, 0.5, 1.0)
THRESHOLDS, CONFIDENCES = v50.THRESHOLDS, v50.CONFIDENCES
MIN_AREA, FUSE_IOU = 20, 0.5
V17_SEED, V17_CELL = "seed1", "mr1_iou030"
V17_TUNE_PQ, V17_AUDIT_PQ = 0.44870271727943206, 0.4707634285309831
PROMOTION_DELTA = 0.010
SHARDS = 4


def key(extra_set, alpha, beta, t, c):
    return f"{extra_set}/a{alpha:.2f}_b{beta:.2f}/t{t:.2f}_c{c:.2f}"


def control_key(extra_set, t, c):
    return key(extra_set, 0.0, 0.0, t, c)


# ---------------------------------------------------------------- V52 extra source
def read_v52(extra_set, name):
    """Load V52 dumped instance proposals for one tune/audit file.

    Returns (rles, scores). RLEs are 2048x2048 pycocotools dicts (counts bytes),
    already disjoint (connected components). score = mean filament probability.
    """
    path = V52_CACHE / extra_set / (name + ".json.gz")
    with gzip.open(path, "rt") as stream:
        raw = json.load(stream)
    if raw["file"] != name:
        raise RuntimeError(f"Stale/invalid V52 cache {extra_set}/{name}")
    rles = [dict(size=list(r["size"]), counts=r["counts"].encode()) for r in raw["rles"]]
    for r in rles:
        if r["size"] != [2048, 2048]:
            raise RuntimeError(f"Unexpected V52 RLE geometry in {extra_set}/{name}")
    return rles, list(raw["scores"])


# ---------------------------------------------------------------- fusion
def fuse(mask_utils, v17_rles, extra_rles, extra_scores):
    """Matched extra score per V17 proposal + indices of unmatched extras."""
    matched = np.zeros(len(v17_rles))
    unmatched = list(range(len(extra_rles)))
    if v17_rles and extra_rles:
        iou = np.asarray(mask_utils.iou(v17_rles, extra_rles, [0] * len(extra_rles)))
        hit = iou >= FUSE_IOU
        escores = np.asarray(extra_scores)
        matched = np.where(hit.any(axis=1), (hit * escores[None, :]).max(axis=1), 0.0)
        unmatched = [j for j in range(len(extra_rles)) if not hit[:, j].any()]
    return matched, unmatched


def score_image(rt, mask_utils, name, gray, gt_by_image, candidates, totals):
    core, base = rt["core"], rt["base"]
    views = v50.read_signed_views(name, "train", V17_CELL, v50.DETECTORS[V17_SEED])
    v17_rles, v17_scores = rt["tta"].cluster_views(views, v50.POLICY, v50.MATCH_IOU)
    v17_scores = np.asarray(v17_scores, dtype=float)
    v17_decoded = [mask_utils.decode(r) for r in v17_rles]
    for set_name in EXTRA_SETS:
        e_rles, e_scores = read_v52(set_name, name)
        matched, unmatched = fuse(mask_utils, list(v17_rles), e_rles, e_scores)
        # Refine V17 and the unmatched V52 crops in separate batches so V17's crops see
        # exactly the same batch composition as V50 (keeps the control bit-identical).
        masks, bounds, probs = [], [], []
        for group in (v17_decoded, [mask_utils.decode(e_rles[j]) for j in unmatched]):
            if group:
                b, _, pr = core.crop_probabilities(base, rt["model"], gray, group, True)
                masks += group
                bounds += b
                probs += pr
        n17 = len(v17_rles)
        e_add = np.asarray([e_scores[j] for j in unmatched], dtype=float)
        for alpha in ALPHAS:
            s17 = (1 - alpha) * v17_scores + alpha * matched
            for beta in BETAS:
                idx = list(range(n17)) + ([n17 + i for i in range(len(e_add))] if beta else [])
                scores = np.concatenate([s17, beta * e_add]) if beta else s17
                for t in THRESHOLDS:
                    if idx:
                        rles, kept_scores = core.clean_refined_predictions(
                            [masks[i] for i in idx], list(scores[: len(idx)]),
                            [bounds[i] for i in idx], [probs[i] for i in idx], t, v50.MODE)
                    else:
                        rles, kept_scores = [], np.asarray([])
                    for c in CONFIDENCES:
                        k = key(set_name, alpha, beta, t, c)
                        if candidates is not None and k not in candidates:
                            continue
                        keep = (np.asarray(kept_scores) >= c if len(kept_scores)
                                else np.zeros(0, bool))
                        kept = [r for r, f in zip(rles, keep, strict=True) if f]
                        total = totals.setdefault(k, np.zeros(4))
                        for gt in gt_by_image.values():
                            iou = (mask_utils.iou(kept, gt, [0] * len(gt)).T if kept and gt
                                   else np.zeros((len(gt), len(kept))))
                            total += core.pq_counts(iou, len(gt), len(kept))


def evaluate(shard, group, candidates=None, limit=0):
    from pycocotools import mask as mask_utils

    rt = v50.runtime_state()
    _, tune, audit, _, by_file = rt["split"]
    names = sorted(tune if group == "tune" else audit)[shard::SHARDS]
    names = names[:limit] if limit else names
    totals: dict[str, np.ndarray] = {}
    started = time.monotonic()
    for index, name in enumerate(names, 1):
        gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        gt_by_image = {}
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            gt = [rt["core"].annotation_rle(info, e) for e in rt["annotations"].get(image_id, [])]
            gt_by_image[image_id] = [r for r in gt if mask_utils.area(r) > 0]
        score_image(rt, mask_utils, name, gray, gt_by_image, candidates, totals)
        print(f"eval {group} shard={shard} {index}/{len(names)} "
              f"elapsed={time.monotonic() - started:.0f}s", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / f"{group}_shard{shard}.json"
    out.write_text(json.dumps(dict(group=group, shard=shard, files=names, limit=limit,
                                   totals={k: v.tolist() for k, v in totals.items()})) + "\n")
    return out


def merge_parts(group, expected_files):
    parts = [json.loads((OUTPUT / f"{group}_shard{s}.json").read_text()) for s in range(SHARDS)]
    if any(p["limit"] for p in parts):
        raise RuntimeError("Refusing to merge limited (smoke) shards")
    files = [f for p in parts for f in p["files"]]
    if len(files) != expected_files or len(set(files)) != len(files):
        raise RuntimeError("Shards do not cover the population exactly once")
    totals: dict[str, np.ndarray] = {}
    for part in parts:
        for k, v in part["totals"].items():
            totals.setdefault(k, np.zeros(4))
            totals[k] += np.asarray(v)
    return totals


def merge():
    totals = merge_parts("tune", 94)
    # Control (alpha=0, beta=0) for EACH extra_set must reproduce V50/V17 exactly.
    ref = json.loads((v50.OUTPUT / "tune_merged.json").read_text())["totals"]
    for set_name in EXTRA_SETS:
        for t in THRESHOLDS:
            for c in CONFIDENCES:
                mine = totals[control_key(set_name, t, c)]
                theirs = ref[f"{V17_SEED}/{V17_CELL}/t{t:.2f}_c{c:.2f}"]
                if abs(mine[0] - theirs[0]) > 1e-6 or list(mine[1:]) != list(theirs[1:]):
                    raise RuntimeError(f"Control mismatch {set_name} t{t} c{c}: {mine} vs {theirs}")
    ranked = sorted(totals, key=lambda k: (-v50.pq(totals[k]), k))
    selected = ranked[0]
    best_pq = v50.pq(totals[selected])
    by_policy = {}
    for k in ranked:
        policy = k.rsplit("/", 1)[0]
        by_policy.setdefault(policy, dict(key=k, pq=v50.pq(totals[k]), counts=totals[k].tolist()))
    report = dict(
        experiment="v53_v17_plus_v52_cldice_fusion",
        control_verified=True, candidates=len(totals),
        selected=selected, selected_counts=totals[selected].tolist(), selected_pq=best_pq,
        delta_vs_v17=best_pq - V17_TUNE_PQ, promotion_bar=V17_TUNE_PQ + PROMOTION_DELTA,
        audit_authorized=bool(best_pq >= V17_TUNE_PQ + PROMOTION_DELTA),
        best_per_policy=by_policy, top15=[(k, v50.pq(totals[k])) for k in ranked[:15]],
        reads_audit=False, reads_test=False, submits=False,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "tune_merged.json").write_text(json.dumps(
        dict(report=report, totals={k: v.tolist() for k, v in totals.items()}), indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "best_per_policy"}, indent=2))
    return report


def selected_for_audit():
    report = json.loads((OUTPUT / "tune_merged.json").read_text())["report"]
    if not report["audit_authorized"]:
        raise RuntimeError("Tune promotion gate not passed; audit stays sealed")
    return report["selected"]


def audit_merge():
    selected = selected_for_audit()
    totals = merge_parts("audit", 47)
    if set(totals) != {selected}:
        raise RuntimeError("Audit evaluated something other than the selected candidate")
    audit_pq = v50.pq(totals[selected])
    result = dict(selected=selected, audit_counts=totals[selected].tolist(), audit_pq=audit_pq,
                  v17_audit_pq=V17_AUDIT_PQ, audit_passed=audit_pq >= V17_AUDIT_PQ)
    (OUTPUT / "audit_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("eval", "audit"):
        p = sub.add_parser(command)
        p.add_argument("--shard", type=int, required=True, choices=range(SHARDS))
        p.add_argument("--limit", type=int, default=0)
    sub.add_parser("merge")
    sub.add_parser("audit-merge")
    args = parser.parse_args()
    if args.command == "eval":
        print(evaluate(args.shard, "tune", limit=args.limit))
    elif args.command == "merge":
        merge()
    elif args.command == "audit":
        selected = selected_for_audit()
        print(evaluate(args.shard, "audit", candidates={selected}))
    else:
        audit_merge()


if __name__ == "__main__":
    main()
