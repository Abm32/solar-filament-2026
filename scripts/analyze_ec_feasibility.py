"""CPU-only feasibility test for E-C (learned proposal-quality / IoU head).

Question
--------
E-C proposes replacing the fixed 0.40 consensus-confidence gate with a learned
per-proposal quality score, ranking proposals by predicted mask IoU at a MATCHED
accepted-proposal budget. Its predeclared falsification criterion is:

    "if the learned scorer at the matched budget cannot beat the 0.40 gate ...
     (i.e. the 278 pool is not separable from FP by learnable features), then
     recall recovery is genuinely detector-limited and effort must shift to V12."

This script tests that separability BEFORE spending any GPU, using only local
artifacts. It trains nothing on GPU, runs no inference, exports no submission.

Method
------
For each of the 94 tuning files, reconstruct the exact canonical proposal
pipeline from the frozen V6 caches using the repo's own `cluster_views`
(flip_consensus) helper -- the same function `analyze_proposal_research.py`
uses -- to obtain grouped proposals and their consensus scores. Then label each
grouped proposal by its best IoU against the canonical per-annotator GT:

    positive  : best GT IoU > 0.5   (a proposal that, if accepted, is a real match)
    negative  : best GT IoU <= 0.5  (a proposal that, if accepted, becomes FP)

Then ask two questions with purely geometric / agreement features that require
no learned weights and no GPU:

  Q1  Ranking quality. Does any cheap feature rank positives above negatives
      better than the consensus score already does? Measured by ROC AUC,
      computed exactly via the Mann-Whitney U statistic (no sklearn needed).

  Q2  Matched-budget simulation. At exactly the same number of accepted
      proposals as the 0.40 gate produces, how many more positives does ranking
      by each feature admit? This is the quantity E-C would have to beat.

Honesty constraints
-------------------
* This is an ORACLE-LABELLED ranking study on the tuning set. Features are
  evaluated on the same data whose labels they are scored against, so every AUC
  here is an OPTIMISTIC upper bound for a fixed feature. It cannot be reported
  as a model result.
* A LEARNED head could exceed the best single fixed feature by combining them,
  but it is very unlikely to exceed a strong single-feature AUC by a large
  margin using these same inputs. So this study is informative mainly in the
  NEGATIVE direction: if no feature separates the pool at all, E-C's premise is
  weak and the budget should shift to V12.
* Labels use grouped-proposal-vs-GT IoU, i.e. proposal geometry BEFORE the
  refiner. The deployed refiner improves geometry (563 oracle accepted matches
  vs 629 actual TP), so positives here are a conservative proxy.
* GT is kept per annotator and never unioned; categories are never dropped.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ec_feasibility_2026-09-06"
CACHE = ROOT / "outputs/kaggle_detector_tta/detector_proposal_cache"
ACCEPT_CONF = 0.40
MATCH_IOU = 0.5


def source_function(path: Path, name: str, namespace: dict):
    """Use the exact local pure helper without importing GPU dependencies."""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace[name]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact ROC AUC via the rank-sum (Mann-Whitney U) identity, ties averaged."""
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def build_features(grouped: list, scores: list, n_views: int) -> pd.DataFrame:
    """Cheap geometric / agreement features per grouped proposal. No GPU, no weights."""
    areas = mask_utils.area(grouped).astype(float)
    rows = []
    for idx, rle in enumerate(grouped):
        mask = mask_utils.decode(rle)
        ys, xs = np.nonzero(mask)
        area = float(areas[idx])
        h = float(ys.max() - ys.min() + 1) if len(ys) else 1.0
        w = float(xs.max() - xs.min() + 1) if len(xs) else 1.0
        bbox_area = h * w
        # PCA elongation of the mask point cloud.
        if len(ys) > 2:
            pts = np.stack([xs.astype(float), ys.astype(float)])
            pts = pts - pts.mean(axis=1, keepdims=True)
            eig = np.linalg.eigvalsh(np.cov(pts))
            eig = np.clip(eig, 1e-9, None)
            elong = float(np.sqrt(eig[-1] / eig[0]))
        else:
            elong = 1.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        perim = float(sum(cv2.arcLength(c, True) for c in contours)) or 1.0
        n_components = len(contours)
        rows.append(dict(
            consensus_score=float(scores[idx]),
            area=area,
            log_area=float(np.log(max(area, 1.0))),
            fill_ratio=area / bbox_area if bbox_area > 0 else 0.0,
            elongation=elong,
            compactness=float(4.0 * np.pi * area / (perim ** 2)),
            solidity_proxy=area / perim,
            n_components=float(n_components),
            # Consensus score scaled by view count == fraction of views agreeing
            # when cluster_views averages support; kept explicit for clarity.
            view_support=float(scores[idx]) * n_views,
        ))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (ROOT / "outputs/kaggle_v10_quality/inference_manifest.json").read_text())
    names = manifest["tune_files"]
    assert len(names) == 94 and set(names).isdisjoint(manifest["audit_files"])

    annotation_path = next(
        (ROOT / "data/raw").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    assert hashlib.sha256(annotation_path.read_bytes()).hexdigest() == \
        manifest["annotation_sha256"], "annotation file identity mismatch"
    data = json.loads(annotation_path.read_text())
    infos, annotations = defaultdict(list), defaultdict(list)
    for info in data["images"]:
        infos[info["file_name"]].append(info)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)

    ns = {"np": np, "mask_utils": mask_utils,
          "POLICIES": ("single", "flip_max", "flip_consensus"),
          "CFG": SimpleNamespace(crop_context=1.8, min_crop=96)}
    cluster = source_function(ROOT / "kaggle_inference/detector_tta.py",
                             "cluster_views", ns)

    frames, cache_hashes = [], {}
    for number, name in enumerate(names, 1):
        remote = ("/kaggle/input/competitions/filament-segmentation-2026/"
                  "MAGFiLO_1.0_Kaggle_2026/train/train_images/" + name)
        key = hashlib.sha256(remote.encode()).hexdigest()[:16] + ".json.gz"
        path = CACHE / key
        cache_hashes[str(path.relative_to(ROOT))] = \
            hashlib.sha256(path.read_bytes()).hexdigest()
        views = json.loads(gzip.decompress(path.read_bytes()))
        pairs = [(v["rles"], v["scores"]) for v in views]
        grouped, scores = cluster(pairs, "flip_consensus")
        if not grouped:
            continue
        feats = build_features(grouped, scores, len(views))
        feats["file_name"] = name
        feats["proposal_index"] = np.arange(len(grouped))

        # Label each grouped proposal per annotator entry separately. A proposal
        # is positive for an annotator if its best IoU against that annotator's
        # GT exceeds 0.5. Annotator entries are scored separately by the metric,
        # so a proposal is "useful" if it matches under ANY annotator entry, and
        # we record the per-annotator best as well.
        best_iou = np.zeros(len(grouped))
        for info in infos[name]:
            anns = annotations[info["id"]]
            gt = [mask_utils.merge(mask_utils.frPyObjects(
                a["segmentation"], info["height"], info["width"])) for a in anns]
            if not gt:
                continue
            ious = mask_utils.iou(grouped, gt, [0] * len(gt))
            best_iou = np.maximum(best_iou, np.asarray(ious).max(axis=1))
        feats["best_gt_iou"] = best_iou
        feats["label"] = (best_iou > MATCH_IOU).astype(int)
        frames.append(feats)
        if number % 20 == 0:
            print(f"Processed {number}/{len(names)} tuning files", flush=True)

    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(OUT / "proposal_pool.csv", index=False)

    accepted = frame.consensus_score >= ACCEPT_CONF
    budget = int(accepted.sum())
    baseline_pos = int(frame.loc[accepted, "label"].sum())
    baseline_neg = budget - baseline_pos
    rejected_pos = int(frame.loc[~accepted, "label"].sum())

    feature_cols = ["consensus_score", "log_area", "fill_ratio", "elongation",
                    "compactness", "solidity_proxy", "n_components",
                    "view_support", "area"]
    results = {}
    for col in feature_cols:
        values = frame[col].to_numpy(dtype=float)
        auc = roc_auc(values, frame.label.to_numpy())
        # Matched-budget: take the top `budget` proposals by this feature.
        order = np.argsort(-values, kind="mergesort")[:budget]
        admitted_pos = int(frame.label.to_numpy()[order].sum())
        results[col] = {
            "auc": auc,
            "matched_budget_positives": admitted_pos,
            "delta_vs_gate": admitted_pos - baseline_pos,
        }

    summary = {
        "scope": ("94 tuning files, frozen V6 flip_consensus grouped proposals, "
                  "oracle-labelled by best per-annotator GT IoU > 0.5"),
        "warning": ("ORACLE-LABELLED FIXED-FEATURE RANKING STUDY on the tuning "
                    "set. AUC values are optimistic upper bounds for each fixed "
                    "feature and are NOT model results, NOT PQ, and NOT a public "
                    "score forecast. Labels use pre-refiner proposal geometry."),
        "n_proposals": int(len(frame)),
        "n_positive": int(frame.label.sum()),
        "n_negative": int((frame.label == 0).sum()),
        "gate": {
            "accept_confidence": ACCEPT_CONF,
            "accepted_budget": budget,
            "accepted_positives": baseline_pos,
            "accepted_negatives": baseline_neg,
            "positives_rejected_by_gate": rejected_pos,
            "precision": baseline_pos / budget if budget else 0.0,
        },
        "features": results,
        "cache_files_hashed": len(cache_hashes),
        "annotation_sha256": manifest["annotation_sha256"],
    }
    best = max((k for k in results if not np.isnan(results[k]["auc"])),
               key=lambda k: results[k]["auc"])
    summary["best_feature_by_auc"] = {"name": best, **results[best]}
    summary["consensus_auc"] = results["consensus_score"]["auc"]
    summary["best_matched_budget_gain"] = max(
        r["delta_vs_gate"] for r in results.values())

    (OUT / "ec_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT / "cache_hashes.json").write_text(
        json.dumps(cache_hashes, indent=2, sort_keys=True) + "\n")

    print(json.dumps({k: v for k, v in summary.items()
                      if k != "features"}, indent=2, sort_keys=True))
    print("\nPer-feature ranking (AUC, matched-budget positives vs gate):")
    for col, r in sorted(results.items(), key=lambda kv: -(kv[1]["auc"] or 0)):
        print(f"  {col:18s} auc={r['auc']:.4f}  "
              f"pos@budget={r['matched_budget_positives']:4d}  "
              f"delta={r['delta_vs_gate']:+d}")


if __name__ == "__main__":
    main()
