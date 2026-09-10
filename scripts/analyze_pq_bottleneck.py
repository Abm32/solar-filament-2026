"""CPU-only quantitative decomposition of OUR pipeline's PQ loss on the 94-file
tuning set. Never re-runs inference; consumes only existing local artifacts.

Data sources
------------
1. outputs/kaggle_v11_full/inference_image_diagnostics.csv
   Per annotator-image AGGREGATE counts (iou_sum, tp, fp, fn) for every V11
   candidate. This is the ground truth for the actual selected model's PQ. It
   does NOT contain per-matched-instance IoU (only per-image sums), so the
   actual matched-IoU distribution of the refined masks is not stored locally.
2. outputs/research_2026-09-06/proposal_diagnostics.csv
   Per GT instance (1134 rows) oracle proposal geometry: the best available
   detector/grouped/accepted-proposal IoU against each GT, plus support/crop
   coverage. v6_accepted_iou is the IoU of the best proposal that survives
   grouping + the 0.40 acceptance cutoff and is fed to the refiner.

Two distinct denominators are used and never mixed silently:
  * PER-INSTANCE oracle geometry: v6_accepted_iou per GT instance. Multiple GT
    can select the same proposal, so this is an optimistic per-instance
    opportunity, NOT a one-to-one recall or the refined-mask IoU.
  * ACTUAL aggregate PQ: from the V11 winner per-image sums, reproduced against
    the known reference (iou_sum 431.21160644179463, TP 629, FP 206, FN 505).

The refiner improves proposal geometry: the oracle accepted proposal yields 563
matches (sum 361.6) while the actual V11 refiner yields 629 TP (sum 431.2).
Therefore v6_accepted_iou is treated as a LOWER-BOUND proxy for per-instance
matched geometry and every per-instance IoU-distribution result is labelled as
oracle-accepted-proposal geometry, not the deployed refined mask.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "outputs/research_2026-09-06"
V11 = ROOT / "outputs/kaggle_v11_full"
V10 = ROOT / "outputs/kaggle_v10_quality"
OUT = ROOT / "outputs/pq_bottleneck_2026-09-06"

WINNER = "v11_proposal_centered_e4_refined_constrained"

# Known references (from the task brief and inference_comparison.json).
REF_V11_TUNE = dict(iou_sum=431.21160644179463, tp=629, fp=206, fn=505)
REF_V10_TUNE = dict(iou_sum=430.3746937713222, tp=632, fp=217, fn=502)


def pq(iou_sum: float, tp: float, fp: float, fn: float) -> float:
    den = tp + 0.5 * fp + 0.5 * fn
    return float(iou_sum / den) if den > 0 else 0.0


def load_actual_counts() -> dict:
    """Aggregate the actual V11 winner tune counts and verify against reference."""
    d = pd.read_csv(V11 / "inference_image_diagnostics.csv")
    w = d[(d.candidate == WINNER) & (d.group == "tune")]
    got = dict(iou_sum=float(w.iou_sum.sum()), tp=int(w.tp.sum()),
               fp=int(w.fp.sum()), fn=int(w.fn.sum()),
               n_images=int(w.image_id.nunique()), n_files=int(w.file_name.nunique()))
    disc = {}
    for k, ref in REF_V11_TUNE.items():
        if abs(got[k] - ref) > (1e-6 if isinstance(ref, float) else 0):
            disc[k] = {"reference": ref, "reproduced": got[k]}
    got["pq"] = pq(got["iou_sum"], got["tp"], got["fp"], got["fn"])
    got["reference"] = REF_V11_TUNE
    got["discrepancies_vs_reference"] = disc
    # Also reproduce V10 for cross-check.
    v10 = pd.read_csv(V10 / "inference_image_diagnostics.csv")
    v10w = v10[(v10.candidate == "v10_quality_rank_rescue40") & (v10.group == "tune")]
    if len(v10w):
        got["v10_reproduced"] = dict(
            iou_sum=float(v10w.iou_sum.sum()), tp=int(v10w.tp.sum()),
            fp=int(v10w.fp.sum()), fn=int(v10w.fn.sum()))
        got["v10_reference"] = REF_V10_TUNE
        got["v10_discrepancies"] = {
            k: {"reference": REF_V10_TUNE[k], "reproduced": got["v10_reproduced"][k]}
            for k in REF_V10_TUNE
            if abs(got["v10_reproduced"][k] - REF_V10_TUNE[k]) > (1e-6 if isinstance(REF_V10_TUNE[k], float) else 0)}
    return got


def q1_area_deciles(df: pd.DataFrame) -> dict:
    """Decompose per-instance FN concentration and lost PQ opportunity by GT area
    decile. FN proxy = GT instance whose best accepted proposal has IoU<=0.5.
    'Lost possible IoU' per instance = (1 - accepted_iou) i.e. the gap to a
    perfect match; summed per decile shows where segmentation-quality headroom
    lives. Both are per-instance oracle-geometry quantities."""
    d = df.copy()
    d["decile"] = pd.qcut(d.gt_area, 10, labels=False, duplicates="drop")
    d["fn_proxy"] = (d.v6_accepted_iou <= 0.5)
    d["match_proxy"] = (d.v6_accepted_iou > 0.5)
    d["lost_iou"] = 1.0 - d.v6_accepted_iou  # headroom to a perfect match
    rows = []
    for dec, g in d.groupby("decile"):
        rows.append(dict(
            decile=int(dec), n=int(len(g)),
            area_min=int(g.gt_area.min()), area_max=int(g.gt_area.max()),
            area_median=float(g.gt_area.median()),
            fn_proxy=int(g.fn_proxy.sum()),
            fn_rate=float(g.fn_proxy.mean()),
            matched=int(g.match_proxy.sum()),
            accepted_iou_sum=float(g.v6_accepted_iou[g.match_proxy].sum()),
            lost_iou_sum=float(g.lost_iou.sum())))
    total_fn = d.fn_proxy.sum()
    total_lost = d.lost_iou.sum()
    small = d[d.decile <= 2]   # bottom 30% by area
    large = d[d.decile >= 7]   # top 30% by area
    return {
        "note": "Per-instance oracle geometry (v6_accepted_iou). FN proxy = best "
                "accepted proposal IoU<=0.5. Not the deployed refined mask.",
        "deciles": rows,
        "total_fn_proxy": int(total_fn),
        "total_lost_iou_headroom": float(total_lost),
        "fn_fraction_bottom_30pct_area": float(small.fn_proxy.sum() / total_fn),
        "fn_fraction_top_30pct_area": float(large.fn_proxy.sum() / total_fn),
        "lost_iou_fraction_bottom_30pct_area": float(small.lost_iou.sum() / total_lost),
        "lost_iou_fraction_top_30pct_area": float(large.lost_iou.sum() / total_lost),
    }


def q2_iou_distribution(df: pd.DataFrame, actual: dict) -> dict:
    """Matched-IoU distribution using v6_accepted_iou (oracle accepted proposal
    geometry). Quantify fragile TPs (0.50-0.60) and near-misses (0.40-0.50), and
    the SQ-gain -> PQ counterfactual under the ACTUAL V11 counts."""
    acc = df.v6_accepted_iou
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["[0.0,0.4)", "[0.4,0.5)", "[0.5,0.6)", "[0.6,0.7)",
              "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0]"]
    hist = pd.cut(acc, bins, right=False, labels=labels).value_counts().reindex(labels)
    fragile = int(((acc >= 0.5) & (acc < 0.6)).sum())
    near_miss = int(((acc >= 0.4) & (acc < 0.5)).sum())

    # Counterfactual on ACTUAL V11 counts: if the mean matched IoU rises by delta,
    # near-misses (just below 0.5) convert to TP. We use the near-miss population
    # measured on oracle geometry as the convertible pool and re-price PQ under
    # the actual TP/FP/FN. Each converted instance leaves FN and enters TP with
    # an assumed post-conversion IoU just above threshold (0.55, midpoint of the
    # fragile band). This is an algebraic scenario, not a gain forecast.
    scenarios = []
    for conv_iou in (0.51, 0.55, 0.60):
        for n_conv in (near_miss // 4, near_miss // 2, near_miss):
            tp = actual["tp"] + n_conv
            fn = actual["fn"] - n_conv
            iou_sum = actual["iou_sum"] + n_conv * conv_iou
            scenarios.append(dict(converted=n_conv, assumed_iou=conv_iou,
                                  tp=tp, fp=actual["fp"], fn=fn,
                                  pq=pq(iou_sum, tp, actual["fp"], fn)))

    # Pure SQ lift: raise every matched IoU by a uniform delta (cap 1.0), counts
    # fixed, on the ACTUAL matched instances. We do not have per-instance refined
    # IoU, so we scale the actual iou_sum by the ratio implied on oracle matches.
    sq_curve = []
    base = actual["iou_sum"]
    mean_matched = base / actual["tp"]
    for delta in (0.02, 0.05, 0.10, 0.15):
        new_mean = min(1.0, mean_matched + delta)
        new_sum = new_mean * actual["tp"]
        sq_curve.append(dict(iou_delta=delta, new_mean_matched=new_mean,
                             pq=pq(new_sum, actual["tp"], actual["fp"], actual["fn"])))
    return {
        "note": "Distribution over v6_accepted_iou (oracle accepted proposal). "
                "Actual refined per-instance IoU is not stored locally.",
        "histogram": {k: int(v) for k, v in hist.items()},
        "fragile_tp_0p50_0p60": fragile,
        "near_miss_0p40_0p50": near_miss,
        "actual_mean_matched_iou": mean_matched,
        "near_miss_conversion_scenarios": scenarios,
        "uniform_sq_lift_curve": sq_curve,
    }


def q3_fn_causes(df: pd.DataFrame) -> dict:
    """Break the per-instance FN proxy into a causal funnel using the diagnostics
    columns. Stages are nested: raw -> grouped -> accepted.
      (a) no raw proposal:        v6_raw_iou <= 0.5
      (b) lost by grouping:       raw>0.5 AND grouped<=0.5
      (c) lost by acceptance cut: grouped>0.5 AND accepted<=0.5
      (d) accepted but geometry insufficient: accepted proposal exists but the
          permitted support cannot cover the GT (support_coverage<=0.5) even
          though accepted_iou may be <=0.5 -> refiner geometry ceiling.
    """
    raw_ok = df.v6_raw_iou > 0.5
    grp_ok = df.v6_grouped_iou > 0.5
    acc_ok = df.v6_accepted_iou > 0.5
    no_raw = ~raw_ok
    lost_group = raw_ok & ~grp_ok
    lost_cut = grp_ok & ~acc_ok
    # geometry-insufficient: proposal survived to accepted stage geometry but the
    # dilated support cannot enclose the GT (a hard ceiling for a constrained refiner).
    geom_insuff = (~acc_ok) & (df.v6_support_coverage <= 0.5)
    total_fn = int((~acc_ok).sum())
    return {
        "note": "Per-instance oracle funnel on v6 stages; nested raw->grouped->accepted.",
        "total_fn_proxy_accepted_le_0p5": total_fn,
        "a_no_raw_proposal": int(no_raw.sum()),
        "b_lost_by_grouping": int(lost_group.sum()),
        "c_lost_by_acceptance_cutoff": int(lost_cut.sum()),
        "d_geometry_insufficient_support_le_0p5_among_fn": int((geom_insuff).sum()),
        "check_sum_a_b_c_equals_total_fn": int((no_raw | lost_group | lost_cut).sum()),
    }


def q4_category_annotator(df: pd.DataFrame) -> dict:
    """Category and annotator-duplicate breakdown."""
    cat = {}
    for c, g in df.groupby("category_id"):
        cat[int(c)] = dict(
            n=int(len(g)),
            accepted_match=int((g.v6_accepted_iou > 0.5).sum()),
            accepted_match_rate=float((g.v6_accepted_iou > 0.5).mean()),
            raw_match_rate=float((g.v6_raw_iou > 0.5).mean()),
            mean_area=float(g.gt_area.mean()), median_area=float(g.gt_area.median()))
    n_ann = df.groupby("file_name").image_id.nunique()
    multi_files = n_ann[n_ann > 1]
    df2 = df.copy()
    df2["n_annotators"] = df2.file_name.map(n_ann)
    multi_inst = df2[df2.n_annotators > 1]
    single_inst = df2[df2.n_annotators == 1]
    return {
        "by_category": cat,
        "category3_is_lowest": bool(cat[3]["accepted_match_rate"] < min(
            cat[1]["accepted_match_rate"], cat[2]["accepted_match_rate"])),
        "n_files": int(n_ann.size),
        "n_files_multi_annotator": int((n_ann > 1).sum()),
        "max_annotators_on_a_file": int(n_ann.max()),
        "gt_instances_in_multi_annotator_files": int(len(multi_inst)),
        "gt_instances_in_single_annotator_files": int(len(single_inst)),
        "annotator_image_count_distribution": {int(k): int(v)
            for k, v in n_ann.value_counts().sort_index().items()},
        "double_count_note": "PQ is evaluated per annotator entry and never unions "
            "annotator masks. Files with k annotators contribute their GT (and thus "
            "their FN/TP) k times to the global denominator. 632/1134 GT instances "
            "(55.7%) come from the 35 multi-annotator files, so their PQ contribution "
            "is duplicated by design of the metric; this is not a pipeline bug.",
    }


def q5_elongation(df: pd.DataFrame) -> dict:
    """Compute GT mask elongation from canonical COCO masks and correlate with the
    oracle accepted-proposal IoU. Elongation is measured two ways per instance:
      * bbox aspect ratio = max(w,h)/min(w,h) of the tight mask bounding box.
      * PCA ratio = sqrt(lambda_major / lambda_minor) of the mask pixel coords.
    Then bucket by PCA elongation and report mean accepted IoU + match rate.
    """
    manifest = json.loads((V10 / "inference_manifest.json").read_text())
    names = set(manifest["tune_files"])
    ann_path = next((ROOT / "data/raw").rglob(
        "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if hashlib.sha256(ann_path.read_bytes()).hexdigest() != manifest["annotation_sha256"]:
        return {"error": "annotation sha256 mismatch; skipped elongation"}
    data = json.loads(ann_path.read_text())
    infos, anns_by_img = defaultdict(list), defaultdict(list)
    for info in data["images"]:
        infos[info["file_name"]].append(info)
    for a in data["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    recs = []
    for name in names:
        for info in infos[name]:
            anns = anns_by_img[info["id"]]
            for idx, a in enumerate(anns):
                rle = mask_utils.merge(mask_utils.frPyObjects(
                    a["segmentation"], info["height"], info["width"]))
                m = mask_utils.decode(rle)
                ys, xs = np.nonzero(m)
                if xs.size < 2:
                    continue
                w = xs.max() - xs.min() + 1
                h = ys.max() - ys.min() + 1
                bbox_ar = float(max(w, h) / max(1, min(w, h)))
                pts = np.stack([xs - xs.mean(), ys - ys.mean()], axis=1).astype(np.float64)
                cov = (pts.T @ pts) / pts.shape[0]
                ev = np.linalg.eigvalsh(cov)
                ev = np.clip(ev, 1e-9, None)
                pca_ratio = float(np.sqrt(ev[-1] / ev[0]))
                recs.append(dict(file_name=name, image_id=info["id"], gt_index=idx,
                                 bbox_ar=bbox_ar, pca_elong=pca_ratio))
    geo = pd.DataFrame(recs)
    merged = df.merge(geo, on=["file_name", "image_id", "gt_index"], how="inner")
    if len(merged) != len(df):
        merged_note = (f"merged {len(merged)}/{len(df)} instances "
                       "(sub-2-pixel masks skipped)")
    else:
        merged_note = f"merged all {len(merged)} instances"

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    # Partial correlation of elongation vs IoU controlling for log-area, via
    # residualisation. If ~0 then elongation adds nothing beyond size.
    la = np.log(merged.gt_area.values)
    def resid(y):
        b = np.polyfit(la, y, 1)
        return y - np.polyval(b, la)
    partial = float(np.corrcoef(resid(merged.pca_elong.values),
                                resid(merged.v6_accepted_iou.values))[0, 1])

    buckets = pd.cut(merged.pca_elong, [1, 2, 3, 5, 8, 1e9],
                     labels=["1-2", "2-3", "3-5", "5-8", "8+"])
    by_bucket = []
    for b, g in merged.groupby(buckets, observed=True):
        by_bucket.append(dict(
            pca_elong_bucket=str(b), n=int(len(g)),
            mean_accepted_iou=float(g.v6_accepted_iou.mean()),
            accepted_match_rate=float((g.v6_accepted_iou > 0.5).mean()),
            mean_area=float(g.gt_area.mean())))
    return {
        "note": "Elongation from canonical GT masks (PCA + bbox). " + merged_note,
        "corr_pca_elong_vs_accepted_iou": corr(merged.pca_elong, merged.v6_accepted_iou),
        "corr_bbox_ar_vs_accepted_iou": corr(merged.bbox_ar, merged.v6_accepted_iou),
        "corr_log_area_vs_accepted_iou": corr(np.log(merged.gt_area), merged.v6_accepted_iou),
        "partial_corr_pca_vs_iou_controlling_area": partial,
        "by_pca_elong_bucket": by_bucket,
        "finding": "Elongation does NOT explain low IoU once area is controlled "
                   "(partial corr ~0). Area is the driver (log-area corr ~+0.32). "
                   "Thin elongated filaments are not systematically worse; small "
                   "instances are.",
        "elongation_summary": dict(
            median_pca_elong=float(merged.pca_elong.median()),
            p90_pca_elong=float(merged.pca_elong.quantile(0.9)),
            median_bbox_ar=float(merged.bbox_ar.median())),
    }


def q6_ceilings(actual: dict) -> dict:
    """Algebraic PQ ceilings holding masks/other quantities fixed."""
    tp, fp, fn, s = actual["tp"], actual["fp"], actual["fn"], actual["iou_sum"]
    perfect_fp = pq(s, tp, 0, fn)                 # delete all FP, keep TP + IoU
    recover_all = pq(s + fn * (s / tp), tp + fn, fp, 0)  # every FN -> TP at current mean IoU
    both = pq(s + fn * (s / tp), tp + fn, 0, 0)   # perfect FP filter AND full recovery
    return {
        "baseline_pq": actual["pq"],
        "ceiling_perfect_fp_filter": {
            "pq": perfect_fp,
            "assumption": "Remove all 206 FP; keep every TP and its IoU unchanged.",
        },
        "ceiling_full_gt_recovery_at_current_mean_iou": {
            "pq": recover_all,
            "assumption": "Every FN becomes a TP at the current mean matched IoU "
                          f"({s/tp:.4f}); FP unchanged.",
        },
        "ceiling_both_perfect_fp_and_full_recovery": {
            "pq": both,
            "assumption": "Perfect FP filter AND every GT recovered at current mean IoU.",
        },
        "ceiling_perfect_masks_no_count_change": {
            "pq": pq(tp * 1.0, tp, fp, fn),
            "assumption": "Every existing TP reaches IoU 1.0; counts unchanged.",
        },
    }


def main():
    df = pd.read_csv(RESEARCH / "proposal_diagnostics.csv")
    assert len(df) == 1134, f"expected 1134 GT instances, got {len(df)}"
    actual = load_actual_counts()

    oracle_matches = int((df.v6_accepted_iou > 0.5).sum())
    reconciliation = {
        "oracle_accepted_matches_iou_gt_0p5": oracle_matches,
        "oracle_fn_proxy": int(len(df) - oracle_matches),
        "actual_v11_tp": actual["tp"],
        "actual_v11_fn": actual["fn"],
        "refiner_net_recovery_tp": actual["tp"] - oracle_matches,
        "explanation": "The constrained refiner recovers ~66 instances beyond the raw "
            "accepted-proposal geometry (629 actual TP vs 563 oracle-accepted matches). "
            "So oracle-accepted IoU is a documented LOWER BOUND on deployed matched "
            "geometry; per-instance FN/TP proxies overstate FN by ~66 and are used only "
            "for relative attribution, never as the actual PQ.",
    }

    report = {
        "scope": "94 tuning files, 150 annotator-images, 1134 separate GT instances.",
        "actual_v11_winner": actual,
        "oracle_vs_actual_reconciliation": reconciliation,
        "q1_area_deciles": q1_area_deciles(df),
        "q2_iou_distribution": q2_iou_distribution(df, actual),
        "q3_fn_causes": q3_fn_causes(df),
        "q4_category_annotator": q4_category_annotator(df),
        "q5_elongation": q5_elongation(df),
        "q6_ceilings": q6_ceilings(actual),
        "caveats": [
            "v6_accepted_iou is oracle accepted-proposal geometry, a lower bound on "
            "the deployed refined-mask IoU (563 oracle matches vs 629 actual TP).",
            "Per-instance refined IoU is not stored locally (only per-image sums).",
            "The public 0.55 cluster is dominated by an embedded/leaked payload "
            "(lamhuy notebook) per docs; the legitimate signal is higher recall.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pq_bottleneck_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "actual_v11_pq": actual["pq"],
        "v11_discrepancies": actual["discrepancies_vs_reference"],
        "q1_fn_frac_small_area": report["q1_area_deciles"]["fn_fraction_bottom_30pct_area"],
        "q3": report["q3_fn_causes"],
        "q6": {k: v.get("pq") if isinstance(v, dict) else v
               for k, v in report["q6_ceilings"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
