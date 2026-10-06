"""V60C-0: is the oracle gain recoverable by CONSENSUS instead of greedy confidence-NMS?

V60B proved the add-only funnel keeps 108/149 tile-only GT through confidence+verifier+
geometry, then cross-source NMS collapses 108 -> 1. The correct mask EXISTS in the surviving
pool; greedy max-confidence NMS just picks a spurious neighbor. The tile detector sees
PARTIAL VIEWS of the same filament from many overlapping windows -> single-image confidence
is the wrong ranking signal; MULTI-VIEW CONSENSUS may be the right one.

This is the DECISIVE CHEAP DIAGNOSTIC (no learned model). For every tile-only GT (reachable
by a tile @IoU0.5, missed by V17), take its CLUSTER of overlapping tile proposals and ask:

 (1) ORACLE-RANK: the "correct" proposal = argmax IoU-to-GT in the cluster. What is its rank
     within the cluster under
        - detector confidence   (what NMS uses)
        - consensus score        (mean IoU-agreement with overlapping cluster neighbors)
     If consensus ranks it ~1-2 while confidence buries it ~10-20, the information we threw
     away is recoverable -> justifies C0/C1. If not, abandon the localization-ranker line.

 (2) C0 CONSENSUS MASK: per cluster, pixel-wise occupancy consensus (fraction of cluster
     members covering each pixel) thresholded at tau; measure its IoU-to-GT vs the best single
     proposal. Does an unlearned consensus mask reconstruct the filament?

Clusters are formed on the TILE-ONLY proposal pool per image by single-link overlap
(IoU >= CLUSTER_IOU), then kept only if the cluster's best proposal matches a tile-only GT.

Firewall: reuses fuse_v60b.collect (V17 signed caches + FIT-only OOF tile detectors + SWEFil
verifier). Tune labels used only to define "correct proposal" + score the diagnostic. No test,
no submission.

  CUDA_VISIBLE_DEVICES=N python scripts/diagnose_v60c_consensus.py \
      --tile-weights outputs/v60_tile_{0..3}/weights/v60_tile_best.pt \
      --verifier outputs/v58_verifier_swefil/weights/verifier_best.pt \
      --annotations <ann> --out outputs/v60c0.json --tile 768 --overlap 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import fuse_v60b as F  # noqa: E402
import train_v52_cldice_semantic as v52  # noqa: E402
import train_v56_maskrcnn as v56  # noqa: E402

CLUSTER_IOU = 0.10          # single-link overlap to group partial-view proposals
CONSENSUS_TAUS = (0.2, 0.3, 0.4, 0.5)
GT_MATCH = 0.5


def _clusters(tt, thr):
    """Single-link connected components over tile-tile IoU >= thr. Returns list of idx lists."""
    n = tt.shape[0]
    seen = np.zeros(n, bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            nb = np.where((tt[u] >= thr) & (~seen))[0]
            for v in nb:
                seen[v] = True
                stack.append(int(v))
        comps.append(comp)
    return comps


def _rank_of(values, target_idx_in_cluster, cluster):
    """Rank (1=best) of target within cluster when sorting cluster by values desc."""
    order = sorted(cluster, key=lambda i: -values[i])
    return order.index(target_idx_in_cluster) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile-weights", type=Path, nargs="+", required=True)
    ap.add_argument("--verifier", type=Path, default=None)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=float, default=0.3)
    args = ap.parse_args()
    if v52.sha256(args.annotations) != v56.ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")

    per_image = F.collect(args)

    conf_ranks, cons_ranks, ver_ranks = [], [], []
    best_single_iou, consensus_iou = [], {t: [] for t in CONSENSUS_TAUS}
    n_gt = n_clusters = 0
    for d in per_image:
        F._ensure_precompute(d)
        tiles = d["tiles"]
        if not tiles:
            continue
        tt = d["tt"]
        sc, ve = d["tile_scores"], d["tile_ver"]
        tog = d["tile_only_gt"]
        all_gt = d["all_gt"]
        if not len(all_gt) or not tog.any():
            continue
        tile_masks = d["tile_masks"]
        tvg = F._ious(tile_masks, all_gt)              # (n_tile, n_gt)
        # consensus score per tile = mean IoU with its overlapping neighbors (>=CLUSTER_IOU)
        cons = np.zeros(len(tiles), np.float32)
        for i in range(len(tiles)):
            nb = tt[i] >= CLUSTER_IOU
            nb[i] = False
            cons[i] = float(tt[i][nb].mean()) if nb.any() else 0.0
        tog_idx = set(np.where(tog)[0].tolist())
        for comp in _clusters(tt, CLUSTER_IOU):
            # which tile-only GT does this cluster's best proposal correspond to?
            sub = tvg[comp]                            # (|comp|, n_gt)
            gi = int(sub.max(0).argmax())              # GT best-covered by the cluster
            if gi not in tog_idx or sub[:, gi].max() < GT_MATCH:
                continue
            n_clusters += 1
            n_gt += 1
            # "correct" proposal = cluster member with best IoU to this GT
            local_best = comp[int(sub[:, gi].argmax())]
            best_single_iou.append(float(sub[:, gi].max()))
            conf_ranks.append(_rank_of(sc, local_best, comp))
            cons_ranks.append(_rank_of(cons, local_best, comp))
            if ve is not None:
                ver_ranks.append(_rank_of(ve, local_best, comp))
            # C0 consensus mask over the cluster
            stack = np.stack([tile_masks[j].astype(np.float32) for j in comp])
            occ = stack.mean(0)                        # fraction of members covering each pixel
            for tau in CONSENSUS_TAUS:
                cm = occ >= tau
                inter = float((cm & all_gt[gi]).sum())
                union = float((cm | all_gt[gi]).sum())
                consensus_iou[tau].append(inter / union if union else 0.0)

    def med(x):
        return float(np.median(x)) if x else 0.0

    def frac_top2(x):
        return float(np.mean([r <= 2 for r in x])) if x else 0.0

    report = {
        "n_tile_only_clusters": n_clusters,
        "rank_of_correct_proposal_within_cluster": {
            "confidence": {"median": med(conf_ranks), "frac_top2": frac_top2(conf_ranks)},
            "consensus": {"median": med(cons_ranks), "frac_top2": frac_top2(cons_ranks)},
            "verifier": {"median": med(ver_ranks), "frac_top2": frac_top2(ver_ranks)},
        },
        "best_single_proposal_iou_to_gt_median": med(best_single_iou),
        "c0_consensus_mask_iou_to_gt_median": {str(t): med(consensus_iou[t])
                                               for t in CONSENSUS_TAUS},
        "c0_consensus_recovers_at_iou0.5": {
            str(t): float(np.mean([v >= 0.5 for v in consensus_iou[t]]))
            if consensus_iou[t] else 0.0 for t in CONSENSUS_TAUS},
        "reads_test": False,
    }
    # verdict: does consensus clearly out-rank confidence for the correct proposal?
    cm_, fm_ = report["rank_of_correct_proposal_within_cluster"]["consensus"]["median"], \
        report["rank_of_correct_proposal_within_cluster"]["confidence"]["median"]
    best_c0 = max(report["c0_consensus_recovers_at_iou0.5"].values())
    if cm_ and fm_ and cm_ <= fm_ - 3:
        verdict = "consensus_ranks_better_build_c1"
    elif best_c0 >= 0.4:
        verdict = "c0_mask_recovers_build_c0"
    else:
        verdict = "consensus_no_signal_abandon_line"
    report["verdict"] = verdict
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
