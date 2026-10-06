"""V60C-0 PQ test: does CLUSTER consensus-masking (not ranking) convert the tile oracle to PQ?

V60C-0 diagnostic verdict: consensus RANKING is refuted (correct proposal ~rank 5-6 under
every signal) BUT an occupancy-consensus MASK over each cluster recovers ~42% of tile-only GT
at IoU>=0.5 (tau~0.25) without needing to pick a proposal. This is the LAST cheap test before
any learned solution: collapse each near-duplicate proposal cloud into ONE consensus object
and measure whether that controls FP where V60B add-only could not (V60B gave ~1 TP per
hundreds of FP).

C0 policy (brutally narrow, NO training):
  final = V17 (trusted base, untouched)
  for each tile-proposal CLUSTER (single-link IoU>=0.10, same as the diagnostic):
      consensus = (mean of member bool masks) >= TAU          # TAU = 0.25 exactly
      if consensus empty: skip
      if cluster_conf (max member score) < conf_thr: skip
      if geometry absurd (area<15 or elong>25): skip
      if max IoU(consensus, every V17 instance) >= overlap_max: skip   # weak/absent only
      add consensus as ONE instance
  score V17-identical per-annotator PQ (512px) vs the 0.4487 bar.

KEY METRIC: dTP/dFP vs the V17-only baseline (C0's purpose is to flip V60B's ~1-TP-per-100-FP
ratio by emitting one object per cluster instead of a cloud). Also report consensus_only_TP.

Gates (user): PQ<=0.4487 kill tile branch · 0.4487<PQ<0.4517 weak->go to V60A refiner ·
PQ>=0.4517 tune tau/admission · PQ>=~0.458 serious result.

Firewall: reuses fuse_v60b.collect (V17 signed caches + FIT-only OOF tiles + SWEFil verifier).
Tune labels only to score PQ. No test, no submission, NO training.

  CUDA_VISIBLE_DEVICES=N python scripts/fuse_v60c0.py \
      --tile-weights outputs/v60_tile_{0..3}/weights/v60_tile_best.pt \
      --verifier outputs/v58_verifier_swefil/weights/verifier_best.pt \
      --annotations <ann> --out outputs/v60c0_pq.json --tile 768 --overlap 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import diagnose_v60c_consensus as C  # noqa: E402
import fuse_v60b as F  # noqa: E402
import train_v52_cldice_semantic as v52  # noqa: E402
import train_v56_maskrcnn as v56  # noqa: E402

TAU = 0.25                       # exact tau from the diagnostic (~42% recovery)
CLUSTER_IOU = C.CLUSTER_IOU      # 0.10, identical to the diagnostic
CONF_THRS = (0.0, 0.15, 0.25, 0.35, 0.5)
OVERLAP_MAX = (0.10, 0.25)


def _consensus_instances(d, tau, conf_thr):
    """One occupancy-consensus mask per tile cluster passing conf+geometry. Returns bool masks."""
    F._ensure_precompute(d)
    tiles = d["tiles"]
    if not tiles:
        return []
    tt = d["tt"]
    sc = d["tile_scores"]
    tile_masks = d["tile_masks"]
    out = []
    for comp in C._clusters(tt, CLUSTER_IOU):
        if float(sc[comp].max()) < conf_thr:          # cluster confidence = max member score
            continue
        stack = np.stack([tile_masks[j].astype(np.float32) for j in comp])
        cm = (stack.mean(0) >= tau)
        a = int(cm.sum())
        if a < 15:
            continue
        ys, xs = np.nonzero(cm)
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        if max(w, h) / max(min(w, h), 1) > 25:
            continue
        out.append(cm)
    return out


def score_c0(per_image, tau, conf_thr, overlap_max):
    """V17 base + add-only consensus instances -> V17-identical per-annotator PQ."""
    iou_sum = tp = fp = fn = 0
    cons_only_tp = 0
    added_total = 0
    for d in per_image:
        v17 = d["v17"]
        cons = _consensus_instances(d, tau, conf_thr)
        # add-only: keep consensus instance only in weak/absent V17 regions
        kept_cons = []
        for cm in cons:
            if v17:
                vmax = F._ious([cm], v17).max()
                if vmax >= overlap_max:
                    continue
            kept_cons.append(cm)
        added_total += len(kept_cons)
        kept = list(v17) + kept_cons
        n_v17 = len(v17)
        # ----- PQ (V17-identical: owned-pixel suppression, per-annotator) -----
        owned = None
        kept2, src = [], []
        for idx, m in enumerate(kept):
            if owned is None:
                owned = np.zeros_like(m)
            mm = m & ~owned
            if int(mm.sum()) < 1:
                continue
            owned |= mm
            kept2.append(mm)
            src.append("v17" if idx < n_v17 else "cons")
        kf = (np.stack([m.ravel() for m in kept2]).astype(np.float32)
              if kept2 else np.zeros((0, 1), np.float32))
        for gmasks in d["gt_sets"]:
            if not kept2 and not gmasks:
                continue
            if not kept2:
                fn += len(gmasks)
                continue
            if not gmasks:
                fp += len(kept2)
                continue
            gf = np.stack([g.ravel() for g in gmasks]).astype(np.float32)
            iou = kf @ gf.T
            iou = iou / (kf.sum(1)[:, None] + gf.sum(1)[None, :] - iou + 1e-9)
            hits = sorted(np.argwhere(iou > 0.5).tolist(),
                          key=lambda pg: -iou[pg[0], pg[1]])
            up, ug = set(), set()
            for p, g in hits:
                if p not in up and g not in ug:
                    up.add(p)
                    ug.add(g)
                    iou_sum += float(iou[p, g])
                    if src[p] == "cons":
                        cons_only_tp += 1
            tp += len(up)
            fp += len(kept2) - len(up)
            fn += len(gmasks) - len(ug)
    return {"pq": F.pq(iou_sum, tp, fp, fn), "tp": tp, "fp": fp, "fn": fn,
            "consensus_only_tp": cons_only_tp, "consensus_instances_added": added_total}


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
    baseline = F.score_fusion(per_image, conf_thr=1.01, ver_thr=0.0, overlap_max=0.0)

    results = {}
    best = (-1, None)
    for cf in CONF_THRS:
        for om in OVERLAP_MAX:
            key = f"tau{TAU}_cf{cf}_om{om}"
            r = score_c0(per_image, TAU, cf, om)
            r["delta"] = round(r["pq"] - baseline["pq"], 4)
            r["d_tp"] = r["tp"] - baseline["tp"]
            r["d_fp"] = r["fp"] - baseline["fp"]
            r["d_fn"] = r["fn"] - baseline["fn"]
            r["dtp_per_dfp"] = (round(r["d_tp"] / r["d_fp"], 4)
                                if r["d_fp"] else float("inf") if r["d_tp"] else 0.0)
            results[key] = r
            if r["pq"] > best[0]:
                best = (r["pq"], key)

    bk = best[1]
    bpq = results[bk]["pq"]
    if bpq <= 0.4487:
        verdict = "kill_tile_branch"
    elif bpq < 0.4517:
        verdict = "weak_go_to_v60a_refiner"
    elif bpq < 0.458:
        verdict = "continue_tune_tau_admission"
    else:
        verdict = "serious_result"
    report = {"baseline_v17_only": baseline, "tau": TAU, "best_policy": bk,
              "best": results[bk], "delta_vs_baseline": results[bk]["delta"],
              "pq_vs_bar_0.4487": round(bpq - 0.4487, 4), "verdict": verdict,
              "all": results, "reads_test": False}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("baseline_v17_only", "best_policy", "best",
                       "delta_vs_baseline", "pq_vs_bar_0.4487", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
