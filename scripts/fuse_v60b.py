"""V60B: convert the V60A tiling oracle recall (union 0.8598 vs V17 0.728) into real PQ.

The oracle proved ~149 tile-only GT (small/compact, mean area ~115px) are reachable by the
native-res tile detector that the downscaled full-disk V17 never proposes. That is ORACLE
recall (GT is somewhere in a 120-200-proposal/image pool). This harness answers the only
question that matters next: can those tile-only recoveries be converted into PQ WITHOUT
drowning in FP?

DESIGN PRINCIPLE (user): do NOT let the noisy tile detector modify good V17 predictions.
V17 is the trusted base and is never altered. Tiles only ADD evidence where V17 is weak or
absent. Cleanest possible test = ADD-ONLY:

    final = V17
    for tile_candidate (sorted by score):
        if max-IoU with every V17 instance < overlap_max        (weak/absent region)
        and tile_score   > conf_thr
        and verifier_prob > ver_thr                              (verifier ONLY on tiles)
        and passes geometry prior                                (area/elong sane)
        and does not collide with an already-added tile          (cross-source NMS)
            add candidate

The SWEFil verifier is applied ONLY to the tile stream (same weak/aggressive character as
the OOF detectors where it gave +0.028; it HURT V17 -0.07, so V17 is never reranked).

PQ scored V17-identically (per-annotator greedy IoU>0.5, owned-pixel suppression, 512px) so
numbers are directly comparable to the V17 baseline 0.4455 and the bar 0.4487.

FUNNEL LOG (where oracle gain leaks): for each filter stage we log #tile-only proposals
surviving AND #distinct true tile-only GT still covered (GT diagnostic only; NEVER used for
selection). Stages: raw -> confidence -> verifier -> geometry -> cross-source-NMS -> added.

GATES (user, delta PQ over V17 baseline):
  < +0.003 oracle gain not selectable; rethink · +0.003-0.01 promising, tune fusion ·
  +0.01-0.02 strong -> bring in V60A refiner · > +0.02 breakthrough pipeline.

Firewall: V17 from signed caches, tile detectors FIT-only OOF, verifier SWEFil+FIT-OOF.
Tune labels only to score PQ + the funnel GT diagnostic. No test, no submission.

  CUDA_VISIBLE_DEVICES=N python scripts/fuse_v60b.py \
      --tile-weights outputs/v60_tile_0/weights/v60_tile_best.pt ... \
      --verifier outputs/v58_verifier_swefil/weights/verifier_best.pt \
      --out outputs/v60b_fusion.json --tile 768 --overlap 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import build_v58_oof_candidates as B  # noqa: E402
import reproduce_v17_from_cache as v50  # noqa: E402
import train_v52_cldice_semantic as v52  # noqa: E402
import train_v56_maskrcnn as v56  # noqa: E402
import train_v58_verifier as V  # noqa: E402
import v60_tiling as T  # noqa: E402

FULL = 2048
SMALL = 512
V17_SEED, V17_CELL = "seed1", "mr1_iou030"
T_REFINE = 0.45

# sweep grids (kept small; add-only is the primary experiment)
CONF_THRS = (0.15, 0.25, 0.35, 0.50)
VER_THRS = (0.0, 0.5, 0.7)              # 0.0 = verifier off (isolate its contribution)
OVERLAP_MAX = (0.10, 0.25)             # "weak/absent region" definition (max IoU vs any V17)
NMS_IOU = 0.5                          # cross-source tile-vs-tile suppression


def pq(iou_sum, tp, fp, fn):
    d = tp + 0.5 * (fp + fn)
    return float(iou_sum / d) if d else 0.0


def _ious(a_list, b_list):
    """IoU matrix between two lists of flat bool masks; (len_a, len_b)."""
    if not a_list or not b_list:
        return np.zeros((len(a_list), len(b_list)), np.float32)
    af = np.stack([a.ravel() for a in a_list]).astype(np.float32)
    bf = np.stack([b.ravel() for b in b_list]).astype(np.float32)
    inter = af @ bf.T
    return inter / (af.sum(1)[:, None] + bf.sum(1)[None, :] - inter + 1e-9)


def tile_detect_scored(torch, cv2, model, device, gray, tile, overlap, conf=0.05):
    """Like v60_tiling.tile_detect_global but keeps per-proposal score + box geometry."""
    out = []
    for ox in T.tile_origins(tile, overlap):
        for oy in T.tile_origins(tile, overlap):
            crop = gray[oy:oy + tile, ox:ox + tile]
            xt = (torch.from_numpy(crop.astype(np.float32) / 255.0)[None]
                  .repeat(3, 1, 1).to(device))
            with torch.no_grad(), torch.cuda.amp.autocast():
                o = model([xt])[0]
            sc = o["scores"].cpu().numpy()
            mk = (o["masks"][:, 0].cpu().numpy() >= 0.5).astype(np.uint8)
            for s, m in zip(sc, mk, strict=True):
                if s < conf:
                    continue
                full = np.zeros((FULL, FULL), np.uint8)
                full[oy:oy + tile, ox:ox + tile] = m
                if int(full.sum()) < T.MIN_INST_AREA:
                    continue
                small = cv2.resize(full, (SMALL, SMALL),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                ys, xs = np.nonzero(small)
                if len(xs) == 0:
                    continue
                w = xs.max() - xs.min() + 1
                h = ys.max() - ys.min() + 1
                out.append({"mask": small, "score": float(s),
                            "area": int(small.sum()),
                            "elong": max(w, h) / max(min(w, h), 1)})
    return out


def collect(args):
    """Per tune file: V17 kept instances (+scores) + scored tile proposals + GT sets."""
    import cv2
    import torch
    from pycocotools import mask as mask_utils

    rt = v50.runtime_state()
    core, tta, base = rt["core"], rt["tta"], rt["base"]
    _, tune, _, _, by_file = rt["split"]
    device = torch.device("cuda:0")

    tile_models = []
    for wk in args.tile_weights:
        ck = torch.load(wk, map_location=device, weights_only=False)
        m = v56.build_model(torch, ck.get("tile", args.tile)).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        tile_models.append(m)

    # verifier (optional; only scores the tile stream)
    vmodel = vmean = vstd = vkeys = None
    if args.verifier is not None:
        vck = torch.load(args.verifier, map_location=device, weights_only=False)
        vmodel = V.build_verifier(torch, len(vck["feature_keys"])).to(device)
        vmodel.load_state_dict(vck["model"])
        vmodel.eval()
        vmean, vstd = np.asarray(vck["mean"]), np.asarray(vck["std"])
        vkeys = vck["feature_keys"]

    per_image = []
    for name in sorted(tune):
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        disk = B.disk_params(gray, np)
        # ----- V17 trusted base (refined + scored) -----
        views = v50.read_signed_views(name, "train", V17_CELL, v50.DETECTORS[V17_SEED])
        proposals, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        v17_masks, v17_scores = [], []
        if proposals:
            masks = [mask_utils.decode(r) for r in proposals]
            bounds, _, probs = core.crop_probabilities(base, rt["model"], gray, masks, True)
            rles, dsc = core.clean_refined_predictions(masks, scores, bounds, probs,
                                                       T_REFINE, v50.MODE)
            for r, s in zip(rles, dsc, strict=True):
                mm = mask_utils.decode(r)
                if mm.ndim == 3:
                    mm = mm[..., 0]
                sm = cv2.resize(mm, (SMALL, SMALL),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                if int(sm.sum()) >= 2:
                    v17_masks.append(sm)
                    v17_scores.append(float(s))
        # ----- tile proposal stream (scored) -----
        tiles = []
        for m in tile_models:
            tiles += tile_detect_scored(torch, cv2, m, device, gray, args.tile, args.overlap)
        # verifier prob for each tile proposal (crop+feature extraction = training-identical)
        if vmodel is not None and tiles:
            crops, feats = [], []
            for tp_ in tiles:
                # recover a full-res box from the small mask for context cropping
                ys, xs = np.nonzero(tp_["mask"])
                sx = FULL / SMALL
                box = [float(xs.min()) * sx, float(ys.min()) * sx,
                       (float(xs.max()) + 1) * sx, (float(ys.max()) + 1) * sx]
                full = cv2.resize(tp_["mask"].astype(np.uint8), (FULL, FULL),
                                  interpolation=cv2.INTER_NEAREST)
                ft = B.candidate_features(np, box, tp_["score"], full, gray, disk)
                feats.append([ft[kk] for kk in vkeys])
                cxb, cyb = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                half = max(box[2] - box[0], box[3] - box[1]) * B.CONTEXT / 2
                a0, b0 = int(max(0, cxb - half)), int(max(0, cyb - half))
                c0, d0 = int(min(FULL, cxb + half)), int(min(FULL, cyb + half))
                cr = gray[b0:d0, a0:c0]
                cr = (cv2.resize(cr, (B.CROP, B.CROP), interpolation=cv2.INTER_AREA)
                      if cr.size else np.zeros((B.CROP, B.CROP), np.uint8))
                crops.append(cr.astype(np.uint8))
            ci = torch.tensor(np.stack(crops), dtype=torch.float32)[:, None]
            ci = (ci / 255.0).repeat(1, 3, 1, 1).to(device)
            fz = torch.tensor((np.asarray(feats) - vmean) / vstd,
                              dtype=torch.float32).to(device)
            with torch.no_grad():
                vs = torch.sigmoid(vmodel(ci, fz)).cpu().numpy()
            for tp_, vprob in zip(tiles, vs, strict=True):
                tp_["ver"] = float(vprob)
        else:
            for tp_ in tiles:
                tp_["ver"] = 1.0        # verifier off -> never blocks
        # ----- GT sets (per annotator) + tile-only GT tagging (diagnostic only) -----
        gt_sets, all_gt = [], []
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            gm = []
            for e in rt["annotations"].get(image_id, []):
                r = core.annotation_rle(info, e)
                if mask_utils.area(r) > 0:
                    d = mask_utils.decode(r)
                    if d.ndim == 3:
                        d = d[..., 0]
                    sm = cv2.resize(d, (SMALL, SMALL),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                    gm.append(sm)
                    all_gt.append(sm)
            gt_sets.append(gm)
        # which GT are tile-only reachable (tile hits >=0.5, V17 misses <0.5) — DIAGNOSTIC
        tile_only_gt = _tile_only_gt_mask(v17_masks, tiles, all_gt)
        # ---- precompute packed masks + IoU matrices ONCE (hot-path optimization) ----
        tile_masks = [t["mask"] for t in tiles]
        tile_scores = np.asarray([t["score"] for t in tiles], np.float32)
        tile_ver = np.asarray([t["ver"] for t in tiles], np.float32)
        tile_area = np.asarray([t["area"] for t in tiles], np.float32)
        tile_elong = np.asarray([t["elong"] for t in tiles], np.float32)
        tv = _ious(tile_masks, v17_masks)      # (n_tile, n_v17) tile-vs-V17
        tt = _ious(tile_masks, tile_masks)     # (n_tile, n_tile) tile-vs-tile (NMS)
        per_image.append({"v17": v17_masks, "v17s": np.asarray(v17_scores),
                          "tiles": tiles, "gt_sets": gt_sets,
                          "all_gt": all_gt, "tile_only_gt": tile_only_gt,
                          "tile_masks": tile_masks, "tile_scores": tile_scores,
                          "tile_ver": tile_ver, "tile_area": tile_area,
                          "tile_elong": tile_elong, "tv": tv, "tt": tt})
        print(f"collected {name} v17={len(v17_masks)} tiles={len(tiles)} "
              f"tile_only_gt={int(tile_only_gt.sum())}", flush=True)
    return per_image


def _tile_only_gt_mask(v17_masks, tiles, all_gt):
    """Boolean per-GT: reachable by a tile (IoU>=0.5) AND missed by V17 (<0.5). Diagnostic."""
    if not all_gt:
        return np.zeros(0, bool)
    tv = _ious([t["mask"] for t in tiles], all_gt)
    vv = _ious(v17_masks, all_gt)
    best_t = tv.max(0) if tv.size else np.zeros(len(all_gt))
    best_v = vv.max(0) if vv.size else np.zeros(len(all_gt))
    return (best_t >= 0.5) & (best_v < 0.5)


def _ensure_precompute(d):
    """Fill tv/tt/tile_* on a per-image dict if absent (keeps unit tests simple)."""
    if "tv" in d:
        return d
    tiles = d["tiles"]
    tile_masks = [t["mask"] for t in tiles]
    d["tile_masks"] = tile_masks
    d["tile_scores"] = np.asarray([t["score"] for t in tiles], np.float32)
    d["tile_ver"] = np.asarray([t["ver"] for t in tiles], np.float32)
    d["tile_area"] = np.asarray([t["area"] for t in tiles], np.float32)
    d["tile_elong"] = np.asarray([t["elong"] for t in tiles], np.float32)
    d["tv"] = _ious(tile_masks, d["v17"])
    d["tt"] = _ious(tile_masks, tile_masks)
    return d


def _select_added(d, conf_thr, ver_thr, overlap_max):
    """Add-only greedy selection using PRECOMPUTED IoU matrices. Returns list of tile idx."""
    _ensure_precompute(d)
    tv, tt = d["tv"], d["tt"]
    sc, ve = d["tile_scores"], d["tile_ver"]
    ar, el = d["tile_area"], d["tile_elong"]
    n_v17 = len(d["v17"])
    order = np.argsort(-sc)
    added = []
    for i in order:
        if sc[i] < conf_thr or ve[i] < ver_thr:
            continue
        if ar[i] < 15 or el[i] > 25:
            continue
        # add-only: weak/absent region (low IoU vs every V17 instance)
        if n_v17 and tv[i].max() >= overlap_max:
            continue
        # cross-source NMS vs already-added tiles
        if added and tt[i, added].max() >= NMS_IOU:
            continue
        added.append(int(i))
    return added


def score_fusion(per_image, conf_thr, ver_thr, overlap_max):
    """Add-only fusion -> V17-identical per-annotator PQ (precomputed IoU hot path)."""
    iou_sum = tp = fp = fn = 0
    for d in per_image:
        added = _select_added(d, conf_thr, ver_thr, overlap_max)
        kept = list(d["v17"]) + [d["tile_masks"][i] for i in added]
        # ----- PQ (V17-identical: owned-pixel suppression, per-annotator) -----
        owned = None
        kept2 = []
        for m in kept:
            if owned is None:
                owned = np.zeros_like(m)
            mm = m & ~owned
            if int(mm.sum()) < 1:
                continue
            owned |= mm
            kept2.append(mm)
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
            tp += len(up)
            fp += len(kept2) - len(up)
            fn += len(gmasks) - len(ug)
    return {"pq": pq(iou_sum, tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}


def build_funnel(per_image, conf_thr, ver_thr, overlap_max):
    """Per-image funnel: #tile proposals + #distinct tile-only GT covered at each stage.

    Uses precomputed arrays. tile-vs-GT IoU computed once per image here (diagnostic only).
    """
    stages = ["raw", "conf", "ver", "geom", "nms"]
    prop = dict.fromkeys(stages, 0)
    gtcov = {s: 0 for s in stages}
    for d in per_image:
        _ensure_precompute(d)
        tog = d["tile_only_gt"]
        all_gt = d["all_gt"]
        tiles = d["tiles"]
        tvg = (_ious(d["tile_masks"], all_gt) if tiles and len(all_gt) else None)
        sc, ve = d["tile_scores"], d["tile_ver"]
        ar, el, tv, tt = d["tile_area"], d["tile_elong"], d["tv"], d["tt"]
        n_v17 = len(d["v17"])
        surviving = {s: set() for s in stages}
        added = []
        for i in np.argsort(-sc):
            i = int(i)
            surviving["raw"].add(i)
            if sc[i] < conf_thr:
                continue
            surviving["conf"].add(i)
            if ve[i] < ver_thr:
                continue
            surviving["ver"].add(i)
            if ar[i] < 15 or el[i] > 25:
                continue
            surviving["geom"].add(i)
            if n_v17 and tv[i].max() >= overlap_max:
                continue
            if added and tt[i, added].max() >= NMS_IOU:
                continue
            surviving["nms"].add(i)
            added.append(i)
        for s in stages:
            prop[s] += len(surviving[s])
            if tvg is not None and tog.any():
                covered = set()
                for i in surviving[s]:
                    gi = np.argwhere((tvg[i] >= 0.5) & tog).ravel()
                    covered.update(gi.tolist())
                gtcov[s] += len(covered)
    return {"proposals": prop, "tile_only_gt_covered": gtcov}


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

    per_image = collect(args)

    # V17-only baseline (no tiles added) = conf_thr above max tile score
    baseline = score_fusion(per_image, conf_thr=1.01, ver_thr=0.0, overlap_max=0.0)

    results = {}
    best = (-1, None)
    for cf in CONF_THRS:
        for vt in VER_THRS:
            for om in OVERLAP_MAX:
                key = f"cf{cf}_vt{vt}_om{om}"
                r = score_fusion(per_image, cf, vt, om)
                r["delta"] = round(r["pq"] - baseline["pq"], 4)
                results[key] = r
                if r["pq"] > best[0]:
                    best = (r["pq"], key)

    bk = best[1]
    # parse best key for funnel
    cf = float(bk.split("cf")[1].split("_")[0])
    vt = float(bk.split("vt")[1].split("_")[0])
    om = float(bk.split("om")[1])
    funnel = build_funnel(per_image, cf, vt, om)

    delta = results[bk]["delta"]
    if delta < 0.003:
        verdict = "not_selectable"
    elif delta < 0.01:
        verdict = "promising"
    elif delta < 0.02:
        verdict = "strong_bring_refiner"
    else:
        verdict = "breakthrough"
    report = {"baseline_v17_only": baseline, "best_policy": bk, "best": results[bk],
              "delta": delta, "verdict": verdict, "funnel_at_best": funnel,
              "all": results, "reads_test": False}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("baseline_v17_only", "best_policy", "best", "delta",
                       "verdict", "funnel_at_best")}, indent=2))


if __name__ == "__main__":
    main()
