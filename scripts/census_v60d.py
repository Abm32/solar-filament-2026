"""V60D census: how big is the proposal CLOUD, and where does its duplication come from?

The tile detector has oracle recall 0.86 (union) but emits ~150 proposals/image. Before training
anything to be "less duplicative" we need to know WHICH duplication exists, because a per-tile
one-to-one assignment can only remove duplicates that come from the same (model, tile):

    within-group   : several proposals from ONE model on ONE tile hit the same GT
    cross-tile     : the same filament seen in 2-4 overlapping tiles (0.3 overlap)
    cross-model    : the 4 OOF fold models each fire on the same filament

Per (detector-set, confidence floor, mask-NMS level) we report, against the tune GT (the same
per-annotator convention as the PQ harness, GT total ~1134):
    proposals/image · unique GT covered @IoU0.5 · duplicates per covered GT (n_match-1)
    mean within-group matches per covered GT · mean matched groups per covered GT · FP/image
with FP = a proposal that matches NO GT (of any annotator) at IoU>=0.5.

Reference row: V17 with its REAL operating cut (refined score >= 0.30), not the uncut pool.

Everything is cached (outputs/census_cache) so variants are compared on identical V17/GT.
Firewall: FIT-only tile weights, V17 from signed caches, tune labels only to score. No test.

  CUDA_VISIBLE_DEVICES=N python scripts/census_v60d.py \
      --annotations <ann> --out outputs/v60d_census_base.json \
      --set base_f0=outputs/v60_tile_0/weights/v60_tile_best.pt \
      --set base_union4=w0,w1,w2,w3
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import reproduce_v17_from_cache as v50  # noqa: E402
import train_v52_cldice_semantic as v52  # noqa: E402
import train_v56_maskrcnn as v56  # noqa: E402
import v60_tiling as T  # noqa: E402

FULL = 2048
SMALL = 512
T_REFINE = 0.45
V17_SEED, V17_CELL = "seed1", "mr1_iou030"
V17_CUT = 0.30
CONFS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
NMS_LEVELS = (None, 0.5, 0.3)


# ----------------------------------------------------------------- pure metrics (numpy)
def select(score, iou_pp, conf, nms_iou):
    """Indices kept at a confidence floor, optionally greedy mask-NMS (by score)."""
    idx = np.where(score >= conf)[0]
    idx = idx[np.argsort(-score[idx], kind="stable")]
    if nms_iou is None or len(idx) == 0:
        return idx
    kept = []
    for i in idx:
        if not kept or iou_pp[i, kept].max() < nms_iou:
            kept.append(int(i))
    return np.asarray(kept, int)


def census_metrics(files, conf, nms_iou, match=0.5):
    """Aggregate the cloud metrics over files. Each file dict needs:
    score (N,), gid (N,), iou_pg (N,G), iou_pp (N,N), gt_ann (G,)."""
    n_props = n_gt = n_cov = n_fp = 0
    sum_dups = sum_within = sum_groups = 0.0
    for f in files:
        idx = select(f["score"], f["iou_pp"], conf, nms_iou)
        n_props += len(idx)
        g = f["iou_pg"].shape[1]
        n_gt += g
        if len(idx) == 0:
            continue
        hit = f["iou_pg"][idx] >= match                          # (K, G)
        n_fp += int((~hit.any(1)).sum())
        gids = f["gid"][idx]
        for j in range(g):
            m = np.where(hit[:, j])[0]
            if len(m) == 0:
                continue
            n_cov += 1
            sum_dups += len(m) - 1
            _, counts = np.unique(gids[m], return_counts=True)
            sum_within += counts.max()
            sum_groups += len(counts)
    nf = max(len(files), 1)
    cov = max(n_cov, 1)
    return {"proposals_per_image": round(n_props / nf, 2),
            "gt_total": n_gt, "gt_covered": n_cov,
            "covered_frac": round(n_cov / max(n_gt, 1), 4),
            "dups_per_covered_gt": round(sum_dups / cov, 3),
            "within_group_per_covered_gt": round(sum_within / cov, 3),
            "groups_per_covered_gt": round(sum_groups / cov, 3),
            "fp_per_image": round(n_fp / nf, 2)}


# ----------------------------------------------------------------- GPU IoU / detection
def iou_matrix(torch, device, a, b):
    """IoU between two stacks of flat bool masks (numpy (n,D),(m,D)) -> numpy (n,m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    at = torch.from_numpy(a).to(device).float()
    bt = torch.from_numpy(b).to(device).float()
    inter = at @ bt.T
    union = at.sum(1)[:, None] + bt.sum(1)[None, :] - inter
    return (inter / (union + 1e-9)).cpu().numpy()


def detect_tiles(torch, cv2, model, device, gray, tile, overlap, conf=0.05):
    """All tile detections >= conf as dicts (small bool mask, score, ox, oy)."""
    out = []
    for ox in T.tile_origins(tile, overlap):
        for oy in T.tile_origins(tile, overlap):
            crop = gray[oy:oy + tile, ox:ox + tile]
            xt = torch.from_numpy(crop.astype(np.float32) / 255.0)[None].repeat(3, 1, 1)
            with torch.no_grad(), torch.cuda.amp.autocast():
                o = model([xt.to(device)])[0]
            sc = o["scores"].cpu().numpy()
            mk = (o["masks"][:, 0].cpu().numpy() >= 0.5).astype(np.uint8)
            for s, m in zip(sc, mk, strict=True):
                if s < conf:
                    continue
                full = np.zeros((FULL, FULL), np.uint8)
                full[oy:oy + tile, ox:ox + tile] = m
                if int(full.sum()) < T.MIN_INST_AREA:
                    continue
                sm = cv2.resize(full, (SMALL, SMALL), interpolation=cv2.INTER_NEAREST)
                if sm.sum() == 0:
                    continue
                out.append({"m": np.packbits(sm.ravel()), "s": float(s), "ox": ox, "oy": oy})
    return out


def unpack(packed):
    return np.unpackbits(packed)[:SMALL * SMALL].astype(bool)


def load_models(torch, device, weights, tile):
    models = []
    for w in weights:
        ck = torch.load(w, map_location=device, weights_only=False)
        m = v56.build_model(torch, ck.get("tile", tile)).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
    return models


def collect_gt_and_v17(args, torch, cv2, device, cache_dir):
    """Tune GT (per-annotator, small bool) and V17 pool with refined scores. Cached."""
    path = cache_dir / "gt_v17.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    from pycocotools import mask as mask_utils
    rt = v50.runtime_state()
    core, tta, base = rt["core"], rt["tta"], rt["base"]
    _, tune, _, _, by_file = rt["split"]
    out = {}
    for name in sorted(tune):
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        views = v50.read_signed_views(name, "train", V17_CELL, v50.DETECTORS[V17_SEED])
        proposals, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        v17 = []
        if proposals:
            masks = [mask_utils.decode(r) for r in proposals]
            bounds, _, probs = core.crop_probabilities(base, rt["model"], gray, masks, True)
            rles, dsc = core.clean_refined_predictions(masks, scores, bounds, probs,
                                                       T_REFINE, v50.MODE)
            for r, s in zip(rles, dsc, strict=True):
                mm = mask_utils.decode(r)
                mm = mm[..., 0] if mm.ndim == 3 else mm
                sm = cv2.resize(mm.astype(np.uint8), (SMALL, SMALL),
                                interpolation=cv2.INTER_NEAREST)
                if sm.sum() >= 2:
                    v17.append({"m": np.packbits(sm.ravel()), "s": float(s), "ox": 0, "oy": 0})
        gts = []
        for ai, image_id in enumerate(sorted(by_file[name])):
            info = rt["images"][image_id]
            for e in rt["annotations"].get(image_id, []):
                rr = core.annotation_rle(info, e)
                if mask_utils.area(rr) > 0:
                    dd = mask_utils.decode(rr)
                    dd = dd[..., 0] if dd.ndim == 3 else dd
                    sm = cv2.resize(dd.astype(np.uint8), (SMALL, SMALL),
                                    interpolation=cv2.INTER_NEAREST)
                    gts.append({"m": np.packbits(sm.ravel()), "ann": ai})
        out[name] = {"v17": v17, "gt": gts}
        print(f"gt/v17 {name} v17={len(v17)} gt={len(gts)}", flush=True)
    path.write_bytes(pickle.dumps(out))
    return out


def collect_tiles(args, torch, cv2, device, cache_dir, tag, weights, names):
    path = cache_dir / f"tiles_{tag}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    models = load_models(torch, device, weights, args.tile)
    from reproduce_v17_from_cache import runtime_state
    base = runtime_state()["base"]
    out = {}
    for name in names:
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        dets = []
        for mi, m in enumerate(models):
            for d in detect_tiles(torch, cv2, m, device, gray, args.tile, args.overlap):
                d["model"] = mi
                dets.append(d)
        out[name] = dets
        print(f"tiles[{tag}] {name} n={len(dets)}", flush=True)
    path.write_bytes(pickle.dumps(out))
    return out


BLOB_FRAC = 0.5                       # a detection covering >= this fraction of a tile is a blob
TILE_SMALL_AREA = (768 * SMALL / FULL) ** 2


def build_files(torch, device, per_file_dets, gt_v17, names, v17_cut=None, drop_blobs=False):
    """Precompute IoU matrices per file for a detection source (tile dets or the V17 pool).

    drop_blobs removes whole-tile blobs (the baseline's empty-tile training artefact) so the
    cloud can be compared with and without them.
    """
    files = []
    for name in names:
        dets = per_file_dets[name]
        if v17_cut is not None:
            dets = [d for d in dets if d["s"] >= v17_cut]
        if drop_blobs:
            dets = [d for d in dets
                    if int(unpack(d["m"]).sum()) < BLOB_FRAC * TILE_SMALL_AREA]
        gts = gt_v17[name]["gt"]
        if dets:
            a = np.stack([unpack(d["m"]) for d in dets])
        else:
            a = np.zeros((0, SMALL * SMALL), bool)
        b = (np.stack([unpack(g["m"]) for g in gts]) if gts
             else np.zeros((0, SMALL * SMALL), bool))
        gmap = {}
        gid = np.asarray([gmap.setdefault((d.get("model", 0), d["ox"], d["oy"]), len(gmap))
                          for d in dets], int)
        files.append({"score": np.asarray([d["s"] for d in dets], np.float32), "gid": gid,
                      "iou_pg": iou_matrix(torch, device, a, b),
                      "iou_pp": iou_matrix(torch, device, a, a),
                      "gt_ann": np.asarray([g["ann"] for g in gts], int)})
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets",
                    help="NAME=w1[,w2,...] tile-detector weights (union if several)")
    ap.add_argument("--cache", type=Path, default=Path("outputs/census_cache"))
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=float, default=0.3)
    args = ap.parse_args()
    if v52.sha256(args.annotations) != v56.ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")

    import cv2
    import torch
    device = torch.device("cuda:0")
    args.cache.mkdir(parents=True, exist_ok=True)
    gt_v17 = collect_gt_and_v17(args, torch, cv2, device, args.cache)
    names = sorted(gt_v17)

    report = {"v17_cut": V17_CUT, "n_files": len(names), "rows": {}, "reads_test": False}
    v17_dets = {n: gt_v17[n]["v17"] for n in names}
    for label, cut in (("v17_c0.30", V17_CUT), ("v17_uncut", None)):
        files = build_files(torch, device, v17_dets, gt_v17, names, v17_cut=cut)
        report["rows"][label] = {"conf0_nms-": census_metrics(files, 0.0, None)}
    for spec in args.sets:
        tag, wl = spec.split("=", 1)
        weights = [Path(w) for w in wl.split(",")]
        dets = collect_tiles(args, torch, cv2, device, args.cache, tag, weights, names)
        rows = {}
        for variant, drop in (("", False), ("_noblob", True)):
            files = build_files(torch, device, dets, gt_v17, names, drop_blobs=drop)
            for conf in CONFS:
                for nms in NMS_LEVELS:
                    key = f"conf{conf}_nms{nms if nms is not None else '-'}{variant}"
                    rows[key] = census_metrics(files, conf, nms)
        report["rows"][tag] = rows
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: (v if k.startswith("v17") else "see file")
                      for k, v in report["rows"].items()}, indent=2))


if __name__ == "__main__":
    main()
