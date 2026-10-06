"""V51: fuse held-out-clean V44 YOLO11-seg proposals into the exact V17 pipeline.

V17 (tune 0.448703, public 0.39) = V16 arm_r1 seed1 @ box iou 0.3, 4-view
flip_consensus proposals -> V11 proposal_centered_e4 refiner -> t0.45_c0.30.
V50 reproduced it bit-for-bit from signed caches. V51 asks whether V44 models
(yolo11{s,m,l}-seg @2048, trained ONLY on the 566 fit files) add value on top of it.

Extra sources (declared): s = /mnt/v44 best (200 ep), m = /mnt/v44_m best,
l = /mnt/v44_l best. The V21 full-refit detector is EXCLUDED: it trained on tune.

Per extra model: 4 flip views @ imgsz 2048, conf floor 0.05, NMS iou 0.6,
retina_masks False + bilinear upsample (= V44 eval path), reduced by flip_consensus
(= V17's policy). Models in an extra set are then clustered across models with the
same cluster_views (models as "views"), so the extra score = cross-model consensus.

Fusion (declared grid, selection by TUNE PQ only):
  extra_set in {s, sml}
  alpha in {0, 0.25, 0.5}: V17 proposal score -> (1-a)*s17 + a*e, e = best
      matching (IoU>=0.5) extra-cluster score (0 if unmatched)  [FP suppression]
  beta  in {0, 0.5, 1.0}: unmatched extra clusters added with score beta*e
      (beta=0 adds none)  [recall]
  threshold x confidence = V17's frozen 6 x 7 grid.
alpha=0, beta=0 is the V17 control and must reproduce V50 counts exactly.

Gates (predeclared): PROMOTE only if tune PQ >= 0.448703 + 0.010 (E024: smaller
local gains never moved public). Audit (47 files) is opened ONCE for the single
selected candidate, only if the tune gate passes, and must be >= V17 audit 0.470763.
No test inference / CSV / submission here.

  python scripts/v51_fusion.py cache --shard N     (CUDA_VISIBLE_DEVICES=N)
  python scripts/v51_fusion.py eval  --shard N
  python scripts/v51_fusion.py merge
  python scripts/v51_fusion.py audit --shard N     (only if merge authorizes)
  python scripts/v51_fusion.py audit-merge
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
WORK = Path("/mnt/v51")
OUTPUT = ROOT / "outputs/v51_fusion"
EXTRA_MODELS = {
    "s": Path("/mnt/v44/runs/v44/weights/best.pt"),
    "m": Path("/mnt/v44_m/runs/v44/weights/best.pt"),
    "l": Path("/mnt/v44_l/runs/v44/weights/best.pt"),
}
EXTRA_SETS = {"s": ("s",), "sml": ("s", "m", "l")}
ALPHAS = (0.0, 0.25, 0.5)
BETAS = (0.0, 0.5, 1.0)
THRESHOLDS, CONFIDENCES = v50.THRESHOLDS, v50.CONFIDENCES
IMGSZ, CONF_FLOOR, NMS_IOU, MIN_AREA = 2048, 0.05, 0.6, 20
FUSE_IOU = 0.5
V17_SEED, V17_CELL = "seed1", "mr1_iou030"
V17_TUNE_PQ, V17_AUDIT_PQ = 0.44870271727943206, 0.4707634285309831
PROMOTION_DELTA = 0.010
SHARDS = 4


def key(extra_set, alpha, beta, t, c):
    return f"{extra_set}/a{alpha:.2f}_b{beta:.2f}/t{t:.2f}_c{c:.2f}"


def control_key(t, c):
    return key("s", 0.0, 0.0, t, c)


# ---------------------------------------------------------------- caching
def model_sha():
    return {name: v50.sha256(path) for name, path in EXTRA_MODELS.items()}


def cache_file(model, name):
    return WORK / "cache" / model / (name + ".json.gz")


def generate_cache(shard, group):
    from pycocotools import mask as mask_utils
    from ultralytics import YOLO

    rt = v50.runtime_state()
    _, tune, audit, _, _ = rt["split"]
    names = sorted(tune if group == "tune" else audit)[shard::SHARDS]
    shas = model_sha()
    models = {n: YOLO(str(p)) for n, p in EXTRA_MODELS.items()}
    started = time.monotonic()
    for index, name in enumerate(names, 1):
        gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for model_name, model in models.items():
            path = cache_file(model_name, name)
            if path.exists():
                continue
            views = []
            for axes in ((), (1,), (0,), (0, 1)):
                result = model.predict(source=rt["tta"].flip_array(bgr, axes), imgsz=IMGSZ,
                                       conf=CONF_FLOOR, iou=NMS_IOU, max_det=1000,
                                       retina_masks=False, verbose=False, device=0)[0]
                rles, scores = [], []
                if result.masks is not None:
                    data = result.masks.data.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy().astype(float).tolist()
                    for mask, conf in zip(data, confs, strict=True):
                        if mask.shape != (2048, 2048):
                            mask = cv2.resize(mask.astype(np.float32), (2048, 2048),
                                              interpolation=cv2.INTER_LINEAR)
                        binary = rt["tta"].flip_array((mask >= 0.5).astype(np.uint8), axes)
                        if int(binary.sum()) < MIN_AREA:
                            continue
                        rle = mask_utils.encode(np.asfortranarray(binary))
                        rles.append(dict(size=list(rle["size"]), counts=rle["counts"].decode()))
                        scores.append(conf)
                views.append(dict(rles=rles, scores=scores))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt") as stream:
                json.dump(dict(model=model_name, sha256=shas[model_name], file=name,
                               imgsz=IMGSZ, conf=CONF_FLOOR, iou=NMS_IOU, views=views), stream)
            tmp.replace(path)
        print(f"cache {group} shard={shard} {index}/{len(names)} "
              f"elapsed={time.monotonic() - started:.0f}s", flush=True)


def read_cache(model, name, expected_sha):
    with gzip.open(cache_file(model, name), "rt") as stream:
        raw = json.load(stream)
    if raw["sha256"] != expected_sha or raw["file"] != name or len(raw["views"]) != 4:
        raise RuntimeError(f"Stale/invalid V51 cache {model}/{name}")
    return [([dict(size=r["size"], counts=r["counts"].encode()) for r in v["rles"]],
             v["scores"]) for v in raw["views"]]


# ---------------------------------------------------------------- fusion
def extra_clusters(rt, name, members, shas):
    """flip_consensus per model, then cross-model consensus (models act as views)."""
    per_model = [rt["tta"].cluster_views(read_cache(m, name, shas[m]), v50.POLICY, v50.MATCH_IOU)
                 for m in members]
    if len(per_model) == 1:
        return per_model[0]
    return rt["tta"].cluster_views(per_model, v50.POLICY, v50.MATCH_IOU)


def fuse(mask_utils, v17_rles, v17_scores, extra_rles, extra_scores):
    """Return matched extra score for each V17 proposal and the unmatched extras."""
    matched = np.zeros(len(v17_rles))
    unmatched = list(range(len(extra_rles)))
    if v17_rles and extra_rles:
        iou = np.asarray(mask_utils.iou(v17_rles, extra_rles, [0] * len(extra_rles)))
        hit = iou >= FUSE_IOU
        escores = np.asarray(extra_scores)
        matched = np.where(hit.any(axis=1), (hit * escores[None, :]).max(axis=1), 0.0)
        unmatched = [j for j in range(len(extra_rles)) if not hit[:, j].any()]
    return matched, unmatched


def score_image(rt, mask_utils, name, gray, gt_by_image, shas, candidates, totals):
    core, base = rt["core"], rt["base"]
    views = v50.read_signed_views(name, "train", V17_CELL, v50.DETECTORS[V17_SEED])
    v17_rles, v17_scores = rt["tta"].cluster_views(views, v50.POLICY, v50.MATCH_IOU)
    v17_scores = np.asarray(v17_scores, dtype=float)
    for set_name, members in EXTRA_SETS.items():
        e_rles, e_scores = extra_clusters(rt, name, members, shas)
        matched, unmatched = fuse(mask_utils, list(v17_rles), v17_scores, e_rles, e_scores)
        # Refine V17 and extra proposals in separate batches so V17's crops see exactly
        # the same batch composition as V50 (keeps the control bit-identical).
        masks, bounds, probs = [], [], []
        for group in (list(v17_rles), [e_rles[j] for j in unmatched]):
            if group:
                decoded = [mask_utils.decode(r) for r in group]
                b, _, pr = core.crop_probabilities(base, rt["model"], gray, decoded, True)
                masks += decoded
                bounds += b
                probs += pr
        n17 = len(v17_rles)
        e_add = np.asarray([e_scores[j] for j in unmatched], dtype=float)
        for alpha in ALPHAS:
            s17 = (1 - alpha) * v17_scores + alpha * matched
            for beta in BETAS:
                if set_name != "s" and alpha == 0 and beta == 0:
                    continue  # identical to the control; counted once under "s"
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
    shas = model_sha()
    totals: dict[str, np.ndarray] = {}
    started = time.monotonic()
    for index, name in enumerate(names, 1):
        gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        gt_by_image = {}
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            gt = [rt["core"].annotation_rle(info, e) for e in rt["annotations"].get(image_id, [])]
            gt_by_image[image_id] = [r for r in gt if mask_utils.area(r) > 0]
        score_image(rt, mask_utils, name, gray, gt_by_image, shas, candidates, totals)
        print(f"eval {group} shard={shard} {index}/{len(names)} "
              f"elapsed={time.monotonic() - started:.0f}s", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / f"{group}_shard{shard}.json"
    out.write_text(json.dumps(dict(group=group, shard=shard, files=names, limit=limit,
                                   model_sha256=shas,
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
    return totals, parts[0]["model_sha256"]


def merge():
    totals, shas = merge_parts("tune", 94)
    ref = json.loads((v50.OUTPUT / "tune_merged.json").read_text())["totals"]
    for t in THRESHOLDS:
        for c in CONFIDENCES:
            mine = totals[control_key(t, c)]
            theirs = ref[f"{V17_SEED}/{V17_CELL}/t{t:.2f}_c{c:.2f}"]
            if abs(mine[0] - theirs[0]) > 1e-6 or list(mine[1:]) != list(theirs[1:]):
                raise RuntimeError(f"Control mismatch at t{t} c{c}: {mine} vs {theirs}")
    ranked = sorted(totals, key=lambda k: (-v50.pq(totals[k]), k))
    selected = ranked[0]
    best_pq = v50.pq(totals[selected])
    control = totals[control_key(0.45, 0.30)]
    by_policy = {}
    for k in ranked:
        policy = k.rsplit("/", 1)[0]
        by_policy.setdefault(policy, dict(key=k, pq=v50.pq(totals[k]),
                                          counts=totals[k].tolist()))
    report = dict(
        experiment="v51_v17_plus_v44_fusion", model_sha256=shas,
        control_verified=True, control_counts=control.tolist(), control_pq=v50.pq(control),
        candidates=len(totals), selected=selected, selected_counts=totals[selected].tolist(),
        selected_pq=best_pq, delta_vs_v17=best_pq - V17_TUNE_PQ,
        promotion_bar=V17_TUNE_PQ + PROMOTION_DELTA,
        audit_authorized=bool(best_pq >= V17_TUNE_PQ + PROMOTION_DELTA),
        best_per_policy=by_policy, top10=[(k, v50.pq(totals[k])) for k in ranked[:10]],
        reads_audit=False, reads_test=False, submits=False,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "tune_merged.json").write_text(json.dumps(
        dict(report=report, totals={k: v.tolist() for k, v in totals.items()}), indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "best_per_policy"}, indent=2))
    print(json.dumps({p: round(v["pq"], 6) for p, v in by_policy.items()}, indent=1))
    return report


def selected_for_audit():
    report = json.loads((OUTPUT / "tune_merged.json").read_text())["report"]
    if not report["audit_authorized"]:
        raise RuntimeError("Tune promotion gate not passed; audit stays sealed")
    return report["selected"]


def audit_merge():
    selected = selected_for_audit()
    totals, _ = merge_parts("audit", 47)
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
    for command in ("cache", "eval", "audit"):
        p = sub.add_parser(command)
        p.add_argument("--shard", type=int, required=True, choices=range(SHARDS))
        p.add_argument("--limit", type=int, default=0)
    sub.add_parser("merge")
    sub.add_parser("audit-merge")
    args = parser.parse_args()
    if args.command == "cache":
        generate_cache(args.shard, "tune")
    elif args.command == "eval":
        print(evaluate(args.shard, "tune", limit=args.limit))
    elif args.command == "merge":
        merge()
    elif args.command == "audit":
        selected = selected_for_audit()
        generate_cache(args.shard, "audit")
        print(evaluate(args.shard, "audit", candidates={selected}))
    else:
        audit_merge()


if __name__ == "__main__":
    main()
