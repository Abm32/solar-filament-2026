"""V100 stage 1 (CPU): the V84 soft extract (real refined pipeline, per-candidate features, per-record IoU) plus a
5-channel 128 px crop per candidate for an image-based selector. Reproduces outputs/v84/extract_soft.pkl per-file
counts exactly (asserted). Train images only; never reads test.

Per candidate (t0.50 cell, score >= v83 FLOOR 0.15), crop = the refiner's own square box (context 1.8, min 96):
  ch0 gray, ch1 refiner probability (4-flip TTA), ch2 candidate mask after ownership, ch3 union of the other
  alive candidates' masks, ch4 detector seed mask. All uint8 0..255, resized to CROP px (nearest for masks).
  CUDA_VISIBLE_DEVICES= PROCS=22 python scripts/v100_extract.py [limit]   -> outputs/v100/extract.pkl
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from multiprocessing import Pool, set_start_method
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import audit_v80_extract as E  # noqa: E402
import train_v73_refiner as V73  # noqa: E402
import v83_extract as X  # noqa: E402
import v84_extract as X84  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v100"
CROP, T = 128, 0.50


def crops_for(gray, P_list, post, alive, cand, bounds, masks):
    import cv2
    from pycocotools import mask as mu

    dec = {i: mu.decode(post[i]) for i in alive}
    out = np.zeros((len(cand), 5, CROP, CROP), np.uint8)
    for k, c in enumerate(cand):
        i = alive[c]
        x0, y0, x1, y1 = bounds[i]
        others = np.zeros_like(dec[i])
        for j in alive:
            if j != i:
                others |= dec[j]
        P = cv2.resize(P_list[i], (x1 - x0, y1 - y0))
        chans = (gray[y0:y1, x0:x1], np.clip(P * 255, 0, 255).astype(np.uint8), dec[i][y0:y1, x0:x1] * 255,
                 others[y0:y1, x0:x1] * 255, masks[i][y0:y1, x0:x1] * 255)
        for ch, a in enumerate(chans):
            out[k, ch] = cv2.resize(a, (CROP, CROP), interpolation=cv2.INTER_AREA if ch < 2 else cv2.INTER_NEAREST)
    return out


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = E.RT, E.RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    recs = []
    for image_id in sorted(rt["by_file"][name], key=str):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        recs.append([r for r in rl if mu.area(r) > 0])
    views = V73.R.K.read_cache_views(V73.R.O.cache_path(V73.RECIPE, fold, name))
    groups = E.groups_with_ids(views)
    scores = [g["score"] for g in groups]
    disk = E.disk_fit(gray)
    masks, bounds, probs = [], [], []
    if groups:
        masks = [mu.decode(g["rle"]) for g in groups]
        bounds, _, probs = core.crop_probabilities(rt["base"], E.MODELS[fold], gray, masks, True)
    X.T = T
    post, feats = X.refine_feats(masks, scores, bounds, probs, gray, disk) if groups else ([], [])
    alive = [i for i in range(len(groups)) if post[i] is not None]
    keep = [c for c, i in enumerate(alive) if scores[i] >= X.CUT]
    I_all = [E.iou_mat(g, [post[i] for i in alive]) for g in recs]
    tot = np.zeros(4)
    for ri, g in enumerate(recs):
        tot += core.pq_counts(I_all[ri][:, keep] if g else np.zeros((0, len(keep))), len(g), len(keep))
    cand, rows = X.candidate_rows(groups, scores, alive, feats)
    for c, f in zip(cand, rows):
        f["area"] = float(mu.area(post[alive[c]]))
    iou = [I_all[ri][:, cand] if g else np.zeros((0, len(cand))) for ri, g in enumerate(recs)]
    crops = crops_for(gray, probs, post, alive, cand, bounds, masks) if cand else np.zeros((0, 5, CROP, CROP), np.uint8)
    return dict(name=name, fold=fold, tot=tot.tolist(), rows=rows, iou=iou, n_gt=[len(g) for g in recs],
                scores=[scores[alive[c]] for c in cand], crops=crops)


def main():
    import torch
    set_start_method("fork")
    V73.DEV = "cpu"
    E.RT = V73.runtime()
    for f in range(5):
        m = V73.build_model(E.RT["base"], "imagenet", "resnet18")
        m.load_state_dict(torch.load(ROOT / (X84.CKPT["soft"].format(f) + "_refiner.pt"), map_location="cpu",
                                     weights_only=True)["model"])
        E.MODELS[f] = m.eval()
    fmap = V73.R.fold_of_file()
    names = sorted(fmap)
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if lim:
        names = names[:: max(1, len(names) // lim)][:lim]
    ref = {r["name"]: r for r in pickle.load(open(ROOT / "outputs/v84/extract_soft.pkl", "rb"))[0.5]}
    with torch.no_grad(), Pool(int(os.environ.get("PROCS", "22")), initializer=E.init) as p:
        res = p.map(work, [(n, fmap[n]) for n in names], chunksize=1)
    worst = max(max(abs(a - b) for a, b in zip(r["tot"], ref[r["name"]]["tot"])) for r in res)
    nrow = sum(len(r["rows"]) for r in res)
    print("files", len(res), "max file count diff vs extract_soft", worst, "candidates", nrow,
          "rows match", all(len(r["rows"]) == len(ref[r["name"]]["rows"]) for r in res), flush=True)
    assert worst <= 1e-6, "pipeline reproduction failed"
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / f"extract{'_smoke' if lim else ''}.pkl", "wb"), protocol=4)
    json.dump({"files": len(res), "candidates": nrow, "crop": CROP, "t": T, "max_file_diff": worst},
              open(OUT / f"extract{'_smoke' if lim else ''}.json", "w"))


if __name__ == "__main__":
    main()
