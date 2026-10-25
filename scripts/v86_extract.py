"""V86 stage 1 (CPU): V84 soft/t0.50 refined candidates for every source (v84_extract path), with post-ownership
RLEs and GT RLEs kept for the pooled selection in stage 2.
Pre-registration: docs/v86_pooled_selector_preregistration_2026-10-12.md
  CUDA_VISIBLE_DEVICES= python scripts/v86_extract.py [limit]
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

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v86"
SOURCES = ["p2first", "p2first_1280", "baseline", "p2first_cp", "baseline_seed2"]
T = 0.50


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = E.RT, E.RT["core"]
    X.T = T
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    recs = []
    for image_id in sorted(rt["by_file"][name], key=str):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        recs.append([r for r in rl if mu.area(r) > 0])
    disk = E.disk_fit(gray)
    out = dict(name=name, fold=fold, gts=recs, n_gt=[len(g) for g in recs], src={})
    for src in SOURCES:
        views = V73.R.K.read_cache_views(V73.R.O.cache_path(src, fold, name))
        groups = E.groups_with_ids(views)
        scores = [g["score"] for g in groups]
        if groups:
            masks = [mu.decode(g["rle"]) for g in groups]
            bounds, _, probs = core.crop_probabilities(rt["base"], E.MODELS[fold], gray, masks, True)
            post, feats = X.refine_feats(masks, scores, bounds, probs, gray, disk)
        else:
            post, feats = [], []
        alive = [i for i in range(len(groups)) if post[i] is not None]
        keep = [c for c, i in enumerate(alive) if scores[i] >= X.CUT]
        I_all = [E.iou_mat(g, [post[i] for i in alive]) for g in recs]
        tot = np.zeros(4)
        for ri, g in enumerate(recs):
            tot += core.pq_counts(I_all[ri][:, keep] if g else np.zeros((0, len(keep))), len(g), len(keep))
        cand, rows = X.candidate_rows(groups, scores, alive, feats)
        for c, f in zip(cand, rows):
            f["area"] = float(mu.area(post[alive[c]]))
        out["src"][src] = dict(tot=tot.tolist(), rows=rows, rles=[post[alive[c]] for c in cand],
                               iou=[I_all[ri][:, cand] if g else np.zeros((0, len(cand))) for ri, g in enumerate(recs)],
                               scores=[scores[alive[c]] for c in cand])
    return out


def main():
    import torch
    set_start_method("fork")
    V73.DEV = "cpu"
    E.RT = V73.runtime()
    for f in range(5):
        m = V73.build_model(E.RT["base"], "imagenet", "resnet18")
        m.load_state_dict(torch.load(ROOT / f"outputs/v84/soft_fold{f}_refiner.pt", map_location="cpu",
                                     weights_only=True)["model"])
        E.MODELS[f] = m.eval()
    fmap = V73.R.fold_of_file()
    names = sorted(fmap)
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if lim:
        names = names[:: max(1, len(names) // lim)][:lim]
    ref = {}
    for f in range(5):
        ref.update(json.loads((ROOT / f"outputs/v84/soft_fold{f}.json").read_text())["per_file"])
    with torch.no_grad(), Pool(int(os.environ.get("V86_PROCS", "48")), initializer=E.init) as p:
        res = p.map(work, [(n, fmap[n]) for n in names], chunksize=1)
    diffs = [max(abs(a - b) for a, b in zip(r["src"]["p2first"]["tot"], ref[r["name"]]["k7"]["t0.50_c0.30"]))
             for r in res]
    for s in SOURCES:
        tot = np.sum([r["src"][s]["tot"] for r in res], 0)
        print(s, "files", len(res), "pq@c0.30", V73.R.K.pq_value(tot),
              "cands", sum(len(r["src"][s]["rows"]) for r in res), flush=True)
    print("p2first max file diff vs v84 soft t0.50_c0.30", max(diffs), flush=True)
    assert max(diffs) <= 1e-3, "pipeline reproduction failed"
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / f"extract{'_smoke' if lim else ''}.pkl", "wb"))


if __name__ == "__main__":
    main()
