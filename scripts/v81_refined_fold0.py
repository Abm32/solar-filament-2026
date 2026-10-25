"""V81 refined-pipeline gate (CPU only): p2first vs p2first_thin views on the 141 fold-0 files.
views -> flip consensus (A.groups_with_ids) -> V73 base fold-0 refiner (k7, t0.45) -> cut 0.30 -> per-record PQ.
Reuses audit_v80_extract (which reproduced outputs/v73/base_fold*.json). Asserts p2first == V73 base fold 0.
  CUDA_VISIBLE_DEVICES= python scripts/v81_refined_fold0.py
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool, set_start_method
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_v80_extract as A  # noqa: E402
import v71_suppress as S  # noqa: E402

V73, ROOT = A.V73, A.ROOT
RECIPES = ["p2first", "p2first_thin"]


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, recipe = job
    rt, core = A.RT, A.RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    recs = []
    for image_id in sorted(rt["by_file"][name], key=str):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        recs.append([r for r in rl if mu.area(r) > 0])
    views = V73.R.K.read_cache_views(V73.R.O.cache_path(recipe, 0, name))
    groups = A.groups_with_ids(views)
    scores = [g["score"] for g in groups]
    post = []
    if groups:
        masks = [mu.decode(g["rle"]) for g in groups]
        bounds, _, probs = core.crop_probabilities(rt["base"], A.MODELS[0], gray, masks, True)
        _, post = A.refine_instrumented(masks, scores, bounds, probs)
    keep = [post[i] for i in range(len(groups)) if post[i] is not None and scores[i] >= A.CUT]
    ka = [float(mu.area(r)) for r in keep]
    out, ratios = {}, []
    for tag, mn in (("all", 0), ("min200", 200)):
        kl = [r for r, a in zip(keep, ka) if a >= mn]
        tot = np.zeros(4)
        for g in recs:
            ik = A.iou_mat(g, kl)
            tot += core.pq_counts(ik if kl else np.zeros((len(g), 0)), len(g), len(kl))
            if tag == "all" and g and kl:
                for gi, j in zip(*np.where(ik > 0.5)):
                    ratios.append(ka[j] / float(mu.area(g[gi])))
        out[tag] = tot.tolist()
    return name, recipe, out, ratios, ka


def main():
    import torch
    set_start_method("fork")
    V73.DEV = "cpu"
    A.RT = V73.runtime()
    m = V73.build_model(A.RT["base"], "imagenet", "resnet18")
    m.load_state_dict(torch.load(ROOT / "outputs/v73/base_fold0_refiner.pt", map_location="cpu",
                                 weights_only=True)["model"])
    A.MODELS[0] = m.eval()
    names = sorted(json.loads((ROOT / "outputs/v61_folds.json").read_text())["folds"]["0"])
    assert len(names) == 141, len(names)
    with torch.no_grad(), Pool(int(os.environ.get("PROCS", "20")), initializer=A.init) as p:
        res = p.map(work, [(n, r) for r in RECIPES for n in names], chunksize=1)
    C = {r: {t: np.asarray([x[2][t] for n in names for x in res if x[0] == n and x[1] == r])
             for t in ("all", "min200")} for r in RECIPES}
    ref = json.loads((ROOT / "outputs/v73/base_fold0.json").read_text())
    refc = np.asarray([ref["per_file"][n]["k7"]["t0.45_c0.30"] for n in names])
    diff = float(np.abs(C["p2first"]["all"] - refc).max())
    repro = dict(max_file_diff=diff, pq=V73.R.K.pq_value(C["p2first"]["all"].sum(0)),
                 ref_pq=ref["k7_t0.45_c0.30"]["pq"], ok=diff < 1e-4)
    print("repro", repro, flush=True)
    rep = {"files": len(names), "reproduction_vs_v73_base_fold0": repro}
    for r in RECIPES:
        rat = np.asarray([v for x in res if x[1] == r for v in x[3]])
        areas = np.asarray([v for x in res if x[1] == r for v in x[4]])
        rep[r] = dict(S.summ(C[r]["all"]), min200=S.summ(C[r]["min200"]),
                      matched_area_ratio_pred_over_gt=dict(mean=float(rat.mean()), median=float(np.median(rat)),
                                                           n=len(rat)),
                      kept_pred_area_median=float(np.median(areas)), n_kept=len(areas))
    d = S.boot(C["p2first"]["all"], C["p2first_thin"]["all"], seed=81)
    rep["primary"] = d
    rep["secondary_min200"] = S.boot(C["p2first"]["min200"], C["p2first_thin"]["min200"], seed=81)
    rep["verdict"] = ("CONTINUE" if d["delta"] >= 0.010 and d["p_gt_0"] >= 0.90
                      else "KILL" if d["delta"] < 0.005 else "INCONCLUSIVE")
    if not repro["ok"]:
        rep["verdict"] = "REPRO_FAIL"
    out = ROOT / "outputs/v81/refined_fold0.json"
    out.write_text(json.dumps(rep, indent=2, default=float) + "\n")
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()
