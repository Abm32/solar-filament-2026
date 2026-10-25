"""V96 refined-pipeline gate per annotator subset (CPU only): p2first vs style arms on the 141 fold-0 files.
Subsets: all records, hard (010101/010401), g01 (group 01), rest. usage: v96_refined_fold0.py ARM [ARM...]
views -> flip consensus (A.groups_with_ids) -> V73 base fold-0 refiner (k7, t0.45) -> cut 0.30 -> per-record PQ.
Reuses audit_v80_extract (which reproduced outputs/v73/base_fold*.json). Asserts p2first == V73 base fold 0.
  CUDA_VISIBLE_DEVICES= python scripts/v94_refined_fold0.py
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
RECIPES = ["p2first"] + sys.argv[1:]
SUBSETS = {"all": lambda a: True, "hard": lambda a: a in ("010101", "010401"), "g01": lambda a: a.startswith("01"),
           "rest": lambda a: not a.startswith("01")}
PRIMARY = {"p2first_hard": "hard", "p2first_g01": "g01", "p2first_more": "all"}


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, recipe = job
    rt, core = A.RT, A.RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    recs, tags = [], []
    for image_id in sorted(rt["by_file"][name], key=str):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        recs.append([r for r in rl if mu.area(r) > 0])
        tags.append(str(image_id).split("-", 1)[0])
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
    out, ratios = {s: np.zeros(4) for s in SUBSETS}, []
    for g, ann in zip(recs, tags):
        ik = A.iou_mat(g, keep)
        c = core.pq_counts(ik if keep else np.zeros((len(g), 0)), len(g), len(keep))
        for s, ok in SUBSETS.items():
            if ok(ann): out[s] += c
        if g and keep:
            for gi, j in zip(*np.where(ik > 0.5)):
                ratios.append(ka[j] / float(mu.area(g[gi])))
    return name, recipe, {s: v.tolist() for s, v in out.items()}, ratios, ka


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
             for t in SUBSETS} for r in RECIPES}
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
        rep[r] = {t: S.summ(C[r][t]) for t in SUBSETS}
        rep[r]["matched_area_ratio_pred_over_gt"] = dict(mean=float(rat.mean()), median=float(np.median(rat)), n=len(rat))
        rep[r]["kept_pred_area_median"] = float(np.median(areas)); rep[r]["n_kept"] = len(areas)
    rep["verdict"] = {}
    for r in RECIPES[1:]:
        rep[f"delta_{r}"] = {t: S.boot(C["p2first"][t], C[r][t], seed=96) for t in SUBSETS}
        d = rep[f"delta_{r}"][PRIMARY.get(r, "all")]
        rep["verdict"][r] = ("CONTINUE" if d["delta"] >= 0.020 and d["p_gt_0"] >= 0.90
                             else "KILL" if d["delta"] < 0.005 else "INCONCLUSIVE")
        if not repro["ok"]:
            rep["verdict"][r] = "REPRO_FAIL"
    out = ROOT / "outputs/v96/refined_fold0.json"
    out.write_text(json.dumps(rep, indent=2, default=float) + "\n")
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()
