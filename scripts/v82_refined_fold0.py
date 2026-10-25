"""V82 gate (CPU only): V81 refined fold-0 harness with each arm's refiner swapped in.
p2first views -> flip consensus -> refiner (k7, t sweep) -> cut 0.30 -> per-record PQ, 141 files.
Asserts base refiner at t0.45 == V73 base fold 0 (0.453259). Primary: t0.45, arm - base.
  CUDA_VISIBLE_DEVICES= python scripts/v82_refined_fold0.py
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
TS = (0.35, 0.40, 0.45, 0.50, 0.55)
MODELS = {"base": ROOT / "outputs/v73/base_fold0_refiner.pt",
          **{a: ROOT / f"outputs/v82/{a}_fold0_refiner.pt" for a in ("soft", "union", "qweight")},
          "base_rerun": ROOT / "outputs/v82_rerun/base_fold0_refiner.pt"}  # noise control


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, arm = job
    rt, core = A.RT, A.RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    recs = []
    for image_id in sorted(rt["by_file"][name], key=str):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        recs.append([r for r in rl if mu.area(r) > 0])
    groups = A.groups_with_ids(V73.R.K.read_cache_views(V73.R.O.cache_path("p2first", 0, name)))
    scores = [g["score"] for g in groups]
    if groups:
        masks = [mu.decode(g["rle"]) for g in groups]
        bounds, _, probs = core.crop_probabilities(rt["base"], A.MODELS[arm], gray, masks, True)
    out = {}
    for t in TS:
        post = []
        if groups:
            A.T = t
            _, post = A.refine_instrumented(masks, scores, bounds, probs)
        keep = [post[i] for i in range(len(groups)) if post[i] is not None and scores[i] >= A.CUT]
        tot = np.zeros(4)
        for g in recs:
            ik = A.iou_mat(g, keep)
            tot += core.pq_counts(ik if keep else np.zeros((len(g), 0)), len(g), len(keep))
        out[t] = tot.tolist()
    return name, arm, out


def main():
    import torch
    set_start_method("fork")
    V73.DEV = "cpu"
    A.RT = V73.runtime()
    arms = [a for a, p in MODELS.items() if p.exists()]
    for a in arms:
        m = V73.build_model(A.RT["base"], "imagenet", "resnet18")
        m.load_state_dict(torch.load(MODELS[a], map_location="cpu", weights_only=True)["model"])
        A.MODELS[a] = m.eval()
    names = sorted(json.loads((ROOT / "outputs/v61_folds.json").read_text())["folds"]["0"])
    assert len(names) == 141, len(names)
    with torch.no_grad(), Pool(int(os.environ.get("PROCS", "20")), initializer=A.init) as p:
        res = p.map(work, [(n, a) for a in arms for n in names], chunksize=1)
    C = {a: {t: np.asarray([x[2][t] for n in names for x in res if x[0] == n and x[1] == a])
             for t in TS} for a in arms}
    ref = json.loads((ROOT / "outputs/v73/base_fold0.json").read_text())
    refc = np.asarray([ref["per_file"][n]["k7"]["t0.45_c0.30"] for n in names])
    diff = float(np.abs(C["base"][0.45] - refc).max())
    rep = {"files": len(names), "reproduction_vs_v73_base_fold0": dict(
        max_file_diff=diff, pq=V73.R.K.pq_value(C["base"][0.45].sum(0)),
        ref_pq=ref["k7_t0.45_c0.30"]["pq"], ok=diff < 1e-4), "arms": {}}
    print("repro", rep["reproduction_vs_v73_base_fold0"], flush=True)
    for a in arms:
        r = rep["arms"][a] = {"sweep": {f"t{t:.2f}": S.summ(C[a][t]) for t in TS}}
        # cross-check: training job's own GPU scoring of the same cells
        tj = MODELS[a].with_name(f"{a.removesuffix('_rerun')}_fold0.json")
        if a != "base" and tj.exists():
            pf = json.loads(tj.read_text())["per_file"]
            r["max_diff_vs_training_json_t0.45"] = float(np.abs(
                C[a][0.45] - np.asarray([pf[n]["k7"]["t0.45_c0.30"] for n in names])).max())
        if a == "base":
            continue
        d = r["primary_t0.45"] = S.boot(C["base"][0.45], C[a][0.45], seed=82)
        r["verdict"] = ("CONTINUE" if d["delta"] >= 0.010 and d["p_gt_0"] >= 0.90
                        else "KILL" if d["delta"] < 0.005 else "INCONCLUSIVE")
        bt = max(TS, key=lambda t: V73.R.K.pq_value(C[a][t].sum(0)))
        r["exploratory_best_t"] = dict(t=bt, vs_base_t045=S.boot(C["base"][0.45], C[a][bt], seed=82),
                                       vs_base_same_t=S.boot(C["base"][bt], C[a][bt], seed=82))
    if not rep["reproduction_vs_v73_base_fold0"]["ok"]:
        for r in rep["arms"].values():
            r["verdict"] = "REPRO_FAIL"
    out = ROOT / "outputs/v82/refined_fold0.json"
    out.write_text(json.dumps(rep, indent=2, default=float) + "\n")
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()
