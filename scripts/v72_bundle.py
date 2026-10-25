"""V72 post-process bundle (MIN200 + V71 R3 + V71 R2). Pre-registration:
docs/v72_bundle_preregistration_2026-10-10.md. CPU only.
  python scripts/v72_bundle.py cache_base   # baseline + V11 refiner on CPU -> outputs/v72/cache_baseline.pkl
  python scripts/v72_bundle.py eval
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v71_suppress as S  # noqa: E402  (imports the CPU-patched probe module)

P, K, E = S.P, S.K, S.E
OUT = E.ROOT / "outputs/v72"
CUT = 0.30


def init_base():
    import torch
    torch.set_num_threads(1)
    P.RT = P.v50.runtime_state()
    # the probe patches build_model to an unloaded net; load V11 exactly as v50 does, on CPU
    m = P.RT["base"].CropUNet(pretrained=False)
    m.load_state_dict(torch.load(P.v50.CANONICAL_REFINER, map_location="cpu", weights_only=True))
    m.load_state_dict(torch.load(P.v50.V11_REFINER, map_location="cpu",
                                 weights_only=True)["model"])
    P.RT["model"] = m.eval()


def base_work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = P.RT, P.RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    gts = []
    for image_id in sorted(rt["split"][4][name]):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        gts.append([r for r in rl if mu.area(r) > 0])
    views = K.read_cache_views(E.cache_path("baseline", fold, name))
    props, scores = rt["tta"].cluster_views(views, P.v50.POLICY, P.v50.MATCH_IOU)
    preds = []
    if props:
        masks = [mu.decode(r) for r in props]
        bounds, _, probs = core.crop_probabilities(rt["base"], rt["model"], gray, masks, True)
        rles, fs = core.clean_refined_predictions(masks, scores, bounds, probs, 0.45, P.v50.MODE)
        for r, s in zip(rles, fs):
            if s >= CUT:
                preds.append(dict(rle=r, score=float(s),
                                  **S.stats(mu.decode(r).astype(bool), gray)))
    return name, dict(fold=fold, gts=gts, preds=preds)


def cmd_cache_base():
    from multiprocessing import set_start_method
    set_start_method("fork")
    ref = json.loads((E.ROOT / "outputs/v63/r3.json").read_text())["per_file"]
    fmap = P.R.fold_of_file()
    names = sorted(ref)
    with Pool(int(os.environ.get("V72_PROCS", "40")), initializer=init_base) as p:
        data = dict(p.map(base_work, [(n, fmap[n]) for n in names], chunksize=1))
    t = np.sum([S.count(data[n], {}) for n in names], 0)
    print("baseline+V11 totals", t, K.pq_value(t))
    assert (int(t[1]), int(t[2]), int(t[3])) == (1004, 464, 589), t
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "cache_baseline.pkl", "wb") as fh:
        pickle.dump(data, fh)


# rules: index -> drop predicate on one cached prediction (fixed, pre-registered)
def rule_a(p):
    return p["area"] < 200


def rule_b(p):
    return p["contrast"] < 0.06 and p["score"] < 0.5


def rule_c(p, area=300):
    return p["area"] < area and p["score"] < 0.5


def pred_status(d):
    """Per prediction: [records matched (TP), records unmatched (FP), sum IoU of matches]."""
    from pycocotools import mask as mu
    r = [p["rle"] for p in d["preds"]]
    st = np.zeros((len(r), 3))
    for gt in d["gts"]:
        if not r:
            break
        iou = mu.iou(r, gt, [0] * len(gt)) if gt else np.zeros((len(r), 1))
        best = iou.max(1)
        st[:, 0] += best > 0.5
        st[:, 1] += best <= 0.5
        st[:, 2] += np.where(best > 0.5, best, 0)
    return st


def drop_sets(d, xfit):
    c_area = 150 if (xfit and d["fold"] == 4) else 300
    ps = d["preds"]
    return {"a": {i for i, p in enumerate(ps) if rule_a(p)},
            "b": {i for i, p in enumerate(ps) if rule_b(p)},
            "c": {i for i, p in enumerate(ps) if rule_c(p, c_area)}}


def arm_counts(data, names, rules, xfit=False):
    out = []
    for n in names:
        d = data[n]
        drop = set().union(*(drop_sets(d, xfit)[r] for r in rules)) if rules else set()
        keep = [i for i in range(len(d["preds"])) if i not in drop]
        out.append(K.score_file([d["preds"][i]["rle"] for i in keep],
                                [d["preds"][i]["score"] for i in keep], d["gts"], (CUT,))[CUT])
    return np.asarray(out)


def interaction(data, names):
    regions = {}
    for n in names:
        d = data[n]
        st = pred_status(d)
        ds = drop_sets(d, False)
        for i in range(len(d["preds"])):
            key = "".join(r for r in "abc" if i in ds[r])
            if not key:
                continue
            g = regions.setdefault(key, np.zeros(4))
            g += [1, *st[i]]
    return {k: dict(preds=int(v[0]), tp_records=int(v[1]), fp_records=int(v[2]))
            for k, v in sorted(regions.items())}


def evaluate(data, ref_counts=None, seed=72):
    names = sorted(data)
    fo = np.asarray([data[n]["fold"] for n in names])
    base = arm_counts(data, names, ())
    if ref_counts is not None:
        t = base.sum(0)
        assert (int(t[1]), int(t[2]), int(t[3])) == ref_counts, t
    arms = {"bundle": ("a", "b", "c"), "loo_bc(no MIN200)": ("b", "c"),
            "loo_ac(no R3)": ("a", "c"), "loo_ab(no R2)": ("a", "b"),
            "only_a MIN200": ("a",), "only_b R3": ("b",), "only_c R2": ("c",)}
    rep = {"files": len(names), "baseline": S.summ(base)}
    for k, rules in arms.items():
        c = arm_counts(data, names, rules)
        rep[k] = dict(S.summ(c), vs_base=S.boot(base, c, seed))
        if k == "bundle":
            rep[k]["per_fold_delta"] = {int(f): K.pq_value(c[fo == f].sum(0)) -
                                        K.pq_value(base[fo == f].sum(0)) for f in range(5)}
    c = arm_counts(data, names, ("a", "b", "c"), xfit=True)
    rep["bundle_xfit"] = dict(S.summ(c), vs_base=S.boot(base, c, seed))
    rep["sum_of_singles"] = sum(rep[k]["vs_base"]["delta"] for k in arms if k.startswith("only"))
    rep["venn"] = interaction(data, names)
    return rep


def cmd_eval():
    with open(E.ROOT / "outputs/v71/cache.pkl", "rb") as fh:
        r3 = pickle.load(fh)
    rep = {"r3": evaluate(r3, (1020, 448, 573))}
    g = rep["r3"]["bundle"]["vs_base"]
    rep["verdict"] = "ADOPT" if g["delta"] >= 0.004 and g["p_gt_0"] >= 0.90 else "DO NOT ADOPT"
    bp = OUT / "cache_baseline.pkl"
    if bp.exists():
        with open(bp, "rb") as fh:
            rep["baseline_v11_context"] = evaluate(pickle.load(fh), (1004, 464, 589))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


def demo():
    p = dict(area=250, score=0.4, contrast=0.1)
    assert not rule_a(p) and not rule_b(p) and rule_c(p) and not rule_c(p, 150)
    assert rule_a(dict(p, area=199)) and rule_b(dict(p, contrast=0.05))
    assert not rule_b(dict(p, contrast=0.05, score=0.5))


if __name__ == "__main__":
    demo()
    {"cache_base": cmd_cache_base, "eval": cmd_eval}[sys.argv[1]]()
