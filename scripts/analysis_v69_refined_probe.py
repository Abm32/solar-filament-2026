"""V69 probe on the REAL pipeline: p2first OOF proposals -> r3 refiner (fold-0 trained on the 566
fit files, outputs/v63/r3_refiner.pt) -> V17 scoring, on the 141 canonical held-out files.
CPU ONLY (monkeypatches the CUDA-only loaders). Variants change the proposal masks fed to the
refiner (vote75, erode1) or erode the refined output masks (post_e1 square, post_ex cross).
`rep` must reproduce outputs/v63/r3.json per-file counts exactly (asserted at t0.45/c0.30)."""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "kaggle_inference"))
import torch  # noqa: E402

torch.cuda.is_available = lambda: True  # runtime_state's guard only; nothing below uses CUDA
import v11_refiner as v11  # noqa: E402

v11.build_model = lambda base, *a, **k: base.CropUNet(pretrained=False).eval()
import analysis_v69_fusion as F  # noqa: E402
import eval_v61_oof as O  # noqa: E402
import eval_v62_refiner as R  # noqa: E402
import reproduce_v17_from_cache as v50  # noqa: E402
import v61_toolkit as K  # noqa: E402

R3 = O.ROOT / "outputs/v63/r3_refiner.pt"
CELLS = [(0.45, c) for c in v50.CONFIDENCES]
RT = None


def init():
    global RT
    torch.set_num_threads(1)
    RT = v50.runtime_state()
    m = RT["base"].CropUNet(pretrained=False)
    m.load_state_dict(torch.load(R3, map_location="cpu", weights_only=True)["model"])
    RT["model"] = m.eval()


def erode(rles, cross):
    import cv2
    from pycocotools import mask as mu
    k = cv2.getStructuringElement(cv2.MORPH_CROSS if cross else cv2.MORPH_RECT, (3, 3))
    out = []
    for r in rles:
        m = cv2.erode(mu.decode(r), k)
        out.append(K.encode(m) if m.sum() >= K.MIN_AREA else r)
    return out


def counts(rles, scores, gt_by_image, core):
    from pycocotools import mask as mu
    out = {}
    for t, c in CELLS:
        keep = [r for r, s in zip(rles, scores) if s >= c]
        tot = np.zeros(4)
        for gt in gt_by_image.values():
            iou = (mu.iou(keep, gt, [0] * len(gt)).T if keep and gt
                   else np.zeros((len(gt), len(keep))))
            tot += core.pq_counts(iou, len(gt), len(keep))
        out[f"t{t:.2f}_c{c:.2f}"] = tot.tolist()
    return out


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = RT, RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    gt = {}
    for image_id in sorted(rt["split"][4][name]):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        gt[image_id] = [r for r in rl if mu.area(r) > 0]
    views = K.read_cache_views(O.cache_path("p2first", fold, name))
    groups = F.groups_of(views)
    scores = [sum(g["scores"]) / len(views) for g in groups]
    order = sorted(range(len(groups)), key=lambda i: -scores[i])
    scores = [scores[i] for i in order]
    res = {}
    for var in ("rep", "vote75", "erode1"):
        props = [F.fuse(groups[i], var) for i in order]
        if not props:
            res[var] = counts([], [], gt, core)
            continue
        masks = [mu.decode(r) for r in props]
        bounds, _, probs = core.crop_probabilities(rt["base"], rt["model"], gray, masks, True)
        rles, fs = core.clean_refined_predictions(masks, scores, bounds, probs, 0.45, v50.MODE)
        res[var] = counts(rles, fs, gt, core)
        if var == "rep":
            res["post_e1"] = counts(erode(rles, False), fs, gt, core)
            res["post_ex"] = counts(erode(rles, True), fs, gt, core)
    return name, res


def main():
    from multiprocessing import set_start_method
    set_start_method("fork")
    ref = json.loads((O.ROOT / "outputs/v63/r3.json").read_text())["per_file"]
    fmap = R.fold_of_file()
    names = sorted(ref)[: int(os.environ.get("V69_LIMIT", len(ref)))]
    with Pool(int(os.environ.get("V69_PROCS", "32")), initializer=init) as p:
        res = dict(p.map(work, [(n, fmap[n]) for n in names], chunksize=1))
    for n in names:
        assert np.allclose(res[n]["rep"]["t0.45_c0.30"], ref[n]["t0.45_c0.30"]), n
    rep, rng = {}, np.random.default_rng(69)
    for var in res[names[0]]:
        rep[var] = {}
        for key in res[names[0]][var]:
            t = np.sum([res[n][var][key] for n in names], 0)
            rep[var][key] = dict(pq=K.pq_value(t), sq=t[0] / t[1], tp=int(t[1]), fp=int(t[2]),
                                 fn=int(t[3]))
        a = np.asarray([res[n]["rep"]["t0.45_c0.30"] for n in names])
        b = np.asarray([res[n][var]["t0.45_c0.30"] for n in names])
        ds = []
        for _ in range(2000):
            i = rng.integers(0, len(a), len(a))
            ds.append(K.pq_value(b[i].sum(0)) - K.pq_value(a[i].sum(0)))
        rep[var]["delta_v17cell"] = [K.pq_value(b.sum(0)) - K.pq_value(a.sum(0)),
                                     float(np.percentile(ds, 2.5)),
                                     float(np.percentile(ds, 97.5)),
                                     float(np.mean(np.asarray(ds) > 0))]
        print(var, rep[var]["t0.45_c0.30"], rep[var]["delta_v17cell"], flush=True)
    (O.ROOT / "outputs/v69_analysis/p2first/refined_probe.json").write_text(
        json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
