"""V95 counterfactual grid (VM-only, CPU, never touches test): PQ of V84 OOF with predictions filtered at
min-area P, scored against train GT with annotations filtered at min-area G, for every (G, P) pair.
G = the H_local world "test labels lack filaments < G px"; P = the submission-side knob.
One host overlap pass (exact host kernel), then slicing per pair. Annotator-images with no GT left after
filtering are dropped from the table (same as scripts/v95_local_world.py).
usage: v95_minarea_grid.py OUT.json"""
import json, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd, torch
from pycocotools import mask as mask_utils
torch.set_num_threads(16)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
src = Path.home().joinpath("v69_kernels/selfeval/src.txt").read_text()
host = {}
for c in [c.split("\n", 1)[1] for c in src.split("### [code]")[1:]]:
    c = c.split("### [markdown]")[0]
    if "import torch" in c or "def get_overlap_df" in c or "def get_pq_score" in c:
        exec(c, host)
out = Path(sys.argv[1])
doc = json.loads((RAW / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text()); anns = {}
for a in doc["annotations"]: anns.setdefault(a["image_id"], []).append(a)
rows = []
for i in doc["images"]:
    for j, a in enumerate(anns.get(i["id"], []), 1):
        rle = mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], 2048, 2048))
        rows.append({"filament_id": f"{i['id']}_{j}", "segmentation_rle": rle["counts"].decode(), "area": int(mask_utils.area(rle))})
gt = pd.DataFrame(rows)
pred = pd.read_csv(ROOT / "outputs/v92/oof_v84.csv", dtype=str, keep_default_na=False)
pred["area"] = [int(mask_utils.area({"size": [2048, 2048], "counts": c.encode()})) for c in pred.segmentation_rle]
print("GT", len(gt), "pred", len(pred), flush=True)
ov = host["get_overlap_df"](gt[["filament_id", "segmentation_rle"]], pred[["filament_id", "segmentation_rle"]])
print("overlap rows", len(ov), flush=True)
gt_key = gt.filament_id.str.split("_", n=1).str[0]; pred_key = pred.filament_id.str.split("_", n=1).str[0]
ga = {k: gt.area[gt_key == k].to_numpy() for k in ov.annotator_image}
pa = {k: pred.area[pred_key == k.split("-", 1)[1]].to_numpy() for k in ov.annotator_image}
for r in ov.itertuples(index=False):
    assert r.iou_matrix.shape == (len(ga[r.annotator_image]), len(pa[r.annotator_image])), r.annotator_image

def pq(G, P):
    sub = []
    for r in ov.itertuples(index=False):
        gm, pm = ga[r.annotator_image] >= G, pa[r.annotator_image] >= P
        if gm.sum() == 0: continue
        m = r.iou_matrix[torch.from_numpy(gm)][:, torch.from_numpy(pm)]
        sub.append({"iou_matrix": m, "n_gt": int(gm.sum()), "n_pred": int(pm.sum())})
    return host["get_pq_score"](pd.DataFrame(sub))

GS = [0, 250, 500, 750, 1000, 1250, 1500, 2000, 3000]
PS = [0, 300, 500, 750, 1000, 1250, 1500, 2000, 3000]
res = {"G": GS, "P": PS, "gt_kept": {str(G): int((gt.area >= G).sum()) for G in GS},
       "pred_kept": {str(P): int((pred.area >= P).sum()) for P in PS}, "pq": {}}
for G in GS:
    res["pq"][str(G)] = {str(P): pq(G, P) for P in PS}
    print(G, {P: round(v, 4) for P, v in res["pq"][str(G)].items()}, flush=True)
out.write_text(json.dumps(res, indent=1) + "\n")
base = res["pq"]["0"]["0"]
print(f"\nbase OOF {base:.4f}; row = GT world G, cols = pred minarea P")
print("G     kept  " + " ".join(f"{P:>7}" for P in PS) + "   gap(P=0)  bestP  bestPQ  d_local(best)")
for G in GS:
    r = res["pq"][str(G)]; bp = max(PS, key=lambda P: r[str(P)])
    print(f"{G:<5} {res['gt_kept'][str(G)]:>5} " + " ".join(f"{r[str(P)]:7.4f}" for P in PS)
          + f"   {r['0'] - base:+.4f}   {bp:>5}  {r[str(bp)]:.4f}  {r[str(bp)] - r['0']:+.4f}")
