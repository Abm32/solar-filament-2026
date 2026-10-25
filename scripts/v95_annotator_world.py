"""V95: per-annotator view of V84 OOF (VM-only, CPU, never touches test).
Host scores per annotator-image, so each real annotator is a real "world" of label style. For each annotator:
images, GT count, share of GT < 900 px, median GT area, PQ of V84 against that annotator's images only,
and the best prediction min-area P for that annotator. Caches the host overlap table (outputs/v95/overlap_v84.pt).
usage: v95_annotator_world.py OUT.json"""
import json, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd, torch
from pycocotools import mask as mask_utils
torch.set_num_threads(16)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
CACHE = ROOT / "outputs/v95/overlap_v84.pt"
src = Path.home().joinpath("v69_kernels/selfeval/src.txt").read_text()
host = {}
for c in [c.split("\n", 1)[1] for c in src.split("### [code]")[1:]]:
    c = c.split("### [markdown]")[0]
    if "import torch" in c or "def get_overlap_df" in c or "def get_pq_score" in c:
        exec(c, host)
out = Path(sys.argv[1])
PS = [0, 300, 500, 750, 1000, 1250, 1500]

if CACHE.exists():
    rows = torch.load(CACHE)
else:
    doc = json.loads((RAW / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text()); anns = {}
    for a in doc["annotations"]: anns.setdefault(a["image_id"], []).append(a)
    g = []
    for i in doc["images"]:
        for j, a in enumerate(anns.get(i["id"], []), 1):
            rle = mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], 2048, 2048))
            g.append({"filament_id": f"{i['id']}_{j}", "segmentation_rle": rle["counts"].decode(), "area": int(mask_utils.area(rle))})
    gt = pd.DataFrame(g)
    pred = pd.read_csv(ROOT / "outputs/v92/oof_v84.csv", dtype=str, keep_default_na=False)
    pred["area"] = [int(mask_utils.area({"size": [2048, 2048], "counts": c.encode()})) for c in pred.segmentation_rle]
    ov = host["get_overlap_df"](gt[["filament_id", "segmentation_rle"]], pred[["filament_id", "segmentation_rle"]])
    gk = gt.filament_id.str.split("_", n=1).str[0]; pk = pred.filament_id.str.split("_", n=1).str[0]
    rows = []
    for r in ov.itertuples(index=False):
        k = r.annotator_image
        rows.append({"key": k, "iou": r.iou_matrix, "ga": gt.area[gk == k].to_numpy(), "pa": pred.area[pk == k.split("-", 1)[1]].to_numpy()})
        assert r.iou_matrix.shape == (len(rows[-1]["ga"]), len(rows[-1]["pa"])), k
    torch.save(rows, CACHE)
print("annotator-images", len(rows), flush=True)

def pq(sel, P=0, G=0):
    sub = []
    for r in sel:
        gm, pm = r["ga"] >= G, r["pa"] >= P
        if gm.sum() == 0: continue
        sub.append({"iou_matrix": r["iou"][torch.from_numpy(gm)][:, torch.from_numpy(pm)], "n_gt": int(gm.sum()), "n_pred": int(pm.sum())})
    return host["get_pq_score"](pd.DataFrame(sub))

by_ann = {}
for r in rows: by_ann.setdefault(r["key"].split("-", 1)[0], []).append(r)
res = {"base": pq(rows), "annotators": {}}
for a, sel in sorted(by_ann.items(), key=lambda kv: -len(kv[1])):
    ga = np.concatenate([r["ga"] for r in sel]); npred = sum(len(r["pa"]) for r in sel)
    pqs = {str(P): pq(sel, P) for P in PS}; bp = max(PS, key=lambda P: pqs[str(P)])
    res["annotators"][a] = {"images": len(sel), "gt": int(len(ga)), "gt_per_image": len(ga) / len(sel), "pred_per_image": npred / len(sel),
                            "frac_lt900": float((ga < 900).mean()), "median_area": float(np.median(ga)), "pq": pqs, "best_P": bp}
# worlds made of real annotators: bottom / top tercile by small-filament share (weighted by images)
ann = sorted(res["annotators"].items(), key=lambda kv: kv[1]["frac_lt900"])
n = len(ann) // 3
for name, part in [("low_small_tercile", ann[:n]), ("high_small_tercile", ann[-n:])]:
    sel = [r for a, _ in part for r in by_ann[a]]
    res[name] = {"annotators": [a for a, _ in part], "images": len(sel), "pq": {str(P): pq(sel, P) for P in PS}}
out.write_text(json.dumps(res, indent=1) + "\n")
print(f"base {res['base']:.4f}")
print("ann      imgs    gt  gt/img pr/img  <900  medA  " + " ".join(f"{P:>6}" for P in PS) + "  bestP")
for a, d in sorted(res["annotators"].items(), key=lambda kv: kv[1]["frac_lt900"]):
    print(f"{a:<8} {d['images']:>4} {d['gt']:>5} {d['gt_per_image']:6.2f} {d['pred_per_image']:6.2f} {d['frac_lt900']:5.2f} {d['median_area']:5.0f}  "
          + " ".join(f"{d['pq'][str(P)]:6.3f}" for P in PS) + f"  {d['best_P']:>5}")
for name in ["low_small_tercile", "high_small_tercile"]:
    d = res[name]; print(name, d["images"], "imgs", {P: round(d["pq"][str(P)], 4) for P in PS})
