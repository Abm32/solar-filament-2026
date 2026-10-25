"""V92 sim: public PQ of the V87 fold CSVs (V84 path, OOF) under three ways Kaggle could split the solution
into public/private halves: by image, by annotator record, or by GT row. Host self-eval code verbatim.
CPU, train labels only.
  CUDA_VISIBLE_DEVICES= python scripts/v92_rowsplit_sim.py
"""
import json
import os
import random
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pandas as pd
import torch
from pycocotools import mask as mask_utils

torch.set_num_threads(8)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v92"
ANN = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
src = Path.home().joinpath("v69_kernels/selfeval/src.txt").read_text()
host = {}
for c in [c.split("\n", 1)[1] for c in src.split("### [code]")[1:]]:
    c = c.split("### [markdown]")[0]
    if "import torch" in c or "def get_overlap_df" in c or "def get_pq_score" in c:
        exec(c, host)
doc = json.loads(ANN.read_text())
anns = {}
for a in doc["annotations"]:
    anns.setdefault(a["image_id"], []).append(a)
gts = []
for info in doc["images"]:
    for j, a in enumerate(anns.get(info["id"], []), 1):
        r = mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], 2048, 2048))
        gts.append({"filament_id": f"{info['id']}_{j}", "segmentation_rle": r["counts"].decode(),
                    "image": info["file_name"][:-5], "rec": info["id"]})
gt = pd.DataFrame(gts)
pred = pd.concat([pd.read_csv(ROOT / f"outputs/v87/fold{k}_v84.csv", dtype=str, keep_default_na=False) for k in range(5)])
pred_img = pred.filament_id.str.split("_").str[0]


def pq(g, p):
    return host["get_pq_score"](host["get_overlap_df"](g[["filament_id", "segmentation_rle"]].reset_index(drop=True),
                                                       p.reset_index(drop=True)))


res = {"full": pq(gt, pred)}
rng = random.Random(0)
for name, key in [("by_image", "image"), ("by_record", "rec")]:
    units = sorted(gt[key].unique())
    rng.shuffle(units)
    half = set(units[: len(units) // 2])
    g = gt[gt[key].isin(half)]
    res[name] = pq(g, pred[pred_img.isin(set(g["image"]))])
for seed in range(2):
    rr = random.Random(seed)
    g = gt[[rr.random() < 0.5 for _ in range(len(gt))]]
    res[f"by_row_s{seed}"] = pq(g, pred)
print(json.dumps(res, indent=1))
OUT.mkdir(exist_ok=True)
(OUT / "rowsplit_sim.json").write_text(json.dumps(res, indent=1) + "\n")
