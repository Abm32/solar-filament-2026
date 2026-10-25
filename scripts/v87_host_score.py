"""V87: score the v87 export-path CSVs with the HOST self-evaluation code (verbatim cells, as audit_v79_roundtrip)
against host-format train GT (one block per annotator record). CPU, train labels only.
  CUDA_VISIBLE_DEVICES= python scripts/v87_host_score.py
"""
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pandas as pd
import torch
from pycocotools import mask as mask_utils

torch.set_num_threads(8)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v87"
ANN = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
src = Path.home().joinpath("v69_kernels/selfeval/src.txt").read_text()
host = {}
for c in [c.split("\n", 1)[1] for c in src.split("### [code]")[1:]]:
    c = c.split("### [markdown]")[0]
    if "import torch" in c or "def get_overlap_df" in c or "def get_pq_score" in c:
        exec(c, host)
folds = {int(k): v for k, v in json.loads((ROOT / "outputs/v61_folds.json").read_text())["folds"].items()}
doc = json.loads(ANN.read_text())
res = {}
for arm in ("v84", "v65"):
    preds, gts = [], []
    for k, names in folds.items():
        p = OUT / f"fold{k}_{arm}.csv"
        if not p.exists():
            continue
        names = set(names)
        preds.append(pd.read_csv(p, dtype=str, keep_default_na=False))
        anns = {}
        for a in doc["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        for info in doc["images"]:
            if info["file_name"] in names:
                for j, a in enumerate(anns.get(info["id"], []), 1):
                    r = mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], 2048, 2048))
                    gts.append({"filament_id": f"{info['id']}_{j}", "segmentation_rle": r["counts"].decode()})
        res.setdefault(arm, {})[f"fold{k}"] = host["get_pq_score"](host["get_overlap_df"](
            pd.DataFrame(gts[-sum(len(anns.get(i['id'], [])) for i in doc['images'] if i['file_name'] in names):]),
            preds[-1]))
    res[arm]["pooled"] = host["get_pq_score"](host["get_overlap_df"](pd.DataFrame(gts), pd.concat(preds)))
(OUT / "host_score.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
