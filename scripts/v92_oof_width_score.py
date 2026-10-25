"""V92: host-scorer PQ of the OOF V84 CSV and its 1-px dilated / eroded variants (outputs/v92/oof_*.csv). CPU."""
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pandas as pd
import torch
from pycocotools import mask as mask_utils

torch.set_num_threads(8)
ROOT = Path(__file__).resolve().parents[1]
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
gt = pd.DataFrame([{"filament_id": f"{i['id']}_{j}",
                    "segmentation_rle": mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], 2048, 2048))["counts"].decode()}
                   for i in doc["images"] for j, a in enumerate(anns.get(i["id"], []), 1)])
res = {}
for n in ["v84", "dilate1", "erode1"]:
    p = pd.read_csv(ROOT / f"outputs/v92/oof_{n}.csv", dtype=str, keep_default_na=False)
    res[n] = host["get_pq_score"](host["get_overlap_df"](gt, p))
    print(n, res[n], flush=True)
(ROOT / "outputs/v92/oof_width.json").write_text(json.dumps(res, indent=1) + "\n")
