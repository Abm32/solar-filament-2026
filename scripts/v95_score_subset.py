"""V95: host-scorer PQ of the OOF V84 CSV restricted to file subsets (year bins, sites): GT and predictions both
filtered. Prints PQ, G (gt polygons), P (pred rows) per subset -> N = PQ*(G+P)/2. CPU."""
import json, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pandas as pd, torch
from pycocotools import mask as mask_utils
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
src = Path.home().joinpath("v69_kernels/selfeval/src.txt").read_text()
host = {}
for c in [c.split("\n", 1)[1] for c in src.split("### [code]")[1:]]:
    c = c.split("### [markdown]")[0]
    if "import torch" in c or "def get_overlap_df" in c or "def get_pq_score" in c:
        exec(c, host)
doc = json.loads(ANN.read_text()); anns = {}
for a in doc["annotations"]: anns.setdefault(a["image_id"], []).append(a)
gt = pd.DataFrame([{"filament_id": f"{i['id']}_{j}", "file": i["file_name"][:-5],
                    "segmentation_rle": mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], 2048, 2048))["counts"].decode()}
                   for i in doc["images"] for j, a in enumerate(anns.get(i["id"], []), 1)])
pred = pd.read_csv(ROOT / "outputs/v92/oof_v84.csv", dtype=str, keep_default_na=False)
pred["file"] = pred.filament_id.str.rsplit("_", n=1).str[0]
subsets = {"Y1112": lambda f: f[:4] in ("2011", "2012"), "Y1314": lambda f: f[:4] in ("2013", "2014"),
           "Y1517": lambda f: f[:4] in ("2015", "2016", "2017"), "Y1822": lambda f: f[:4] >= "2018",
           "Y2122": lambda f: f[:4] >= "2021"}
for s in ("Bh", "Ch", "Lh", "Mh", "Th", "Uh"): subsets[s] = (lambda s: lambda f: f.endswith(s))(s)
which = sys.argv[1:] or list(subsets)
out = ROOT / "outputs/v95/oof_subsets.json"; res = json.loads(out.read_text()) if out.exists() else {}
files = sorted({i["file_name"][:-5] for i in doc["images"]})
for k in which:
    fs = {f for f in files if subsets[k](f)}
    g, p = gt[gt.file.isin(fs)], pred[pred.file.isin(fs)]
    pq = host["get_pq_score"](host["get_overlap_df"](g.drop(columns="file"), p.drop(columns="file")))
    res[k] = {"files": len(fs), "pq": pq, "G": len(g), "P": len(p), "N": pq * (len(g) + len(p)) / 2}
    print(k, res[k], flush=True); out.write_text(json.dumps(res, indent=1) + "\n")
