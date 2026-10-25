"""V95 counterfactual: score V84 OOF and a probe against train GT with the same object class removed
(the H_local world "test labels lack class X"). usage: v95_local_world.py OUT.json CLASS
CLASS in minarea500 minarea1000 maxrr080. d_local = PQ(probe | GT\\X) - PQ(v84 | GT\\X) is the probe's gain
if test truly lacks X; compare with the uniform-world delta OOF_probe - OOF_v84 (outputs/v95/oof_scores.json).
Reads train images for maxrr only. Never touches test."""
import json, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd, torch
from pycocotools import mask as mask_utils
torch.set_num_threads(8)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RAW = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
src = Path.home().joinpath("v69_kernels/selfeval/src.txt").read_text()
host = {}
for c in [c.split("\n", 1)[1] for c in src.split("### [code]")[1:]]:
    c = c.split("### [markdown]")[0]
    if "import torch" in c or "def get_overlap_df" in c or "def get_pq_score" in c:
        exec(c, host)
out, cls = Path(sys.argv[1]), sys.argv[2]
doc = json.loads((RAW / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text()); anns = {}
for a in doc["annotations"]: anns.setdefault(a["image_id"], []).append(a)
disks = {}

def disk(fname):
    if fname not in disks:
        import cv2
        from v94_limb import fit_disk
        f = fit_disk(cv2.imread(str(RAW / "train_images" / fname), cv2.IMREAD_GRAYSCALE))
        disks[fname] = (f["cx"], f["cy"], f["radius"])
    return disks[fname]

rows, kept, dropped = [], 0, 0
for i in doc["images"]:
    for j, a in enumerate(anns.get(i["id"], []), 1):
        rle = mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], 2048, 2048))
        if cls.startswith("minarea"):
            keep = mask_utils.area(rle) >= int(cls[7:])
        elif cls == "maxrr080":
            cx, cy, R = disk(i["file_name"]); ys, xs = np.nonzero(mask_utils.decode(rle))
            keep = np.hypot(xs.mean() - cx, ys.mean() - cy) / R <= 0.80
        else:
            raise SystemExit(f"unknown class {cls}")
        if keep: rows.append({"filament_id": f"{i['id']}_{j}", "segmentation_rle": rle["counts"].decode()}); kept += 1
        else: dropped += 1
gt = pd.DataFrame(rows)
print(cls, "GT kept", kept, "dropped", dropped, flush=True)
r = {"gt_kept": kept, "gt_dropped": dropped}
for name, f in [("v84", ROOT / "outputs/v92/oof_v84.csv"), ("probe", ROOT / f"outputs/v95/oof/{cls}.csv")]:
    p = pd.read_csv(f, dtype=str, keep_default_na=False)
    r[name] = host["get_pq_score"](host["get_overlap_df"](gt, p)); print(cls, name, r[name], flush=True)
r["d_local"] = r["probe"] - r["v84"]
res = json.loads(out.read_text()) if out.exists() else {}
res[cls] = r; out.write_text(json.dumps(res, indent=1) + "\n"); print(cls, r, flush=True)
