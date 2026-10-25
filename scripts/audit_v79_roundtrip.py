"""V79 audit: submission round trip + metric parity + public-subset noise. CPU only, train labels only.
  CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/audit_v79_roundtrip.py
Writes outputs/v79_audit/roundtrip.json (+ the two CSVs it scores).
"""
import collections
import json
import os
import pickle
import re
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

torch.set_num_threads(8)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v61_toolkit as K  # noqa: E402

OUT = ROOT / "outputs/v79_audit"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
ANN = DATA / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
res = {}

# --- host scorer: exec the official self-evaluation notebook code cells verbatim
src = Path.home().joinpath("v69_kernels/selfeval/src.txt").read_text()
cells = [c.split("\n", 1)[1] for c in src.split("### [code]")[1:]]
host = {}
for c in cells:
    c = c.split("### [markdown]")[0]
    if "import torch" in c or "def get_overlap_df" in c or "def get_pq_score" in c:
        exec(c, host)


def host_pq(pred_csv, gt_csv):
    pred = pd.read_csv(pred_csv, dtype=str, keep_default_na=False)
    gt = pd.read_csv(gt_csv, dtype=str, keep_default_na=False)
    df = host["get_overlap_df"](gt, pred)
    hit = [m > 0.5 for m in df.iou_matrix]
    tp = sum(int(h.sum()) for h in hit)
    return host["get_pq_score"](df), dict(records=len(df), tp=tp)


def pq(c):
    return c[0] / (c[1] + 0.5 * c[2] + 0.5 * c[3])


# --- A. GT rasterisation convention: host says annotation["area"] = pycocotools(polygon->RLE)
doc = json.loads(ANN.read_text())
imgs = {i["id"]: i for i in doc["images"]}
d_area = [abs(float(mask_utils.area(mask_utils.merge(mask_utils.frPyObjects(
    a["segmentation"], 2048, 2048)))) - a["area"]) for a in doc["annotations"]]
res["gt_area_field_vs_frPyObjects"] = dict(n=len(d_area), exact=int(sum(x < 0.5 for x in d_area)),
                                           max_abs_diff=float(max(d_area)))

# --- B. round trip of OOF refined preds (v71 cache, p2first+r3, t0.45 c0.30) through export writer
cache = pickle.load(open(ROOT / "outputs/v71/cache.pkl", "rb"))
for min_area in (20, 200):
    rows, counts = [], np.zeros(4)
    for name, d in sorted(cache.items()):
        preds = [p["rle"] for p in d["preds"]]  # already ownership-cleaned, conf >= 0.30
        # ---- verbatim from scripts/export_v74.py main() ----
        preds = [r for r in preds if int(mask_utils.area(r)) >= min_area]
        occupied = np.zeros((2048, 2048), dtype=bool)
        for number, rle in enumerate(preds, 1):
            mask = mask_utils.decode(rle).astype(bool)
            if int(mask.sum()) < min_area or np.any(occupied & mask):
                raise RuntimeError(f"invalid area/overlap for {name}")
            occupied |= mask
            rows.append({"filament_id": f"{Path(name).stem}_{number}",
                         "segmentation_rle": rle["counts"].decode()})
        # ----
        counts += K.score_file(preds, [1.0] * len(preds), d["gts"], (0.3,))[0.3]
    pcsv = OUT / f"oof141_pred_min{min_area}.csv"
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(pcsv, index=False)
    if min_area == 20:  # host-format solution from the polygons (one block per annotator record)
        grows = []
        for iid, info in imgs.items():
            if info["file_name"] not in cache:
                continue
            anns = [a for a in doc["annotations"] if a["image_id"] == iid]
            for k, a in enumerate(anns, 1):
                r = mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], 2048, 2048))
                grows.append({"filament_id": f"{iid}_{k}", "segmentation_rle": r["counts"].decode()})
        gcsv = OUT / "oof141_gt_hostformat.csv"
        pd.DataFrame(grows).to_csv(gcsv, index=False)
    hp, hinfo = host_pq(pcsv, gcsv)
    res[f"roundtrip_min{min_area}"] = dict(internal_pq=pq(counts), internal_tp=int(counts[1]),
                                           host_csv_pq=hp, **hinfo, rows=len(rows))

# --- C. submitted V65 CSV: parse exactly as host, integrity + distribution vs OOF
sub = pd.read_csv(ROOT / "outputs/v65/export/submission_v65_p2first_r3all.csv", dtype=str,
                  keep_default_na=False)
test_stems = {p.stem for p in (DATA / "test/test_images").glob("*.jpeg")}
ids = sub.filament_id.str.split("_", n=1).str[0]
areas, overl, per_img = [], 0, collections.Counter(ids)
for stem, g in sub.groupby(ids):
    m = host["rles_to_layers"](g.segmentation_rle.tolist())
    areas += m.reshape(len(m), -1).sum(1).tolist()
    overl += int((m.sum(0) > 1).sum())
oof_areas = [float(mask_utils.area(p["rle"])) for d in cache.values() for p in d["preds"]]
res["v65_submission"] = dict(
    rows=len(sub), images_with_preds=len(per_img), unknown_ids=len(set(ids) - test_stems),
    duplicate_ids=int(sub.filament_id.duplicated().sum()), overlapping_px=overl,
    preds_per_test_image=len(sub) / 180, preds_per_oof_image=len(oof_areas) / len(cache),
    median_area_test=float(np.median(areas)), median_area_oof=float(np.median(oof_areas)),
    frac_area_lt200_test=float(np.mean(np.array(areas) < 200)),
    frac_area_lt200_oof=float(np.mean(np.array(oof_areas) < 200)))

# --- D. site/year mix train vs test
def mix(names):
    return dict(collections.Counter(re.match(r"(\d{4})\d+(\w)h", n).groups()[1] for n in names)), \
        dict(collections.Counter(n[:4] for n in names))
res["site_year_train"] = mix([p.name for p in (DATA / "train/train_images").glob("*.jpeg")])
res["site_year_test"] = mix(sorted(test_stems))

# --- E. public-board sampling noise: PQ over random 90-image subsets
rng = np.random.default_rng(79)
lines = [json.loads(x) for x in Path.home().joinpath(
    "v69_kernels/audit/p2first_perfile.jsonl").read_text().splitlines()]
arr = np.array([x["host_035"] for x in lines])
sub90 = [pq(arr[rng.choice(len(arr), 90, replace=False)].sum(0)) for _ in range(5000)]
r141 = np.array([K.score_file([p["rle"] for p in d["preds"] if p["area"] >= 200],
                              [1.0] * sum(p["area"] >= 200 for p in d["preds"]), d["gts"], (0.3,))[0.3]
                 for d in cache.values()])
s141 = [pq(r141[rng.choice(len(r141), 90, replace=False)].sum(0)) for _ in range(5000)]
res["public90_noise"] = dict(
    direct707_full=pq(arr.sum(0)), direct707_sd90=float(np.std(sub90)),
    direct707_p05_p95=[float(np.percentile(sub90, 5)), float(np.percentile(sub90, 95))],
    refined141_full=pq(r141.sum(0)), refined141_sd90_wo_repl=float(np.std(s141)),
    refined_sd90_fpc_corrected=float(np.std(s141) / np.sqrt((141 - 90) / (141 - 1))),
)
(OUT / "roundtrip.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
