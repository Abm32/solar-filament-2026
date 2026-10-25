"""V95: why are some annotators hard for V84? (VM-only, CPU, never touches test)
(a) per annotator: FN split by max IoU with any prediction (0 / <0.3 / 0.3-0.5) and FP split by max IoU with any GT;
(b) inter-annotator agreement: on images with >=2 annotators, PQ(annotator A labels as pred | annotator B labels as GT)
    with the host PQ rule (pairs with IoU>0.5), pooled per annotator-as-GT. This is the ceiling a perfect
    "other-annotator" model reaches on that annotator.
usage: v95_annotator_style.py OUT.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from pycocotools import mask as mask_utils
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
out = Path(sys.argv[1])
rows = torch.load(ROOT / "outputs/v95/overlap_v84.pt", weights_only=False)
by = {}
for r in rows: by.setdefault(r["key"].split("-", 1)[0], []).append(r)
res = {"fnfp": {}, "agreement": {}}
print("ann      imgs   PQ    FN  fn_iou0 fn<.3 fn.3-.5   FP  fp_iou0 fp<.3 fp.3-.5")
for a, sel in sorted(by.items(), key=lambda kv: -len(kv[1])):
    if len(sel) < 10: continue
    fn = []; fp = []; ious = []; TP = 0
    for r in sel:
        m = r["iou"]; hit = m > 0.5; ious += m[hit].tolist()
        if m.shape[1]: fn += m.max(1).values[hit.sum(1) == 0].tolist()
        else: fn += [0.0] * m.shape[0]
        if m.shape[0]: fp += m.max(0).values[hit.sum(0) == 0].tolist()
        else: fp += [0.0] * m.shape[1]
    fn, fp = np.array(fn), np.array(fp)
    pq = sum(ious) / (len(ious) + 0.5 * len(fp) + 0.5 * len(fn))
    d = {"images": len(sel), "pq": pq, "FN": len(fn), "fn_iou0": int((fn == 0).sum()), "fn_lt03": int(((fn > 0) & (fn < 0.3)).sum()),
         "fn_03_05": int((fn >= 0.3).sum()), "FP": len(fp), "fp_iou0": int((fp == 0).sum()), "fp_lt03": int(((fp > 0) & (fp < 0.3)).sum()), "fp_03_05": int((fp >= 0.3).sum())}
    res["fnfp"][a] = d
    print(f"{a:<8} {len(sel):>4} {pq:.3f} {d['FN']:5d} {d['fn_iou0']:7d} {d['fn_lt03']:5d} {d['fn_03_05']:7d} {d['FP']:5d} {d['fp_iou0']:7d} {d['fp_lt03']:5d} {d['fp_03_05']:7d}")

doc = json.loads((RAW / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text()); anns = {}
for a in doc["annotations"]: anns.setdefault(a["image_id"], []).append(a)
by_file = {}
for i in doc["images"]: by_file.setdefault(i["file_name"], []).append(i["id"])
rle = {}
def rles(iid):
    if iid not in rle:
        rle[iid] = [mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], 2048, 2048)) for a in anns.get(iid, [])]
    return rle[iid]
acc = {}  # annotator-as-GT -> [sum_iou, tp, fp, fn, n_pairs]
pair_pq = {}
for f, ids in by_file.items():
    if len(ids) < 2: continue
    for g in ids:
        for p in ids:
            if g == p: continue
            G, P = rles(g), rles(p)
            if not G: continue
            m = np.array(mask_utils.iou(G, P, [0] * len(P))).reshape(len(G), len(P)) if P else np.zeros((len(G), 0))
            hit = m > 0.5
            s = [m[hit].sum(), int(hit.sum()), int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum()), 1]
            ag = g.split("-", 1)[0]
            acc[ag] = [x + y for x, y in zip(acc.get(ag, [0] * 5), s)]
print("\nimages with >=2 annotators:", sum(len(v) >= 2 for v in by_file.values()), "of", len(by_file))
print("ann(as GT)  pairs  PQ_other_annotators  V84_PQ")
for a, (si, tp, fp, fn, n) in sorted(acc.items(), key=lambda kv: -kv[1][4]):
    pq = si / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0
    res["agreement"][a] = {"pairs": n, "pq_other": pq, "tp": tp, "fp": fp, "fn": fn}
    v84 = res["fnfp"].get(a, {}).get("pq")
    print(f"{a:<10} {n:5d}  {pq:.3f}               {v84 if v84 is None else round(v84, 3)}")
out.write_text(json.dumps(res, indent=1) + "\n")
