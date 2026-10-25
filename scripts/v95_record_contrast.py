"""V95: are any TRAIN records 'dead' (polygons not on dark structure)? Uses the V94 limb-flattened PNGs so the
disk is flat. Per record: ring contrast = (mean in a 15-px ring around the GT union - mean inside) / disk std,
for the original geometry and flipped/transposed geometries (misregistration check). CPU, train only."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
LIMB = ROOT / "outputs/v94/images"
doc = json.loads((D / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
anns = {}
for a in doc["annotations"]: anns.setdefault(a["image_id"], []).append(a)
def one(im):
    g = cv2.imread(str(LIMB / (Path(im["file_name"]).stem + ".png")), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    disk = g > 0
    sd = g[disk].std()
    polys = anns.get(im["id"], [])
    m = np.zeros((2048, 2048), bool)
    for a in polys: m |= mu.decode(mu.merge(mu.frPyObjects(a["segmentation"], 2048, 2048))).astype(bool)
    def c(mask):
        inside = mask & disk
        if not inside.any(): return float("nan")
        ring = ndi.binary_dilation(mask, iterations=15) & ~mask & disk
        return float((g[ring].mean() - g[inside].mean()) / sd)
    return {"record": im["id"], "file": im["file_name"], "n_poly": len(polys), "gt_px": int(m.sum()),
            "c_orig": c(m), "c_lr": c(m[:, ::-1]), "c_ud": c(m[::-1]), "c_rot180": c(m[::-1, ::-1]), "c_T": c(m.T)}
if __name__ == "__main__":
    with Pool(24) as p: rows = p.map(one, doc["images"], chunksize=8)
    out = ROOT / "outputs/v95/record_contrast.json"; out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(rows, indent=0) + "\n")
    c = np.array([r["c_orig"] for r in rows]); alt = np.array([max(r["c_lr"], r["c_ud"], r["c_rot180"], r["c_T"]) for r in rows])
    print("records", len(rows), "ring contrast quantiles p1/p5/p10/p25/p50:", np.nanpercentile(c, [1, 5, 10, 25, 50]).round(2))
    print("alt-geometry contrast p50/p90/p99:", np.nanpercentile(alt, [50, 90, 99]).round(2))
    for t in (0.1, 0.2, 0.3): print(f"records with c_orig < {t}:", int((c < t).sum()))
    print("alt better than orig by >0.3:", int((alt - c > 0.3).sum()))
    for r in sorted(rows, key=lambda r: r["c_orig"])[:12]: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
