"""V95: one-knob perturbations of a submission-format CSV (OOF 707-file CSV or the 180-image export).
Ownership order = row order within an image (earlier row wins). Output stays pixel-disjoint. CPU only.
  python scripts/v95_probe_ops.py SRC DST OP [ARG]
OP: shift DX,DY | minarea A | topk K | dropsite XX (e.g. Bh) | dropyears 2011,2012 | maxrr Q (drop masks whose centroid r/R > Q)
    | merge G (union masks of an image that touch after dilating each by G px; merged mask = union of originals)
maxrr reads the image (train_images or test_images by stem) to fit the disk; nothing else reads images.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MIN_AREA = 200
src, dst, op = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
arg = sys.argv[4] if len(sys.argv) > 4 else None

IMG_DIRS = [ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images",
            ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"]


def disk(stem):
    import cv2
    from v94_limb import fit_disk
    for d in IMG_DIRS:
        p = d / f"{stem}.jpeg"
        if p.exists():
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            f = fit_disk(g)
            return f["cx"], f["cy"], f["radius"]
    raise FileNotFoundError(stem)


def dec(c):
    return mask_utils.decode({"size": [2048, 2048], "counts": c.encode()}).astype(bool)


def shift(m, dx, dy):
    out = np.zeros_like(m)
    H, W = m.shape
    ys, xs = slice(max(dy, 0), H + min(dy, 0)), slice(max(dx, 0), W + min(dx, 0))
    yd, xd = slice(max(-dy, 0), H + min(-dy, 0)), slice(max(-dx, 0), W + min(-dx, 0))
    out[ys, xs] = m[yd, xd]
    return out


df = pd.read_csv(src, dtype=str, keep_default_na=False)
df["image"] = df.filament_id.str.rsplit("_", n=1).str[0]
rows, stats = [], {"op": op, "arg": arg, "in": len(df), "out": 0, "dropped": 0}
for image, g in df.groupby("image", sort=False):
    masks = [dec(c) for c in g.segmentation_rle]
    if op == "dropsite":
        if image.endswith(arg):
            stats["dropped"] += len(masks); continue
    elif op == "dropyears":  # comma list of 4-digit years
        if image[:4] in arg.split(","):
            stats["dropped"] += len(masks); continue
    elif op == "topk":
        k = int(arg); stats["dropped"] += max(0, len(masks) - k); masks = masks[:k]
    elif op == "minarea":
        a = int(arg); keep = [m for m in masks if m.sum() >= a]; stats["dropped"] += len(masks) - len(keep); masks = keep
    elif op == "shift":
        dx, dy = map(int, arg.split(",")); masks = [shift(m, dx, dy) for m in masks]
    elif op == "maxrr":
        q = float(arg); cx, cy, R = disk(image); keep = []
        for m in masks:
            ys, xs = np.nonzero(m)
            if np.hypot(xs.mean() - cx, ys.mean() - cy) / R <= q: keep.append(m)
        stats["dropped"] += len(masks) - len(keep); masks = keep
    elif op == "merge":
        gpx = int(arg)
        if len(masks) > 1:
            st = ndi.generate_binary_structure(2, 1)
            fat = [ndi.binary_dilation(m, st, iterations=gpx) for m in masks]
            n = len(masks); parent = list(range(n))
            def find(i):
                while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
                return i
            for i in range(n):
                for j in range(i + 1, n):
                    if (fat[i] & fat[j]).any(): parent[find(i)] = find(j)
            groups = {}
            for i in range(n): groups.setdefault(find(i), []).append(i)
            merged = []
            for idx in sorted(groups.values(), key=min):  # keep best-score order via smallest original index
                u = np.zeros_like(masks[0])
                for i in idx: u |= masks[i]
                merged.append(u)
            stats["dropped"] += n - len(merged); masks = merged
    else:
        raise SystemExit(f"unknown op {op}")
    occupied = np.zeros((2048, 2048), bool); n = 0
    for m in masks:
        m = m & ~occupied
        if m.sum() < MIN_AREA: stats["dropped"] += 1; continue
        occupied |= m; n += 1
        rows.append({"filament_id": f"{image}_{n}",
                     "segmentation_rle": mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()})
stats["out"] = len(rows)
dst.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(dst, index=False)
print(stats)
