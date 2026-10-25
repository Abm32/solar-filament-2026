"""V81 label geometry: area-matched ("thin") polygons and per-file union-of-annotators labels.

Imported by train_v61_fold.py (after scripts/patch_v81.py) to build a recipe-specific YOLO dataset
dir `<work>/yolo_fold<K>_<kind>`; never touches the baseline `yolo_fold<K>` dir. Train json only.

  python scripts/v81_labels.py fidelity [N]          # CPU: Ultralytics-rasterised area ratio / IoU vs GT
  python scripts/v81_labels.py stats --fold 0        # CPU: union label counts on fold-0 train files
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

S = 2048
SHRINK = 1.0       # px at 2048 (inward vertex offset), fixed by the V81 CPU fidelity check before training
MATCH_IOU = 0.5    # cross-annotator duplicate threshold (same as the V79 oracles)


def gt_mask(poly) -> np.ndarray:
    return mu.decode(mu.frPyObjects([poly], S, S))[:, :, 0]


def offset(poly, d: float):
    """Move every vertex d px inward along its miter (mitre length capped at 2d). Sub-pixel exact."""
    p = np.asarray(poly, np.float64).reshape(-1, 2)
    p = p[np.any(p != np.roll(p, -1, 0), axis=1)]           # drop repeated vertices
    if len(p) < 3 or d == 0:
        return list(poly)
    x, y = p[:, 0], p[:, 1]
    sgn = np.sign(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) or 1.0   # + = CCW in image coords
    e = np.roll(p, -1, 0) - p
    n = sgn * np.stack([-e[:, 1], e[:, 0]], 1) / np.linalg.norm(e, axis=1, keepdims=True)  # inward
    a, b = np.roll(n, 1, 0), n                                  # normals of edges into / out of vertex
    m = (a + b) / np.maximum(1 + np.sum(a * b, 1, keepdims=True), 0.25)  # |m| <= 2 (cap at 2d)
    return (p + d * m).ravel().tolist()


def union_file(records: list[list]) -> tuple[list, dict]:
    """records: per annotator record, a list of polygons. Returns (union polygons, stats).

    Greedy cross-record clustering (IoU > MATCH_IOU, at most one member per record, same as
    audit_v79 multi_oracles); each cluster contributes its lower-median-area member's polygon.
    """
    R = len(records)
    rles = [[mu.frPyObjects([p], S, S)[0] for p in rec] for rec in records]
    area = [[float(mu.area(r)) for r in rec] for rec in rles]
    cross = {(i, j): mu.iou(rles[i], rles[j], [0] * len(rles[j]))
             for i in range(R) for j in range(R) if i != j and rles[i] and rles[j]}
    done, out, sizes = set(), [], []
    for r in range(R):
        for k in range(len(rles[r])):
            if (r, k) in done:
                continue
            c = {r: k}
            done.add((r, k))
            for r2 in range(R):
                if r2 == r or (r, r2) not in cross:
                    continue
                row = cross[(r, r2)][k]
                for k2 in np.argsort(-row):
                    if row[k2] > MATCH_IOU and (r2, int(k2)) not in done:
                        c[r2] = int(k2)
                        done.add((r2, int(k2)))
                        break
            members = sorted(c.items(), key=lambda rk: area[rk[0]][rk[1]])
            rr, kk = members[(len(members) - 1) // 2]
            out.append(records[rr][kk])
            sizes.append(len(c))
    n = sum(len(x) for x in records)
    st = {"records": R, "per_record": [len(x) for x in records], "union": len(out),
          "merged_duplicates": n - len(out),
          # an unmatched instance is background in each of the R-len(c) other copies
          "conflicts_removed": sum(R - s for s in sizes) if R > 1 else 0,
          "singletons": sum(s == 1 for s in sizes) if R > 1 else 0}
    return out, st


def build(kind: str, train_files, data: Path, ann_path: Path, image_dir: Path, polygon_line) -> dict:
    """Write <data>/images/train (symlinks), labels/train, data.yaml, manifest.json."""
    assert kind in ("thin", "union"), kind
    manifest = data / "manifest.json"
    keep = set(train_files)
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["train_files"] == len(keep) and saved["kind"] == kind:
            return saved
        raise RuntimeError(f"refusing to overwrite differing dataset {data}")
    coco = json.loads(ann_path.read_text())
    anns: dict = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    files: dict = {}
    for info in sorted(coco["images"], key=lambda i: str(i["id"])):
        if info["file_name"] in keep:
            assert (info["width"], info["height"]) == (S, S), info["id"]
            files.setdefault(info["file_name"], []).append(info["id"])
    if set(files) != keep:
        raise RuntimeError("train files missing from annotation json")
    img_out, lab_out = data / "images/train", data / "labels/train"
    img_out.mkdir(parents=True)
    lab_out.mkdir(parents=True)
    tot = {"images": 0, "label_lines": 0, "records": 0, "per_record_lines": 0,
           "merged_duplicates": 0, "conflicts_removed": 0, "singletons_multi": 0}
    for name, ids in sorted(files.items()):
        recs = [[p for a in anns.get(i, []) for p in a["segmentation"] if len(p) >= 6] for i in ids]
        tot["records"] += len(ids)
        tot["per_record_lines"] += sum(map(len, recs))
        p = Path(name)
        if kind == "thin":
            items = []
            for i, rec in zip(ids, recs):
                polys = [offset(q, SHRINK) for q in rec]
                items.append((f"{p.stem}_ann{i}{p.suffix}", polys))
        else:
            polys, st = union_file(recs)
            for k in ("merged_duplicates", "conflicts_removed"):
                tot[k] += st[k]
            tot["singletons_multi"] += st["singletons"]
            items = [(f"{p.stem}_union{p.suffix}", polys)]
        for target, polys in items:
            (img_out / target).symlink_to(image_dir / name)
            lines = [polygon_line(q, S, S) for q in polys]
            (lab_out / f"{Path(target).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            tot["images"] += 1
            tot["label_lines"] += len(lines)
    (data / "data.yaml").write_text(
        f"path: {data}\ntrain: images/train\nval: images/train\nnames:\n  0: filament\n")
    saved = dict(kind=kind, train_files=len(keep), shrink_px=SHRINK if kind == "thin" else None,
                 match_iou=MATCH_IOU if kind == "union" else None, **tot)
    manifest.write_text(json.dumps(saved, indent=2) + "\n")
    return saved


def _fid(poly):
    """audit_v79 'yolo1024' path: polygon_line rounding -> polygon2mask @1024 -> bilinear up >= 0.5."""
    from ultralytics.data.utils import polygon2mask
    g = gt_mask(poly).astype(bool)
    res = {}
    for name, q in (("orig", poly), *((f"shrink{d}", offset(poly, d)) for d in (0.5, 0.75, 1.0))):
        for ratio in (1, 2):  # stage-2 (mask_ratio 1, 1024) and stage-1 (mask_ratio 2, 512) targets
            seg = np.round(np.clip(np.asarray(q, np.float64).reshape(-1, 2) / S, 0, 1), 6) * 1024
            m = polygon2mask((1024, 1024), [seg.reshape(-1)], downsample_ratio=ratio)
            pm = cv2.resize(m.astype(np.float32), (S, S), interpolation=cv2.INTER_LINEAR) >= 0.5
            u = np.logical_or(pm, g).sum()
            res[f"{name}@{1024 // ratio}"] = (float(pm.sum() / g.sum()), float(np.logical_and(pm, g).sum() / u))
    return res


def _main():
    from multiprocessing import Pool
    root = Path(__file__).resolve().parents[1]
    ann = root / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = json.loads(ann.read_text())
    if sys.argv[1] == "fidelity":
        sq = np.asarray(offset([0, 0, 10, 0, 10, 10, 0, 10], 1.0)).reshape(-1, 2)   # either orientation
        assert np.allclose(sq.min(0), 1) and np.allclose(sq.max(0), 9), sq
        assert np.allclose(np.asarray(offset([0, 0, 0, 10, 10, 10, 10, 0], 1.0)).reshape(-1, 2).min(0), 1)
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        polys = [a["segmentation"][0] for a in coco["annotations"]]
        idx = np.random.default_rng(81).choice(len(polys), min(n, len(polys)), replace=False)
        with Pool(32) as pool:
            rows = pool.map(_fid, [polys[i] for i in idx], chunksize=8)
        rep = {k: {"median_area_ratio": round(float(np.median([r[k][0] for r in rows])), 4),
                   "mean_iou": round(float(np.mean([r[k][1] for r in rows])), 4),
                   "frac_iou_le_0_5": round(float(np.mean([r[k][1] <= 0.5 for r in rows])), 4)}
               for k in rows[0]}
        print(json.dumps({"n": len(rows), **rep}, indent=2))
    else:  # stats: union counts on the fold-0 train files, in memory
        fold = int(sys.argv[sys.argv.index("--fold") + 1])
        m = json.loads((root / "outputs/v61_folds.json").read_text())
        keep = {f for k, v in m["folds"].items() if int(k) != fold for f in v}
        anns: dict = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a["segmentation"][0])
        files: dict = {}
        for info in coco["images"]:
            if info["file_name"] in keep:
                files.setdefault(info["file_name"], []).append(anns.get(info["id"], []))
        with Pool(32) as pool:
            sts = pool.map(_union_stats, list(files.values()), chunksize=4)
        multi = [s for s in sts if s["records"] > 1]
        print(json.dumps({
            "files": len(sts), "records": sum(s["records"] for s in sts),
            "per_record_instances": sum(sum(s["per_record"]) for s in sts),
            "union_instances": sum(s["union"] for s in sts),
            "multi_record_files": len(multi),
            "multi_mean_per_record": round(float(np.mean([np.mean(s["per_record"]) for s in multi])), 2),
            "multi_mean_union": round(float(np.mean([s["union"] for s in multi])), 2),
            "multi_union_over_max_record": round(float(np.mean([s["union"] / max(max(s["per_record"]), 1) for s in multi])), 3),
            "merged_duplicates": sum(s["merged_duplicates"] for s in sts),
            "singletons_in_multi_files": sum(s["singletons"] for s in multi),
            "conflicts_removed": sum(s["conflicts_removed"] for s in sts),
            "empty_records": sum(x == 0 for s in sts for x in s["per_record"]),
            "mean_instances_per_image_record": round(sum(sum(s["per_record"]) for s in sts) / sum(s["records"] for s in sts), 2),
            "mean_instances_per_image_union": round(sum(s["union"] for s in sts) / len(sts), 2),
        }, indent=2))


def _union_stats(recs):
    return union_file(recs)[1]


if __name__ == "__main__":
    _main()
