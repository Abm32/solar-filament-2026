"""V60D pre-check: how often does the baseline tile dataset hit its "empty tile" hack?

v60_tiling.make_tile_dataset replaces an EMPTY tile's targets (no GT instance >= MIN_INST_AREA
pixels inside the crop) with a FULL-TILE box and an all-ones mask labelled "filament". If empty
tiles are common, the baseline detector was trained to call whole empty tiles a filament,
which would distort both its false-positive behaviour and its score calibration.

Counts, for the fold-K training files (FIT-only, no tune/test):
  * tiles total, tiles with >=1 GT instance, EMPTY tiles (the hack fires)
  * per-tile GT-instance count distribution

  python scripts/diagnose_v60d_empty_tiles.py --fold 0 --annotations <ann> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import train_v52_cldice_semantic as v52  # noqa: E402
import train_v56_maskrcnn as v56  # noqa: E402
import train_v58_fold as fold  # noqa: E402
import v60_tiling as T  # noqa: E402


def main():
    from pycocotools import mask as mask_utils

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=float, default=0.3)
    args = ap.parse_args()
    if v52.sha256(args.annotations) != v56.ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")

    doc = json.loads(args.annotations.read_text())
    images = {str(r["id"]): r for r in doc["images"]}
    annotations = v52.load_annotations(doc)
    by_file = v52.canonical_split(doc)[2]
    files, _ = fold.fold_train_files(doc, args.fold)
    recs = v52.records_for(files, by_file, images, annotations)
    origins = T.tile_origins(args.tile, args.overlap)

    n_tiles = n_empty = 0
    counts = []
    for _name, image_id in recs:
        info = images[image_id]
        masks = []
        for ann in annotations[image_id]:
            m = mask_utils.decode(v52.annotation_rle(mask_utils, info, ann))
            masks.append(m[..., 0] if m.ndim == 3 else m)
        for ox in origins:
            for oy in origins:
                k = sum(int(m[oy:oy + args.tile, ox:ox + args.tile].sum()) >= T.MIN_INST_AREA
                        for m in masks)
                n_tiles += 1
                n_empty += int(k == 0)
                counts.append(k)
    counts = np.asarray(counts)
    report = {"fold": args.fold, "records": len(recs), "tiles": n_tiles,
              "empty_tiles_hack_fires": n_empty,
              "empty_frac": round(n_empty / max(n_tiles, 1), 4),
              "mean_instances_per_nonempty_tile": float(counts[counts > 0].mean()),
              "p50_p90_max_instances": [float(np.percentile(counts, 50)),
                                        float(np.percentile(counts, 90)), int(counts.max())],
              "reads_tune": False, "reads_test": False}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
