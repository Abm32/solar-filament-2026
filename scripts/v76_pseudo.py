"""V76: fold-K teacher pseudo-labels on unlabelled GONG archive frames (training only).
Pre-registration: docs/v76_noisy_student_preregistration_2026-10-10.md

Teacher = the p2first fold-K detector (never saw fold-K labels), 4-flip views exactly as
eval_v61_oof (V17 cell), flip-consensus groups (score = sum(view scores) / 4). Keep groups with
score >= CONF, erode 1 px (3x3 cross: direct masks are ~1 px too wide, V69 audit), pixel
ownership in score order, drop < MIN_AREA, write YOLO polygon labels. Frames with no kept mask
are skipped (an empty frame is more likely a teacher miss than a clean negative).
  python scripts/v76_pseudo.py --fold 0 --frames /mnt/v76/frames --out /mnt/v76/pseudo_fold0 \
      [--device cpu] [--shard-index i --num-shards n]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import analysis_v69_fusion as F  # noqa: E402
import train_v21_full_detector as v21  # noqa: E402
import v61_toolkit as K  # noqa: E402

CONF = 0.35  # teacher operating point (V69 audit: Dinkelbach optimum 0.30-0.35)
MIN_AREA = 200
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


def pseudo_masks(views):
    import cv2
    from pycocotools import mask as mu
    groups = F.groups_of(views)
    scored = sorted(((sum(g["scores"]) / len(views), g["rle"]) for g in groups),
                    key=lambda x: -x[0])
    occupied, kept = None, []
    for s, rle in scored:
        if s < CONF:
            break
        m = cv2.erode(mu.decode(rle), CROSS).astype(bool)
        if occupied is None:
            occupied = np.zeros_like(m)
        m &= ~occupied
        if m.sum() < MIN_AREA:
            continue
        occupied |= m
        kept.append((s, m))
    return kept


def polygons(m):
    import cv2
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1).tolist() for c in cs if len(c) >= 3 and cv2.contourArea(c) >= 20]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, required=True, choices=range(5))
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shard-index", type=int, default=0, dest="shard_index")
    ap.add_argument("--num-shards", type=int, default=1, dest="num_shards")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from ultralytics import YOLO
    weights = Path(f"/mnt/v61/p2first_fold{args.fold}/mask_ratio1/final.pt")
    model = YOLO(str(weights))
    model.to(args.device)
    frames = sorted(args.frames.glob("*.jp*g"))[args.shard_index::args.num_shards]
    frames = frames[: args.limit] if args.limit else frames
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    stats = []
    for i, f in enumerate(frames, 1):
        lab = args.out / "labels" / f"{f.stem}.txt"
        done = args.out / "done" / f"{f.stem}.json"
        if done.exists():
            continue
        try:
            views = K.generate_views(model, f, iou=0.3, imgsz=1024)
        except ValueError:  # not a 2048x2048 full disk
            rec = dict(frame=f.name, status="bad_image")
        else:
            kept = pseudo_masks(views)
            lines = [v21.polygon_line(p, 2048, 2048) for _, m in kept for p in polygons(m)]
            rec = dict(frame=f.name, status="kept" if lines else "empty", masks=len(kept),
                       lines=len(lines), scores=[round(s, 4) for s, _ in kept])
            if lines:
                lab.write_text("\n".join(lines) + "\n")
                link = args.out / "images" / f.name
                if not link.exists():
                    link.symlink_to(f.resolve())
        done.parent.mkdir(exist_ok=True)
        done.write_text(json.dumps(rec))
        stats.append(rec)
        if i % 25 == 0:
            print(f"shard {args.shard_index}: {i}/{len(frames)}", flush=True)
    print(json.dumps({"shard": args.shard_index, "frames": len(frames), "teacher": str(weights),
                      "kept": sum(r["status"] == "kept" for r in stats)}))


if __name__ == "__main__":
    main()
