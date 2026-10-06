"""Count pretrained-weight transfer at the REAL training shape (nc=1), per module group.

The trainer rebuilds the model with nc=1 (single_cls) and loads the init checkpoint through
Ultralytics' intersect_dicts (same key AND same shape). Detect sizes its classification branch as
c3 = max(ch[0], min(nc, 100)), so at nc=1 c3 == ch[0] = channels of the FIRST head input. In the
stock graph ch[0] is P3 (128 at scale s) == COCO's c3 (max(128, 80)). In `p2first` ch[0] is P2
(64), so every classification-branch tensor has a different shape than COCO and is NOT transferred.
This script reports exactly what transfers for: baseline, p2last, p2first.

  CUDA_VISIBLE_DEVICES="" python scripts/_v62_transfer_nc1.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def group(key: str) -> str:
    m = re.match(r"^model\.(\d+)\.(.*)$", key)
    if not m:
        return "other"
    idx, rest = int(m.group(1)), m.group(2)
    if idx <= 10:
        return "backbone(0-10)"
    if idx <= 22:
        return "neck(11-22)"
    if idx <= 25:
        return "new P2 neck(23-25)" if idx > 22 and "cv" not in rest else "head"
    for pre in ("cv2.", "cv3.", "cv4.", "proto.", "dfl."):
        if rest.startswith(pre):
            return "head." + pre.strip(".")
    return "head.other"


def main():
    import v62_arch
    import v62_init
    from ultralytics import YOLO
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.torch_utils import intersect_dicts

    v62_arch.write_all()
    coco = YOLO("yolo11s-seg.pt").model.state_dict()
    for name, cfg, variant in (("baseline", "yolo11s-seg.yaml", None),
                               ("p2last", v62_arch.CONFIGS / "yolo11s-p2last-seg.yaml", "p2last"),
                               ("p2first", v62_arch.CONFIGS / "yolo11s-p2first-seg.yaml",
                                "p2first")):
        net = SegmentationModel(str(cfg), ch=3, nc=1, verbose=False)
        new_sd = net.state_dict()
        if variant is None:
            src = coco
        else:
            # remap COCO keys the same way build() does, but compare shapes at nc=1
            src = {}
            for k, v in coco.items():
                nk = v62_init.remap_key(k, variant)
                if nk is not None:
                    src[nk] = v
        ok = intersect_dicts(src, new_sd)
        tot, got = Counter(), Counter()
        for k in new_sd:
            g = group(k)
            tot[g] += 1
            got[g] += int(k in ok)
        total_t, total_g = sum(tot.values()), sum(got.values())
        print(f"\n{name}: transferred {total_g}/{total_t} tensors at nc=1")
        for g in sorted(tot):
            print(f"   {g:22s} {got[g]:4d}/{tot[g]:4d}")


if __name__ == "__main__":
    main()
