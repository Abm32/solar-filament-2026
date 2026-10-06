"""Recover interrupted V30 training from hash-pinned epoch 25 to epoch 40."""
from __future__ import annotations
import argparse, hashlib, os
from pathlib import Path

CHECKPOINT_SHA256 = "46c0c478bc129471e19ccfb6d83bec2b099b89e220151eb0fb05df888da76e0f"
TOTAL_EPOCHS = 40


def sha256(path: Path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): d.update(block)
    return d.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    a = p.parse_args()
    if sha256(a.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("V30 recovery checkpoint changed")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from ultralytics import YOLO
    model = YOLO(str(a.checkpoint))
    # Ultralytics resume restores its stored 40-epoch contract and continues at 27.
    model.train(resume=True, device=[0, 1, 2, 3])


if __name__ == "__main__": main()
