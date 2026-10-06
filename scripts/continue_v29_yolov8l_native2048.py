"""Archived V29 continuation attempt; intentionally refuses to launch.

The 2026-09-13 attempt passed ``epochs=40`` to an Ultralytics resume from the
optimizer-bearing epoch-15 checkpoint. Ultralytics 8.4.138 restored the stored
20-epoch checkpoint contract, ignored that override, and only ran epochs 17-20.
Therefore no V29 40-epoch model exists. Keep this guard so the failed recovery
mechanism cannot silently be reused or misreported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CHECKPOINT_SHA256 = "d26cdb664b62cae949701805b9771864ab4208c43d7c5649e2f59fe7469842c8"
ATTEMPTED_TOTAL_EPOCHS = 40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("V29 optimizer-bearing continuation checkpoint changed")
    contract = json.loads(args.contract.read_text())
    frozen = contract["arguments"]
    required = {"imgsz": 2048, "mask_ratio": 4, "single_cls": True, "batch": 4}
    if any(frozen.get(key) != value for key, value in required.items()):
        raise RuntimeError("V29 original training contract changed")
    raise RuntimeError(
        "V29 continuation is retired: Ultralytics resume restored the stored "
        "20-epoch contract and did not continue to 40. No 40-epoch V29 result exists."
    )


if __name__ == "__main__":
    main()
