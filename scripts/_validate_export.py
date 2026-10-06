"""Validate an export CSV (or shard) with the repo validator against the real test image stems.

  python scripts/_validate_export.py <csv> [--allow-partial]

--allow-partial permits a CSV that covers only a subset of the 180 test stems (smoke shards);
a full submission must NOT use it.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reproduce_v17_from_cache as v50  # noqa: E402
import validate_submission as V  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("csv", type=Path)
ap.add_argument("--allow-partial", action="store_true")
args = ap.parse_args()
stems = sorted(p.stem for p in (v50.DATA_ROOT / "test/test_images").glob("*.jpeg"))
report = V.validate_submission(args.csv, stems)
print(json.dumps(report, indent=2))
if not args.allow_partial and report["populated_images"] < 1:
    sys.exit("no populated images")
print("full test coverage:", len(stems), "stems known; populated:", report["populated_images"])
