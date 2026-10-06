"""Direct-detector (no refiner) p2first - baseline on the SAME 141 canonical held-out files used by
eval_v62_refiner.py, so the refiner's effect on the gain can be separated from subset effects.

Also reports the same contrast on the complementary 566 files (refiner-SEEN files; shown only to
compare the direct gain, never to evaluate a refiner).

  python scripts/analyze_v62_subset_direct.py --out outputs/v62_subset_direct.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import eval_v61_compare as C  # noqa: E402
import eval_v62_5fold as F  # noqa: E402
import reproduce_v17_from_cache as v50  # noqa: E402
import v61_toolkit as K  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default="baseline")
    ap.add_argument("--b", default="p2first")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rt = v50.runtime_state()
    canon = set(rt["split"][0])
    da, db = F.pooled_detail(args.a), F.pooled_detail(args.b)
    names = [f["name"] for f in da]
    assert names == [f["name"] for f in db]
    conf = F.PRIMARY_CONF
    ca, cb = F.counts_at(da, conf), F.counts_at(db, conf)
    mask = np.asarray([n in canon for n in names])
    rep = {"primary_conf": conf}
    for tag, m in (("canonical_141_refiner_unseen", mask), ("other_566_refiner_seen", ~mask),
                   ("all_707", np.ones_like(mask))):
        pb = C.paired_bootstrap(ca[m], cb[m])
        rep[tag] = {"files": int(m.sum()), "pq_a": K.pq_value(ca[m].sum(0)),
                    "pq_b": K.pq_value(cb[m].sum(0)), **pb}
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    for k, v in rep.items():
        if isinstance(v, dict):
            print(f"{k:32s} files {v['files']:4d}  A {v['pq_a']:.4f}  B {v['pq_b']:.4f}  "
                  f"delta {v['delta']:+.4f}  CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
