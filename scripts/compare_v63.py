"""Pair the V63 refiner arms on the 141 canonical held-out files at the fixed V17 cell.

Pre-registered (written before any V63 run):
  PRIMARY  r1 (p2first proposals, V11 recipe re-fit) - v11_baseline (current V17-equivalent).
  PASS iff delta >= +0.010 AND file-bootstrap P(delta > 0) >= 0.90.
  Reported, never gating: r1 - r2 (detector effect, matched refiner recipe), r2 - v11_baseline
  (re-fit effect), r1 - r1s2 (refiner seed noise), r3 - r1 and r3 - v11_baseline (12 epochs).

  python scripts/compare_v63.py --dir outputs/v63
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CELL = "t0.45_c0.30"
BAR_DELTA, BAR_P = 0.010, 0.90
CONTRASTS = [("primary", "v11_baseline", "r1"), ("detector_matched", "r2", "r1"),
             ("refit_effect", "v11_baseline", "r2"), ("refiner_seed_noise", "r1s2", "r1"),
             ("longer_training", "r1", "r3"), ("r3_vs_current", "v11_baseline", "r3"),
             ("v11_p2first_vs_current", "v11_baseline", "v11_p2first")]


def pq(t):
    d = t[1] + 0.5 * (t[2] + t[3])
    return float(t[0] / d) if d else 0.0


def paired(a, b, seed=63, n=4000):
    rng = np.random.default_rng(seed)
    d = [pq(b[i].sum(0)) - pq(a[i].sum(0))
         for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return {"delta": pq(b.sum(0)) - pq(a.sum(0)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_gt_0": float(np.mean(np.asarray(d) > 0))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--pairs", nargs="+", default=[], help="extra a:b contrasts (b - a)")
    args = ap.parse_args()
    contrasts = CONTRASTS + [(f"{a}->{b}", a, b) for a, b in (p.split(":") for p in args.pairs)]
    arms = {p.stem: json.loads(p.read_text()) for p in sorted(args.dir.glob("*.json"))
            if p.stem != "compare"}
    names = sorted(next(iter(arms.values()))["per_file"])
    arr = {k: np.asarray([v["per_file"][n][CELL] for n in names]) for k, v in arms.items()
           if sorted(v["per_file"]) == names}
    rep = {"files": len(names), "cell": CELL,
           "arms": {k: {"pq": pq(a.sum(0)), "tp": int(a.sum(0)[1]), "fp": int(a.sum(0)[2]),
                        "fn": int(a.sum(0)[3])} for k, a in arr.items()},
           "contrasts": {}}
    for label, a, b in contrasts:
        if a in arr and b in arr:
            rep["contrasts"][label] = {"a": a, "b": b, **paired(arr[a], arr[b])}
    p = rep["contrasts"].get("primary")
    rep["primary_passed"] = bool(p and p["delta"] >= BAR_DELTA and p["p_gt_0"] >= BAR_P)
    (args.dir / "compare.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
