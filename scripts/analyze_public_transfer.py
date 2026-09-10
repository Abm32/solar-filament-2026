"""Does local tune PQ still predict the public score? (zero GPU)

`ceiling_analysis_2026-09-07.md` §1 asserts a *stable* local->public offset of
-0.050 +- 0.003 across four submissions, and derives the whole strategy from it:
"public 0.55 requires local tune PQ ~= 0.60". Every experiment since (V14, V15,
V16, V17) has been justified by that arithmetic.

V17 is the fifth calibration point and the first that can falsify the model:
local 0.448703, which under a -0.050 offset predicts public ~0.399, and under the
tightest observed offset (-0.0473) predicts 0.4014. It scored **0.39**.

This script asks three questions, all from data already on disk:

  Q1. Is ANY affine model public = a*local + b consistent with all five
      submissions under two-decimal display? If the feasible region is empty,
      local tune PQ is not an affine predictor of public score and the -0.050
      offset cannot be used to set targets.

  Q2. What does the feasible region imply for the target? If a << 1, then
      reaching local 0.60 does NOT reach public 0.55, and `ceiling_analysis` §8's
      entire expected-value calculation is wrong.

  Q3. Is tune-set selection overfitting? Across V17's 84 predeclared candidates,
      does the tune-argmax agree with the audit-argmax, and how large is the tune
      spread relative to the margin we have been treating as signal? Five
      submissions in a row selected the *edge* of the confidence grid (c0.30),
      which is a classic overfitting signature.

Usage:
    python scripts/analyze_public_transfer.py
    python scripts/analyze_public_transfer.py --json outputs/public_transfer.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The five measured (local tune PQ, public score) pairs. Sources:
#   E015  docs/experiments.md E015, submission 56026663
#   V11g  E020, submission 56050554
#   V11p  E019, submission 56049359
#   V13   E024, submission 56076685
#   V17   outputs/kaggle_v17_full/v17_report.json, submission 56103401
SUBMISSIONS = [
    ("E015 consensus TTA", 0.432823, 0.38, "56026663"),
    ("V11 gt_centered_e4", 0.437313, 0.39, "56050554"),
    ("V11 proposal_centered_e4", 0.438001, 0.39, "56049359"),
    ("V13 compose", 0.441777, 0.39, "56076685"),
    ("V17 compose", 0.448703, 0.39, "56103401"),
]

V13_TUNE = 0.441777
V13_AUDIT = 0.464644


def display_interval(public, mode):
    """The interval of true scores consistent with a displayed 2-decimal value."""
    if mode == "round":
        return (public - 0.005, public + 0.005)
    if mode == "truncate":
        return (public, public + 0.010)
    raise ValueError(mode)


def constant_offset_feasibility(mode):
    """Q1a: is a CONSTANT offset (a=1) consistent with all five points?"""
    lo, hi = -1.0, 1.0
    per_point = []
    for name, local, public, ref in SUBMISSIONS:
        plo, phi = display_interval(public, mode)
        # public = local - offset  =>  offset = local - public
        olo, ohi = local - phi, local - plo
        per_point.append({"name": name, "local": local, "public": public,
                          "ref": ref, "offset_lo": olo, "offset_hi": ohi})
        lo, hi = max(lo, olo), min(hi, ohi)
    return {"mode": mode, "per_point": per_point,
            "intersection_lo": lo, "intersection_hi": hi,
            "feasible": lo < hi}


def affine_feasibility(mode, a_grid):
    """Q1b/Q2: for each slope a, is there an intercept b fitting all five points?

    For a fixed a, each point constrains b to [plo - a*local, phi - a*local].
    Feasible iff the intersection over points is non-empty.
    """
    feasible = []
    for a in a_grid:
        lo, hi = -10.0, 10.0
        for _name, local, public, _ref in SUBMISSIONS:
            plo, phi = display_interval(public, mode)
            lo, hi = max(lo, plo - a * local), min(hi, phi - a * local)
        if lo < hi:
            feasible.append({"a": a, "b_lo": lo, "b_hi": hi,
                             # what local PQ would be needed for public 0.55?
                             "local_needed_for_0_55": (0.55 - (lo + hi) / 2) / a if a > 0 else None,
                             # what public would local 0.60 produce?
                             "public_at_local_0_60": a * 0.60 + (lo + hi) / 2})
    return feasible


def selection_overfitting():
    """Q3: tune-argmax vs audit-argmax across V17's 84 predeclared candidates."""
    path = REPO / "outputs/kaggle_v17_full/v17_results.json"
    if not path.exists():
        return {"available": False, "reason": f"missing {path}"}
    data = json.loads(path.read_text())
    out = {"available": True, "seeds": {}}
    for seed, cells in data["composition"].items():
        rows = []
        for cell, groups in cells.items():
            row = {"cell": cell}
            for group in ("tune", "audit"):
                iou_sum, tp, fp, fn = groups[group]
                denom = tp + 0.5 * fp + 0.5 * fn
                row[f"{group}_pq"] = iou_sum / denom if denom else 0.0
                if group == "tune":
                    row["tune_tp"], row["tune_fp"], row["tune_fn"] = tp, fp, fn
                    row["tune_recall"] = tp / (tp + fn) if (tp + fn) else 0.0
                    row["tune_mean_iou"] = iou_sum / tp if tp else 0.0
            rows.append(row)
        rows.sort(key=lambda r: -r["tune_pq"])
        best_tune = rows[0]
        best_audit = max(rows, key=lambda r: r["audit_pq"])
        # how many candidates sit within the E026 training-noise floor of the top?
        NOISE = 0.007677
        within = [r for r in rows if best_tune["tune_pq"] - r["tune_pq"] <= NOISE]
        # is the selected cell on the boundary of the confidence grid?
        confs = sorted({c.split("_c")[1] for c in cells})
        thresholds = sorted({c.split("/")[1].split("_c")[0].replace("t", "") for c in cells})
        sel_conf = best_tune["cell"].split("_c")[1]
        sel_thr = best_tune["cell"].split("/")[1].split("_c")[0].replace("t", "")
        out["seeds"][seed] = {
            "n_candidates": len(rows),
            "selected_by_tune": best_tune["cell"],
            "selected_tune_pq": best_tune["tune_pq"],
            "selected_audit_pq": best_tune["audit_pq"],
            "best_by_audit": best_audit["cell"],
            "best_audit_pq": best_audit["audit_pq"],
            "tune_and_audit_agree": best_tune["cell"] == best_audit["cell"],
            "audit_cost_of_tune_selection": best_audit["audit_pq"] - best_tune["audit_pq"],
            "tune_pq_range": [rows[-1]["tune_pq"], rows[0]["tune_pq"]],
            "n_within_noise_floor_of_top": len(within),
            "noise_floor_used": NOISE,
            "confidence_grid": confs,
            "threshold_grid": thresholds,
            "selected_confidence": sel_conf,
            "selected_threshold": sel_thr,
            "selected_conf_is_grid_edge": sel_conf in (confs[0], confs[-1]),
            "selected_thr_is_grid_edge": sel_thr in (thresholds[0], thresholds[-1]),
            "top5": [{"cell": r["cell"], "tune_pq": r["tune_pq"], "audit_pq": r["audit_pq"],
                      "tune_fp": r["tune_fp"], "tune_recall": r["tune_recall"]} for r in rows[:5]],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report = {"submissions": [
        {"name": n, "local_tune_pq": l, "public": p, "ref": r} for n, l, p, r in SUBMISSIONS
    ]}

    print("=" * 78)
    print("Q1. Is a CONSTANT local->public offset consistent with all five points?")
    print("=" * 78)
    report["constant_offset"] = {}
    for mode in ("round", "truncate"):
        res = constant_offset_feasibility(mode)
        report["constant_offset"][mode] = res
        print(f"\n  display model: {mode}")
        for p in res["per_point"]:
            print(f"    {p['name']:26} local={p['local']:.6f} public={p['public']:.2f}"
                  f"  -> offset in ({p['offset_lo']:+.6f}, {p['offset_hi']:+.6f}]")
        verdict = "FEASIBLE" if res["feasible"] else "EMPTY -> constant offset IMPOSSIBLE"
        print(f"    intersection: ({res['intersection_lo']:+.6f}, "
              f"{res['intersection_hi']:+.6f}]  => {verdict}")

    print()
    print("=" * 78)
    print("Q2. Which affine models public = a*local + b survive all five points?")
    print("=" * 78)
    a_grid = [round(x * 0.01, 2) for x in range(0, 201)]
    report["affine"] = {}
    for mode in ("round", "truncate"):
        feas = affine_feasibility(mode, a_grid)
        report["affine"][mode] = {
            "n_feasible_slopes": len(feas),
            "slope_min": min((f["a"] for f in feas), default=None),
            "slope_max": max((f["a"] for f in feas), default=None),
            "feasible": feas,
        }
        print(f"\n  display model: {mode}")
        if not feas:
            print("    no affine model fits -> relationship is not even monotone-affine")
            continue
        amin, amax = min(f["a"] for f in feas), max(f["a"] for f in feas)
        print(f"    feasible slopes a in [{amin:.2f}, {amax:.2f}]  "
              f"({len(feas)} of {len(a_grid)} tested)")
        print(f"    slope a=1.00 (the assumed constant offset) is "
              f"{'FEASIBLE' if any(abs(f['a'] - 1.0) < 1e-9 for f in feas) else 'RULED OUT'}")
        print("    implications at the extremes of the feasible set:")
        for f in (feas[0], feas[len(feas) // 2], feas[-1]):
            need = f["local_needed_for_0_55"]
            need_s = f"{need:.4f}" if need is not None else "n/a"
            print(f"      a={f['a']:.2f} b~{(f['b_lo'] + f['b_hi']) / 2:+.4f}"
                  f"  local needed for public 0.55: {need_s}"
                  f"  |  public at local 0.60: {f['public_at_local_0_60']:.4f}")

    print()
    print("=" * 78)
    print("Q3. Is tune-set candidate selection overfitting?")
    print("=" * 78)
    sel = selection_overfitting()
    report["selection"] = sel
    if not sel["available"]:
        print(f"  unavailable: {sel['reason']}")
    else:
        for seed, s in sel["seeds"].items():
            print(f"\n  {seed}: {s['n_candidates']} candidates")
            print(f"    tune-argmax : {s['selected_by_tune']:24} "
                  f"tune={s['selected_tune_pq']:.6f} audit={s['selected_audit_pq']:.6f}")
            print(f"    audit-argmax: {s['best_by_audit']:24} "
                  f"audit={s['best_audit_pq']:.6f}")
            print(f"    tune and audit agree on the winner: {s['tune_and_audit_agree']}")
            print(f"    audit left on the table by tune-selection: "
                  f"{s['audit_cost_of_tune_selection']:+.6f}")
            print(f"    candidates within the E026 noise floor "
                  f"({s['noise_floor_used']:.6f}) of the top: "
                  f"{s['n_within_noise_floor_of_top']} of {s['n_candidates']}")
            print(f"    selected confidence {s['selected_confidence']} "
                  f"(grid {s['confidence_grid'][0]}..{s['confidence_grid'][-1]}) "
                  f"-> grid edge: {s['selected_conf_is_grid_edge']}")
            print(f"    selected threshold {s['selected_threshold']} "
                  f"(grid {s['threshold_grid'][0]}..{s['threshold_grid'][-1]}) "
                  f"-> grid edge: {s['selected_thr_is_grid_edge']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
