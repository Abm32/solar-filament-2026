"""Does V16's `mask_ratio=1` result replicate across two independent seeds?

Seed 1 (`outputs/kaggle_v16_full`, seed=42) produced a confusing pair of facts: arm_r1
(mask_ratio=1) broke the 0.8734 raw-proposal ceiling, but arm_r2 -- same mask_ratio as
the frozen control, included purely as a harness check -- scored 0.0077 tune PQ BELOW
that control. A clean re-run of unchanged settings should reproduce the control closely.
It didn't, which meant run-to-run noise was at least as large as the arm_r1-vs-arm_r2
gap and the PQ comparison could not be trusted in isolation.

Seed 2 (`outputs/kaggle_v16_full_seed2`, seed=20260908) is the replication that prices
that noise floor. This script does three things the raw selection JSONs cannot:

  1. Reports both seeds side by side on tune PQ and raw proposal IoU.
  2. Runs a PAIRED per-image bootstrap of arm_r1 vs the frozen control, per seed, using
     the same protocol as E023/E024 (10,000 resamples, seed 42, resample FILES). Point
     estimates are what misled seed 1; intervals are the check.
  3. Applies the sign test across seeds: a lever that is real should move the same
     direction in both draws, on both PQ and raw IoU.

Zero GPU. Reads the two runs' `v16_image_diagnostics.csv` and `v16_selection.json`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CONTROL = "v16_frozen_control"
ARM_R1 = "v16_arm_r1"
ARM_R2 = "v16_arm_r2"
RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


def read_diagnostics(path):
    """Per-(candidate, group) -> {file_name: (iou_sum, tp, fp, fn)}.

    PQ is recognition-quality weighted: iou_sum / (tp + 0.5*fp + 0.5*fn), so the
    bootstrap must resample whole files and re-aggregate, never average per-image PQ.

    A single `file_name` carries SEVERAL rows -- one per `image_id`, i.e. per annotator
    variant of the same image. Those rows must be SUMMED into one per-file record, not
    overwritten: the resampling unit is the file (that is what the 94/47 tune/audit split
    is grouped by), and dropping variants would both understate the counts and silently
    disagree with `v16_selection.json`. Summing here reproduces the shipped tune/audit PQ
    exactly, which `_assert_matches_selection` verifies before any bootstrap runs.
    """
    out = {}
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["candidate"], row["group"])
            bucket = out.setdefault(key, {})
            current = bucket.get(row["file_name"], (0.0, 0.0, 0.0, 0.0))
            bucket[row["file_name"]] = (
                current[0] + float(row["iou_sum"]),
                current[1] + float(row["tp"]),
                current[2] + float(row["fp"]),
                current[3] + float(row["fn"]))
    return out


def pq(rows):
    iou_sum = sum(r[0] for r in rows)
    tp = sum(r[1] for r in rows)
    fp = sum(r[2] for r in rows)
    fn = sum(r[3] for r in rows)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return iou_sum / denominator if denominator > 0 else 0.0


def paired_bootstrap(arm, control, files, resamples=RESAMPLES, seed=BOOTSTRAP_SEED):
    """Resample files with replacement; recompute both PQs on the same resample."""
    rng = np.random.default_rng(seed)
    names = list(files)
    index = np.arange(len(names))
    deltas = np.empty(resamples, dtype=np.float64)
    for trial in range(resamples):
        picked = rng.choice(index, size=len(names), replace=True)
        chosen = [names[i] for i in picked]
        deltas[trial] = pq([arm[n] for n in chosen]) - pq([control[n] for n in chosen])
    observed = pq([arm[n] for n in names]) - pq([control[n] for n in names])
    return {
        "observed_delta": observed,
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "p_delta_gt_0": float((deltas > 0).mean()),
        "p_delta_ge_0p005": float((deltas >= 0.005).mean()),
        "resamples": resamples,
        "bootstrap_seed": seed,
    }


def _assert_matches_selection(diagnostics, selection, label):
    """The re-aggregated CSV must reproduce v16_selection.json's PQ exactly.

    V14's headline metric was invalid because a diagnostic was silently misaligned with
    the population it claimed to describe. This guard makes that failure mode loud: if
    the per-file aggregation here disagrees with the shipped selection numbers, the
    bootstrap below is measuring the wrong thing and the run aborts instead of
    reporting a confident interval about nothing.
    """
    expected = {
        (CONTROL, "tune"): selection["harness_check"]["control_tune_pq"],
        (ARM_R2, "tune"): selection["harness_check"]["arm_r2_tune_pq"],
        (ARM_R1, "tune"): selection["gate"]["selected_tune_pq"],
        (ARM_R1, "audit"): selection["gate"]["selected_audit_pq"],
    }
    for key, want in expected.items():
        got = pq(list(diagnostics[key].values()))
        if abs(got - want) > 1e-9:
            raise AssertionError(
                f"{label}: {key[0]}/{key[1]} PQ from diagnostics {got:.9f} != "
                f"selection {want:.9f}; per-file aggregation is wrong.")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed1", type=Path, default=root / "outputs/kaggle_v16_full")
    parser.add_argument("--seed2", type=Path, default=root / "outputs/kaggle_v16_full_seed2")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/v16_proposal_analysis/seed_replication.json")
    args = parser.parse_args()

    runs = {}
    for label, base in (("seed1", args.seed1), ("seed2", args.seed2)):
        diagnostics = read_diagnostics(base / "v16_image_diagnostics.csv")
        selection = json.loads((base / "v16_selection.json").read_text())
        _assert_matches_selection(diagnostics, selection, label)
        runs[label] = {"diagnostics": diagnostics, "selection": selection}

    report = {}
    for label, data in runs.items():
        selection = data["selection"]
        diagnostics = data["diagnostics"]
        raw = selection["raw_proposal_summaries"]
        entry = {
            "run_identity": selection["run_identity"],
            "control_tune_pq": selection["harness_check"]["control_tune_pq"],
            "arm_r2_tune_pq": selection["harness_check"]["arm_r2_tune_pq"],
            "arm_r2_vs_control_delta": selection["harness_check"]["arm_r2_vs_control_delta"],
            "arm_r1_tune_pq": selection["gate"]["selected_tune_pq"],
            "arm_r1_audit_pq": selection["gate"]["selected_audit_pq"],
            "raw_mean": {k: raw[k]["mean"] for k in raw},
            "raw_max": {k: raw[k]["max"] for k in raw},
            "promotable": selection["gate"]["promotable"],
        }
        entry["arm_r1_vs_control_tune"] = entry["arm_r1_tune_pq"] - entry["control_tune_pq"]
        entry["arm_r1_vs_arm_r2_tune"] = entry["arm_r1_tune_pq"] - entry["arm_r2_tune_pq"]

        for group in ("tune", "audit"):
            control = diagnostics[(CONTROL, group)]
            for arm_name, arm_key in (("arm_r1", ARM_R1), ("arm_r2", ARM_R2)):
                arm = diagnostics[(arm_key, group)]
                shared = sorted(set(arm) & set(control))
                assert len(shared) == len(control) == len(arm), (
                    f"{label}/{group}/{arm_name}: file sets differ")
                entry[f"bootstrap_{arm_name}_vs_control_{group}"] = paired_bootstrap(
                    arm, control, shared)
        report[label] = entry

    # Cross-seed sign test: a real lever moves the same way in independent draws.
    signs = {}
    for axis, getter in (
        ("arm_r1_vs_control_tune_pq", lambda e: e["arm_r1_vs_control_tune"]),
        ("arm_r1_vs_arm_r2_tune_pq", lambda e: e["arm_r1_vs_arm_r2_tune"]),
        ("arm_r1_raw_mean_vs_control",
         lambda e: e["raw_mean"][ARM_R1] - e["raw_mean"][CONTROL]),
        ("arm_r1_raw_max_vs_control",
         lambda e: e["raw_max"][ARM_R1] - e["raw_max"][CONTROL]),
    ):
        values = {label: getter(entry) for label, entry in report.items()}
        signs[axis] = {
            **values,
            "both_positive": all(v > 0 for v in values.values()),
            "consistent_sign": len({v > 0 for v in values.values()}) == 1,
        }
    report["cross_seed_sign_test"] = signs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print("V16 seed replication — tune PQ")
    header = f"{'':28} {'seed1':>12} {'seed2':>12}"
    print(header)
    print("-" * len(header))
    for name, key in (("frozen control", "control_tune_pq"),
                      ("arm_r2 (mask_ratio=2)", "arm_r2_tune_pq"),
                      ("arm_r1 (mask_ratio=1)", "arm_r1_tune_pq")):
        print(f"{name:28} {report['seed1'][key]:12.6f} {report['seed2'][key]:12.6f}")
    print(f"{'harness delta (r2-control)':28} "
          f"{report['seed1']['arm_r2_vs_control_delta']:12.6f} "
          f"{report['seed2']['arm_r2_vs_control_delta']:12.6f}")
    print(f"{'arm_r1 - control':28} "
          f"{report['seed1']['arm_r1_vs_control_tune']:12.6f} "
          f"{report['seed2']['arm_r1_vs_control_tune']:12.6f}")

    print("\nRaw proposal IoU (pre-refiner, 1593 proposals)")
    print(header)
    print("-" * len(header))
    for name, field, key in (("control mean", "raw_mean", CONTROL),
                             ("arm_r1 mean", "raw_mean", ARM_R1),
                             ("control max", "raw_max", CONTROL),
                             ("arm_r1 max", "raw_max", ARM_R1)):
        print(f"{name:28} {report['seed1'][field][key]:12.6f} "
              f"{report['seed2'][field][key]:12.6f}")

    print("\nPaired per-image bootstrap, arm_r1 vs frozen control "
          f"({RESAMPLES} resamples, seed {BOOTSTRAP_SEED})")
    for label in ("seed1", "seed2"):
        for group in ("tune", "audit"):
            b = report[label][f"bootstrap_arm_r1_vs_control_{group}"]
            print(f"  {label}/{group:5} delta {b['observed_delta']:+.6f} "
                  f"CI95 [{b['ci95_low']:+.6f}, {b['ci95_high']:+.6f}] "
                  f"P(>0)={b['p_delta_gt_0']:.3f}")

    print("\nCross-seed sign test")
    for axis, values in signs.items():
        print(f"  {axis:34} seed1 {values['seed1']:+.6f}  seed2 {values['seed2']:+.6f}  "
              f"consistent={values['consistent_sign']} both_positive={values['both_positive']}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
