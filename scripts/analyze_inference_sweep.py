"""Compare a tune-selected inference candidate with V6 using paired image bootstraps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL = "r1024_refined_free_t0.65_tta0_c0.40"
COUNTS = ["iou_sum", "tp", "fp", "fn"]


def pq(totals):
    denom = totals[..., 1] + 0.5 * (totals[..., 2] + totals[..., 3])
    return np.divide(totals[..., 0], denom, out=np.zeros_like(denom), where=denom > 0)


def compare_group(diagnostics, candidate, group, samples=4000, control_candidate=CONTROL):
    frame = (diagnostics[diagnostics.group.isin(["tune", "audit"])] if group == "all"
             else diagnostics[diagnostics.group == group])
    challenger = frame[frame.candidate == candidate].groupby("file_name")[COUNTS].sum()
    control = frame[frame.candidate == control_candidate].groupby("file_name")[COUNTS].sum()
    if challenger.empty or set(challenger.index) != set(control.index):
        raise ValueError(f"Incomplete paired evaluation for {candidate}, group={group}")
    a = challenger.sort_index().to_numpy(dtype=float)
    b = control.sort_index().to_numpy(dtype=float)
    # All annotations for one image stay together in each resampled unit.
    indices = np.random.default_rng(42).integers(0, len(a), size=(samples, len(a)))
    differences = pq(a[indices].sum(axis=1)) - pq(b[indices].sum(axis=1))
    delta = float(pq(a.sum(axis=0)) - pq(b.sum(axis=0)))
    return {"images": len(a), "candidate_pq": float(pq(a.sum(axis=0))),
            "control_pq": float(pq(b.sum(axis=0))), "delta_pq": delta,
            "delta_95pct_interval": np.quantile(differences, [0.025, 0.975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-dir", type=Path,
                        help="Compare with a previous run's tune-selected model")
    args = parser.parse_args()
    manifest = json.loads((args.output / "inference_manifest.json").read_text())
    if manifest["smoke_limit"]:
        raise ValueError("A smoke test cannot establish model improvement")
    if manifest.get("refiner_epochs", 0) and args.baseline_dir is None:
        raise ValueError("Fine-tuned models require a separate baseline run")
    candidates = pd.read_csv(args.output / "inference_candidates.csv")
    diagnostics = pd.read_csv(args.output / "inference_image_diagnostics.csv")
    selected = candidates.sort_values(["tune_pq", "candidate"], ascending=[False, True]).iloc[0]
    control_name = CONTROL
    comparison_key = selected.candidate
    if args.baseline_dir is None:
        control = candidates[candidates.candidate == CONTROL].iloc[0]
    else:
        baseline_manifest = json.loads((args.baseline_dir / "inference_manifest.json").read_text())
        for field in ("tune_files", "audit_files", "target_rasterization"):
            if manifest[field] != baseline_manifest[field]:
                raise ValueError(f"Baseline mismatch: {field}")
        baseline_candidates = pd.read_csv(args.baseline_dir / "inference_candidates.csv")
        control = baseline_candidates.sort_values(
            ["tune_pq", "candidate"], ascending=[False, True]).iloc[0]
        baseline_diagnostics = pd.read_csv(args.baseline_dir / "inference_image_diagnostics.csv")
        current_rows = diagnostics[diagnostics.candidate == selected.candidate].copy()
        baseline_rows = baseline_diagnostics[baseline_diagnostics.candidate == control.candidate].copy()
        comparison_key = "fine_tuned_candidate"
        current_rows["candidate"] = comparison_key
        baseline_rows["candidate"] = CONTROL
        diagnostics = pd.concat([current_rows, baseline_rows], ignore_index=True)
        control_name = f"{args.baseline_dir.name}/{control.candidate}"
    report = {
        "selected_on": "tune_pq only",
        "candidate": selected.candidate,
        "config": {field: selected[field].item() if isinstance(selected[field], np.generic)
                   else selected[field] for field in
                   ("resolution", "mode", "threshold", "tta", "confidence")},
        "control": control_name,
        "comparison": {group: compare_group(diagnostics, comparison_key, group)
                       for group in ("tune", "audit", "all")},
        "legacy_candidate_pq": float(selected.legacy_pq),
        "legacy_control_pq": float(control.legacy_pq),
        "target_rasterization": manifest["target_rasterization"],
        "audit_caveat": manifest["audit_caveat"],
        "public_score": "unmeasured",
    }
    if "detector_policy" in selected:
        report["config"]["detector_policy"] = selected.detector_policy
    (args.output / "inference_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("\nBest candidate at each resolution (selected on tune only):")
    print(candidates.sort_values("tune_pq", ascending=False).groupby("resolution").head(1)[
        ["candidate", "tune_pq", "audit_pq", "all_pq", "legacy_pq"]].to_string(index=False))


if __name__ == "__main__":
    main()
