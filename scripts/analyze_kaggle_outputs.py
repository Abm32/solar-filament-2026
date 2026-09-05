from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_key_values(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_validation_variants(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    pattern = re.compile(
        r"mode=(?P<mode>\S+) conf=(?P<conf>\S+) val_pq=(?P<val_pq>\S+) "
        r"tp=(?P<tp>\d+) fp=(?P<fp>\d+) fn=(?P<fn>\d+)"
        r"(?: precision=(?P<precision>\S+) recall=(?P<recall>\S+))?"
    )
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        row = match.groupdict()
        tp = int(row["tp"])
        fp = int(row["fp"])
        fn = int(row["fn"])
        rows.append(
            {
                "mode": row["mode"],
                "conf": float(row["conf"]),
                "val_pq": float(row["val_pq"]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": (
                    float(row["precision"]) if row["precision"] else tp / (tp + fp) if tp + fp else 0.0
                ),
                "recall": (
                    float(row["recall"]) if row["recall"] else tp / (tp + fn) if tp + fn else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def best_training_row(path: Path) -> dict[str, str | float | int] | None:
    if not path.exists():
        return None
    results = pd.read_csv(path)
    mask_col = "metrics/mAP50(M)"
    if mask_col not in results:
        return None
    best_index = results[mask_col].astype(float).idxmax()
    best = results.loc[best_index]
    last = results.iloc[-1]
    return {
        "best_epoch": int(best["epoch"]),
        "best_mask_mAP50": float(best[mask_col]),
        "last_epoch": int(last["epoch"]),
        "last_mask_mAP50": float(last[mask_col]),
    }


def submission_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return max(sum(1 for _ in path.open()) - 1, 0)


def analyze_output(output_dir: Path) -> None:
    print(f"output_dir={output_dir}")
    summary = parse_key_values(output_dir / "validation_summary.txt")
    diagnostics = parse_key_values(output_dir / "submission_diagnostics.txt")
    for key in ["best_conf", "best_mode", "best_pq", "best_tp", "best_fp", "best_fn"]:
        if key in summary:
            print(f"{key}={summary[key]}")
    for key in ["rows", "populated_images", "missing_images", "median_instances", "mode", "conf"]:
        if key in diagnostics:
            print(f"submission_{key}={diagnostics[key]}")

    training = best_training_row(output_dir / "runs_filament" / "yolo_seg_optimized" / "results.csv")
    if training:
        for key, value in training.items():
            print(f"{key}={value}")

    candidates_path = output_dir / "validation_candidates.csv"
    if candidates_path.exists():
        candidates = pd.read_csv(candidates_path)
    else:
        candidates = parse_validation_variants(output_dir / "validation_variants.txt")
    if not candidates.empty:
        candidates = candidates.sort_values("val_pq", ascending=False)
        print("\ntop_validation_candidates:")
        print(candidates.head(8).to_string(index=False))
        recall_sorted = candidates.sort_values(["recall", "val_pq"], ascending=False)
        if "recall" in recall_sorted and recall_sorted["recall"].notna().any():
            print("\ntop_recall_candidates:")
            print(recall_sorted.head(5).to_string(index=False))

    submission_files = sorted(output_dir.glob("submission*.csv"))
    if submission_files:
        print("\nsubmission_files:")
        for path in submission_files:
            print(f"{path.name}: rows={submission_rows(path)}")
            diag = parse_key_values(path.with_suffix("").with_name(path.stem + "_diagnostics.txt"))
            if diag:
                print(
                    "  "
                    + " ".join(
                        f"{key}={diag[key]}"
                        for key in ["mode", "conf", "populated_images", "missing_images", "median_instances"]
                        if key in diag
                    )
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    analyze_output(args.output_dir)


if __name__ == "__main__":
    main()
