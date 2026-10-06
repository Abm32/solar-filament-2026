"""V61/V62 run monitor: liveness + results freshness + log tail for every tracked training run.

A run is OK only if its process exists, results.csv advanced recently, and its log has no
Traceback / OutOfMemoryError / CUDA error. Anything else prints PROBLEM with the reason.

  python scripts/_v61_monitor.py            (run on the VM)
"""
from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

BASE = Path("/mnt/v61")
LOGS = Path.home() / "solar-filament-2026/outputs"
RUNS = [  # (label, run dir, argv fragment, log files)
    ("V62 p2last f0", "p2last_fold0", "--fold 0 --recipe p2last ", ["v62_p2last_fold0.log"]),
    ("V62 p2first f0", "p2first_fold0", "--fold 0 --recipe p2first ", ["v62_p2first_fold0.log"]),
    ("V62 p2last s2", "p2last_seed2_fold0", "--fold 0 --recipe p2last_seed2",
     ["v62_p2last_seed2_fold0.log"]),
    ("V62 p2first s2", "p2first_seed2_fold0", "--fold 0 --recipe p2first_seed2",
     ["v62_p2first_seed2_fold0.log"]),
]
STALE_SECONDS = 15 * 60
BAD = ("Traceback", "OutOfMemoryError", "CUDA error")


def alive(fragment: str) -> int:
    out = subprocess.run(["pgrep", "-f", f"train_v61_fold.py {fragment}"],
                         capture_output=True, text=True).stdout.split()
    return len(out)


def latest_results(run: Path):
    for stage in ("mask_ratio1", "base"):
        p = run / stage / "run" / "results.csv"
        if p.exists():
            rows = list(csv.reader(p.open()))
            return stage, int(rows[-1][0]), p.stat().st_mtime
    return None, 0, 0.0


def log_problems(names) -> list[str]:
    found = []
    for n in names:
        p = LOGS / n
        if not p.exists():
            continue
        text = p.read_text(errors="ignore").replace("\r", "\n")
        tail = "\n".join(text.splitlines()[-400:])
        found += [f"{n}: contains {bad!r}" for bad in BAD if bad in tail]
    return found


def main():
    now = time.time()
    for label, d, frag, logs in RUNS:
        run = BASE / d
        if (run / "fold_result.json").exists():
            print(f"{label:16s} COMPLETE")
            continue
        n = alive(frag)
        stage, ep, mtime = latest_results(run)
        age = now - mtime if mtime else float("inf")
        problems = []
        if n == 0:
            problems.append("no live process")
        if age > STALE_SECONDS and stage is not None:
            problems.append(f"results.csv stale {age / 60:.0f} min")
        problems += log_problems(logs)
        status = "OK" if not problems else "PROBLEM: " + "; ".join(problems)
        total = 20 if stage == "mask_ratio1" else 80
        print(f"{label:16s} {stage or 'starting':11s} epoch {ep}/{total}  procs={n}  "
              f"last_update={age / 60:.0f} min ago  {status}")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout
        print("gpu:", gpu.replace("\n", "; "))
    except OSError:
        pass


if __name__ == "__main__":
    main()
