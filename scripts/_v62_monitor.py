"""V62 5-fold run monitor: liveness + results freshness + log tail for p2first folds 1-4.

OK only if the process exists, results.csv advanced in the last 15 min (once started), and the
log has no Traceback / OutOfMemoryError / CUDA error.

  python scripts/_v62_monitor.py            (run on the VM)
"""
from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

BASE = Path("/mnt/v61")
LOGS = Path.home() / "solar-filament-2026/outputs"
RECIPE = "p2first"
FOLDS = (1, 2, 3, 4)
STALE_SECONDS = 15 * 60
BAD = ("Traceback", "OutOfMemoryError", "CUDA error")


def alive(fold: int) -> int:
    out = subprocess.run(["pgrep", "-f", f"train_v61_fold.py --fold {fold} --recipe {RECIPE} "],
                         capture_output=True, text=True).stdout.split()
    return len(out)


def latest(run: Path):
    for stage, total in (("mask_ratio1", 20), ("base", 80)):
        p = run / stage / "run" / "results.csv"
        if p.exists():
            rows = list(csv.reader(p.open()))
            h = [x.strip() for x in rows[0]]
            t = float(rows[-1][h.index("time")])
            ep = int(rows[-1][0])
            return stage, ep, total, t, p.stat().st_mtime
    return None, 0, 0, 0.0, 0.0


def main():
    now = time.time()
    for k in FOLDS:
        run = BASE / f"{RECIPE}_fold{k}"
        log = LOGS / f"v62_{RECIPE}_fold{k}.log"
        if (run / "fold_result.json").exists():
            print(f"fold{k} COMPLETE")
            continue
        n = alive(k)
        stage, ep, total, t, mtime = latest(run)
        age = now - mtime if mtime else float("inf")
        problems = []
        if n == 0:
            problems.append("no live process")
        if stage is not None and age > STALE_SECONDS:
            problems.append(f"results.csv stale {age / 60:.0f} min")
        if log.exists():
            tail = "\n".join(log.read_text(errors="ignore").replace("\r", "\n")
                             .splitlines()[-400:])
            problems += [f"log contains {b!r}" for b in BAD if b in tail]
        eta = f" ETA {(total - ep) * t / ep / 60:.0f} min" if ep else ""
        status = "OK" if not problems else "PROBLEM: " + "; ".join(problems)
        print(f"fold{k} {stage or 'starting':11s} epoch {ep}/{total}{eta}  procs={n}  "
              f"last_update={age / 60:.0f} min ago  {status}")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    print("gpu:", gpu.replace("\n", "; "))


if __name__ == "__main__":
    main()
