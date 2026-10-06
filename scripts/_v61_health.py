"""Quick health/ETA check for V61 wave-2 trainings (reads results.csv only)."""
import csv
import sys

BASE = "/mnt/v61"
RUNS = [("B focal_dice", "focal_dice_fold0"), ("A' seed2", "baseline_seed2_fold0"),
        ("A4 baseline f4", "baseline_fold4")]


def main():
    for tag, d in RUNS:
        path = f"{BASE}/{d}/base/run/results.csv"
        try:
            with open(path) as f:
                rows = list(csv.reader(f))
        except FileNotFoundError:
            print(tag, "no results yet")
            continue
        head = [x.strip() for x in rows[0]]
        body = rows[1:]
        last = body[-1]
        ep = int(last[0])
        t = float(last[head.index("time")])
        nan = any(x.strip().lower() in ("nan", "inf", "-inf") for r in body for x in r)
        seg = float(last[head.index("train/seg_loss")])
        box = float(last[head.index("train/box_loss")])
        total = t / ep * 80
        print("%-16s epoch %2d/80  %4.0fs/ep  box %.3f  seg %.3f  nan=%s  "
              "base stage done in ~%d min"
              % (tag, ep, t / ep, box, seg, nan, (total - t) / 60))


if __name__ == "__main__":
    sys.exit(main())
