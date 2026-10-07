"""Stage-2 progress for the V65 resume (p2first_cp folds 1-4): epoch, speed, ETA, NaN, freshness."""
import csv
import os
import time

now = time.time()
for k in (1, 2, 3, 4):
    p = f"/mnt/v61/p2first_cp_fold{k}/mask_ratio1/run/results.csv"
    if not os.path.exists(p):
        print(f"fold{k}: no stage-2 results yet")
        continue
    rows = list(csv.reader(open(p)))
    h = [x.strip() for x in rows[0]]
    last = rows[-1]
    t = float(last[h.index("time")])
    ep = int(last[0])
    nan = any(x.strip().lower() in ("nan", "inf", "-inf") for r in rows for x in r)
    age = (now - os.path.getmtime(p)) / 60
    print(f"fold{k}: stage2 epoch {ep}/20  {t / ep:.0f}s/ep  ETA {(20 - ep) * t / ep:.0f}s "
          f"(~{(20 - ep) * t / ep / 60:.0f} min)  nan={nan}  last_update {age:.0f} min ago")
