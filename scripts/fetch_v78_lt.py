"""V78 label transfer: fetch same-site GONG H-alpha frames near each labelled TRAIN file.

Buckets m5/m2/m1/p1/p2/p5 = nearest same-site archive frame within +-30 s of t0 + offset (never t0
itself). Frames within +-1 h of any TEST image timestamp are dropped (host ruling, thread 748045).
Saved as <out>/<bucket>/<original kaggle file name>; manifest.tsv lists name, bucket, url, dt (s).
Usage: python -I fetch_v78_lt.py train_names.txt test_names.txt OUTDIR
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_v69_nbr import BASE, day_frames, get  # noqa: E402

OFFSETS = {"m5": -5, "m2": -2, "m1": -1, "p1": 1, "p2": 2, "p5": 5}
TOL_S = 30


def ts(name):
    return datetime.strptime(name[:14], "%Y%m%d%H%M%S")


def plan(name, test_ts):
    t0, site = ts(name), name[14]
    frames = []
    for d in sorted({(t0 + timedelta(minutes=k)).strftime("%Y%m%d") for k in (-6, 0, 6)}):
        frames += day_frames(d)
    jobs = []
    for b, off in OFFSETS.items():
        tgt = t0 + timedelta(minutes=off)
        c = [(abs((t - tgt).total_seconds()), t) for t, s in frames
             if s == site and t != t0 and abs((t - tgt).total_seconds()) <= TOL_S]
        if not c:
            continue
        t = min(c)[1]
        if any(abs((t - u).total_seconds()) <= 3600 for u in test_ts):
            continue
        jobs.append((b, f"{BASE}/{t:%Y%m}/{t:%Y%m%d}/{t:%Y%m%d%H%M%S}{site}h.jpg", (t - t0).total_seconds()))
    return name, jobs


def main():
    names = Path(sys.argv[1]).read_text().split()
    test_ts = [ts(n) for n in Path(sys.argv[2]).read_text().split()]
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        plans = list(ex.map(lambda n: plan(n, test_ts), names))
    jobs, log = [], []
    for n, js in plans:
        for b, url, dt in js:
            jobs.append((out / b / n, url))
            log.append(f"{n}\t{b}\t{url}\t{dt:.0f}")
    (out / "manifest.tsv").write_text("\n".join(log) + "\n")
    print("planned", len(jobs), "frames for", len(names), "files", flush=True)

    def fetch(j):
        p, url = j
        if p.exists():
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(get(url))
    with ThreadPoolExecutor(4) as ex:
        for i, _ in enumerate(ex.map(fetch, jobs), 1):
            if i % 100 == 0:
                print(i, "/", len(jobs), flush=True)
    print("done", len(jobs), "frames for", len(names), "files")


if __name__ == "__main__":
    main()
