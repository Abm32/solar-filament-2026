"""V69 neighbour-frame check: fetch GONG H-alpha frames near each fold-0 held-out TRAIN file.

Buckets (nearest archive frame to target time, within +-90 s): same site at 0 (archive copy of the
labelled frame itself, re-encoding control), -2, +2, -10, +10, -30, +30 min; and 'xsite' = nearest
frame from a DIFFERENT site within 15 min. Saved as <out>/<bucket>/<original kaggle file name>.
Only fold-0 TRAIN file names are read. Polite: 4 workers, one listing per day, retries.
Usage: python -I fetch_v69_nbr.py fold0.txt OUTDIR
"""
import re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

BASE = "https://gong2.nso.edu/HA/hag"
OFFSETS = {"o0": 0, "m2": -2, "p2": 2, "m10": -10, "p10": 10, "m30": -30, "p30": 30}
PAT = re.compile(r'href="(\d{14})([A-Z])h\.jpg"')


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            time.sleep(3 * (i + 1))
    raise RuntimeError(url)


listings = {}
def day_frames(day):
    if day not in listings:
        html = get(f"{BASE}/{day[:6]}/{day}/").decode("latin1")
        listings[day] = [(datetime.strptime(t, "%Y%m%d%H%M%S"), s) for t, s in PAT.findall(html)]
    return listings[day]


def plan(name):
    t0 = datetime.strptime(name[:14], "%Y%m%d%H%M%S"); site = name[14]
    frames = []
    for d in {(t0 + timedelta(minutes=k)).strftime("%Y%m%d") for k in (-31, 0, 31)}:
        frames += day_frames(d)
    jobs = []
    for b, off in OFFSETS.items():
        tgt = t0 + timedelta(minutes=off)
        c = [(abs((t - tgt).total_seconds()), t, s) for t, s in frames if s == site]
        c = [x for x in c if x[0] <= 90 and (off == 0 or x[1] != t0)]
        if c:
            jobs.append((b, min(c)[1], site))
    c = [(abs((t - t0).total_seconds()), t, s) for t, s in frames if s != site]
    c = [x for x in c if x[0] <= 900]
    if c:
        jobs.append(("xsite", min(c)[1], min(c)[2]))
    return [(b, f"{BASE}/{t:%Y%m}/{t:%Y%m%d}/{t:%Y%m%d%H%M%S}{s}h.jpg", (t - t0).total_seconds())
            for b, t, s in jobs]


def main():
    names = Path(sys.argv[1]).read_text().split()
    out = Path(sys.argv[2])
    jobs, log = [], []
    for n in names:
        for b, url, dt in plan(n):
            jobs.append((out / b / n, url)); log.append(f"{n}\t{b}\t{url}\t{dt:.0f}")
    (out).mkdir(parents=True, exist_ok=True)
    (out / "manifest.tsv").write_text("\n".join(log) + "\n")
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
