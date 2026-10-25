"""V67: fetch co-temporal GONG LOS magnetograms for TRAIN H-alpha frames only (training-only aux).

Host rule (discussion 743521, docs/kaggle_discussion_scan_2026-10-04.md): co-temporal GONG
magnetograms may be used as a TRAINING-ONLY auxiliary signal, never at inference. This script reads
the TRAIN directory only; it never lists or touches test images, so nothing here can reach test.

For each train file YYYYMMDDHHMMSS<station>h.jpeg it lists https://gong2.nso.edu/oQR/zqa/YYYYMM/
bbzqaYYMMDD/, picks the file whose time (HHMM) is nearest the H-alpha time, downloads it, and writes
a manifest with the time gap. Falls back to the quick-reduce product (bqa) if zqa has no file that
day. Public NSO data; polite (3 workers, small sleep, retries). Resumable: existing files are kept.

  python scripts/fetch_v67_magnetograms.py --limit 10     # smoke test
  python scripts/fetch_v67_magnetograms.py                # all 707 train files
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
OUT = ROOT / "data/external/gong_bz"
BASE = "https://gong2.nso.edu/oQR"
# GONG magnetogram directories are PER SITE; the H-alpha station letter is the same site, so the
# co-temporal magnetogram is the one from the same station (B=Big Bear, C=Cerro Tololo,
# L=Learmonth, M=Mauna Loa, T=Teide, U=Udaipur).
SITE = {"B": "bb", "C": "ct", "L": "le", "M": "ml", "T": "td", "U": "ud"}
PRODUCTS = (("zqa", "zqa"), ("bqa", "bqa"))   # (directory, filename product code)
NAME = re.compile(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})([A-Za-z])h\.jpe?g$")
_LISTINGS: dict[str, list[str]] = {}


def http_get(url: str, tries: int = 4, timeout: int = 60) -> bytes | None:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:  # noqa: BLE001  network flake: back off and retry
            pass
        time.sleep(2 * (k + 1))
    return None


def day_files(product: str, prefix: str, y: str, m: str, d: str) -> list[str]:
    key = f"{product}/{prefix}/{y}{m}{d}"
    if key not in _LISTINGS:
        raw = http_get(f"{BASE}/{product}/{y}{m}/{prefix}{y[2:]}{m}{d}/")
        names = re.findall(rf'href="({prefix}{y[2:]}{m}{d}t\d{{4}}\.fits\.gz)"',
                           raw.decode("utf8", "ignore")) if raw else []
        _LISTINGS[key] = sorted(set(names))
    return _LISTINGS[key]


def file_minutes(fname: str) -> int:
    """Minute-of-day from a name like bbzqa110113t1554.fits.gz."""
    hhmm = re.search(r"t(\d{4})\.fits", fname).group(1)
    return int(hhmm[:2]) * 60 + int(hhmm[2:])


def pick(name: str) -> dict:
    mt = NAME.match(name)
    y, m, d, hh, mi, _ss, st = mt.groups()
    target = int(hh) * 60 + int(mi)
    for product, code in PRODUCTS:
        prefix = SITE[st.upper()] + code
        files = day_files(product, prefix, y, m, d)
        if not files:
            continue
        best = min(files, key=lambda f: abs(file_minutes(f) - target))
        gap = file_minutes(best) - target
        url = f"{BASE}/{product}/{y}{m}/{prefix}{y[2:]}{m}{d}/{best}"
        return dict(image=name, station=st, product=product, file=best, url=url, gap_min=gap)
    return dict(image=name, station=st, product=None, file=None, url=None, gap_min=None)


def fetch(rec: dict) -> dict:
    if not rec["url"]:
        return dict(rec, status="no_magnetogram_that_day")
    dest = OUT / rec["file"]
    if dest.exists() and dest.stat().st_size > 100_000:
        return dict(rec, status="cached", bytes=dest.stat().st_size)
    raw = http_get(rec["url"])
    if raw is None or len(raw) < 100_000:
        return dict(rec, status="download_failed")
    dest.write_bytes(raw)
    time.sleep(0.2)
    return dict(rec, status="ok", bytes=len(raw))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in TRAIN.rglob("*.jp*g"))
    assert names and all(NAME.match(n) for n in names), "unexpected train filenames"
    assert "test" not in str(TRAIN), "refusing to read a test directory"
    if args.limit:
        names = names[:: max(1, len(names) // args.limit)][: args.limit]
    print(f"train files: {len(names)}", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        picks = list(ex.map(pick, names))
        results = list(ex.map(fetch, picks))
    manifest = OUT / ("manifest_smoke.json" if args.limit else "manifest.json")
    manifest.write_text(json.dumps(results, indent=1) + "\n")
    gaps = sorted(abs(r["gap_min"]) for r in results if r["gap_min"] is not None)
    status = {s: sum(r["status"] == s for r in results) for s in {r["status"] for r in results}}
    print("status:", status)
    if gaps:
        print(f"|gap| minutes  median {gaps[len(gaps) // 2]}  p90 {gaps[int(len(gaps) * .9)]}  "
              f"max {gaps[-1]}")
    print("manifest:", manifest)


if __name__ == "__main__":
    main()
