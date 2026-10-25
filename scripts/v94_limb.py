"""V94: limb-darkening division as detector input (no CLAHE, no blur). CPU only.
Pre-registration: docs/v94_limb_division_preregistration_2026-10-19.md

Per image: Otsu disk fit, 256-bin radial median profile (Gaussian-smoothed, sigma 15 bins),
divide by the interpolated profile, p2-p98 scale to uint8, zero off-disk. This is the V23
transform (train_v23_radial_clahe.py, fit_disk/radial_profile copied verbatim) minus the
3x3 blur and CLAHE, which competitors report as harmful.

  python scripts/v94_limb.py --src <dir of 2048x2048 jpeg> --out outputs/v94/images
  python scripts/v94_limb.py --selfcheck <one image>
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

PROFILE_SIZE = 512
RADIAL_BINS = 256
PROFILE_SIGMA_BINS = 15.0
DISK_LIMIT = 0.98
CENTER_TOLERANCE_FRACTION = 0.06


def fit_disk(gray: np.ndarray) -> dict:
    if gray.shape != (2048, 2048):
        raise ValueError(f"expected 2048x2048 image, found {gray.shape}")
    work = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("solar disk contour not found")
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    filled = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(filled, [largest], -1, 1, cv2.FILLED)
    disk_fraction = float(filled.mean())
    if abs(cx - 1023.5) > CENTER_TOLERANCE_FRACTION * 2048 or abs(cy - 1023.5) > CENTER_TOLERANCE_FRACTION * 2048:
        raise ValueError(f"disk center outside frozen tolerance: {(cx, cy)}")
    if not 0.35 * 2048 <= radius <= 0.55 * 2048:
        raise ValueError(f"disk radius outside frozen tolerance: {radius}")
    if not 0.45 <= disk_fraction <= 0.80:
        raise ValueError(f"disk fraction outside frozen tolerance: {disk_fraction}")
    return {"cx": float(cx), "cy": float(cy), "radius": float(radius),
            "disk_fraction": disk_fraction}


def radial_profile(gray: np.ndarray, disk: dict) -> np.ndarray:
    small = cv2.resize(gray, (PROFILE_SIZE, PROFILE_SIZE), interpolation=cv2.INTER_AREA)
    scale = PROFILE_SIZE / 2048.0
    cx, cy, radius = disk["cx"] * scale, disk["cy"] * scale, disk["radius"] * scale
    yy, xx = np.indices(small.shape, dtype=np.float32)
    radial = np.hypot(xx - cx, yy - cy) / radius
    valid = radial <= DISK_LIMIT
    indices = np.minimum((radial * RADIAL_BINS).astype(np.int32), RADIAL_BINS - 1)
    profile = np.full(RADIAL_BINS, np.nan, dtype=np.float32)
    for index in range(RADIAL_BINS):
        values = small[valid & (indices == index)]
        if values.size:
            profile[index] = float(np.median(values))
    finite = np.flatnonzero(np.isfinite(profile))
    if len(finite) < RADIAL_BINS * 0.9:
        raise ValueError(f"radial profile has only {len(finite)} finite bins")
    missing = np.flatnonzero(~np.isfinite(profile))
    profile[missing] = np.interp(missing, finite, profile[finite])
    profile = cv2.GaussianBlur(profile[None, :], (0, 0), sigmaX=PROFILE_SIGMA_BINS,
                               borderType=cv2.BORDER_REPLICATE)[0]
    if float(profile.min()) <= 1.0:
        raise ValueError("radial profile contains invalid background")
    return profile


def transform(gray: np.ndarray) -> tuple[np.ndarray, dict]:
    disk = fit_disk(gray)
    profile = radial_profile(gray, disk)
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    radial = np.hypot(xx - disk["cx"], yy - disk["cy"]) / disk["radius"]
    mask = radial <= DISK_LIMIT
    coordinates = np.clip(radial * (RADIAL_BINS - 1), 0, RADIAL_BINS - 1)
    low = np.floor(coordinates).astype(np.int32)
    high = np.minimum(low + 1, RADIAL_BINS - 1)
    fraction = coordinates - low
    background = profile[low] * (1.0 - fraction) + profile[high] * fraction
    flattened = gray.astype(np.float32) / np.maximum(background, 1.0) * float(np.median(profile))
    lo, hi = np.percentile(flattened[mask], (2, 98))
    out = np.clip((flattened - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
    out[~mask] = 0
    return out, {**disk, "p02": float(lo), "p98": float(hi)}


def work(job):
    src, dst = job
    gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    out, diag = transform(gray)
    if not cv2.imwrite(str(dst), out):
        raise RuntimeError(f"write failed: {dst}")
    return src.name, diag


def selfcheck(path: Path) -> None:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    out, disk = transform(gray)
    assert out.shape == gray.shape and out.dtype == np.uint8
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    radial = np.hypot(xx - disk["cx"], yy - disk["cy"]) / disk["radius"]
    assert out[radial > DISK_LIMIT].max() == 0, "off-disk not zeroed"
    rings = [float(np.median(out[(radial >= a) & (radial < a + 0.1)])) for a in np.arange(0.0, 0.9, 0.1)]
    raw = [float(np.median(gray[(radial >= a) & (radial < a + 0.1)])) for a in np.arange(0.0, 0.9, 0.1)]
    spread = (max(rings) - min(rings)) / np.median(rings)
    print({"raw_ring_medians": raw, "out_ring_medians": rings, "out_spread": spread})
    assert spread < 0.10, "limb division left a radial gradient"
    print("selfcheck ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--selfcheck", type=Path)
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args.selfcheck)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("*.jpeg"))
    jobs = [(f, args.out / (f.stem + ".png")) for f in files if not (args.out / (f.stem + ".png")).exists()]
    with Pool(args.procs) as p:
        diag = dict(p.map(work, jobs, chunksize=4))
    done = sum((args.out / (f.stem + ".png")).exists() for f in files)
    print(json.dumps({"src": str(args.src), "files": len(files), "transformed_now": len(jobs), "present": done}))
    if done != len(files):
        raise RuntimeError("missing transformed images")
    dpath = args.out.parent / f"transform_diag_{args.src.parent.name}.json"
    old = json.loads(dpath.read_text()) if dpath.exists() else {}
    dpath.write_text(json.dumps({**old, **diag}, indent=1) + "\n")


if __name__ == "__main__":
    main()
