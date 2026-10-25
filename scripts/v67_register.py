"""V67: register a GONG LOS magnetogram (860x860, Gauss) onto a 2048x2048 GONG H-alpha frame.

Both are full-disk, north-up views of the same Sun from the same GONG site a few minutes apart, so
a similarity transform (scale + translation) from disk circle to disk circle is sufficient. The
H-alpha disk circle is fitted from the image itself; the magnetogram circle comes from its header
(CRPIX1/2 = disk centre, FNDLMBMI/MA = limb radius). Array orientation (row flip) is decided by a
fixed convention verified in tests/diagnostics (flip rows: FITS row 0 is the bottom).

Only TRAIN images are ever passed here. Nothing in this module reads labels or test data.
"""
from __future__ import annotations

import gzip
import io

import cv2
import numpy as np

SMALL = 512            # fit the H-alpha disk on a 4x downsampled copy (fast, sub-pixel enough)


def _circle_lsq(pts: np.ndarray) -> tuple[float, float, float]:
    """Algebraic least-squares circle through points (N,2) as (x, y)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0], sol[1]
    return float(cx), float(cy), float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))


def _rough_disk(gray: np.ndarray) -> tuple[float, float, float]:
    """Coarse disk (cx, cy, r): component containing the centre above 0.7 x central median."""
    h = gray.shape[0]
    small = cv2.resize(gray, (SMALL, SMALL), interpolation=cv2.INTER_AREA).astype(np.float32)
    blur = cv2.GaussianBlur(small, (0, 0), 2)
    c = SMALL // 2
    ref = float(np.median(blur[c - 60:c + 60, c - 60:c + 60]))
    _n, lab = cv2.connectedComponents((blur > 0.7 * ref).astype(np.uint8), connectivity=8)
    if lab[c, c] == 0:
        raise ValueError("disk centre is background")
    comp = (lab == lab[c, c]).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    k = h / SMALL
    return cx * k, cy * k, r * k


def fit_disk(gray: np.ndarray, n_angles: int = 72) -> tuple[float, float, float]:
    """H-alpha disk circle (cx, cy, r) in full-resolution pixels.

    A global threshold either swallows the dim limb halo (radius too big) or cuts inside the limb
    (too small), so: start from a coarse disk, then for each of `n_angles` rays find the radius of
    the steepest intensity DROP (the physical limb edge) within +-12% of the coarse radius, and fit
    a circle to those edge points with iterative outlier trimming.
    """
    cx0, cy0, r0 = _rough_disk(gray)
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.5)
    radii = np.arange(0.88 * r0, 1.12 * r0, 1.0)
    pts = []
    for t in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        xs = cx0 + radii * np.cos(t)
        ys = cy0 + radii * np.sin(t)
        ok = (xs >= 0) & (xs < gray.shape[1] - 1) & (ys >= 0) & (ys < gray.shape[0] - 1)
        if ok.sum() < 20:
            continue
        prof = cv2.remap(g, xs[ok].astype(np.float32)[None], ys[ok].astype(np.float32)[None],
                         cv2.INTER_LINEAR)[0]
        d = np.gradient(prof)
        i = int(np.argmin(d))
        pts.append((xs[ok][i], ys[ok][i]))
    pts = np.asarray(pts, dtype=np.float64)
    for _ in range(5):
        cx, cy, r = _circle_lsq(pts)
        res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        keep = res <= max(2.5 * float(np.median(res)), 1.5)
        if keep.all() or keep.sum() < 20:
            break
        pts = pts[keep]
    return float(cx), float(cy), float(r)


def read_mag(path) -> tuple[np.ndarray, dict]:
    """-> (Bz array [rows,cols] Gauss, geometry dict). Needs astropy (not a project dependency)."""
    from astropy.io import fits

    with gzip.open(path) as f:
        hdul = fits.open(io.BytesIO(f.read()))
    hd = hdul[0].header
    geom = dict(cx=float(hd["CRPIX1"]) - 1.0, cy=float(hd["CRPIX2"]) - 1.0,   # 0-based pixel
                r=0.5 * (float(hd["FNDLMBMI"]) + float(hd["FNDLMBMA"])),
                time=str(hd.get("TIME-OBS", "")))
    return np.asarray(hdul[0].data, dtype=np.float32), geom


def warp_mag(bz: np.ndarray, geom: dict, disk: tuple[float, float, float],
             out_size: int = 2048, flip_rows: bool = True) -> np.ndarray:
    """Resample Bz onto the H-alpha pixel grid (disk circle -> disk circle). Off-disk = 0."""
    if flip_rows:
        bz = bz[::-1]
        cy_m = (bz.shape[0] - 1) - geom["cy"]
    else:
        cy_m = geom["cy"]
    cx_h, cy_h, r_h = disk
    s = geom["r"] / r_h                                   # magnetogram px per H-alpha px
    mat = np.array([[s, 0.0, geom["cx"] - s * cx_h],
                    [0.0, s, cy_m - s * cy_h]], dtype=np.float32)
    out = cv2.warpAffine(np.ascontiguousarray(bz), mat, (out_size, out_size),
                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    yy, xx = np.mgrid[:out_size, :out_size]
    on_disk = (xx - cx_h) ** 2 + (yy - cy_h) ** 2 <= (0.985 * r_h) ** 2
    return np.where(on_disk, out, 0.0).astype(np.float32)
