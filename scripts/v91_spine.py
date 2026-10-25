"""V91 spine-first dense instance segmentation (docs/v91_spine_first_preregistration_2026-10-12.md).

No detector / refiner / COCO-HF weights: ResNet18 U-Net (CropUNet) from our in-domain MIM SSL state predicts
soft foreground + expert-spine heatmap at native res; instances = spine components, pixels assigned geodesically.
Train labels only (707 files); never reads test images/labels.

  python scripts/v91_spine.py demo                     # decoder self-check (synthetic, asserts)
  python scripts/v91_spine.py prep                     # /mnt/v91 soft targets + gt.pkl (CPU)
  python scripts/v91_spine.py oracle                   # G1: decode GT soft targets, 707 files
  CUDA_VISIBLE_DEVICES=0 python scripts/v91_spine.py train --fold 0
  CUDA_VISIBLE_DEVICES=0 python scripts/v91_spine.py infer --fold 0
  python scripts/v91_spine.py cache --fold 0           # decode grid + match stats -> /mnt/v91/cache_f0.pkl
  python scripts/v91_spine.py g2 | g3
  CUDA_VISIBLE_DEVICES=0 python scripts/v91_spine.py train --fold 0 --sanity   # S0 overfit check (8 files)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
ANN = RAW / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
DATA, OUT = Path("/mnt/v91"), ROOT / "outputs/v91"
SIZE, TILE, SIGMA = 2048, 1024, 2.5
NAMES = sorted(os.listdir(RAW / "train_images"))  # rows 0..706 of DATA/imgs.u8 (built by prep)
GRID = [(tf, ts) for tf in (0.3, 0.4, 0.5) for ts in (0.3, 0.5)]
K8 = np.ones((3, 3), bool)


def folds():
    return {int(k): v for k, v in json.loads((ROOT / "outputs/v61_folds.json").read_text())["folds"].items()}


def mm(name, mode="r", n=len(NAMES), ch=()):
    return np.memmap(DATA / name, np.uint8, mode, shape=(n, *ch, SIZE, SIZE))


def imgs():
    return mm("imgs.u8")


# ---------------------------------------------------------------- decoder
def decode(pf, ps, tf, ts, min_len=8):
    """Probabilities -> (label image, features[n,5]). Instances = spine seeds + seedless fg components."""
    from skimage.morphology import skeletonize
    from skimage.segmentation import watershed
    F = pf >= tf
    S = (ps >= ts) & F
    seeds, n = ndi.label(S, K8)
    sk = skeletonize(S)
    sklen = np.bincount(seeds[sk], minlength=n + 1)
    keep = sklen >= min_len
    keep[0] = False
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    seeds, n = remap[seeds], int(keep.sum())
    lab = watershed(np.zeros(F.shape, np.uint8), seeds, mask=F) if n else np.zeros(F.shape, np.int32)
    rest, m = ndi.label(F & (lab == 0), K8)
    lab = np.where(rest > 0, rest + n, lab).astype(np.int32)
    N = n + m
    flat = lab.ravel()
    area = np.bincount(flat, minlength=N + 1)[1:].astype(np.float64)
    mpf = np.bincount(flat, pf.ravel(), N + 1)[1:] / np.maximum(area, 1)
    skl = lab[sk & (seeds > 0)]
    slen = np.bincount(skl, minlength=N + 1)[1:].astype(np.float64)
    msp = np.bincount(skl, ps[sk & (seeds > 0)], N + 1)[1:] / np.maximum(slen, 1)
    mxp = ndi.maximum(ps, lab, np.arange(1, N + 1)) if N else np.zeros(0)
    feats = np.stack([mpf, msp, np.asarray(mxp, float), np.log(area + 1), np.log(slen + 1)], 1) if N \
        else np.zeros((0, 5))
    return lab, feats


def hits(lab, n, gt_masks):
    """Sparse IoU>0.5 entries (g, j, iou) of label image vs GT masks (bool stack). j is 0-based."""
    area = np.bincount(lab.ravel(), minlength=n + 1)
    out = []
    for g, gm in enumerate(gt_masks):
        inter = np.bincount(lab[gm], minlength=n + 1)
        iou = inter / np.maximum(area + gm.sum() - inter, 1)
        for j in np.where(iou[1:] > 0.5)[0]:
            out.append((g, int(j), float(iou[j + 1])))
    return out


def stats(recs, cols_ok):
    """Host-exact pooled PQ pieces for one file. recs: list of (n_gt, [hit-lists per column group]);
    cols_ok: per group bool array of kept columns. Returns [sumIoU, TP, FP, FN]."""
    s = np.zeros(4)
    for n_gt, groups in recs:
        hit_g = set()
        for hl, ok in zip(groups, cols_ok):
            hit_j = set()
            for g, j, iou in hl:
                if ok[j]:
                    s[0] += iou
                    s[1] += 1
                    hit_j.add(j)
                    hit_g.add(g)
            s[2] += ok.sum() - len(hit_j)
        s[3] += n_gt - len(hit_g)
    return s


def pq(s):
    s = np.asarray(s).reshape(-1, 4).sum(0)
    return s[0] / max(s[1] + 0.5 * s[2] + 0.5 * s[3], 1e-9)


def demo():
    import cv2
    yy, xx = np.mgrid[0:256, 0:256]
    truth = np.zeros((256, 256), np.int32)
    spine = np.zeros((256, 256), np.uint8)
    for k, y0 in enumerate((80, 100)):  # two near-parallel curved filaments, 20 px apart, touching fg
        pts = np.stack([np.arange(30, 200), y0 + 8 * np.sin(np.arange(30, 200) / 30)], 1).astype(np.int32)
        cv2.polylines(spine, [pts], False, 1)
        m = np.zeros_like(spine)
        cv2.polylines(m, [pts], False, 1, thickness=11)
        truth[(m > 0) & (truth == 0)] = k + 1
    truth[(xx - 200) ** 2 + (yy - 200) ** 2 < 100] = 3  # spine-less blob -> seedless instance
    d = ndi.distance_transform_edt(spine == 0)
    ps = np.exp(-d ** 2 / (2 * SIGMA ** 2)) * (truth > 0)
    pf = (truth > 0).astype(np.float64)
    lab, feats = decode(pf, ps, 0.5, 0.5)
    assert lab.max() == 3 and feats.shape == (3, 5), (lab.max(), feats.shape)
    for t in (1, 2, 3):
        h = hits(lab, 3, [truth == t])
        assert len(h) == 1 and h[0][2] > 0.9, (t, h)
    rec = [(3, [hits(lab, 3, [truth == t for t in (1, 2, 3)])])]
    assert abs(pq(stats(rec, [np.ones(3, bool)])) - np.mean([h[2] for h in rec[0][1][0]])) < 1e-9
    assert pq(stats(rec, [np.array([1, 1, 0], bool)])) < pq(stats(rec, [np.ones(3, bool)]))
    print("demo ok", [round(h[2], 3) for h in rec[0][1][0]])


# ---------------------------------------------------------------- data
def parse(s):
    return json.loads(s) if isinstance(s, str) else s


def records():
    doc = json.loads(ANN.read_text())
    anns = {}
    for a in doc["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    recs = {}
    for info in doc["images"]:
        recs.setdefault(info["file_name"], []).append(anns.get(info["id"], []))
    return recs


def prep_one(job):
    import cv2
    from pycocotools import mask as mu
    i, recs = job
    fg = np.zeros((SIZE, SIZE), np.float32)
    sp = np.zeros((SIZE, SIZE), np.float32)
    gt, pts = [], []
    for rec in recs:
        rles, union, line = [], np.zeros((SIZE, SIZE), bool), np.zeros((SIZE, SIZE), np.uint8)
        for a in rec:
            r = mu.merge(mu.frPyObjects(parse(a["segmentation"]), SIZE, SIZE))
            rles.append(r)
            union |= mu.decode(r).astype(bool)
            p = np.round(np.asarray(parse(a["spine"]), np.float64).reshape(-1, 2)).astype(np.int32)
            cv2.polylines(line, [p], False, 1)
            pts.append(p)
        gt.append(rles)
        fg += union
        if line.any():
            sp += np.exp(-ndi.distance_transform_edt(line == 0) ** 2 / (2 * SIGMA ** 2))
    n = max(len(recs), 1)
    g = cv2.imread(str(RAW / "train_images" / NAMES[i]), cv2.IMREAD_GRAYSCALE)
    assert g is not None and g.shape == (SIZE, SIZE), NAMES[i]
    im = mm("imgs.u8", "r+")
    im[i] = g
    im.flush()
    f, s = mm("fg.u8", "r+"), mm("sp.u8", "r+")
    f[i] = np.round(255 * fg / n).astype(np.uint8)
    s[i] = np.round(255 * np.clip(sp / n, 0, 1)).astype(np.uint8)
    f.flush(), s.flush()
    p = np.clip(np.concatenate(pts), 0, SIZE - 1).astype(np.int16) if pts else np.zeros((0, 2), np.int16)
    return i, gt, p


def cmd_prep(_):
    from multiprocessing import Pool
    recs = records()
    assert set(recs) == set(NAMES)
    mm("imgs.u8", "w+").flush(), mm("fg.u8", "w+").flush(), mm("sp.u8", "w+").flush()
    with Pool(32) as pool:
        res = list(pool.imap_unordered(prep_one, [(i, recs[f]) for i, f in enumerate(NAMES)], chunksize=2))
    res.sort(key=lambda r: r[0])
    pickle.dump({"gt": [r[1] for r in res], "pts": [r[2] for r in res]}, open(DATA / "gt.pkl", "wb"))
    f = mm("fg.u8")
    print(json.dumps({"files": len(res), "records": sum(len(r[1]) for r in res),
                      "gt": sum(len(x) for r in res for x in r[1]),
                      "fg_frac": float(np.mean([(f[i] > 0).mean() for i in range(0, 707, 20)]))}))


def gt_masks(rles):
    from pycocotools import mask as mu
    return [mu.decode(r).astype(bool) for r in rles]


# ---------------------------------------------------------------- G1 oracle
def oracle_one(i):
    G = pickle.load(open(DATA / "gt.pkl", "rb"))["gt"][i]
    lab, feats = decode(mm("fg.u8")[i] / 255.0, mm("sp.u8")[i] / 255.0, 0.5, 0.5)
    n = len(feats)
    recs = [(len(r), [hits(lab, n, gt_masks(r))]) for r in G]
    return stats(recs, [np.ones(n, bool)])


def cmd_oracle(_):
    from multiprocessing import Pool
    with Pool(48) as pool:
        S = np.asarray(pool.map(oracle_one, range(len(NAMES)), chunksize=2))
    f0 = [NAMES.index(f) for f in folds()[0]]
    r = {"pq_707": pq(S), "pq_fold0": pq(S[f0]), "tp_fp_fn": S.sum(0)[1:].astype(int).tolist(),
         "sq": S.sum(0)[0] / S.sum(0)[1], "g1_pass": bool(pq(S) >= 0.85)}
    (OUT / "g1_oracle.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


# ---------------------------------------------------------------- model
def build_model(ssl=True):
    import torch
    spec = importlib.util.spec_from_file_location("solar_base", ROOT / "outputs/v6_inference_source/solar_base.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["solar_base"] = m
    spec.loader.exec_module(m)
    net = m.CropUNet(pretrained=False)
    if ssl:
        sd = torch.load(ROOT / "outputs/v89/ssl_mim/ssl_state.pt", map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        assert not unexpected and set(missing) == {"head.weight", "head.bias"}, (missing, unexpected)
    net.head = torch.nn.Sequential(torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(inplace=True),
                                   torch.nn.Conv2d(32, 2, 1))
    torch.nn.init.constant_(net.head[2].bias, -4.6)
    return net


class Tiles:
    def __init__(self, idx, seed, n):
        self.idx, self.seed, self.n, self.d = idx, seed, n, None

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        import torch
        if self.d is None:
            self.d = (imgs(), mm("fg.u8"), mm("sp.u8"), pickle.load(open(DATA / "gt.pkl", "rb"))["pts"])
        im, fg, sp, pts = self.d
        rng = np.random.default_rng([self.seed, k])
        i = int(rng.choice(self.idx))
        if rng.random() < 0.8 and len(pts[i]):
            px, py = pts[i][rng.integers(len(pts[i]))]
            x0, y0 = (int(np.clip(v - rng.integers(32, TILE - 32), 0, SIZE - TILE)) for v in (px, py))
        else:
            x0, y0 = (int(v) for v in rng.integers(0, SIZE - TILE + 1, 2))
        sl = (i, slice(y0, y0 + TILE), slice(x0, x0 + TILE))
        x = np.stack([im[sl], fg[sl], sp[sl]]).astype(np.float32) / 255.0
        x = np.rot90(x, int(rng.integers(4)), (1, 2))
        if rng.random() < 0.5:
            x = x[:, :, ::-1]
        x = np.ascontiguousarray(x)
        if rng.random() < 0.5:
            x[0] = x[0] ** float(rng.uniform(0.85, 1.2))
        if rng.random() < 0.5:
            x[0] = np.clip(x[0] * float(rng.uniform(0.9, 1.1)), 0, 1)
        return torch.from_numpy(x[:1]), torch.from_numpy(x[1:])


def loss_fn(logit, y):
    import torch.nn.functional as F
    logit = logit.float()
    bce = F.binary_cross_entropy_with_logits(logit[:, 0], y[:, 0])
    p0 = logit[:, 0].sigmoid()
    dice = 1 - (2 * (p0 * y[:, 0]).sum() + 1) / (p0.sum() + y[:, 0].sum() + 1)
    p1 = logit[:, 1].sigmoid()
    qfl = (F.binary_cross_entropy_with_logits(logit[:, 1], y[:, 1], reduction="none")
           * (y[:, 1] - p1).abs() ** 2).sum() / y[:, 1].sum().clamp(min=1)
    return bce + dice + qfl, (bce.item(), dice.item(), qfl.item())


def predict(net, i):
    import torch
    x = torch.from_numpy(imgs()[i].astype(np.float32) / 255.0)[None, None].cuda().repeat(1, 3, 1, 1)
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        p = net(x).float().sigmoid()[0].cpu().numpy()
    return np.round(p * 255).astype(np.uint8)


def cmd_train(a):
    import torch
    torch.manual_seed(a.seed)
    f = folds()
    held = set(f[a.fold]) if a.fold in f else set()  # fold 99 = all 707 files (export model)
    idx = [i for i, n in enumerate(NAMES) if n not in held]
    tag = f"f{a.fold}" + ("_sanity" if a.sanity else "")
    if a.sanity:
        idx, a.steps = idx[:8], 600
    od = OUT / tag
    od.mkdir(parents=True, exist_ok=True)
    net = build_model().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    scaler = torch.cuda.amp.GradScaler()
    dl = torch.utils.data.DataLoader(Tiles(idx, a.seed, a.steps * a.bs), batch_size=a.bs, num_workers=8,
                                     persistent_workers=True, prefetch_factor=4)
    log, t0, acc = open(od / "train.log", "a"), time.time(), []
    net.train()
    for step, (x, y) in enumerate(dl):
        x, y = x.cuda(non_blocking=True).repeat(1, 3, 1, 1), y.cuda(non_blocking=True)
        with torch.autocast("cuda", torch.float16):
            out = net(x)
        loss, parts = loss_fn(out, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        acc.append(parts)
        assert np.isfinite(loss.item()), step
        if (step + 1) % 100 == 0:
            pf = out[:, 0].float().sigmoid()
            r = {"step": step + 1, "bce_dice_qfl": np.mean(acc, 0).round(4).tolist(),
                 "fgpix_ratio": float((pf >= 0.5).sum() / (y[:, 0] >= 0.5).sum().clamp(min=1)),
                 "mem": torch.cuda.max_memory_allocated() // 2 ** 20, "s": int(time.time() - t0)}
            log.write(json.dumps(r) + "\n")
            log.flush()
            acc = []
    torch.save(net.state_dict(), od / "model.pt")
    if a.sanity:  # S0: in-sample decode, keep all
        net.eval()
        G = pickle.load(open(DATA / "gt.pkl", "rb"))["gt"]
        S, ratio = [], []
        for i in idx:
            p = predict(net, i)
            lab, feats = decode(p[0] / 255.0, p[1] / 255.0, 0.5, 0.5)
            S.append(stats([(len(r), [hits(lab, len(feats), gt_masks(r))]) for r in G[i]], [np.ones(len(feats), bool)]))
            ratio.append(float((p[0] >= 128).sum() / max((mm("fg.u8")[i] >= 128).sum(), 1)))
        r = {"s0_pq_keepall": pq(S), "fg_ratio": ratio, "tp_fp_fn": np.sum(S, 0)[1:].tolist(),
             "pass": bool(pq(S) >= 0.5 and all(0.3 <= q <= 3 for q in ratio))}
        (od / "s0.json").write_text(json.dumps(r, indent=1))
        log.write(json.dumps(r) + "\n")
    log.write(json.dumps({"done": True, "s": int(time.time() - t0)}) + "\n")


def cmd_infer(a):
    import torch
    net = build_model(ssl=False).cuda()
    net.load_state_dict(torch.load(OUT / f"f{a.fold}" / "model.pt", map_location="cuda", weights_only=True))
    net.eval()
    files = folds()[a.fold]
    out = mm(f"pred_f{a.fold}.u8", "w+", len(files), (2,))
    for k, f in enumerate(files):
        out[k] = predict(net, NAMES.index(f))
    out.flush()
    print("infer done", a.fold, len(files))


# ---------------------------------------------------------------- cache (decode grid + match bookkeeping)
def v84_rles(fold):
    import pandas as pd
    df = pd.read_csv(ROOT / f"outputs/v87/fold{fold}_v84.csv", dtype=str, keep_default_na=False)
    out = {}
    for fid, r in zip(df.filament_id, df.segmentation_rle):
        out.setdefault(fid.rsplit("_", 1)[0], []).append({"size": [SIZE, SIZE], "counts": r.encode()})
    return out


def cache_one(job):
    from pycocotools import mask as mu
    fold, k, fname, v84 = job
    i = NAMES.index(fname)
    G = pickle.load(open(DATA / "gt.pkl", "rb"))["gt"][i]
    p = mm(f"pred_f{fold}.u8", "r", len(folds()[fold]), (2,))[k]
    pf, ps = p[0] / 255.0, p[1] / 255.0
    gms = [gt_masks(r) for r in G]
    v84m = gt_masks(v84)
    U = np.any(v84m, 0) if v84m else np.zeros((SIZE, SIZE), bool)
    h84 = []
    for r in G:
        M = mu.iou(r, v84, [0] * len(v84)) if len(r) and len(v84) else np.zeros((0, 0))
        h84.append([(int(g), int(j), float(M[g, j])) for g, j in zip(*np.where(M > 0.5))])
    fp84 = np.ones(len(v84), bool)
    for hl in h84:
        for _, j, _ in hl:
            fp84[j] = False
    out = {"name": fname, "n_gt": [len(r) for r in G], "h84": h84, "n84": len(v84), "grid": {}}
    for tf, ts in GRID:
        lab, feats = decode(pf, ps, tf, ts)
        n = len(feats)
        area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        frac = np.bincount(lab[U], minlength=n + 1)[1:] / np.maximum(area, 1)
        labm = np.where(U, 0, lab)
        # V91 instances matching a V84 FP (IoU>0.5) -> shared-FP bookkeeping
        same84 = np.zeros(n, bool)
        for g, j, _ in hits(lab, n, [v84m[q] for q in np.where(fp84)[0]]):
            same84[j] = True
        out["grid"][(tf, ts)] = {"feats": feats, "frac": frac, "same84fp": same84,
                                 "h": [hits(lab, n, gm) for gm in gms],
                                 "hm": [hits(labm, n, gm) for gm in gms]}
    return out


def cmd_cache(a):
    from multiprocessing import Pool
    v84 = v84_rles(a.fold)
    files = folds()[a.fold]
    jobs = [(a.fold, k, f, v84.get(Path(f).stem, [])) for k, f in enumerate(files)]
    with Pool(a.procs) as pool:
        res = pool.map(cache_one, jobs, chunksize=1)
    pickle.dump(res, open(DATA / f"cache_f{a.fold}.pkl", "wb"))
    print("cache done", a.fold, len(res))


# ---------------------------------------------------------------- selection / scoring
def targets(c, g):
    """Per-instance soft target m = fraction of records with an IoU>0.5 hit."""
    e = c["grid"][g]
    m = np.zeros(len(e["feats"]))
    for hl in e["h"]:
        for _, j, _ in hl:
            m[j] += 1
    return m / max(len(c["n_gt"]), 1)


def fit_lr(X, y):
    from scipy.optimize import minimize
    mu_, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.c_[(X - mu_) / sd, np.ones(len(X))]

    def f(w):
        z = Z @ w
        return np.sum(np.logaddexp(0, z) - y * z) / len(y) + 1e-3 * w[:-1] @ w[:-1], \
            Z.T @ (1 / (1 + np.exp(-z)) - y) / len(y) + np.r_[2e-3 * w[:-1], 0]
    w = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    return lambda X_: 1 / (1 + np.exp(-(np.c_[(X_ - mu_) / sd, np.ones(len(X_))] @ w)))


TAUS = np.round(np.arange(0.02, 0.96, 0.01), 2)


def file_stats(c, g, keep, mode):
    e = c["grid"][g]
    if mode == "alone":
        return stats([(n, [hl]) for n, hl in zip(c["n_gt"], e["h"])], [keep])
    if mode == "v84":
        return stats([(n, [h8]) for n, h8 in zip(c["n_gt"], c["h84"])], [np.ones(c["n84"], bool)])
    add = keep & (e["frac"] < 0.1)
    return stats([(n, [h8, hm]) for n, h8, hm in zip(c["n_gt"], c["h84"], e["hm"])],
                 [np.ones(c["n84"], bool), add])


def select(fit, mode):
    """On fitting files: best (grid, model, tau) by pooled PQ of `mode`."""
    best = None
    for g in GRID:
        model = fit_lr(np.concatenate([c["grid"][g]["feats"] for c in fit]),
                       np.concatenate([targets(c, g) for c in fit]))
        ms = [model(c["grid"][g]["feats"]) for c in fit]
        for t in TAUS:
            s = pq([file_stats(c, g, m > t, mode) for c, m in zip(fit, ms)])
            if best is None or s > best[0]:
                best = (s, g, model, float(t))
    return best


def apply(sel, files, mode):
    _, g, model, t = sel
    out = []
    for c in files:
        m = model(c["grid"][g]["feats"])
        out.append((file_stats(c, g, m > t, mode), m > t, g))
    return out


def boot(a, b, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(0)
    d = [pq(a[ix]) - pq(b[ix]) for ix in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return float(np.mean(np.asarray(d) > 0))


def orthogonality(cs, kept):
    rec84 = fn84 = fp_sh = fp91 = 0
    for c, (_, keep, g) in zip(cs, kept):
        e = c["grid"][g]
        for hl, h8 in zip(e["h"], c["h84"]):
            g84 = {x[0] for x in h8}
            g91 = {x[0] for x in hl if keep[x[1]]}
            rec84 += len(g91 - g84)
        fn84 += sum(n - len({x[0] for x in h8}) for n, h8 in zip(c["n_gt"], c["h84"]))
        tp_any = np.zeros(len(keep), bool)
        for hl in e["h"]:
            for _, j, _ in hl:
                tp_any[j] = True
        fpj = keep & ~tp_any
        fp91 += int(fpj.sum())
        fp_sh += int((fpj & e["same84fp"]).sum())
    return {"v84_fn_pairs": fn84, "v84_fn_recovered_by_v91": rec84, "v91_fp_never_tp": fp91,
            "v91_fp_shared_with_v84_fp": fp_sh}


def cmd_g2(_):
    cs = pickle.load(open(DATA / "cache_f0.pkl", "rb"))
    halves = [cs[0::2], cs[1::2]]
    res = {}
    base = [file_stats(c, GRID[0], None, "v84") for c in cs]
    res["v84_fold0_check"] = pq(base)
    for mode in ("alone", "fused"):
        sels = [select(halves[1 - h], mode) for h in (0, 1)]
        app = [apply(sels[h], halves[h], mode) for h in (0, 1)]
        out = [None] * len(cs)
        out[0::2], out[1::2] = app[0], app[1]
        S = [o[0] for o in out]
        res[mode] = {"pq": pq(S), "d_vs_v84": pq(S) - pq(base), "p_gt0": boot(S, base),
                     "tp_fp_fn": np.sum(S, 0)[1:].astype(int).tolist(),
                     "selected": [{"fitPQ": s[0], "grid": s[1], "tau": s[3]} for s in sels]}
        if mode == "alone":
            res["orthogonality"] = orthogonality(cs, out)
    a, f = res["alone"], res["fused"]
    res["continue"] = bool(a["pq"] >= 0.45 or (f["d_vs_v84"] >= 0.008 and f["p_gt0"] >= 0.90))
    res["arm"] = "alone" if a["pq"] >= 0.45 else ("fused" if res["continue"] else None)
    (OUT / "g2.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def cmd_g3(a):
    mode = a.mode
    cs = {k: pickle.load(open(DATA / f"cache_f{k}.pkl", "rb")) for k in range(5)}
    S, B, per = [], [], {}
    for k in range(5):
        sel = select([c for q in range(5) if q != k for c in cs[q]], mode)
        s = [o[0] for o in apply(sel, cs[k], mode)]
        b = [file_stats(c, GRID[0], None, "v84") for c in cs[k]]
        per[k] = {"pq": pq(s), "v84": pq(b), "grid": sel[1], "tau": sel[3]}
        S += s
        B += b
    res = {"mode": mode, "pq": pq(S), "v84": pq(B), "d": pq(S) - pq(B), "p_gt0": boot(S, B), "folds": per,
           "tp_fp_fn": np.sum(S, 0)[1:].astype(int).tolist()}
    res["g3_pass"] = bool(res["d"] >= 0.010 and res["p_gt0"] >= 0.90)
    (OUT / f"g3_{mode}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--mode", default="alone")
    ap.add_argument("--sanity", action="store_true")
    a = ap.parse_args()
    {"demo": lambda _: demo(), "prep": cmd_prep, "oracle": cmd_oracle, "train": cmd_train, "infer": cmd_infer,
     "cache": cmd_cache, "g2": cmd_g2, "g3": cmd_g3}[a.cmd](a)
