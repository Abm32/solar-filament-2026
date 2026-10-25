"""V89: domain self-supervised pretraining of the V73 CropUNet refiner (ResNet18 U-Net), ImageNet init.

Pre-registration: docs/v89_ssl_refiner_preregistration_2026-10-12.md. Never reads test images or any label.
Data: 707 train images (pixels only, no annotations) + /mnt/v76/frames (unlabelled GONG archive frames,
V76: train/test dates +-1 day excluded; re-asserted here: every frame > 1 h from every test timestamp).

  python scripts/v89_ssl.py prep                       # -> /mnt/v89/imgs.u8 memmap + meta.npz (CPU)
  CUDA_VISIBLE_DEVICES=0 python scripts/v89_ssl.py train --arm mim  --steps N
  CUDA_VISIBLE_DEVICES=1 python scripts/v89_ssl.py train --arm moco --steps N
  python scripts/v89_ssl.py demo                       # CPU self-check of sampler/masking
Arms:
  mim   masked-image modelling of the FULL U-Net: 32 px patches (8x8 grid on 256 crops), ratio 0.5, masked
        patches filled with the crop mean; target = per-crop standardised intensity; L1 on masked pixels.
  moco  MoCo v2 on the ResNet18 encoder (2048-128 MLP head, queue 16384, T 0.2, m 0.99, shuffled-chunk key BN).
Crops (both arms): centre from a dark-structure pixel (p 0.5; top 0.5% of blur30 - blur4 inside r < 0.9 R) or uniform
on the disk (r < 0.95 R); side log-uniform 96..768 native px, resized to 256 (V73 crop geometry); rot90/flip,
gamma 0.85-1.2, gain 0.9-1.1 (v11.augment_triplet ranges).
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
TRAIN_DIR, TEST_DIR, FRAMES = RAW / "train/train_images", RAW / "test/test_images", Path("/mnt/v76/frames")
DATA = Path("/mnt/v89")
SIZE, S = 2048, 256
N_DARK = 4000


def stamp(name):
    return dt.datetime.strptime(name[:14], "%Y%m%d%H%M%S")


def file_list():
    test = [stamp(f) for f in os.listdir(TEST_DIR)]
    train = sorted(TRAIN_DIR / f for f in os.listdir(TRAIN_DIR))
    frames = sorted(FRAMES / f for f in os.listdir(FRAMES) if f.endswith(".jpg"))
    tnames = set(os.listdir(TEST_DIR))
    for f in train + frames:
        assert f.name not in tnames
    for f in frames:  # host ruling: extra unlabelled frames OK except test timestamps +-1 h
        assert min(abs((stamp(f.name) - t).total_seconds()) for t in test) > 3600, f
    assert len(train) == 707, len(train)
    return train + frames


def prep_one(job):
    import cv2
    i, path = job
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None or g.shape != (SIZE, SIZE):
        return i, None
    b = cv2.GaussianBlur(g, (0, 0), 4)
    _, th = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, cen = cv2.connectedComponentsWithStats(th)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cx, cy = cen[k]
    r = float(np.sqrt(st[k, cv2.CC_STAT_AREA] / np.pi))
    gf = g.astype(np.float32)
    dark = cv2.GaussianBlur(gf, (0, 0), 30) - cv2.GaussianBlur(gf, (0, 0), 4)
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    disk = (xs - cx) ** 2 + (ys - cy) ** 2 < (0.9 * r) ** 2
    thr = np.percentile(dark[disk], 99.5)
    py, px = np.where(disk & (dark > thr))
    sel = np.random.default_rng(i).choice(len(py), min(N_DARK, len(py)), replace=False)
    pts = np.zeros((N_DARK, 2), np.uint16)
    pts[: len(sel)] = np.stack([px[sel], py[sel]], 1)
    pts[len(sel):] = pts[0]
    mm = np.memmap(DATA / "imgs.u8", np.uint8, "r+", shape=(NFILES, SIZE, SIZE))
    mm[i] = g
    mm.flush()
    return i, (cx, cy, r, pts)


def cmd_prep(_):
    global NFILES
    from multiprocessing import Pool
    files = file_list()
    NFILES = len(files)
    np.memmap(DATA / "imgs.u8", np.uint8, "w+", shape=(NFILES, SIZE, SIZE)).flush()
    with Pool(32) as p:
        res = dict(p.imap_unordered(prep_one, list(enumerate(files)), chunksize=4))
    ok = np.asarray([res[i] is not None for i in range(NFILES)])
    geo = np.asarray([res[i][:3] if ok[i] else (0, 0, 0) for i in range(NFILES)], np.float32)
    pts = np.stack([res[i][3] if ok[i] else np.zeros((N_DARK, 2), np.uint16) for i in range(NFILES)])
    np.savez(DATA / "meta.npz", names=np.asarray([f.name for f in files]), ok=ok, geo=geo, pts=pts,
             n=NFILES)
    print(json.dumps({"files": NFILES, "ok": int(ok.sum()), "train": 707, "frames": NFILES - 707,
                      "bad": [files[i].name for i in np.where(~ok)[0]]}))


class Crops:
    """Random crop sampler over the memmapped images (seeded per index, worker-safe)."""

    def __init__(self):
        m = np.load(DATA / "meta.npz")
        self.ok = np.where(m["ok"])[0]
        self.geo, self.pts, n = m["geo"], m["pts"], int(m["n"])
        self.imgs = np.memmap(DATA / "imgs.u8", np.uint8, "r", shape=(n, SIZE, SIZE))

    def bounds(self, rng):
        i = int(rng.choice(self.ok))
        cx, cy, r = self.geo[i]
        if rng.random() < 0.5:
            x, y = self.pts[i][rng.integers(N_DARK)].astype(float)
        else:
            a, d = rng.uniform(0, 2 * np.pi), 0.95 * r * np.sqrt(rng.random())
            x, y = cx + d * np.cos(a), cy + d * np.sin(a)
        side = float(np.exp(rng.uniform(np.log(96), np.log(768))))
        return i, x, y, side

    def crop(self, i, x, y, side):
        import cv2
        side = int(min(max(round(side), 32), SIZE))
        x0 = min(max(round(x - side / 2), 0), SIZE - side)
        y0 = min(max(round(y - side / 2), 0), SIZE - side)
        c = cv2.resize(np.asarray(self.imgs[i, y0:y0 + side, x0:x0 + side]), (S, S),
                       interpolation=cv2.INTER_LINEAR)
        return c.astype(np.float32) / 255.0


def augment(img, rng, strong=False):
    import cv2
    img = np.rot90(img, int(rng.integers(4)))
    if rng.random() < 0.5:
        img = np.flip(img, 1)
    if rng.random() < 0.5:
        img = img ** float(rng.uniform(0.85, 1.2))
    if rng.random() < 0.5:
        img = np.clip(img * float(rng.uniform(0.9, 1.1)), 0, 1)
    if strong:  # moco only: wider photometric + blur + noise
        if rng.random() < 0.8:
            m = img.mean()
            img = np.clip((img - m) * float(rng.uniform(0.6, 1.6)) + m, 0, 1)
        if rng.random() < 0.5:
            img = cv2.GaussianBlur(np.ascontiguousarray(img), (0, 0), float(rng.uniform(0.1, 2.0)))
        if rng.random() < 0.5:
            img = np.clip(img + rng.normal(0, rng.uniform(0, 0.03), img.shape).astype(np.float32), 0, 1)
    return np.ascontiguousarray(img, dtype=np.float32)


def mim_mask(rng, patch=32, ratio=0.5):
    g = S // patch
    m = np.zeros(g * g, np.float32)
    m[rng.choice(g * g, int(round(ratio * g * g)), replace=False)] = 1
    return np.kron(m.reshape(g, g), np.ones((patch, patch), np.float32))


class DS:
    def __init__(self, arm, seed, n):
        self.arm, self.seed, self.n, self.c = arm, seed, n, None

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        import torch
        if self.c is None:
            self.c = Crops()
        rng = np.random.default_rng([self.seed, k])
        i, x, y, side = self.c.bounds(rng)
        if self.arm == "mim":
            img = augment(self.c.crop(i, x, y, side), rng)
            return torch.from_numpy(img[None]), torch.from_numpy(mim_mask(rng)[None])
        views = []
        for _ in range(2):
            s2 = side * float(np.exp(rng.uniform(np.log(0.8), np.log(1.25))))
            dx, dy = rng.uniform(-0.15, 0.15, 2) * side
            views.append(torch.from_numpy(augment(self.c.crop(i, x + dx, y + dy, s2), rng, True)[None]))
        return views[0], views[1]


def load_base():
    spec = importlib.util.spec_from_file_location("solar_base", ROOT / "outputs/v6_inference_source/solar_base.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules['solar_base'] = m
    spec.loader.exec_module(m)
    return m


def encode(m, x):
    x = (x.repeat(1, 3, 1, 1) - m.mean) / m.std
    return m.layer4(m.layer3(m.layer2(m.layer1(m.pool(m.stem(x)))))).mean((2, 3))


def cmd_train(a):
    import copy
    import torch
    F = torch.nn.functional
    torch.manual_seed(a.seed)
    out = ROOT / "outputs/v89" / f"ssl_{a.arm}"
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    net = load_base().CropUNet(pretrained=True).to(dev)  # ImageNet ResNet18 encoder (continued pretraining)
    enc_p = [p for n, p in net.named_parameters() if n.split(".")[0] in ("stem", "layer1", "layer2", "layer3", "layer4")]
    dec_p = [p for n, p in net.named_parameters() if n.split(".")[0] not in ("stem", "layer1", "layer2", "layer3", "layer4")]
    if a.arm == "mim":
        opt = torch.optim.AdamW([{"params": enc_p, "lr": 1e-4}, {"params": dec_p, "lr": 1e-3}], weight_decay=0.05)
    else:
        proj = torch.nn.Sequential(torch.nn.Linear(512, 2048), torch.nn.ReLU(inplace=True),
                                   torch.nn.Linear(2048, 128)).to(dev)
        knet, kproj = copy.deepcopy(net), copy.deepcopy(proj)
        for p in list(knet.parameters()) + list(kproj.parameters()):
            p.requires_grad = False
        queue = F.normalize(torch.randn(16384, 128, device=dev), dim=1)
        qptr = 0
        opt = torch.optim.SGD(enc_p + list(proj.parameters()), lr=0.03, momentum=0.9, weight_decay=1e-4)
    base_lrs = [g["lr"] for g in opt.param_groups]
    loader = torch.utils.data.DataLoader(DS(a.arm, a.seed, a.steps * a.batch), batch_size=a.batch,
                                         shuffle=False, num_workers=a.workers, pin_memory=True,
                                         drop_last=True, persistent_workers=True, prefetch_factor=4)
    scaler = torch.cuda.amp.GradScaler()
    log, t0, run = open(out / "train.log", "a"), time.monotonic(), []
    for step, batch in enumerate(loader, 1):
        f = min(1.0, step / 1000) * 0.5 * (1 + np.cos(np.pi * step / a.steps))  # warmup + cosine
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f
        with torch.autocast("cuda", dtype=torch.float16):
            if a.arm == "mim":
                x, m = (t.to(dev, non_blocking=True) for t in batch)
                mu = x.mean((2, 3), keepdim=True)
                sd = x.std((2, 3), keepdim=True) + 0.02
                pred = net((x * (1 - m) + mu * m).repeat(1, 3, 1, 1)).float()
                loss = ((pred - (x - mu) / sd).abs() * m).sum() / m.sum()
            else:
                x1, x2 = (t.to(dev, non_blocking=True) for t in batch)
                with torch.no_grad():
                    for pq_, pk in zip(list(net.parameters()) + list(proj.parameters()),
                                       list(knet.parameters()) + list(kproj.parameters())):
                        pk.mul_(0.99).add_(pq_.detach(), alpha=0.01)
                    perm = torch.randperm(len(x2), device=dev)  # shuffled-chunk BN (single-GPU shuffle-BN)
                    k = torch.cat([kproj(encode(knet, c)) for c in x2[perm].chunk(4)])
                    k = F.normalize(k.float(), dim=1)[torch.argsort(perm)]
                q = F.normalize(proj(encode(net, x1)).float(), dim=1)
                logits = torch.cat([(q * k).sum(1, keepdim=True), q @ queue.T], 1) / 0.2
                loss = F.cross_entropy(logits, torch.zeros(len(q), dtype=torch.long, device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if a.arm == "moco":
            with torch.no_grad():
                queue[qptr:qptr + len(k)] = k
                qptr = (qptr + len(k)) % len(queue)
        run.append(float(loss))
        if not np.isfinite(run[-1]):
            raise RuntimeError(f"nonfinite loss at step {step}")
        if step % 200 == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(run)), "lr_f": f, "s": round(time.monotonic() - t0)}
            run = []
            print(json.dumps(rec), file=log, flush=True)
        if step % 5000 == 0 or step == a.steps:
            torch.save(net.state_dict(), out / "ssl_state.pt")
    (out / "DONE").write_text(json.dumps({"steps": a.steps, "batch": a.batch, "seconds": time.monotonic() - t0}))


def demo():
    rng = np.random.default_rng(0)
    m = mim_mask(rng)
    assert m.shape == (S, S) and abs(m.mean() - 0.5) < 1e-6
    assert set(np.unique(m[:32, :32])) <= {0.0, 1.0} and len(np.unique(m[:32, :32])) == 1
    img = augment(np.linspace(0, 1, S * S, dtype=np.float32).reshape(S, S), rng, True)
    assert img.shape == (S, S) and img.dtype == np.float32 and 0 <= img.min() and img.max() <= 1
    if (DATA / "meta.npz").exists():
        c = Crops()
        for _ in range(50):
            b = c.bounds(rng)
            assert c.crop(*b).shape == (S, S)
    print("demo ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "train", "demo"))
    ap.add_argument("--arm", choices=("mim", "moco"))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=89)
    a = ap.parse_args()
    if a.cmd == "prep":
        cmd_prep(a)
    elif a.cmd == "train":
        cmd_train(a)
    else:
        demo()
