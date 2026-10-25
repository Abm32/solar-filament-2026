"""V97: annotator-style crop refiner, fold 0. Pre-registration: docs/v97_style_refiner_preregistration_2026-10-24.md.
Never reads test. Recipe = V84 soft (train_v82_refiner_targets --arm soft): stage A 8 ep 3e-4 on GT crops,
stage B 12 ep 1e-4 on p2first fold-0 OOF proposals, seed 42, ImageNet ResNet18 CropUNet, 256 px crops.

Arms
  soft_g01  stage B = group-01 records only; target = mean of the GTs the image's group-01 records matched to p
  cond      all records; CropUNet + zero-init FiLM(token) at the bottleneck and the 1/4 decoder block.
            token 0 = pooled (V84 soft target), token k = annotation group k (per-record GT target); p=0.5 each.
Usage
  CUDA_VISIBLE_DEVICES=1 python scripts/v97_style_refiner.py fold --arm cond --fold 0 [--limit N --epochs-a 1 --epochs-b 1]
  CUDA_VISIBLE_DEVICES= python scripts/v97_style_refiner.py extract TAG CKPT [--token K | --oracle]   # CKPT 'soft' = V84
  python scripts/v97_style_refiner.py gate TAG [TAG ...]
  python scripts/v97_style_refiner.py demo
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v97"
GROUPS = ("01", "02", "03", "04", "05", "06")
SUBSETS = {"all": lambda a: True, "hard": lambda a: a in ("010101", "010401"),
           "g01": lambda a: a.startswith("01"), "rest": lambda a: not a.startswith("01")}
PRIMARY_CELL, PRIMARY_SUBSET = "t050_c030", "g01"
GATE = dict(cont=0.020, p=0.90, kill=0.005)
T, CUT, MINA = 0.50, 0.30, 200


# ----------------------------------------------------------------------------- model
def cond_unet(base):
    """base.CropUNet + FiLM on a 7-token embedding; zero-init so token-agnostic at start == V73 build_model."""
    import torch
    import torchvision
    nn = torch.nn

    class CondUNet(base.CropUNet):
        def __init__(self):
            super().__init__(pretrained=False)
            self.emb = nn.Embedding(1 + len(GROUPS), 64)
            self.film4 = nn.Linear(64, 2 * 512)
            self.film2 = nn.Linear(64, 2 * 64)
            for m in (self.film4, self.film2):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
            self.token = 0

        def forward(self, image, g=None):
            if g is None:
                g = torch.full((image.shape[0],), int(self.token), dtype=torch.long, device=image.device)
            e = self.emb(g)
            g4, b4 = self.film4(e).chunk(2, 1)
            g2, b2 = self.film2(e).chunk(2, 1)
            image = (image - self.mean) / self.std
            x0 = self.stem(image)
            x1 = self.layer1(self.pool(x0))
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            x4 = x4 * (1 + g4[:, :, None, None]) + b4[:, :, None, None]
            x = self.dec4(torch.cat([self.up4(x4), x3], 1))
            x = self.dec3(torch.cat([self.up3(x), x2], 1))
            x = self.dec2(torch.cat([self.up2(x), x1], 1))
            x = x * (1 + g2[:, :, None, None]) + b2[:, :, None, None]
            x = self.dec1(torch.cat([self.up1(x), x0], 1))
            return self.head(self.up0(x))

    model = CondUNet()
    enc = torchvision.models.resnet18(weights="DEFAULT")  # same swap as train_v73_refiner.build_model
    model.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
    model.pool, model.layer1, model.layer2 = enc.maxpool, enc.layer1, enc.layer2
    model.layer3, model.layer4 = enc.layer3, enc.layer4
    return model


def build(arm, base, dev):
    import train_v73_refiner as V73
    if arm == "cond":
        return cond_unet(base).to(dev)
    V73.DEV = dev
    return V73.build_model(base, "imagenet", "resnet18")


def load(arm, ckpt, base, dev="cpu"):
    import torch
    m = build(arm, base, dev)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
    return m.eval()


# ----------------------------------------------------------------------------- records / targets
def gid_of(image_id):
    return str(image_id)[:2]


def gt_records(rt, names):
    """train_v73_refiner.gt_records + the record's group."""
    from pycocotools import mask as mu
    core, base, out = rt["core"], rt["base"], []
    for name in names:
        for image_id in sorted(rt["by_file"].get(name, []), key=str):
            info = rt["images"][image_id]
            for a in rt["annotations"].get(image_id, []):
                rle = core.annotation_rle(info, a)
                if mu.area(rle) > 0:
                    out.append({"name": name, "seed": rle, "target": rle, "gid": gid_of(image_id),
                                "bounds": tuple(base.square_bounds(mu.decode(rle)))})
    return out


def fit_records(rt, recipe, names, fmap):
    """train_v82_refiner_targets.fit_records + per-record group and the groups of the matching records."""
    import train_v73_refiner as V73
    from pycocotools import mask as mu
    v11, v50, R = V73.v11, V73.v50, V73.R
    base, core, tta = rt["base"], rt["core"], rt["tta"]
    records = []
    for i, name in enumerate(names, 1):
        views = R.K.read_cache_views(R.O.cache_path(recipe, fmap[name], name))
        proposals, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        ids = sorted(rt["by_file"].get(name, []), key=str)
        gids = [gid_of(x) for x in ids]
        file_recs, matched = [], {}
        for image_id in ids:
            info = rt["images"][image_id]
            targets = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
            targets = [t for t in targets if mu.area(t) > 0]
            if not targets or not proposals:
                continue
            iou = mu.iou(targets, proposals, [0] * len(proposals))
            for ti, pi in v11.match_positive_proposals(iou, scores):
                seed = mu.decode(proposals[pi])
                file_recs.append({"name": name, "seed": proposals[pi], "target": targets[ti], "pi": pi,
                                  "gid": gid_of(image_id), "bounds": tuple(base.square_bounds(seed))})
                matched.setdefault(pi, []).append((gid_of(image_id), targets[ti]))
        for r in file_recs:
            r["agree"], r["R"], r["Rg"] = matched[r["pi"]], len(ids), gids.count(r["gid"])
        records += file_recs
        if i % 50 == 0:
            print(f"cohort {i}/{len(names)} records={len(records)}", flush=True)
    if not records:
        raise RuntimeError("empty training cohort")
    return records


def target_u8(r, kind):
    """kind: 'gt' (own record), 'soft' (mean over all R records, V84), 'soft_g' (mean over own group's records)."""
    from pycocotools import mask as mu
    if kind == "gt" or "agree" not in r:
        return mu.decode(r["target"]) * np.uint8(255)
    if kind == "soft":
        sel, n = [t for _, t in r["agree"]], r["R"]
    else:
        sel, n = [t for g, t in r["agree"] if g == r["gid"]], r["Rg"]
    m = np.stack([mu.decode(t) for t in sel]).astype(np.float32).sum(0) / n
    return np.rint(m * 255).astype(np.uint8)


class CropDataset:
    """V82 dataset; returns (image, target, token). soft_g01: token 0, group-mean target.
    cond: p=0.5 token=group with GT target, else token 0 with the V84 soft target."""

    def __init__(self, records, train_dir, epoch, seed, size, arm, stage):
        self.records, self.train_dir, self.epoch, self.seed = records, train_dir, epoch, seed
        self.size, self.arm, self.stage, self.cache = size, arm, stage, {}

    def __len__(self):
        return len(self.records)

    def sample(self, r, rng):
        if self.arm == "soft_g01":
            return 0, ("gt" if self.stage == "A" else "soft_g")
        if rng.random() < 0.5:
            return 1 + GROUPS.index(r["gid"]), "gt"
        return 0, ("gt" if self.stage == "A" else "soft")

    def __getitem__(self, index):
        import cv2
        import torch
        import train_v73_refiner as V73
        from pycocotools import mask as mu
        v11 = V73.v11
        r = self.records[index]
        if r["name"] not in self.cache:
            if len(self.cache) > 8:
                self.cache.clear()
            self.cache[r["name"]] = cv2.imread(str(self.train_dir / r["name"]), cv2.IMREAD_GRAYSCALE)
        gray = self.cache[r["name"]]
        rng = np.random.default_rng(self.seed * 7919 + self.epoch * 1_000_003 + index)
        tok, kind = self.sample(r, rng)
        seed, target = mu.decode(r["seed"]), target_u8(r, kind)
        jit = v11.jitter_bounds(r["bounds"], gray.shape, rng)
        trip = v11.crop_triplet(gray, seed, target, jit, self.size)
        if not trip[1].any() or not trip[2].any():
            trip = v11.crop_triplet(gray, seed, target, r["bounds"], self.size)
        image, _, tgt = v11.augment_triplet(*trip, rng)
        return (torch.from_numpy(np.repeat(image[None], 3, axis=0)),
                torch.from_numpy(tgt.astype(np.float32)[None] / 255.0), torch.tensor(tok, dtype=torch.long))


def train(rt, model, records, epochs, lr, seed, size, stage, arm, dev):
    """V82 loop: AdamW wd 1e-4, cosine, 0.5 BCE + 0.5 Dice, batch 16, fp32."""
    import torch
    import train_v73_refiner as V73
    F = torch.nn.functional
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for epoch in range(1, epochs + 1):
        ds = CropDataset(records, rt["base"].CFG.train_dir, epoch, seed, size, arm, stage)
        order = np.random.default_rng(seed + epoch).permutation(len(records)).tolist()
        loader = torch.utils.data.DataLoader(ds, batch_size=V73.v11.BATCH_SIZE, sampler=order,
                                             num_workers=8, pin_memory=dev == "cuda")
        model.train()
        total = n = 0
        t0 = time.monotonic()
        for x, y, g in loader:
            x, y, g = x.to(dev), y.to(dev), g.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(x, g) if arm == "cond" else model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y)
            p = logits.sigmoid()
            dice = 1 - ((2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean()
            loss = 0.5 * bce + 0.5 * dice
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss {stage} epoch {epoch}")
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(x)
            n += len(x)
        sched.step()
        history.append({"stage": stage, "epoch": epoch, "mean_loss": total / n, "seconds": time.monotonic() - t0})
        if dev == "cuda":
            history[-1]["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
        print(json.dumps(history[-1]), flush=True)
    return history


def cmd_fold(args):
    import torch
    import train_v73_refiner as V73
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    V73.DEV = dev
    OUT.mkdir(parents=True, exist_ok=True)
    result = OUT / f"{args.arm}_fold{args.fold}.json"
    if result.exists() and not args.limit:
        print(f"{result} exists; skipping")
        return
    t0 = time.monotonic()
    rt = V73.runtime()
    rt["base"].CFG.crop_size = 256
    fmap = V73.R.fold_of_file()
    assert len(fmap) == 707
    test = sorted(n for n, k in fmap.items() if k == args.fold)
    fit = sorted(n for n, k in fmap.items() if k != args.fold)
    assert not set(test) & set(fit)
    if args.limit:
        fit, test = fit[: args.limit], test[: args.limit]
    rep = {"arm": args.arm, "fold": args.fold, "recipe": V73.RECIPE, "seed": args.seed, "reads_test": False,
           "fit_files": len(fit), "scored_files": len(test), "device": dev, "history": []}
    rt["base"].seed_everything(args.seed)
    model = build(args.arm, rt["base"], dev)
    recs = gt_records(rt, fit)
    rep["records_a"] = len(recs)
    rep["history"] += train(rt, model, recs, args.epochs_a, V73.STAGE_A[1], args.seed, 256, "A", args.arm, dev)
    recs = fit_records(rt, V73.RECIPE, fit, fmap)
    rep["records_b_all"] = len(recs)
    rep["groups_b"] = {g: sum(r["gid"] == g for r in recs) for g in GROUPS}
    if args.arm == "soft_g01":
        recs = [r for r in recs if r["gid"] == "01"]
    rep["records_b"] = len(recs)
    rep["history"] += train(rt, model, recs, args.epochs_b, V73.STAGE_B[1], args.seed, 256, "B", args.arm, dev)
    ck = OUT / f"{args.arm}_fold{args.fold}{'_smoke' if args.limit else ''}_refiner.pt"
    torch.save({"model": model.state_dict(), "arm": args.arm, "fold": args.fold}, ck)
    model.eval()
    rep["per_file"] = {}
    with torch.no_grad():
        for i, name in enumerate(test, 1):
            rep["per_file"][name] = V73.file_counts(rt, model, args.fold, name, kernels=(7,))
            if i % 20 == 0:
                print(f"score {i}/{len(test)}", flush=True)
    tot = np.sum([v["k7"]["t0.50_c0.30"] for v in rep["per_file"].values()], axis=0)
    rep["k7_t0.50_c0.30"] = {"pq": V73.R.K.pq_value(tot), "tp": int(tot[1]), "fp": int(tot[2]), "fn": int(tot[3])}
    rep["elapsed_seconds"] = time.monotonic() - t0
    rep["checkpoint"] = str(ck)
    if not args.limit:
        result.write_text(json.dumps(rep) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_file", "history")}, indent=2))


# ----------------------------------------------------------------------------- extract (CPU, fold 0)
X = E = V73 = None
TOKEN, ORACLE = 0, False


def nearmiss(gt, kept):
    """(iou, inter/pred area, inter/gt area) for GT x kept pairs with 0.3 < IoU <= 0.5."""
    from pycocotools import mask as mu
    out = []
    if not gt or not kept:
        return out
    m = np.asarray(mu.iou(gt, kept, [0] * len(kept)))
    for gi, ki in zip(*np.where((m > 0.3) & (m <= 0.5))):
        inter = float(mu.area(mu.merge([gt[gi], kept[ki]], intersect=True)))
        out.append((float(m[gi, ki]), inter / float(mu.area(kept[ki])), inter / float(mu.area(gt[gi]))))
    return out


def work(job):
    """v84_extract.work at t0.50 with per-record tags, near-miss geometry and token handling."""
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = E.RT, E.RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    recs, tags = [], []
    for image_id in sorted(rt["by_file"][name], key=str):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        recs.append([r for r in rl if mu.area(r) > 0])
        tags.append(str(image_id).split("-", 1)[0])
    views = V73.R.K.read_cache_views(V73.R.O.cache_path(V73.RECIPE, fold, name))
    groups = E.groups_with_ids(views)
    scores = [g["score"] for g in groups]
    disk = E.disk_fit(gray)
    model = E.MODELS[fold]
    tokens = sorted({1 + GROUPS.index(t[:2]) for t in tags}) if ORACLE else [TOKEN]
    runs = {}
    for tok in tokens:
        if hasattr(model, "token"):
            model.token = tok
        post, feats, masks, bounds = [], [], [], []
        if groups:
            masks = [mu.decode(g["rle"]) for g in groups]
            bounds, _, probs = core.crop_probabilities(rt["base"], model, gray, masks, True)
            X.T = T
            post, feats = X.refine_feats(masks, scores, bounds, probs, gray, disk)
        alive = [i for i in range(len(groups)) if post[i] is not None]
        cand, rows = X.candidate_rows(groups, scores, alive, feats)
        for c, f in zip(cand, rows):
            f["area"] = float(mu.area(post[alive[c]]))
        kept = [post[alive[c]] for c, f in zip(cand, rows) if scores[alive[c]] >= CUT and f["area"] >= MINA]
        runs[tok] = dict(alive=alive, post=post, cand=cand, rows=rows, kept=kept)
    out = dict(name=name, fold=fold, tags=tags, n_gt=[len(g) for g in recs])
    run = runs[tokens[0]]
    out["rows"] = run["rows"]
    out["scores"] = [scores[run["alive"][c]] for c in run["cand"]]
    out["iou"], out["nm"], out["tot_rec"] = [], [], []
    for g, tag in zip(recs, tags):
        r = runs[1 + GROUPS.index(tag[:2])] if ORACLE else run
        I = E.iou_mat(g, [r["post"][i] for i in r["alive"]])
        out["iou"].append(I[:, r["cand"]] if g else np.zeros((0, len(r["cand"]))))
        out["nm"].append(nearmiss(g, r["kept"]))
        ik = E.iou_mat(g, r["kept"])
        out["tot_rec"].append([float(v) for v in core.pq_counts(
            ik if g and r["kept"] else np.zeros((len(g), len(r["kept"]))), len(g), len(r["kept"]))])
    out["tot"] = np.sum(out["tot_rec"], 0).tolist() if out["tot_rec"] else [0, 0, 0, 0]
    return out


def cmd_extract(tag, ckpt, token, oracle):
    global X, E, V73, TOKEN, ORACLE
    import torch
    import audit_v80_extract as E_
    import train_v73_refiner as V73_
    import v83_extract as X_
    X, E, V73, TOKEN, ORACLE = X_, E_, V73_, token, oracle
    set_start_method("fork")
    V73.DEV = "cpu"
    E.RT = V73.runtime()
    arm = "cond" if "cond" in Path(ckpt).name else "soft"
    E.MODELS[0] = load(arm, ckpt, E.RT["base"])
    fmap = V73.R.fold_of_file()
    names = sorted(n for n, k in fmap.items() if k == 0)
    assert len(names) == 141
    lim = int(os.environ.get("LIMIT", "0"))
    if lim:
        names = names[:lim]
    with torch.no_grad(), Pool(int(os.environ.get("PROCS", "22")), initializer=E.init) as p:
        res = p.map(work, [(n, 0) for n in names], chunksize=1)
    tot = np.sum([r["tot"] for r in res], 0)
    print(tag, "files", len(res), "t050_c030 pq", V73.R.K.pq_value(tot), tot.tolist(), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / f"extract_{tag}{'_smoke' if lim else ''}.pkl", "wb"))


# ----------------------------------------------------------------------------- gate (CPU)
def tags_by_file():
    raw = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    doc = json.loads(raw.read_text())
    by = {}
    for im in doc["images"]:
        by.setdefault(im["file_name"], []).append(im["id"])
    return {f: [str(i).split("-", 1)[0] for i in sorted(ids, key=str)] for f, ids in by.items()}


def subset_counts(res, keep, fi, tags, ok):
    """v83_rule.file_counts restricted to records whose annotator passes `ok`."""
    out = np.zeros((len(res), 4))
    for f, r in enumerate(res):
        k = keep[fi == f]
        for m, n, tag in zip(r["iou"], r["n_gt"], tags[r["name"]]):
            if not ok(tag):
                continue
            sub = m[:, k] if m.size else np.zeros((n, int(k.sum())))
            if not n or not k.sum():
                out[f] += (0.0, 0, int(k.sum()), n)
                continue
            h = sub > 0.5
            out[f] += (sub[h].sum(), h.sum(), (~h.any(0)).sum(), (~h.any(1)).sum())
    return out


def band_stats(res, keep, fi, tags, ok, nm=None):
    """FN/FP split by best IoU with the kept set (0 / <.3 / .3-.5) and near-miss containment."""
    fn0 = fnl = fnm = fp0 = fpl = fpm = 0
    pa, ga = [], []
    for f, r in enumerate(res):
        k = keep[fi == f]
        for ri, (m, n, tag) in enumerate(zip(r["iou"], r["n_gt"], tags[r["name"]])):
            if not ok(tag):
                continue
            sub = m[:, k] if m.size else np.zeros((n, int(k.sum())))
            if n and k.sum():
                h = sub > 0.5
                bg = sub.max(1)[~h.any(1)]
                bp = sub.max(0)[~h.any(0)]
            else:
                bg, bp = np.zeros(n), np.zeros(int(k.sum()))
            fn0 += int((bg == 0).sum()); fnl += int(((bg > 0) & (bg < 0.3)).sum()); fnm += int((bg >= 0.3).sum())
            fp0 += int((bp == 0).sum()); fpl += int(((bp > 0) & (bp < 0.3)).sum()); fpm += int((bp >= 0.3).sum())
            if nm is not None and "nm" in r:
                for _, p, g in r["nm"][ri]:
                    pa.append(p); ga.append(g)
    d = dict(fn_iou0=fn0, fn_lt03=fnl, fn_03_05=fnm, fp_iou0=fp0, fp_lt03=fpl, fp_03_05=fpm)
    if pa:
        d.update(nm_pairs=len(pa), nm_inter_over_pred=float(np.mean(pa)), nm_inter_over_gt=float(np.mean(ga)))
    return d


def cmd_gate(tags_):
    import v84_bundle as B
    V = B.V
    tags = tags_by_file()
    soft = pickle.load(open(ROOT / "outputs/v84/extract_soft.pkl", "rb"))
    res5 = soft[0.5]
    A5 = B.arrays(res5)
    pred = B.crossfit(A5)
    ref, _, lams = B.run_arm(res5, A5, pred, 1, 1)
    ffold = np.asarray([r["fold"] for r in res5])
    tr = A5["fold"] != 0
    model, lam = V.fit(A5["X"][tr], A5["y"][tr], A5["w"][tr]), lams[0]
    assert np.allclose(model.predict(A5["X"][~tr]), pred[0][~tr]), "demote refit not deterministic"
    ctrl = [r for r in res5 if r["fold"] == 0]
    names = [r["name"] for r in ctrl]
    arms = {"v84_soft": ctrl}
    for t in tags_:
        arms[t] = pickle.load(open(OUT / f"extract_{t}.pkl", "rb"))
        assert [r["name"] for r in arms[t]] == names, t
    C, S = {}, {}
    for a, res in arms.items():
        A = B.arrays(res)
        keep = (A["sc"] >= CUT) & (A["area"] >= MINA)
        full = keep & (model.predict(A["X"]) > lam / 2) if len(A["X"]) else keep
        C[a] = {s: {"t050_c030": subset_counts(res, keep, A["fi"], tags, ok),
                    "full_v84": subset_counts(res, full, A["fi"], tags, ok)} for s, ok in SUBSETS.items()}
        S[a] = {s: band_stats(res, keep, A["fi"], tags, ok, nm=True) for s, ok in SUBSETS.items()}
    repro = float(np.abs(C["v84_soft"]["all"]["full_v84"] - ref[ffold == 0]).max())
    rep = {"files": len(names), "lambda_fold0": lam, "control_repro_max_file_diff_vs_v84_bundle": repro,
           "records": {s: int(sum(ok(t) for n in names for t in tags[n])) for s, ok in SUBSETS.items()},
           "gate": GATE, "primary": [PRIMARY_CELL, PRIMARY_SUBSET], "arms": {}, "verdict": {}}
    assert repro < 1e-9, repro
    for a in arms:
        r = rep["arms"][a] = {}
        for s in SUBSETS:
            r[s] = {c: V.summ(C[a][s][c]) for c in C[a][s]}
            r[s]["bands"] = S[a][s]
            if a != "v84_soft":
                for c in C[a][s]:
                    r[s][f"d_{c}"] = V.boot(C["v84_soft"][s][c], C[a][s][c], seed=97)
        if a != "v84_soft":
            d = r[PRIMARY_SUBSET][f"d_{PRIMARY_CELL}"]
            rep["verdict"][a] = ("CONTINUE" if d["delta"] >= GATE["cont"] and d["p_gt_0"] >= GATE["p"]
                                 else "KILL" if d["delta"] < GATE["kill"] else "INCONCLUSIVE")
    for t in tags_:
        if t.endswith("@00"):
            rep["sanity_cond00_vs_soft_all"] = rep["arms"][t]["all"]["d_t050_c030"]
        if t.endswith("@oracle") or t.endswith("@00"):
            rep["verdict"][t] = "REPORTED"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "arms"}, indent=1, default=float))
    for a in arms:
        print(a, {s: round(rep["arms"][a][s]["t050_c030"]["pq"], 4) for s in SUBSETS},
              {s: round(rep["arms"][a][s].get("d_t050_c030", {}).get("delta", 0), 4) for s in SUBSETS})


# ----------------------------------------------------------------------------- self-check
def demo():
    """CPU check: targets, token sampling, FiLM zero-init == plain forward, near-miss geometry."""
    import torch
    from pycocotools import mask as mu
    a = np.zeros((8, 8), np.uint8, order="F"); a[2:6, 2:6] = 1
    b = np.zeros((8, 8), np.uint8, order="F"); b[2:7, 2:7] = 1
    ra, rb = mu.encode(a), mu.encode(b)
    r = {"target": ra, "gid": "01", "agree": [("01", ra), ("02", rb)], "R": 3, "Rg": 2}
    assert target_u8(r, "gt")[3, 3] == 255 and target_u8(r, "gt")[6, 6] == 0
    s = target_u8(r, "soft"); assert s[3, 3] == 170 and s[6, 6] == 85, s
    g = target_u8(r, "soft_g"); assert g[3, 3] == 128 and g[6, 6] == 0, g  # own group only: ra / 2 records
    ds = CropDataset([r], None, 1, 42, 256, "cond", "B")
    toks = [ds.sample(r, np.random.default_rng(i)) for i in range(200)]
    assert {t for t, _ in toks} == {0, 1} and all((t == 1) == (k == "gt") for t, k in toks)
    assert CropDataset([r], None, 1, 42, 256, "soft_g01", "B").sample(r, None) == (0, "soft_g")
    nm = nearmiss([ra], [rb])  # iou 16/25 = 0.64 -> not a near miss
    assert nm == []
    c = np.zeros((8, 8), np.uint8, order="F"); c[2:6, 4:8] = 1  # inter 8, union 24 -> 0.333
    nm = nearmiss([ra], [mu.encode(c)])
    assert len(nm) == 1 and abs(nm[0][0] - 1 / 3) < 1e-6 and nm[0][1] == 0.5 and nm[0][2] == 0.5, nm
    sys.path.insert(0, str(ROOT / "kaggle_kernel"))
    import importlib
    base = importlib.import_module("train_and_submit")
    m = cond_unet(base).eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        y0 = m(x, torch.tensor([0, 0])); y1 = m(x, torch.tensor([1, 3]))
    assert torch.allclose(y0, y1), "FiLM not zero-initialised"
    print("demo ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fold")
    f.add_argument("--arm", choices=("soft_g01", "cond"), required=True)
    f.add_argument("--fold", type=int, default=0)
    f.add_argument("--epochs-a", type=int, default=8, dest="epochs_a")
    f.add_argument("--epochs-b", type=int, default=12, dest="epochs_b")
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--limit", type=int, default=0)
    e = sub.add_parser("extract")
    e.add_argument("tag"); e.add_argument("ckpt")
    e.add_argument("--token", type=int, default=0); e.add_argument("--oracle", action="store_true")
    g = sub.add_parser("gate"); g.add_argument("tags", nargs="+")
    sub.add_parser("demo")
    args = ap.parse_args()
    if args.cmd == "fold":
        cmd_fold(args)
    elif args.cmd == "extract":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        ck = str(ROOT / "outputs/v84/soft_fold0_refiner.pt") if args.ckpt == "soft" else args.ckpt
        cmd_extract(args.tag, ck, args.token, args.oracle)
    elif args.cmd == "gate":
        cmd_gate(args.tags)
    else:
        demo()


if __name__ == "__main__":
    main()
