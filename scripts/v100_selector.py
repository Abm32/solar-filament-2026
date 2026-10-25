"""V100 arm A: image-based candidate selector (ResNet18 on 5-channel 128 px crops + 28 tabular features), cross-fitted
on the V61 folds, scored through the V84 path (cut 0.30, MIN200, yhat > lambda/2 with the inner Dinkelbach lambda).
Prereg: docs/v100_preregistration_2026-10-25.md. Train images only; never reads test.

  CUDA_VISIBLE_DEVICES=1 python scripts/v100_selector.py train --outer 0 1 2 --seed 100   # outer k + its 4 inner fits
  CUDA_VISIBLE_DEVICES=2 python scripts/v100_selector.py train --outer 3 4 --seed 100
  python scripts/v100_selector.py gate [--seeds 100 101]                                   # -> outputs/v100/gate.json
Each `train` writes outputs/v100/pred_s{seed}_o{k}.npz: yhat (outer, fold-k rows) and inner_j for j != k (fold-j rows
from a model trained on folds not in {k, j}), i.e. exactly the arrays v84_bundle.run_arm consumes.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "outputs/v70/pylib"))
import v83_rule as V  # noqa: E402
import v84_bundle as B  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v100"
EPOCHS, BATCH, LR = 12, 64, 3e-4


def load():
    res = pickle.load(open(OUT / "extract.pkl", "rb"))
    A = B.arrays(res)
    crops = np.concatenate([r["crops"] for r in res if len(r["rows"])])
    assert len(crops) == len(A["y"]), (len(crops), len(A["y"]))
    return res, A, crops


def tab_norm(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


class Net:
    def __init__(self, n_tab, device):
        import torch
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        w = m.conv1.weight.data  # (64,3,7,7)
        conv = torch.nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        conv.weight.data = w.mean(1, keepdim=True).repeat(1, 5, 1, 1) * (3 / 5)
        m.conv1 = conv
        m.fc = torch.nn.Identity()
        self.body = m
        self.tab = torch.nn.Sequential(torch.nn.Linear(n_tab, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(512 + 64, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 1))
        self.mods = torch.nn.ModuleList([self.body, self.tab, self.head]).to(device)

    def __call__(self, img, tab):
        import torch
        return self.head(torch.cat([self.body(img), self.tab(tab)], 1))[:, 0]


def d4(img, rng):
    k, f = int(rng.integers(4)), bool(rng.integers(2))
    x = np.rot90(img, k, axes=(1, 2))
    return np.ascontiguousarray(np.flip(x, 2) if f else x)


def fit_predict(crops, Xt, y, w, tr, seed, device, log):
    """Train on rows `tr`, return predictions (sigmoid) for all rows."""
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(Xt.shape[1], device)
    params = [p for m in net.mods for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    idx = np.where(tr)[0]
    steps = EPOCHS * int(np.ceil(len(idx) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)
    rng = np.random.default_rng(seed)
    yt, wt = torch.tensor(y, dtype=torch.float32), torch.tensor(w / w.mean(), dtype=torch.float32)
    for ep in range(EPOCHS):
        net.mods.train()
        order = rng.permutation(idx)
        tot = 0.0
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            img = torch.tensor(np.stack([d4(crops[i], rng) for i in b]), dtype=torch.float32, device=device) / 255
            tab = torch.tensor(Xt[b], device=device)
            logit = net(img, tab)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, yt[b].to(device), reduction="none")
                    * wt[b].to(device)).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(b)
        log(f"epoch {ep + 1} loss {tot / len(idx):.4f}")
    net.mods.eval()
    out = np.zeros(len(y), np.float32)
    with torch.no_grad():
        for s in range(0, len(y), 256):
            img = torch.tensor(crops[s:s + 256], dtype=torch.float32, device=device) / 255
            tab = torch.tensor(Xt[s:s + 256], device=device)
            p = torch.sigmoid(net(img, tab))  # 4-flip TTA
            for dims in ((2,), (3,), (2, 3)):
                p += torch.sigmoid(net(img.flip(dims), tab))
            out[s:s + 256] = (p / 4).cpu().numpy()
    return out


def cmd_train(args):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res, A, crops = load()
    fold, X, y, w = A["fold"], A["X"], A["y"], A["w"]
    for k in args.outer:
        logf = open(OUT / f"train_s{args.seed}_o{k}.log", "a")
        log = lambda s: (logf.write(s + "\n"), logf.flush())  # noqa: E731
        tr = fold != k
        yhat = fit_predict(crops, tab_norm(X, tr), y, w, tr, args.seed * 10 + k, device, lambda s: log(f"outer{k} {s}"))
        inner = {}
        for j in range(5):
            if j == k:
                continue
            t2 = (fold != k) & (fold != j)
            inner[j] = fit_predict(crops, tab_norm(X, t2), y, w, t2, args.seed * 100 + k * 10 + j, device,
                                   lambda s: log(f"outer{k} inner{j} {s}"))
        np.savez(OUT / f"pred_s{args.seed}_o{k}.npz", yhat=yhat, **{f"inner_{j}": v for j, v in inner.items()})
        log(f"outer{k} done auc_outer {auc(yhat[fold == k], y[fold == k] > 0):.4f}")


def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def assemble(seeds, A):
    """(yhat, inner) in v84_bundle.crossfit format, averaged over seeds."""
    fold = A["fold"]
    yhat, inner = np.zeros(len(fold)), {k: np.zeros(len(fold)) for k in range(5)}
    for seed in seeds:
        for k in range(5):
            z = np.load(OUT / f"pred_s{seed}_o{k}.npz")
            yhat[fold == k] += z["yhat"][fold == k] / len(seeds)
            for j in range(5):
                if j != k:
                    inner[k][fold == j] += z[f"inner_{j}"][fold == j] / len(seeds)
    return yhat, inner


def cmd_gate(args):
    res, A, _ = load()
    y01 = A["y"] > 0
    lgb_pred = B.crossfit(A)
    cnn_pred = assemble(args.seeds, A)
    mix_pred = (0.5 * (lgb_pred[0] + cnn_pred[0]), {k: 0.5 * (lgb_pred[1][k] + cnn_pred[1][k]) for k in range(5)})
    arms = {"v84": lgb_pred, "cnn": cnn_pred, "mix": mix_pred}
    counts, rep = {}, {"files": len(res), "seeds": args.seeds, "arms": {}}
    base_counts = V.file_counts(res, A["sc"] >= B.CUT, A["fi"])
    rep["cut030_no_demote"] = V.summ(base_counts)
    for a, pred in arms.items():
        c, keep, lams = B.run_arm(res, A, pred, 1, 1)
        counts[a] = c
        m = A["sc"] >= B.CUT
        rep["arms"][a] = dict(V.summ(c), lambdas=[float(v) for v in lams], kept=int(keep.sum()),
                              auc_all=auc(pred[0], y01), auc_cut=auc(pred[0][m], y01[m]))
    assert abs(rep["arms"]["v84"]["pq"] - 0.458975) < 2e-4, rep["arms"]["v84"]["pq"]
    # secondary: promote cell (eligible = all candidates >= FLOOR, keep iff yhat > lam/2, lam from the demote arm)
    for a in ("cnn", "mix"):
        pred = arms[a]
        keep = np.zeros(len(A["y"]), bool)
        for k, lam in enumerate(rep["arms"][a]["lambdas"]):
            te = A["fold"] == k
            keep[te] = (A["area"][te] >= B.MINA) & (pred[0][te] > lam / 2)
        rep["arms"][f"{a}_promote"] = V.summ(V.file_counts(res, keep, A["fi"]))
    for a in ("cnn", "mix"):
        rep["arms"][a]["d_vs_v84"] = V.boot(counts["v84"], counts[a], seed=100)
        d = rep["arms"][a]["d_vs_v84"]
        rep["arms"][a]["verdict"] = ("CONTINUE" if d["delta"] >= 0.008 and d["p_gt_0"] >= 0.95
                                     else "INCONCLUSIVE" if d["delta"] >= 0.003 else "KILL")
    (OUT / "gate.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    for a, r in rep["arms"].items():
        line = f"{a:12s} pq={r['pq']:.4f} tp/fp/fn={r['tp']}/{r['fp']}/{r['fn']}"
        if "auc_cut" in r:
            line += f" auc_cut={r['auc_cut']:.3f}"
        if "d_vs_v84" in r:
            d = r["d_vs_v84"]
            line += f" d={d['delta']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] P={d['p_gt_0']:.2f} {r['verdict']}"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--outer", type=int, nargs="+", required=True); t.add_argument("--seed", type=int, default=100)
    g = sub.add_parser("gate"); g.add_argument("--seeds", type=int, nargs="+", default=[100])
    args = ap.parse_args()
    {"train": cmd_train, "gate": cmd_gate}[args.cmd](args)
