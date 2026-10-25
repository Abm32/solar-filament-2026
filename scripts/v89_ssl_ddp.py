"""V89 SSL pretraining, multi-GPU (DDP) version of scripts/v89_ssl.py `train` (user request: combine all GPUs).

Same arms, data stream, global batch, lr schedule and steps as the single-GPU run; only the work is split:
  - each rank takes a contiguous quarter of every global batch (same sample indices as single-GPU);
  - mim: SyncBatchNorm so BN statistics are over the full global batch (as single-GPU);
  - moco: query encoder SyncBN (full-batch stats, as single-GPU); key encoder keeps plain BN and the
    shuffled-chunk key BN is reproduced across GPUs: keys are all-gathered, permuted with one shared perm,
    rank r encodes chunk r (single-GPU used 4 chunks too), keys all-gathered back and un-permuted;
    all ranks enqueue the full global key batch, so queues stay identical.
  torchrun --nproc_per_node 4 scripts/v89_ssl_ddp.py --arm mim --steps 40000 --batch 64
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v89_ssl import DS, ROOT, encode, load_base  # noqa: E402

F = torch.nn.functional
ENC = ("stem", "layer1", "layer2", "layer3", "layer4")


class Query(torch.nn.Module):
    def __init__(self, net, proj):
        super().__init__()
        self.net, self.proj = net, proj

    def forward(self, x):
        return self.proj(encode(self.net, x))


def gather(t):
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.cat(out)


def main(a):
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = "cuda"
    assert a.batch % world == 0
    b = a.batch // world
    torch.manual_seed(a.seed)
    out = ROOT / "outputs/v89" / f"ssl_{a.arm}"
    out.mkdir(parents=True, exist_ok=True)
    net = load_base().CropUNet(pretrained=True).to(dev)
    enc_p = [p for n, p in net.named_parameters() if n.split(".")[0] in ENC]
    dec_p = [p for n, p in net.named_parameters() if n.split(".")[0] not in ENC]
    if a.arm == "mim":
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        enc_p = [p for n, p in net.named_parameters() if n.split(".")[0] in ENC]
        dec_p = [p for n, p in net.named_parameters() if n.split(".")[0] not in ENC]
        model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[torch.cuda.current_device()])
        opt = torch.optim.AdamW([{"params": enc_p, "lr": 1e-4}, {"params": dec_p, "lr": 1e-3}], weight_decay=0.05)
    else:
        proj = torch.nn.Sequential(torch.nn.Linear(512, 2048), torch.nn.ReLU(inplace=True),
                                   torch.nn.Linear(2048, 128)).to(dev)
        knet, kproj = copy.deepcopy(net), copy.deepcopy(proj)  # plain BN (shuffled-chunk key BN)
        for p in list(knet.parameters()) + list(kproj.parameters()):
            p.requires_grad = False
        for p in dec_p:
            p.requires_grad = False  # decoder unused by MoCo (single-GPU run never updated it either)
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        enc_p = [p for n, p in net.named_parameters() if n.split(".")[0] in ENC]
        model = torch.nn.parallel.DistributedDataParallel(Query(net, proj), device_ids=[torch.cuda.current_device()])
        queue = F.normalize(torch.randn(16384, 128, device=dev), dim=1)
        qptr = 0
        opt = torch.optim.SGD(enc_p + list(proj.parameters()), lr=0.03, momentum=0.9, weight_decay=1e-4)
    base_lrs = [g["lr"] for g in opt.param_groups]
    idx = [s * a.batch + rank * b + j for s in range(a.steps) for j in range(b)]  # rank's quarter of each batch
    ds = torch.utils.data.Subset(DS(a.arm, a.seed, a.steps * a.batch), idx)
    loader = torch.utils.data.DataLoader(ds, batch_size=b, shuffle=False, num_workers=a.workers, pin_memory=True,
                                         drop_last=True, persistent_workers=True, prefetch_factor=4)
    scaler = torch.cuda.amp.GradScaler()
    log = open(out / "train.log", "a") if rank == 0 else None
    t0, run = time.monotonic(), []
    gperm = torch.Generator(device=dev).manual_seed(a.seed)
    for step, batch in enumerate(loader, 1):
        f = min(1.0, step / 1000) * 0.5 * (1 + np.cos(np.pi * step / a.steps))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f
        with torch.autocast("cuda", dtype=torch.float16):
            if a.arm == "mim":
                x, m = (t.to(dev, non_blocking=True) for t in batch)
                mu = x.mean((2, 3), keepdim=True)
                sd = x.std((2, 3), keepdim=True) + 0.02
                pred = model((x * (1 - m) + mu * m).repeat(1, 3, 1, 1)).float()
                loss = ((pred - (x - mu) / sd).abs() * m).sum() / m.sum()  # m.sum() equal on every rank
            else:
                x1, x2 = (t.to(dev, non_blocking=True) for t in batch)
                with torch.no_grad():
                    for pq_, pk in zip(list(net.parameters()) + list(proj.parameters()),
                                       list(knet.parameters()) + list(kproj.parameters())):
                        pk.mul_(0.99).add_(pq_.detach(), alpha=0.01)
                    allx = gather(x2)
                    perm = torch.randperm(len(allx), device=dev, generator=gperm)
                    dist.broadcast(perm, 0)
                    kc = kproj(encode(knet, allx[perm].chunk(world)[rank]))
                    k_all = F.normalize(gather(kc.float()), dim=1)[torch.argsort(perm)]
                    k = k_all[rank * b:(rank + 1) * b]
                q = F.normalize(model(x1).float(), dim=1)
                logits = torch.cat([(q * k).sum(1, keepdim=True), q @ queue.T], 1) / 0.2
                loss = F.cross_entropy(logits, torch.zeros(len(q), dtype=torch.long, device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if a.arm == "moco":
            with torch.no_grad():
                queue[qptr:qptr + len(k_all)] = k_all
                qptr = (qptr + len(k_all)) % len(queue)
        lt = loss.detach().float()
        dist.all_reduce(lt)
        run.append(float(lt) / world)
        if not np.isfinite(run[-1]):
            raise RuntimeError(f"nonfinite loss at step {step}")
        if rank == 0 and (step % 200 == 0 or step == a.steps):
            print(json.dumps({"step": step, "loss": float(np.mean(run)), "lr_f": f, "gpus": world,
                              "s": round(time.monotonic() - t0)}), file=log, flush=True)
            run = []
        if rank == 0 and (step % 5000 == 0 or step == a.steps):
            torch.save(net.state_dict(), out / "ssl_state.pt")  # SyncBN keeps BN param/buffer names
    if rank == 0:
        (out / "DONE").write_text(json.dumps({"steps": a.steps, "batch": a.batch, "gpus": world,
                                              "seconds": time.monotonic() - t0}))
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("mim", "moco"), required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=89)
    main(ap.parse_args())
