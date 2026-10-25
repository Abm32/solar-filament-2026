"""V91 S0 probe: decode the sanity model's in-sample predictions under several thresholds (engineering check)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v91_spine as V  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "f0_sanity"
net = V.build_model(ssl=False).cuda()
net.load_state_dict(torch.load(V.OUT / tag / "model.pt", weights_only=True))
net.eval()
held = set(V.folds()[0])
idx = [i for i, n in enumerate(V.NAMES) if n not in held][:8]
G = pickle.load(open(V.DATA / "gt.pkl", "rb"))["gt"]
P = {i: V.predict(net, i) for i in idx}
for tf, ts in [(0.5, 0.5), (0.5, 0.3), (0.5, 0.2), (0.4, 0.3)]:
    S, nseed = [], 0
    for i in idx:
        lab, f = V.decode(P[i][0] / 255.0, P[i][1] / 255.0, tf, ts)
        nseed += int((f[:, 4] > 0).sum())
        S.append(V.stats([(len(r), [V.hits(lab, len(f), V.gt_masks(r))]) for r in G[i]], [np.ones(len(f), bool)]))
        if tf == 0.5 and ts == 0.3:
            S2 = V.stats([(len(r), [V.hits(lab, len(f), V.gt_masks(r))]) for r in G[i]], [f[:, 4] > 0])
            print("seeded_only", V.pq([S2]), S2[1:].tolist())
    print(json.dumps({"tf": tf, "ts": ts, "pq": V.pq(S), "tp_fp_fn": np.sum(S, 0)[1:].tolist(), "nseed": nseed,
                      "gt": sum(len(r) for i in idx for r in G[i]),
                      "sp_max": [int(P[i][1].max()) for i in idx]}))
