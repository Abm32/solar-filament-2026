"""V90 fold-0 screen: probability-averaged ensemble of fold-0 soft refiners (independent seeds / inits).

Pre-registration: docs/v90_refiner_ensemble_preregistration_2026-10-12.md. Reuses scripts/v89_screen.py: the
wrapper returns logit(mean_i sigmoid(m_i(x))), so optimize_inference.crop_probabilities (which applies .sigmoid())
sees the mean probability of the members.

  CUDA_VISIBLE_DEVICES= python scripts/v90_refiner_ens.py extract TAG CKPT [CKPT ...]  # -> outputs/v89/extract_TAG.pkl
  then: python scripts/v89_screen.py gate TAG ...
"""
from __future__ import annotations

import os
import pickle
import sys
from multiprocessing import Pool, set_start_method
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v89_screen as S  # noqa: E402


def main(tag, ckpts):
    import torch

    class Ens(torch.nn.Module):
        def __init__(self, ms):
            super().__init__()
            self.ms = torch.nn.ModuleList(ms)

        def forward(self, x):
            p = torch.stack([m(x).sigmoid() for m in self.ms]).mean(0)
            return torch.logit(p.clamp(1e-6, 1 - 1e-6))

    set_start_method("fork")
    S.V73.DEV = "cpu"
    S.E.RT = S.V73.runtime()
    ms = []
    for c in ckpts:
        m = S.V73.build_model(S.E.RT["base"], "imagenet", "resnet18")
        m.load_state_dict(torch.load(c, map_location="cpu", weights_only=True)["model"])
        ms.append(m.eval())
    S.E.MODELS[0] = Ens(ms).eval()
    fmap = S.V73.R.fold_of_file()
    names = sorted(n for n, k in fmap.items() if k == 0)
    assert len(names) == 141
    with torch.no_grad(), Pool(int(os.environ.get("PROCS", "22")), initializer=S.E.init) as p:
        res = p.map(S.X84.work, [(n, 0) for n in names], chunksize=1)
    pickle.dump({t: [r[t] for r in res] for t in S.X84.TS}, open(S.OUT / f"extract_{tag}.pkl", "wb"))


if __name__ == "__main__":
    assert sys.argv[1] == "extract" and len(sys.argv) >= 4
    main(sys.argv[2], sys.argv[3:])
