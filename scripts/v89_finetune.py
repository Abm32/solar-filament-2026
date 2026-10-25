"""V89: the exact V84 soft refiner recipe (train_v82_refiner_targets.py --arm soft = V73 stage A 8 ep 3e-4 on GT
crops + stage B 12 ep 1e-4 on p2first OOF proposals, seed 42) from an SSL init instead of plain ImageNet.

  --init PATH   v89_ssl state (CropUNet state_dict); --init-kind full (all but the 1x1 head) | encoder (stem+layer1-4)
  CUDA_VISIBLE_DEVICES=0 python scripts/v89_finetune.py --init outputs/v89/ssl_mim/ssl_state.pt --init-kind full \
      fold --arm soft --fold 0 --out outputs/v89/ft_mim
Without --init this is train_v82_refiner_targets.py unchanged (used for the seed-noise control).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pop(flag):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    v = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return v


INIT, KIND = pop("--init"), pop("--init-kind") or "full"
import train_v82_refiner_targets as V82  # noqa: E402  (reads --arm from argv at import)

V73 = V82.V73
ENC = ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")


def select(sd, kind):
    if kind == "encoder":
        return {k: v for k, v in sd.items() if k.startswith(ENC)}
    return {k: v for k, v in sd.items() if not k.startswith("head.")}


def build_model(base, init, encoder, _orig=V73.build_model):
    import torch
    model = _orig(base, init, encoder)
    if INIT:
        assert init == "imagenet" and encoder == "resnet18"
        sd = select(torch.load(INIT, map_location="cpu", weights_only=True), KIND)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        exp = {k for k in model.state_dict() if k not in sd and not k.endswith("num_batches_tracked")}
        assert set(missing) == exp, (missing, exp)
        print(f"V89 init {INIT} kind={KIND}: loaded {len(sd)} tensors, kept fresh {len(missing)}", flush=True)
    return model


if __name__ == "__main__":
    V73.V63.fit_records, V73.CropDataset, V73.train = V82.fit_records, V82.CropDataset, V82.train
    V73.build_model = build_model
    V73.main()
