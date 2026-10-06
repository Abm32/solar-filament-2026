"""V54 step 3: re-score V52 TUNE proposals with the FIT-trained quality gate.

Runs the SAME V52 inference as the tune dump (so the proposal set is identical to
what V53 fused), computes the 10 gate features per proposal, applies the FIT-only
logistic gate (outputs/v54_gate/gate.json), and writes dumps in the V52 cache format
{file, threshold, rles, scores} but with scores = P(TP) from the gate instead of
mean-probability. v53_fusion.py can then consume this dir as an EXTRA_SET.

Firewall: the gate was trained on FIT only; this step merely APPLIES it to tune. We
read tune GT nowhere here. No test / CSV / submission.

  CUDA_VISIBLE_DEVICES=N python scripts/rescore_v54_tune.py \
      --weights outputs/v52_cldice_b48/weights/v52_cldice_best.pt \
      --annotations <ann.json> --images <train_images> --thresh 0.40 \
      --gate outputs/v54_gate/gate.json --out-dir /mnt/v52/cache/base32_gate \
      --shard N --shards 4
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import dump_v54_fit_features as feats  # noqa: E402
import fit_v54_gate as gatelib  # noqa: E402
import train_v52_cldice_semantic as v52  # noqa: E402


def apply_gate(gate, feature_rows):
    """feature_rows: list of [F] in gate feature order -> P(TP) via sigmoid(w.z+b)."""
    if not feature_rows:
        return []
    x = np.asarray(feature_rows, dtype=np.float64)
    mean = np.asarray(gate["mean"])
    std = np.asarray(gate["std"])
    z = (x - mean) / std
    scores = gatelib.sigmoid(z @ np.asarray(gate["weights"]) + float(gate["bias"]))
    return [float(s) for s in scores]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--thresh", type=float, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    if v52.sha256(args.annotations) != v52.ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")

    import cv2
    import torch
    from skimage import measure, morphology

    gate = json.loads(args.gate.read_text())
    if gate["feature_names"] != feats.FEATURE_NAMES:
        raise RuntimeError("gate feature order mismatch")
    if gate.get("reads_tune") is not False:
        raise RuntimeError("gate must be FIT-only (reads_tune=False)")

    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    unet = v52.build_unet(torch.nn)
    model = unet(base=ckpt.get("base", 32)).to(device)
    model.load_state_dict(ckpt["model"])
    res = ckpt.get("res", 1024)

    images = {str(r["id"]): r for r in document["images"]}
    annotations = v52.load_annotations(document)
    _fit, tune, by_file, _ = v52.canonical_split(document)
    recs = v52.records_for(sorted(tune), by_file, images, annotations)
    recs = recs[args.shard::args.shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from pycocotools import mask as mask_utils
    for n, (name, _image_id) in enumerate(recs, 1):
        gray = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
        prob_full = v52.predict_prob_full(torch, np, cv2, model, device,
                                          gray.astype(np.float32), res, tta=not args.no_tta)
        masks, _mp = v52.mask_to_instances(np, cv2, measure, prob_full, args.thresh)
        rows = [[feats.proposal_features(np, cv2, measure, morphology, m, prob_full)[k]
                 for k in feats.FEATURE_NAMES] for m in masks]
        gate_scores = apply_gate(gate, rows)
        rles = []
        for m in masks:
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            rles.append({"size": list(rle["size"]), "counts": rle["counts"].decode()})
        path = args.out_dir / (name + ".json.gz")
        with gzip.open(path, "wt") as stream:
            json.dump({"file": name, "threshold": args.thresh, "rles": rles,
                       "scores": gate_scores, "rescored_by": "v54_gate"}, stream)
        print(f"shard={args.shard} {n}/{len(recs)} {name} props={len(masks)}", flush=True)


if __name__ == "__main__":
    main()
