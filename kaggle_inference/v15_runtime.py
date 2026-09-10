"""GPU runtime for V15. Decision logic and constants live in v15_maskhead.py.

Split so the pure logic (grid, gates, AUC, frontier, cohort checks) stays importable
on a CPU-only box for tests, while everything needing CUDA lives here.

Order of operations, mirroring V12/V13/V14/E-A:
  1. Verify every pinned input (V11's load_runtime does this exhaustively).
  2. Run the FROZEN CONTROL, require E015's exact counts, and require its exported
     CSV to hash to the accepted 0.38 file. Abort otherwise.
  3. Build the mask cohort (byte-identical to V11/V14) and the SCORING cohort, which
     additionally contains unmatched proposals so the IoU head sees negatives.
  4. Train one arm with an ImageNet-initialised ResNet34.
  5. Evaluate the 42 predeclared candidates using V13's EXACT selection: E-A iou=0.3
     proposals, refined_constrained, threshold x confidence. The learned IoU score is
     reported as a keep-fraction frontier and NEVER selects.
  6. Export ONLY if SUBMIT_BAR passes. Never submit.
"""
import gzip
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

MODULE_SHA256 = None
CHECKPOINT_SHA256 = None
RAW_CACHE_SHA256 = None
SEED_BASE = 42


def modules():
    import sys
    return (sys.modules["optimize_inference"], sys.modules["detector_tta"],
            sys.modules["v10_quality"], sys.modules["v11_refiner"],
            sys.modules["v15_maskhead"])


def crop_inputs(gray, proposal_mask, bounds, size):
    """Grayscale crop replicated to 3 channels plus the proposal hint channel."""
    x0, y0, x1, y1 = bounds
    image = cv2.resize(gray[y0:y1, x0:x1], (size, size)).astype(np.float32) / 255.0
    hint = cv2.resize(proposal_mask[y0:y1, x0:x1], (size, size),
                      interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return np.concatenate((np.repeat(image[None], 3, axis=0), hint[None]), axis=0)


class MaskHeadDataset:
    """V11's cohort records and augmentation, plus the proposal hint channel."""

    def __init__(self, v11, records, train_dir, epoch, size):
        self.v11, self.records = v11, records
        self.train_dir, self.epoch, self.size = train_dir, epoch, size
        self.cache = {}

    def __len__(self):
        return len(self.records)

    def _gray(self, name):
        if name not in self.cache:
            gray = cv2.imread(str(self.train_dir / name), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (2048, 2048):
                raise ValueError(f"Cannot read training image {name}")
            if len(self.cache) > 4:
                self.cache.clear()
            self.cache[name] = gray
        return self.cache[name]

    def __getitem__(self, index):
        import torch
        record = self.records[index]
        gray = self._gray(record["file_name"])
        seed = mask_utils.decode(dict(size=record["seed"]["size"],
                                      counts=record["seed"]["counts"].encode()))
        # NEGATIVE records (V14 defect-2 fix) have no GT: the target is an empty mask,
        # so the mask head learns "nothing here" and the IoU head sees a true ~0. They
        # carry no `target` key, so nothing here can silently read GT that is absent.
        negative = record.get("negative", False)
        if negative:
            target = np.zeros_like(seed)
        else:
            target = mask_utils.decode(dict(size=record["target"]["size"],
                                            counts=record["target"]["counts"].encode()))
        # Proposal-centred bounds: at inference only the proposal exists, so training
        # on GT-centred crops would leak a luxury we do not have. E020 showed the two
        # centrings tie, so use the honest one.
        bounds = record["proposal_bounds"]
        rng = np.random.default_rng(SEED_BASE + self.epoch * 1_000_003 + index)
        jittered = self.v11.jitter_bounds(bounds, gray.shape, rng)
        x0, y0, x1, y1 = jittered
        if seed[y0:y1, x0:x1].any() and (negative or target[y0:y1, x0:x1].any()):
            bounds = jittered
        channels = crop_inputs(gray, seed, bounds, self.size)
        x0, y0, x1, y1 = bounds
        mask = cv2.resize(target[y0:y1, x0:x1], (self.size, self.size),
                          interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if not negative and not mask.any():
            raise ValueError(f"Training target vanished on resize: {record['file_name']}")
        if rng.random() < 0.5:
            channels, mask = channels[:, :, ::-1].copy(), mask[:, ::-1].copy()
        if rng.random() < 0.5:
            channels, mask = channels[:, ::-1].copy(), mask[::-1].copy()
        # mask_weight 0 on negatives: they supervise the IoU head ONLY, so the mask
        # objective stays exactly V11's on exactly V11's positive cohort and the
        # meanIoU comparison against V13 remains apples-to-apples.
        weight = 0.0 if negative else 1.0
        return (torch.from_numpy(channels), torch.from_numpy(mask[None]),
                torch.tensor(weight, dtype=torch.float32))


def train(v11, v15, runtime, cohort, output, identity, epochs, warm_start):
    """One arm, ImageNet-initialised. Budgets asserted per epoch."""
    import torch
    base = runtime["base"]
    base.seed_everything(SEED_BASE)
    model, transfer = v15.build_mask_head(base, warm_start)
    if transfer.get("pretrained") is not True:
        raise ValueError("V15 requires a verifiably ImageNet-initialised encoder")
    model = model.to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=v15.LEARNING_RATE,
                                  weight_decay=v15.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for epoch in range(1, epochs + 1):
        checkpoint = output / "checkpoints" / f"{v15.ARM}_e{epoch}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cuda:0", weights_only=True)
            if (saved["run_identity"] != identity
                    or saved["cohort_sha256"] != cohort["cohort_sha256"]
                    or saved["epoch"] != epoch):
                raise ValueError("Checkpoint cannot resume another experiment or cohort")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            history.append(json.loads(checkpoint.with_suffix(".json").read_text()))
            continue
        dataset = MaskHeadDataset(v11, cohort["records"], base.CFG.train_dir, epoch,
                                  v15.CROP_SIZE)
        order = v11.epoch_sample_order(cohort["records"], epoch)
        loader = torch.utils.data.DataLoader(dataset, batch_size=v15.BATCH_SIZE,
                                             sampler=order, num_workers=2,
                                             pin_memory=True, drop_last=False)
        model.train()
        mask_sum = iou_sum = 0.0
        samples = updates = positives = 0
        for channels, target, weight in loader:
            channels, target = channels.to("cuda:0"), target.to("cuda:0")
            weight = weight.to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            outputs, predicted_iou = model(channels)
            if outputs[0].dtype != torch.float32:
                raise ValueError("V15 trains in float32")
            # Mask loss on POSITIVES ONLY, so the mask objective stays byte-comparable
            # to V11/V13 and the meanIoU claim is apples-to-apples. Negatives exist to
            # train the IoU head, which is the V14 defect-2 fix.
            keep = weight > 0
            if keep.any():
                loss_mask = v15.deep_supervision_loss(
                    tuple(o[keep] for o in outputs), target[keep], base)
            else:
                loss_mask = outputs[0].sum() * 0.0
            actual_iou = v15.observed_iou(outputs[0], target)
            loss_iou = v15.iou_head_loss(predicted_iou, actual_iou)
            loss = loss_mask + loss_iou
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite epoch {epoch} loss")
            loss.backward()
            optimizer.step()
            mask_sum += float(loss_mask.detach()) * len(channels)
            iou_sum += float(loss_iou.detach()) * len(channels)
            samples += len(channels)
            positives += int(keep.sum())
            updates += 1
            if updates % 50 == 0:
                print(f"train epoch={epoch}/{epochs} {updates}/{len(loader)} "
                      f"mask={mask_sum/samples:.6f} iou={iou_sum/samples:.6f}", flush=True)
        if samples != len(cohort["records"]) or updates != math.ceil(samples / v15.BATCH_SIZE):
            raise ValueError("Arm violated the matched sample/update budget")
        training = dict(arm=v15.ARM, epoch=epoch, samples=samples, updates=updates,
                        positive_samples=positives,
                        negative_samples=samples - positives,
                        mean_mask_loss=mask_sum / samples, mean_iou_loss=iou_sum / samples,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        sample_order_sha256=v11.digest_json(order),
                        warm_start_transfer=transfer, run_identity=identity)
        scheduler.step()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        v15.write_json(checkpoint.with_suffix(".json"), training)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), epoch=epoch, arm=v15.ARM,
                        run_identity=identity,
                        cohort_sha256=cohort["cohort_sha256"]), checkpoint)
        history.append(training)
        v15.write_json(output / "training_history.json", history)
        print(json.dumps(dict(epoch=epoch, mask_loss=training["mean_mask_loss"],
                              iou_loss=training["mean_iou_loss"])), flush=True)
    return model, history, transfer


def crop_probabilities_and_scores(v15, model, gray, masks, size):
    """Refined probabilities plus the learned IoU scalar, in RAW PROPOSAL ORDER.

    Returning both in the same order is what makes the aligned diagnostic possible;
    V14's defect 3 was mixing this order with the sorted/filtered build order.
    """
    import torch
    base_bounds = [_square_bounds(mask) for mask in masks]
    probabilities, predicted = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(masks), 8):
            chunk = slice(start, start + 8)
            stack = np.stack([crop_inputs(gray, mask, bound, size)
                              for mask, bound in zip(masks[chunk], base_bounds[chunk])])
            channels = torch.from_numpy(stack).float().to("cuda:0")
            total, iou_total = None, 0.0
            # Four-view crop TTA, matching the accepted pipeline's averaging.
            for dims in ((), (2,), (3,), (2, 3)):
                flipped = channels.flip(dims) if dims else channels
                outputs, predicted_iou = model(flipped)
                logits = outputs[0].flip(dims) if dims else outputs[0]
                probability = logits.sigmoid()
                total = probability if total is None else total + probability
                # iou_head_loss trains this scalar directly against observed_iou in
                # [0,1] with plain MSE -- no sigmoid in that loss. Applying one here
                # would mismatch training and compress scores into [0.5,1).
                iou_total = iou_total + predicted_iou.clamp(0.0, 1.0)
            probabilities.extend((total[:, 0] / 4).cpu().numpy())
            predicted.extend((iou_total / 4).cpu().numpy())
    return base_bounds, probabilities, np.asarray(predicted, dtype=float)


_SQUARE_BOUNDS = {}


def _square_bounds(mask):
    """Delegates to the frozen base helper; bound once at runtime."""
    return _SQUARE_BOUNDS["fn"](mask)


def surviving_indices(core, masks, scores, bounds, probabilities, threshold, mode):
    """Replicate clean_refined_predictions' ordering and return SURVIVING INDICES.

    The frozen selection path still calls core.clean_refined_predictions so V13 parity
    is exact; this mirror exists only to recover which proposal produced which built
    instance, and main() asserts the two agree. That is the V14 defect-3 fix.
    """
    if not masks:
        return [], []
    occupied = np.zeros_like(masks[0], dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rles, indices = [], []
    for index in sorted(range(len(scores)), key=lambda i: -scores[i]):
        x0, y0, x1, y1 = bounds[index]
        proposal = masks[index][y0:y1, x0:x1]
        crop = cv2.resize(probabilities[index], (x1 - x0, y1 - y0)) >= threshold
        if mode == "refined_constrained":
            crop &= cv2.dilate(proposal, kernel) > 0
        if not crop.any():
            crop = proposal.astype(bool)
        crop &= ~occupied[y0:y1, x0:x1]
        if crop.sum() < core.MIN_AREA:
            continue
        occupied[y0:y1, x0:x1] |= crop
        rles.append(core.encode_crop(crop, bounds[index], occupied.shape))
        indices.append(index)
    return rles, indices


def evaluate_arm(runtime, core, v15, model, detector, cell, cache_root, thresholds,
                 confidences, arm_name, progress_every=20):
    """Canonical PQ over V13's grid, plus an ALIGNED learned-score diagnostic."""
    base, images, annotations = runtime["base"], runtime["images"], runtime["annotations"]
    held_out, tune, audit, _, by_file = runtime["split"]
    totals = {key: {group: np.zeros(4) for group in ("tune", "audit")}
              for key in (v15.candidate_key(t, c) for t in thresholds for c in confidences)}
    diagnostics, head_rows = [], []
    for index, name in enumerate(sorted(held_out), 1):
        views = arm_views(runtime, core, v15, detector, cell, cache_root, name, "train")
        proposals, scores = runtime["tta"].cluster_views(views, v15.POLICY, v15.MATCH_IOU)
        group = "tune" if name in tune else "audit"
        gt_by_image = {}
        for image_id in sorted(by_file[name]):
            info = images[image_id]
            entries = [core.annotation_rle(info, e) for e in annotations.get(image_id, [])]
            gt_by_image[image_id] = [r for r in entries if mask_utils.area(r) > 0]
        if not proposals:
            for key in totals:
                for image_id, gt in gt_by_image.items():
                    counts = core.pq_counts(np.zeros((len(gt), 0)), len(gt), 0)
                    totals[key][group] += counts
                    diagnostics.append(dict(arm=arm_name, candidate=key, file_name=name,
                                            image_id=image_id, group=group, target="coco",
                                            **dict(zip(v15.COUNTS, counts))))
            continue
        masks = [mask_utils.decode(rle) for rle in proposals]
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (2048, 2048):
            raise ValueError(f"Cannot read {name}")
        bounds, probabilities, predicted_iou = crop_probabilities_and_scores(
            v15, model, gray, masks, v15.CROP_SIZE)
        for threshold in thresholds:
            rles, kept_scores = core.clean_refined_predictions(
                masks, scores, bounds, probabilities, threshold, v15.MODE)
            # Aligned head evidence, at the control threshold only, on tune files.
            if group == "tune" and abs(threshold - 0.40) < 1e-9:
                mirrored, indices = surviving_indices(core, masks, scores, bounds,
                                                      probabilities, threshold, v15.MODE)
                if len(mirrored) != len(rles):
                    raise ValueError("Index mirror disagrees with the frozen refiner")
                every_gt = [r for rles_ in gt_by_image.values() for r in rles_]
                if mirrored and every_gt:
                    actual = mask_utils.iou(mirrored, every_gt,
                                            [0] * len(every_gt)).max(axis=1)
                    for position, proposal_index in enumerate(indices):
                        head_rows.append(dict(
                            file_name=name, proposal_index=int(proposal_index),
                            predicted_iou=float(predicted_iou[proposal_index]),
                            actual_iou=float(actual[position]),
                            detector_confidence=float(scores[proposal_index]),
                            is_match=bool(actual[position] > v15.MATCH_IOU)))
            for confidence in confidences:
                keep = (np.asarray(kept_scores) >= confidence if len(kept_scores)
                        else np.zeros(0, dtype=bool))
                kept = [rle for rle, flag in zip(rles, keep) if flag]
                key = v15.candidate_key(threshold, confidence)
                for image_id, gt in gt_by_image.items():
                    iou = (mask_utils.iou(kept, gt, [0] * len(gt)).T if kept and gt
                           else np.zeros((len(gt), len(kept))))
                    counts = core.pq_counts(iou, len(gt), len(kept))
                    totals[key][group] += counts
                    diagnostics.append(dict(arm=arm_name, candidate=key, file_name=name,
                                            image_id=image_id, group=group, target="coco",
                                            **dict(zip(v15.COUNTS, counts))))
        if index % progress_every == 0 or index == len(held_out):
            best = max(totals, key=lambda k: v15.pq(totals[k]["tune"]))
            axes = v15.axis_metrics(totals[best]["tune"])
            print(f"eval {arm_name} {index}/{len(held_out)} best={best} "
                  f"pq={axes['pq']:.6f} recall={axes['recall']:.4f} "
                  f"iou={axes['mean_matched_iou']:.4f} fp={axes['fp']}", flush=True)
    results = {key: {g: t.tolist() for g, t in groups.items()}
               for key, groups in totals.items()}
    return results, diagnostics, head_rows


def arm_views(runtime, core, v15, detector, cell, cache_root, name, split):
    """Frozen caches for the control; freshly generated signed caches for iou=0.3."""
    if cell["name"] == v15.CONTROL_CELL["name"]:
        return read_views(runtime["inventory"][(split, name)])
    signature = dict(cell=cell, split=split, file_name=name,
                     detector_sha256=runtime["detector_sha256"],
                     resolution=v15.DETECTOR_RESOLUTION,
                     confidence_floor=v15.DETECTOR_CONFIDENCE_FLOOR, views=4)
    path = Path(cache_root) / v15.cache_id(name, split, cell["name"])
    root = runtime["base"].CFG.train_dir if split == "train" else runtime["base"].CFG.test_dir
    return generate_views(runtime["tta"], core, v15, detector, root / name, cell,
                          path, signature)


def generate_views(tta, core, v15, detector, image_path, cell, cache_path, signature):
    """Four flip views at the cell's flags, signed and cached; identical to V13/E-A."""
    cache_path = Path(cache_path)
    marker = cache_path.with_suffix(".signature.json")
    if cache_path.exists():
        if not marker.exists():
            raise ValueError(f"Unsigned existing cache: {cache_path}")
        recorded = json.loads(marker.read_text())
        if recorded["signature"] != signature:
            raise ValueError(f"Stale cache signature: {cache_path}")
        if recorded["sha256"] != v15.sha256(cache_path):
            raise ValueError(f"Changed cache contents: {cache_path}")
        return read_views(cache_path)
    image = cv2.imread(str(image_path))
    if image is None or image.shape[:2] != (2048, 2048):
        raise ValueError(f"Cannot read or unexpected geometry: {image_path}")
    views = []
    for axes in ((), (1,), (0,), (0, 1)):
        result = detector.predict(
            str(image_path) if not axes else tta.flip_array(image, axes),
            imgsz=v15.DETECTOR_RESOLUTION, conf=v15.DETECTOR_CONFIDENCE_FLOOR,
            agnostic_nms=cell["agnostic_nms"], retina_masks=cell["retina_masks"],
            iou=cell["iou"], verbose=False)[0]
        rles, scores = [], []
        if result.masks is not None:
            scores = result.boxes.conf.detach().cpu().numpy().tolist()
            for mask in result.masks.data:
                restored = tta.flip_array(mask.cpu().numpy().astype(np.uint8), axes)
                if restored.shape != image.shape[:2]:
                    raise ValueError("Detector masks must be in image coordinates")
                rles.append(core.encode(restored))
        keep = [i for i, rle in enumerate(rles) if mask_utils.area(rle) > 0]
        views.append(([rles[i] for i in keep], [scores[i] for i in keep]))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt") as stream:
        json.dump([dict(rles=[dict(size=r["size"], counts=r["counts"].decode())
                              for r in rles], scores=s) for rles, s in views], stream)
    v15.write_json(marker, dict(signature=signature, sha256=v15.sha256(cache_path)))
    return read_views(cache_path)


def read_views(path):
    with gzip.open(path, "rt") as stream:
        raw = json.load(stream)
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"Incomplete four-view cache: {path}")
    views = []
    for view in raw:
        rles = [dict(size=r["size"], counts=r["counts"].encode()) for r in view["rles"]]
        scores = view["scores"]
        if len(rles) != len(scores) or any(not np.isfinite(s) or not 0 <= s <= 1
                                           for s in scores):
            raise ValueError(f"Invalid cache scores: {path}")
        if any(r["size"] != [2048, 2048] or mask_utils.area(r) == 0 for r in rles):
            raise ValueError(f"Invalid proposal geometry: {path}")
        views.append((rles, scores))
    return views


def evaluate_control(runtime, core, v11, v15, output):
    """Frozen V11/V13 path at the accepted settings; must reproduce E015 exactly."""
    base, images, annotations = runtime["base"], runtime["images"], runtime["annotations"]
    held_out, tune, _, _, by_file = runtime["split"]
    model = v11.build_model(base, runtime["refiner"])
    totals = {group: np.zeros(4) for group in ("tune", "audit")}
    for index, name in enumerate(sorted(held_out), 1):
        predictions = v11.predict_modes(runtime, model, base.CFG.train_dir / name,
                                        "train", False, [v15.MODE])
        kept = predictions[v15.MODE]
        group = "tune" if name in tune else "audit"
        for image_id in sorted(by_file[name]):
            info = images[image_id]
            gt = [core.annotation_rle(info, a) for a in annotations.get(image_id, [])]
            gt = [r for r in gt if mask_utils.area(r) > 0]
            iou = (mask_utils.iou(kept, gt, [0] * len(gt)).T if kept and gt
                   else np.zeros((len(gt), len(kept))))
            totals[group] += core.pq_counts(iou, len(gt), len(kept))
        if index % 40 == 0 or index == len(held_out):
            print(f"control {index}/{len(held_out)} "
                  f"tune_pq={v15.pq(totals['tune']):.6f}", flush=True)
    counts = {v15.CONTROL_REFERENCE["candidate"]:
              {group: total.tolist() for group, total in totals.items()}}
    verified = v15.verify_control(counts)
    print(json.dumps(dict(control_verified=True, counts=verified)), flush=True)
    del model
    import torch
    torch.cuda.empty_cache()
    return verified


def export(runtime, core, v15, model, detector, cell, cache_root, config, output, filename):
    """180-image test export with the same ownership guarantees as every prior run."""
    base = runtime["base"]
    rows, image_counts = [], {}
    for index, path in enumerate(sorted(base.CFG.test_dir.glob("*.jpeg")), 1):
        views = arm_views(runtime, core, v15, detector, cell, cache_root, path.name, "test")
        proposals, scores = runtime["tta"].cluster_views(views, v15.POLICY, v15.MATCH_IOU)
        predictions = []
        if proposals:
            masks = [mask_utils.decode(rle) for rle in proposals]
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (2048, 2048):
                raise ValueError(f"Cannot read {path}")
            bounds, probabilities, _ = crop_probabilities_and_scores(
                v15, model, gray, masks, v15.CROP_SIZE)
            rles, kept_scores = core.clean_refined_predictions(
                masks, scores, bounds, probabilities, config["threshold"], v15.MODE)
            predictions = [rle for rle, score in zip(rles, kept_scores)
                           if score >= config["confidence"]]
        image_counts[path.name] = len(predictions)
        occupied = np.zeros((2048, 2048), dtype=bool)
        for number, rle in enumerate(predictions, 1):
            mask = mask_utils.decode(rle).astype(bool)
            if mask.sum() < v15.MIN_AREA or (mask & occupied).any():
                raise ValueError("Invalid test prediction area or pixel ownership")
            occupied |= mask
            rows.append(dict(filament_id=f"{path.stem}_{number}",
                             segmentation_rle=rle["counts"].decode()))
        if index % 20 == 0 or index == 180:
            print(f"export {filename} {index}/180 rows={len(rows)}", flush=True)
    if len(image_counts) != 180:
        raise ValueError("Incomplete test coverage during export")
    destination = Path(output) / filename
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(
        destination, index=False)
    return dict(file=filename, rows=len(rows), sha256=v15.sha256(destination),
                config=config, evaluated_images=180,
                empty_images=[n for n, c in image_counts.items() if not c],
                structural_validation="all masks decoded, area>=20, ownership disjoint")


def build_scoring_cohort(runtime, core, v15, cohort, output, progress_every=50):
    """Negative training records: grouped TRAINING proposals that no GT matches.

    This is the real fix for V14 defect 2. V11's cohort is positive-matched only
    (min_iou=0.30), so a head trained on it never sees a bad proposal and collapses to
    a constant, which silently disables anything built on its score.

    Cost note: this reuses the proposal caches `build_training_cohort` just wrote, so
    it adds ZERO detector inference. It never touches runtime["split"]'s by_file, which
    covers held-out validation files only -- the bug that failed the first smoke.
    """
    base, images, annotations = runtime["base"], runtime["images"], runtime["annotations"]
    names = sorted(cohort["training_sources"])
    by_name = {}
    for image_id, info in images.items():
        if info["file_name"] in cohort["training_sources"]:
            by_name.setdefault(info["file_name"], []).append(image_id)
    records, rows = [], []
    for index, name in enumerate(sorted(names), 1):
        cache = Path(output) / cohort["training_sources"][name]["cache"]
        views = read_views(cache)
        proposals, scores = runtime["tta"].cluster_views(views, v15.POLICY)
        if not proposals:
            continue
        every_gt = []
        for image_id in sorted(by_name.get(name, []), key=str):
            info = images[image_id]
            every_gt.extend(r for r in (core.annotation_rle(info, a)
                                        for a in annotations.get(image_id, []))
                            if mask_utils.area(r) > 0)
        if not every_gt:
            continue
        actual = mask_utils.iou(proposals, every_gt, [0] * len(every_gt)).max(axis=1)
        for proposal_index, value in enumerate(actual):
            rows.append(dict(file_name=name, proposal_index=int(proposal_index),
                             actual_iou=float(value),
                             detector_confidence=float(scores[proposal_index])))
            if value >= v15.NEGATIVE_IOU_CEILING:
                continue
            proposal = proposals[proposal_index]
            records.append(dict(
                file_name=name, proposal_id=int(proposal_index), negative=True,
                actual_iou=float(value), confidence=float(scores[proposal_index]),
                proposal_bounds=list(base.square_bounds(mask_utils.decode(proposal))),
                seed=dict(size=proposal["size"], counts=proposal["counts"].decode())))
        if index % progress_every == 0 or index == len(names):
            print(f"scoring cohort {index}/{len(names)} proposals={len(rows)} "
                  f"negatives={len(records)}", flush=True)
    combined = [dict(r, negative=False) for r in cohort["records"]] + records
    health = v15.verify_scoring_cohort(
        [0.0 if r["negative"] else r["matched_iou"] for r in combined])
    health.update(positives=len(cohort["records"]), negatives=len(records),
                  measured_proposals=len(rows))
    print(json.dumps(dict(scoring_cohort=health)), flush=True)
    return combined, rows, health


def main(args):
    import torch
    core, tta, v10, v11, v15 = modules()
    v11.RAW_CACHE_SHA256 = RAW_CACHE_SHA256
    v11.MODULE_SHA256 = MODULE_SHA256
    v15.verify_grid()
    v15.verify_grid_matches_frozen_constants(tta)
    runtime = v11.load_runtime(args)
    runtime["tta"] = tta
    runtime["detector_sha256"] = v10.SOURCE_PINS["v6"]["detector"]
    if runtime["provenance"]["sha256"]["yolo_best.pt"] != runtime["detector_sha256"]:
        raise ValueError("V6 detector is not the pinned frozen detector")
    _SQUARE_BOUNDS["fn"] = runtime["base"].square_bounds
    args.output.mkdir(parents=True, exist_ok=True)

    warm_start = [p for p in Path(args.input_root).rglob("proposal_centered_e4.pt")
                  if v15.sha256(p) == CHECKPOINT_SHA256]
    if len(warm_start) != 1:
        raise ValueError("Expected exactly one pinned V11 proposal_centered_e4.pt")
    warm_start = warm_start[0]

    epochs = args.epochs
    thresholds, confidences = v15.THRESHOLDS, v15.CONFIDENCES
    cohort_names = sorted(runtime["provenance"]["training_files"])
    if args.limit:
        epochs = min(epochs, 2)
        cohort_names = cohort_names[:args.limit]
        thresholds, confidences = (0.40,), (0.30, 0.40)
    identity = v15.digest_json(dict(experiment="v15_maskhead", modules=MODULE_SHA256,
                                    checkpoint=CHECKPOINT_SHA256, cell=v15.CELL,
                                    raw_cache=v15.digest_json(RAW_CACHE_SHA256),
                                    epochs=epochs, limit=args.limit,
                                    thresholds=list(thresholds),
                                    confidences=list(confidences)))
    manifest_path = args.output / "v15_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("run_identity") != identity:
            raise ValueError("Output directory belongs to a different V15 run")
        if previous.get("complete"):
            print(json.dumps(dict(already_complete=True)), flush=True)
            return
    manifest = dict(experiment="v15_maskhead", run_identity=identity, arm=v15.ARM,
                    cell=v15.CELL, control_cell=v15.CONTROL_CELL,
                    v11_source_run=v15.V11_SOURCE_RUN,
                    v11_checkpoint_sha256=CHECKPOINT_SHA256,
                    detector_sha256=runtime["detector_sha256"],
                    control_reference=v15.CONTROL_REFERENCE,
                    accepted_control_csv_sha256=v15.ACCEPTED_CONTROL_CSV_SHA256,
                    baselines=v15.BASELINES, submit_bar=v15.SUBMIT_BAR,
                    promotion_gate=v15.PROMOTION_GATE,
                    declared_candidates=list(v15.declared_candidates()),
                    thresholds=list(thresholds), confidences=list(confidences),
                    policy=v15.POLICY, mode=v15.MODE, match_iou=v15.MATCH_IOU,
                    min_prediction_area=v15.MIN_AREA, crop_size=v15.CROP_SIZE,
                    epochs=epochs, batch_size=v15.BATCH_SIZE, seed=SEED_BASE,
                    learning_rate=v15.LEARNING_RATE, weight_decay=v15.WEIGHT_DECAY,
                    iou_loss_weight=v15.IOU_LOSS_WEIGHT,
                    keep_fractions=list(v15.KEEP_FRACTIONS),
                    min_negative_fraction=v15.MIN_NEGATIVE_FRACTION,
                    imagenet_init_required=True,
                    selection_note="V13's EXACT selection: E-A iou=0.3 proposals + "
                                   "threshold x confidence, refined_constrained. The "
                                   "learned IoU score NEVER selects; it is reported "
                                   "as a keep-fraction frontier only.",
                    focal_tversky_note=v15.focal_tversky_disabled_note(),
                    smoke_limit=args.limit, trains=True, submits=False,
                    versions=runtime["versions"], complete=False)
    v15.write_json(manifest_path, manifest)
    cache_root = args.output / "proposal_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # 1. Frozen control FIRST: E015 counts, and the accepted 0.38 CSV byte-for-byte.
    control_counts = evaluate_control(runtime, core, v11, v15, args.output)
    control_export = None
    if not args.limit:
        control_model = v11.build_model(runtime["base"], runtime["refiner"])
        # The control export MUST go through V11's frozen exporter, which is the code
        # path that produced the accepted 0.38 CSV. Using V15's own exporter here
        # would prove nothing about the accepted file.
        control_export = v11.export_candidate(
            runtime, control_model,
            dict(mode=v15.MODE, threshold=0.40, confidence=0.40, tta=True,
                 conditioned=False),
            args.output, "submission_control.csv")
        v10.verify_control_export(args.output / "submission_control.csv")
        if control_export["sha256"] != v15.ACCEPTED_CONTROL_CSV_SHA256:
            raise ValueError("Control CSV does not reproduce the accepted 0.38 file")
        print(json.dumps(dict(control_csv_verified=True, **control_export)), flush=True)
        del control_model
        torch.cuda.empty_cache()
    manifest.update(control_counts=control_counts, control_verified=True,
                    control_export=control_export)
    v15.write_json(manifest_path, manifest)

    # 2. Mask cohort, byte-identical to V11/V14 so the objective stays comparable.
    cohort = v11.build_training_cohort(runtime, cohort_names, args.output, identity,
                                       sample_limit=0)
    print(json.dumps(dict(cohort_samples=cohort["matched_samples"],
                          cohort_sha256=cohort["cohort_sha256"])), flush=True)

    # 3. Add NEGATIVES for the scoring head and check cohort health BEFORE training,
    #    so a positives-only population aborts instead of shipping a constant head.
    #    Reuses the caches step 2 just wrote, so it costs no detector inference.
    combined, scoring_rows, scoring_health = build_scoring_cohort(
        runtime, core, v15, cohort, args.output)
    pd.DataFrame(scoring_rows).to_csv(args.output / "v15_scoring_cohort.csv", index=False)
    training_cohort = dict(cohort, records=combined,
                           cohort_sha256=v15.digest_json(
                               [r.get("actual_iou", r.get("matched_iou")) for r in combined]),
                           mask_cohort_sha256=cohort["cohort_sha256"])
    manifest.update(scoring_cohort=scoring_health,
                    mask_cohort_sha256=cohort["cohort_sha256"],
                    training_cohort_sha256=training_cohort["cohort_sha256"])
    v15.write_json(manifest_path, manifest)

    # 4. Train one arm with a verifiably ImageNet-initialised encoder.
    detector = runtime["base"].YOLO(str(runtime["source_root"] / "yolo_best.pt"))
    model, history, transfer = train(v11, v15, runtime, training_cohort, args.output,
                                     identity, epochs, warm_start)
    manifest.update(warm_start_transfer=transfer, training_epochs=len(history))
    v15.write_json(manifest_path, manifest)

    # 5. Evaluate the predeclared grid with V13's exact selection.
    results, diagnostics, head_rows = evaluate_arm(
        runtime, core, v15, model, detector, v15.CELL, cache_root, thresholds,
        confidences, v15.ARM)
    pd.DataFrame(diagnostics).to_csv(args.output / "v15_image_diagnostics.csv", index=False)
    head = pd.DataFrame(head_rows)
    head.to_csv(args.output / "v15_iou_head.csv", index=False)
    rows = []
    for key, value in results.items():
        axes, audit_axes = v15.axis_metrics(value["tune"]), v15.axis_metrics(value["audit"])
        rows.append(dict(arm=v15.ARM, candidate=key, tune_pq=axes["pq"],
                         audit_pq=audit_axes["pq"], tune_recall=axes["recall"],
                         tune_mean_matched_iou=axes["mean_matched_iou"],
                         tune_tp=axes["tp"], tune_fp=axes["fp"], tune_fn=axes["fn"],
                         audit_recall=audit_axes["recall"],
                         audit_mean_matched_iou=audit_axes["mean_matched_iou"]))
    frame = pd.DataFrame(rows).sort_values("tune_pq", ascending=False)
    frame.to_csv(args.output / "v15_candidates.csv", index=False)

    selected = v15.select_candidate(results)
    report = v15.evaluate_bars(selected, results)
    report["control_counts"] = control_counts
    report["control_export"] = control_export
    report["warm_start_transfer"] = transfer
    report["scoring_cohort"] = scoring_health
    if len(head):
        alignment = v15.verify_head_alignment(head.predicted_iou, head.actual_iou)
        n_gt = v15.CONTROL_REFERENCE["tune"][1] + v15.CONTROL_REFERENCE["tune"][3]
        report["iou_head"] = dict(
            alignment=alignment,
            consensus_score_auc_baseline=v15.CONSENSUS_SCORE_AUC,
            calibration_error=(None if math.isnan(
                v15.calibration_error(head.predicted_iou, head.actual_iou))
                else v15.calibration_error(head.predicted_iou, head.actual_iou)),
            predicted_min=float(head.predicted_iou.min()),
            predicted_max=float(head.predicted_iou.max()),
            predicted_std=float(head.predicted_iou.std()),
            confidence_auc=v15.ranking_auc(head.detector_confidence, head.is_match),
            keep_fraction_frontier=v15.keep_fraction_frontier(
                head.predicted_iou, head.actual_iou, n_gt),
            note="Diagnostic only. Selection is V13's confidence grid.")
    v15.write_json(args.output / "v15_report.json", report)
    v15.write_json(args.output / "v15_results.json",
                   dict(control={v15.CONTROL: control_counts}, arm=results))
    print(json.dumps(report, indent=2, default=str), flush=True)

    if args.limit:
        manifest.update(complete=True, smoke=True, report=report)
        v15.write_json(manifest_path, manifest)
        print("Smoke finished; no export, no gate decision", flush=True)
        return

    # 6. Export only if SUBMIT_BAR authorises measurement.
    candidate_export = None
    if report["export_authorized"]:
        threshold, confidence = (float(part[1:]) for part in selected.split("_"))
        candidate_export = export(runtime, core, v15, model, detector, v15.CELL,
                                  cache_root, dict(threshold=threshold,
                                                   confidence=confidence),
                                  args.output, "submission.csv")
        if control_export and candidate_export["sha256"] == control_export["sha256"]:
            raise ValueError("Candidate export is byte-identical to the control")
        candidate_export.update(candidate=selected, tune_pq=report["selected_tune_pq"],
                                audit_pq=report["selected_audit_pq"],
                                promotable=report["promotable"],
                                requires_independent_csv_validation=True)
        v15.write_json(args.output / "submission_candidates.json",
                       [dict(control_export or {}, candidate=v15.CONTROL,
                             control_verified=True), candidate_export])
    else:
        print("SUBMIT_BAR not cleared; no CSV exported by design", flush=True)
    del model
    torch.cuda.empty_cache()
    manifest.update(complete=True, selected=selected, report=report,
                    candidate_export=candidate_export)
    v15.write_json(manifest_path, manifest)
    print(json.dumps(dict(selected=selected,
                          export_authorized=report["export_authorized"],
                          promotable=report["promotable"],
                          rows=(candidate_export or {}).get("rows"),
                          sha256=(candidate_export or {}).get("sha256"),
                          complete=True), indent=2), flush=True)
