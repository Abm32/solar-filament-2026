"""V14 runtime: training, dual-axis evaluation, and gated export.

Split from v14_dualhead.py so the pure decision logic (grid, losses, gates, AUC)
stays importable on a CPU-only box for tests, while everything here needs a GPU.

Order of operations is deliberate and mirrors V12/V13/E-A:
  1. Verify every pinned input (V11's load_runtime does this exhaustively).
  2. Run the FROZEN CONTROL and require E015's exact counts. Abort otherwise.
  3. Train both arms on an identical cohort and sample order, asserted.
  4. Evaluate the 21 predeclared candidates per arm. Tune-PQ-only selection.
  5. Export ONLY if the predeclared SUBMIT_BAR passes. Never submit.
"""
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


def modules():
    import sys
    return (sys.modules["optimize_inference"], sys.modules["detector_tta"],
            sys.modules["v10_quality"], sys.modules["v11_refiner"],
            sys.modules["v14_dualhead"])


def crop_inputs(v11, gray, proposal_mask, bounds, size):
    """Grayscale crop replicated to 3 channels plus the proposal hint channel."""
    x0, y0, x1, y1 = bounds
    image = cv2.resize(gray[y0:y1, x0:x1], (size, size)).astype(np.float32) / 255.0
    hint = cv2.resize(proposal_mask[y0:y1, x0:x1], (size, size),
                      interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return np.concatenate((np.repeat(image[None], 3, axis=0), hint[None]), axis=0)


class DualHeadDataset:
    """Reuses V11's cohort records and augmentation; adds the hint channel."""

    def __init__(self, v11, records, train_dir, epoch, size):
        self.v11, self.records = v11, records
        self.train_dir, self.epoch, self.size = train_dir, epoch, size
        self.cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        record = self.records[index]
        name = record["file_name"]
        if name not in self.cache:
            gray = cv2.imread(str(self.train_dir / name), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (2048, 2048):
                raise ValueError(f"Cannot read training image {name}")
            if len(self.cache) > 4:
                self.cache.clear()
            self.cache[name] = gray
        gray = self.cache[name]
        seed = mask_utils.decode(dict(size=record["seed"]["size"],
                                      counts=record["seed"]["counts"].encode()))
        target = mask_utils.decode(dict(size=record["target"]["size"],
                                        counts=record["target"]["counts"].encode()))
        # Proposal-centred bounds: at inference we only ever have the proposal, so
        # training on GT-centred crops would leak a luxury we do not have. V11's E020
        # already showed the two centrings tie, so use the honest one.
        bounds = record["proposal_bounds"]
        rng = np.random.default_rng(SEED_BASE + self.epoch * 1_000_003 + index)
        jittered = self.v11.jitter_bounds(bounds, gray.shape, rng)
        x0, y0, x1, y1 = jittered
        if seed[y0:y1, x0:x1].any() and target[y0:y1, x0:x1].any():
            bounds = jittered
        channels = crop_inputs(self.v11, gray, seed, bounds, self.size)
        x0, y0, x1, y1 = bounds
        mask = cv2.resize(target[y0:y1, x0:x1], (self.size, self.size),
                          interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if not mask.any():
            raise ValueError(f"Training target vanished on resize: {name}")
        if rng.random() < 0.5:
            channels, mask = channels[:, :, ::-1].copy(), mask[:, ::-1].copy()
        if rng.random() < 0.5:
            channels, mask = channels[:, ::-1].copy(), mask[::-1].copy()
        return torch.from_numpy(channels), torch.from_numpy(mask[None])


SEED_BASE = 42


def train_arm(v11, v14, runtime, arm, cohort, output, identity, epochs, warm_start):
    """One arm. Only the loss differs between arms; budgets are asserted identical."""
    import random
    import torch
    base = runtime["base"]
    base.seed_everything(SEED_BASE)
    model = v14.build_dual_head(base, warm_start).to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=v14.LEARNING_RATE,
                                  weight_decay=v14.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for epoch in range(1, epochs + 1):
        checkpoint = output / "checkpoints" / f"{arm}_e{epoch}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cuda:0", weights_only=True)
            if (saved["run_identity"] != identity
                    or saved["cohort_sha256"] != cohort["cohort_sha256"]
                    or saved["epoch"] != epoch or saved["arm"] != arm):
                raise ValueError("Checkpoint cannot resume another experiment or cohort")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            training = json.loads(checkpoint.with_suffix(".json").read_text())
            history.append(training)
            continue
        dataset = DualHeadDataset(v11, cohort["records"], base.CFG.train_dir,
                                  epoch, v14.CROP_SIZE)
        order = v11.epoch_sample_order(cohort["records"], epoch)
        loader = torch.utils.data.DataLoader(dataset, batch_size=v14.BATCH_SIZE,
                                             sampler=order, num_workers=2,
                                             pin_memory=True, drop_last=False)
        model.train()
        mask_sum = iou_sum = 0.0
        samples = updates = 0
        for channels, target in loader:
            channels, target = channels.to("cuda:0"), target.to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            outputs, predicted_iou = model(channels)
            if outputs[0].dtype != torch.float32:
                raise ValueError("V14 trains in float32 for a clean arm comparison")
            loss_mask = v14.deep_supervision_loss(arm, outputs, target, base)
            actual_iou = v14.observed_iou(outputs[0], target)
            loss_iou = v14.iou_head_loss(predicted_iou, actual_iou)
            loss = loss_mask + loss_iou
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {arm} epoch {epoch} loss")
            loss.backward()
            optimizer.step()
            mask_sum += float(loss_mask.detach()) * len(channels)
            iou_sum += float(loss_iou.detach()) * len(channels)
            samples += len(channels)
            updates += 1
            if updates % 50 == 0:
                print(f"train {arm} epoch={epoch}/{epochs} {updates}/{len(loader)} "
                      f"mask={mask_sum/samples:.6f} iou={iou_sum/samples:.6f}", flush=True)
        if samples != len(cohort["records"]) or updates != math.ceil(samples / v14.BATCH_SIZE):
            raise ValueError("Arm violated the matched sample/update budget")
        training = dict(arm=arm, loss=v14.ARM_LOSS[arm], epoch=epoch, samples=samples,
                        updates=updates, mean_mask_loss=mask_sum / samples,
                        mean_iou_loss=iou_sum / samples,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        sample_order_sha256=v11.digest_json(order),
                        run_identity=identity)
        scheduler.step()
        v11.write_json(checkpoint.with_suffix(".json"), training)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), epoch=epoch, arm=arm,
                        run_identity=identity,
                        cohort_sha256=cohort["cohort_sha256"]), checkpoint)
        history.append(training)
        v11.write_json(output / f"training_history_{arm}.json", history)
        print(json.dumps(dict(arm=arm, epoch=epoch,
                              mask_loss=training["mean_mask_loss"],
                              iou_loss=training["mean_iou_loss"])), flush=True)
    return model, history


def predict_image(v10, v11, v14, tta, core, runtime, model, image_path, split, name):
    """Masks plus predicted IoU for every grouped proposal of one image.

    No dilate(proposal) clamp: measured as worth ~0, and keeping it would cap the
    very axis this run targets. Ownership is still enforced at write time.
    """
    import torch
    base = runtime["base"]
    views = v10.read_raw_views(runtime["inventory"][(split, name)])
    proposals, scores = tta.cluster_views(views, v14.POLICY, v14.MATCH_IOU)
    if not proposals:
        return [], np.zeros(0), np.zeros(0)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (2048, 2048):
        raise ValueError(f"Cannot read {image_path}")
    masks = [mask_utils.decode(rle) for rle in proposals]
    bounds = [base.square_bounds(mask) for mask in masks]
    probabilities, predicted = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(masks), 8):
            batch_bounds = bounds[start:start + 8]
            stack = np.stack([crop_inputs(v11, gray, mask, bound, v14.CROP_SIZE)
                              for mask, bound in zip(masks[start:start + 8], batch_bounds)])
            channels = torch.from_numpy(stack).float().to("cuda:0")
            total = None
            iou_total = 0.0
            # Four-view crop TTA, matching the accepted pipeline's averaging.
            for dims in ((), (2,), (3,), (2, 3)):
                flipped = channels.flip(dims) if dims else channels
                outputs, predicted_iou = model(flipped)
                logits = outputs[0].flip(dims) if dims else outputs[0]
                probability = logits.sigmoid()
                total = probability if total is None else total + probability
                # iou_head_loss (v14_dualhead.py) trains this scalar directly
                # against observed_iou in [0, 1] with plain MSE -- no sigmoid
                # anywhere in that loss. Applying sigmoid here at inference would
                # mismatch the training target, compress most scores into [0.5, 1)
                # regardless of quality, and blow the iou_gate filter open. Clamp
                # to [0, 1] only to guard a head that has not yet learned to stay
                # in-range early in training.
                iou_total = iou_total + predicted_iou.clamp(0.0, 1.0)
            probabilities.extend((total[:, 0] / 4).cpu().numpy())
            predicted.extend((iou_total / 4).cpu().numpy())
    return bounds, np.asarray(predicted, dtype=float), probabilities


def instances_for_candidate(core, base, masks, bounds, probabilities, predicted_iou,
                            mask_threshold, iou_gate):
    """Binarise at mask_threshold, keep predicted_iou >= iou_gate, enforce ownership.

    Ranking is by PREDICTED IOU, not detector confidence. That is the mechanism
    aimed at the 278 cutoff-killed instances: a mediocre-confidence proposal with a
    clean mask now survives, and a confident proposal with a bad mask does not.
    """
    keep = [i for i, value in enumerate(predicted_iou) if value >= iou_gate]
    keep.sort(key=lambda i: -predicted_iou[i])
    occupied = np.zeros((2048, 2048), dtype=bool)
    rles, kept_scores = [], []
    for index in keep:
        x0, y0, x1, y1 = bounds[index]
        crop = cv2.resize(probabilities[index], (x1 - x0, y1 - y0)) >= mask_threshold
        if not crop.any():
            crop = masks[index][y0:y1, x0:x1].astype(bool)
        crop &= ~occupied[y0:y1, x0:x1]
        if crop.sum() < 20:
            continue
        occupied[y0:y1, x0:x1] |= crop
        rles.append(core.encode_crop(crop, bounds[index], occupied.shape))
        kept_scores.append(float(predicted_iou[index]))
    return rles, kept_scores


def evaluate_arm(v10, v11, v14, tta, core, runtime, model, arm, output,
                 candidates, progress_every=20):
    """All predeclared candidates for one arm, plus IoU-head diagnostics."""
    base = runtime["base"]
    images, annotations = runtime["images"], runtime["annotations"]
    held_out, tune, audit, _, by_file = runtime["split"]
    totals = {key: {group: np.zeros(4) for group in ("tune", "audit")}
              for key in candidates}
    diagnostics, head_rows = [], []
    for index, name in enumerate(sorted(held_out), 1):
        bounds, predicted_iou, probabilities = predict_image(
            v10, v11, v14, tta, core, runtime, model, base.CFG.train_dir / name,
            "train", name)
        views = v10.read_raw_views(runtime["inventory"][("train", name)])
        proposals, _ = tta.cluster_views(views, v14.POLICY, v14.MATCH_IOU)
        masks = [mask_utils.decode(rle) for rle in proposals]
        group = "tune" if name in tune else "audit"
        gt_by_image = {}
        for image_id in sorted(by_file[name]):
            info = images[image_id]
            entries = [core.annotation_rle(info, a) for a in annotations.get(image_id, [])]
            gt_by_image[image_id] = [r for r in entries if mask_utils.area(r) > 0]
        # IoU-head calibration/AUC evidence, computed once per image against the
        # union of all annotator GT for that file.
        if masks and group == "tune":
            every_gt = [rle for rles in gt_by_image.values() for rle in rles]
            if every_gt:
                built = instances_for_candidate(core, base, masks, bounds, probabilities,
                                                predicted_iou, 0.45, 0.0)[0]
                if built:
                    actual = mask_utils.iou(built, every_gt, [0] * len(every_gt)).max(axis=1)
                    for value, truth in zip(predicted_iou[:len(actual)], actual):
                        head_rows.append(dict(file_name=name, predicted_iou=float(value),
                                              actual_iou=float(truth),
                                              is_match=bool(truth > 0.5)))
        for key in candidates:
            mask_threshold, iou_gate = candidates[key]
            rles, _ = instances_for_candidate(core, base, masks, bounds, probabilities,
                                              predicted_iou, mask_threshold, iou_gate)
            for image_id, gt in gt_by_image.items():
                iou = (mask_utils.iou(rles, gt, [0] * len(gt)).T if rles and gt
                       else np.zeros((len(gt), len(rles))))
                counts = core.pq_counts(iou, len(gt), len(rles))
                totals[key][group] += counts
                diagnostics.append(dict(arm=arm, candidate=key, file_name=name,
                                        image_id=image_id, group=group, target="coco",
                                        **dict(zip(v14.COUNTS, counts))))
        if index % progress_every == 0 or index == len(held_out):
            best = max(totals, key=lambda k: v14.pq(totals[k]["tune"]))
            axes = v14.axis_metrics(totals[best]["tune"])
            print(f"eval {arm} {index}/{len(held_out)} best={best} "
                  f"pq={axes['pq']:.6f} recall={axes['recall']:.4f} "
                  f"iou={axes['mean_matched_iou']:.4f} fp={axes['fp']}", flush=True)
    results = {f"{arm}/{key}": {g: t.tolist() for g, t in groups.items()}
               for key, groups in totals.items()}
    return results, diagnostics, head_rows


def evaluate_control(v10, v11, v14, tta, core, runtime, output):
    """Frozen V11/V13 path at the accepted settings; must reproduce E015 exactly."""
    base = runtime["base"]
    images, annotations = runtime["images"], runtime["annotations"]
    held_out, tune, _, _, by_file = runtime["split"]
    model = v11.build_model(base, runtime["refiner"])
    totals = {group: np.zeros(4) for group in ("tune", "audit")}
    for index, name in enumerate(sorted(held_out), 1):
        predictions = v11.predict_modes(runtime, model, base.CFG.train_dir / name,
                                        "train", False, ["refined_constrained"])
        kept = predictions["refined_constrained"]
        group = "tune" if name in tune else "audit"
        for image_id in sorted(by_file[name]):
            info = images[image_id]
            gt = [core.annotation_rle(info, a) for a in annotations.get(image_id, [])]
            gt = [r for r in gt if mask_utils.area(r) > 0]
            iou = (mask_utils.iou(kept, gt, [0] * len(gt)).T if kept and gt
                   else np.zeros((len(gt), len(kept))))
            totals[group] += core.pq_counts(iou, len(gt), len(kept))
        if index % 40 == 0 or index == len(held_out):
            print(f"control {index}/{len(held_out)} tune_pq="
                  f"{v14.pq(totals['tune']):.6f}", flush=True)
    counts = {group: total.tolist() for group, total in totals.items()}
    v14.verify_control(counts)
    print(json.dumps(dict(control_verified=True, counts=counts)), flush=True)
    del model
    import torch
    torch.cuda.empty_cache()
    return counts


def export(v10, v11, v14, tta, core, runtime, model, config, output, filename):
    """180-image test export with the same ownership guarantees as every prior run."""
    base = runtime["base"]
    rows, image_counts = [], {}
    for index, path in enumerate(sorted(base.CFG.test_dir.glob("*.jpeg")), 1):
        bounds, predicted_iou, probabilities = predict_image(
            v10, v11, v14, tta, core, runtime, model, path, "test", path.name)
        views = v10.read_raw_views(runtime["inventory"][("test", path.name)])
        proposals, _ = tta.cluster_views(views, v14.POLICY, v14.MATCH_IOU)
        masks = [mask_utils.decode(rle) for rle in proposals]
        rles, _ = instances_for_candidate(core, base, masks, bounds, probabilities,
                                         predicted_iou, config["mask_threshold"],
                                         config["iou_gate"])
        image_counts[path.name] = len(rles)
        occupied = np.zeros((2048, 2048), dtype=bool)
        for number, rle in enumerate(rles, 1):
            mask = mask_utils.decode(rle).astype(bool)
            if mask.sum() < v14.MIN_AREA or (mask & occupied).any():
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
    return dict(file=filename, rows=len(rows), sha256=v11.sha256(destination),
                config=config, evaluated_images=180,
                empty_images=[n for n, c in image_counts.items() if not c])


def load_arm_checkpoint(v14, runtime, output, arm, epoch, identity, cohort, warm_start):
    """Rebuild one arm's final model from its checkpoint, with identity checks."""
    import torch
    path = Path(output) / "checkpoints" / f"{arm}_e{epoch}.pt"
    saved = torch.load(path, map_location="cuda:0", weights_only=True)
    if (saved["run_identity"] != identity or saved["arm"] != arm
            or saved["epoch"] != epoch
            or saved["cohort_sha256"] != cohort["cohort_sha256"]):
        raise ValueError(f"Checkpoint {path.name} does not belong to this run/arm/cohort")
    model = v14.build_dual_head(runtime["base"], warm_start).to("cuda:0")
    model.load_state_dict(saved["model"])
    return model.eval()


def main(args):
    import torch
    core, tta, v10, v11, v14 = modules()
    v11.RAW_CACHE_SHA256 = RAW_CACHE_SHA256
    v11.MODULE_SHA256 = MODULE_SHA256
    runtime = v11.load_runtime(args)
    args.output.mkdir(parents=True, exist_ok=True)
    warm_start = None
    matches = [p for p in Path(args.input_root).rglob("proposal_centered_e4.pt")
               if v11.sha256(p) == CHECKPOINT_SHA256]
    if len(matches) != 1:
        raise ValueError("Expected exactly one pinned V11 proposal_centered_e4.pt")
    warm_start = matches[0]

    epochs = args.epochs
    candidates = {v14.candidate_key(t, g): (t, g)
                  for t in v14.MASK_THRESHOLDS for g in v14.IOU_GATES}
    cohort_names = sorted(runtime["provenance"]["training_files"])
    if args.limit:
        epochs = min(epochs, 2)
        cohort_names = cohort_names[:args.limit]
        candidates = {k: candidates[k] for k in ("m0.45_g0.30", "m0.45_g0.40")}
    identity = v11.digest_json(dict(experiment="v14_dualhead", modules=MODULE_SHA256,
                                    checkpoint=CHECKPOINT_SHA256,
                                    raw_cache=v11.digest_json(RAW_CACHE_SHA256),
                                    epochs=epochs, limit=args.limit,
                                    candidates=sorted(candidates)))
    manifest_path = args.output / "v14_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("complete"):
        print(json.dumps(dict(already_complete=True)), flush=True)
        return
    manifest = dict(experiment="v14_dualhead", run_identity=identity,
                    arms=list(v14.ARMS), arm_loss=v14.ARM_LOSS,
                    v11_source_run=v14.V11_SOURCE_RUN,
                    v11_checkpoint_sha256=CHECKPOINT_SHA256,
                    control_reference=v14.CONTROL_REFERENCE,
                    baselines=v14.BASELINES, submit_bar=v14.SUBMIT_BAR,
                    promotion_gate=v14.PROMOTION_GATE,
                    declared_candidates=list(v14.declared_candidates()),
                    evaluated_candidates=sorted(candidates),
                    consensus_score_auc_baseline=v14.CONSENSUS_SCORE_AUC,
                    epochs=epochs, batch_size=v14.BATCH_SIZE, seed=SEED_BASE,
                    learning_rate=v14.LEARNING_RATE, weight_decay=v14.WEIGHT_DECAY,
                    crop_size=v14.CROP_SIZE, iou_loss_weight=v14.IOU_LOSS_WEIGHT,
                    tversky=dict(alpha=v14.TVERSKY_ALPHA, beta=v14.TVERSKY_BETA,
                                 gamma=v14.TVERSKY_GAMMA),
                    proposal_clamp_removed=True,
                    clamp_note="refined_free ~= refined_constrained historically, so "
                               "unclamping alone is worth ~0; the head and loss are "
                               "the actual change.",
                    smoke_limit=args.limit, trains=True, submits=False,
                    versions=runtime["versions"], complete=False)
    v11.write_json(manifest_path, manifest)

    # 1. Frozen control FIRST.
    control_counts = evaluate_control(v10, v11, v14, tta, core, runtime, args.output)

    # 2. Identical cohort for both arms.
    cohort = v11.build_training_cohort(runtime, cohort_names, args.output, identity,
                                       sample_limit=0)
    print(json.dumps(dict(cohort_samples=cohort["matched_samples"],
                          cohort_sha256=cohort["cohort_sha256"])), flush=True)

    results, diagnostics, head_rows, histories = {}, [], [], {}
    for arm in v14.ARMS:
        model, history = train_arm(v11, v14, runtime, arm, cohort, args.output,
                                   identity, epochs, warm_start)
        histories[arm] = history
        arm_results, arm_diagnostics, arm_head = evaluate_arm(
            v10, v11, v14, tta, core, runtime, model, arm, args.output, candidates)
        results.update(arm_results)
        diagnostics.extend(arm_diagnostics)
        head_rows.extend(dict(arm=arm, **row) for row in arm_head)
        del model
        torch.cuda.empty_cache()

    # Arms must have received identical budgets, or the comparison is void.
    for epoch in range(epochs):
        rows = [histories[arm][epoch] for arm in v14.ARMS]
        reference = (rows[0]["samples"], rows[0]["updates"], rows[0]["sample_order_sha256"])
        if any((r["samples"], r["updates"], r["sample_order_sha256"]) != reference
               for r in rows):
            raise ValueError("Arms did not receive matched training budgets")

    pd.DataFrame(diagnostics).to_csv(args.output / "v14_image_diagnostics.csv", index=False)
    head = pd.DataFrame(head_rows)
    head.to_csv(args.output / "v14_iou_head.csv", index=False)
    rows = []
    for key, value in results.items():
        arm, candidate = key.split("/", 1)
        axes = v14.axis_metrics(value["tune"])
        audit_axes = v14.axis_metrics(value["audit"])
        rows.append(dict(arm=arm, candidate=candidate, key=key,
                         tune_pq=axes["pq"], audit_pq=audit_axes["pq"],
                         tune_recall=axes["recall"],
                         tune_mean_matched_iou=axes["mean_matched_iou"],
                         tune_tp=axes["tp"], tune_fp=axes["fp"], tune_fn=axes["fn"],
                         audit_recall=audit_axes["recall"],
                         audit_mean_matched_iou=audit_axes["mean_matched_iou"]))
    frame = pd.DataFrame(rows).sort_values("tune_pq", ascending=False)
    frame.to_csv(args.output / "v14_candidates.csv", index=False)

    selected = v14.select_candidate(results)
    report = v14.evaluate_bars(selected, results)
    if len(head):
        auc = v14.ranking_auc(head.predicted_iou, head.is_match)
        calibration = v14.calibration_error(head.predicted_iou, head.actual_iou)
        # A degenerate head (all-similar scores, or a fold with only one class)
        # is itself a real, reportable finding -- not a crash. json.dumps rejects
        # NaN, so record it as null rather than losing the whole run's report.
        report["iou_head"] = dict(
            auc=None if math.isnan(auc) else auc,
            consensus_score_auc_baseline=v14.CONSENSUS_SCORE_AUC,
            calibration_error=None if math.isnan(calibration) else calibration,
            samples=int(len(head)))
    report["control_counts"] = control_counts
    report["per_arm_best"] = {
        arm: frame[frame.arm == arm].iloc[0][
            ["candidate", "tune_pq", "audit_pq", "tune_recall",
             "tune_mean_matched_iou", "tune_fp"]].to_dict()
        for arm in v14.ARMS if (frame.arm == arm).any()}
    v11.write_json(args.output / "v14_report.json", report)
    v11.write_json(args.output / "v14_results.json",
                   dict(control={v14.CONTROL: control_counts}, arms=results))
    print(json.dumps(report, indent=2, default=str), flush=True)

    if args.limit:
        manifest.update(complete=True, smoke=True, report=report)
        v11.write_json(manifest_path, manifest)
        print("Smoke finished; no export, no gate decision", flush=True)
        return

    candidate_export = None
    if report["export_authorized"]:
        arm, candidate = selected.split("/", 1)
        mask_threshold, iou_gate = candidates[candidate]
        # Reload the SELECTED arm's final checkpoint explicitly. Exporting whichever
        # model happened to be resident after the loop would silently ship the wrong
        # weights when arm_a wins.
        export_model = load_arm_checkpoint(v14, runtime, args.output, arm, epochs,
                                          identity, cohort, warm_start)
        candidate_export = export(v10, v11, v14, tta, core, runtime, export_model,
                                  dict(mask_threshold=mask_threshold, iou_gate=iou_gate),
                                  args.output, "submission.csv")
        candidate_export.update(arm=arm, candidate=candidate,
                                tune_pq=report["selected_tune_pq"],
                                audit_pq=report["selected_audit_pq"],
                                promotable=report["promotable"],
                                requires_independent_csv_validation=True)
        v11.write_json(args.output / "submission_candidates.json", [candidate_export])
        del export_model
        torch.cuda.empty_cache()
    else:
        print("SUBMIT_BAR not cleared; no CSV exported by design", flush=True)
    manifest.update(complete=True, selected=selected, report=report,
                    candidate_export=candidate_export)
    v11.write_json(manifest_path, manifest)
    print(json.dumps(dict(selected=selected,
                          export_authorized=report["export_authorized"],
                          promotable=report["promotable"], complete=True),
                     indent=2), flush=True)
