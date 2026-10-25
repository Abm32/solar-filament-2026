# V91 "spine-first" dense instance segmentation: pre-registration (2026-10-12)

Written before any V91 result exists (no oracle, no training, and no decoding has been run).

## Question

Can a single dense network at native resolution reach or complement the detector+refiner frontier (V84 OOF 0.45898, fold 0 0.45899)? The network predicts foreground and the expert spine. Instances are decoded from spine components with no learned detector. Nothing from the YOLO, box-detector, crop-refiner or COCO/HF-weights stack is reused. The prior is the V31 spine-SDT oracle at PQ 0.969 (E043). The E044/E045 failures were collapse to all-background at 0.14% foreground, plus NaN in DBSCAN. This design removes both causes: losses are normalised by positive mass plus Dice, and the decoder has no clustering step.

## Data and firewall

- Train only: 707 files and 1,154 annotator records from the official train JSON. Folds come from `outputs/v61_folds.json` (fold 0 = 141 files).
- No test pixels or labels, Dataverse test GT, SWEFil, extra GT or leaked CSVs. No submission.
- The `spine` (flat x,y polyline) and `segmentation` fields are parsed with `json.loads`. GT masks are rasterised exactly as the host scorer does (`pycocotools frPyObjects` + merge).

## Targets (per file, native 2048)

- `fg` = mean over the file's annotator records of the union of that record's masks.
- `sp` = mean over records of `exp(-d^2 / (2·2.5^2))`, where `d` is the EDT distance to that record's rasterised spines (cv2.polylines, 1 px).
- No distance/width head. Width comes from the foreground threshold, and pixels are owned by geodesic assignment to the nearest spine (design option (c) is not used).

## Model and training

- The ResNet18 U-Net `CropUNet` (`outputs/v6_inference_source/solar_base.py`) is initialised from `outputs/v89/ssl_mim/ssl_state.pt` (in-domain MIM SSL; every tensor except `head.*`).
- New head: Conv3x3(16→32) + ReLU + Conv1x1(32→2), giving fg and spine logits. Spine bias init −4.6.
- Input: grayscale/255 repeated to 3 channels (CropUNet normalises internally).
- Tiles: 1024×1024 native crops (no resizing). With p=0.8 the crop contains a random GT spine point (uniform offset); otherwise the crop is uniform. Augmentation: rot90, flip, gamma 0.85–1.2, gain 0.9–1.1.
- Loss: BCE(fg) + soft Dice(fg), plus a spine quality-focal loss `Σ|y−p|²·BCE(p,y) / max(1, Σy)`. The spine loss is normalised by positive mass, so the background majority cannot dilute it.
- Optimiser: AdamW lr 3e-4, wd 1e-4, cosine with 200-step warmup, fp16 AMP, batch 4, fixed step budget (6,000 steps, ~42 epochs). The last checkpoint is used; there is no early stopping and the held-out fold is not looked at during training.
- Inference: the full 2048 image in a single fully convolutional pass. This is equivalent to fully overlapping tiles without seams.

## Decoder (no learned detector)

1. `F = P_fg ≥ t_f`, `S = (P_sp ≥ t_s) & F`.
2. Seeds = 8-connected components of `S`. Seeds whose skeleton is shorter than 8 px are dropped. Spur pruning is skipped because skeletonisation preserves topology, so spurs never change seed identity. There is no fragment linking.
3. Each foreground pixel goes to its geodesically nearest seed within `F`: a flooding watershed on a flat image with `mask = F`. Foreground components with no seed become their own instance, so the keep rule can decide on them. The labels are non-overlapping by construction.
4. Per-instance features: mean P_fg, mean P_sp on the skeleton, max P_sp, log area and log skeleton length. A logistic model (numpy/scipy) predicts `m` = the fraction of the file's records in which the instance is a TP (IoU > 0.5).
5. Keep iff `m > τ`. The grid {t_f ∈ {0.3, 0.4, 0.5}} × {t_s ∈ {0.3, 0.5}}, the logistic model and τ are all selected by cross-fitting (below), never on the scored files.

## Scoring

- The pooled host PQ is reimplemented per file: for each record, IoU > 0.5 hits, ΣIoU, FP = unmatched predictions, FN = unmatched GT. It is first verified to reproduce the host result for V84 fold 0 = 0.458988 (`outputs/v87/fold0_v84.csv`).
- Paired file-level bootstrap: 2,000 resamples of files, `P(d > 0)`.

## Cross-fitting

- **G2 (fold 0 only):** the 141 held-out files are split by sorted index parity into halves A and B. The grid, logistic model and τ are fit on A and applied to B, then the reverse. The two halves are pooled for scoring.
- **G3 (5-fold):** for fold k, they are fit on the pooled OOF decodes of the other four folds.

## Gates

- **S0, engineering sanity (not a kill gate):** overfit 8 training images for 600 steps. The model must not collapse: the number of pixels with P_fg ≥ 0.5 must be within 0.3–3× the GT foreground, and the in-sample decoded PQ with all instances kept must be ≥ 0.5. A failure means debugging, not closure.
- **G1, decoder oracle (CPU, sanity):** decode from the GT soft `fg` and `sp` with t_f = t_s = 0.5, keeping all instances. Pooled PQ on all 707 files must be ≥ 0.85. Below that, fix the decoder before training (which is allowed because the decoder is the thing under test).
- **G2, fold-0 model on 141 held-out files:**
  - Report standalone cross-fitted PQ against p2first direct (≈0.40) and V84 (0.45899).
  - Report orthogonality: V84 FN (GT, record) pairs recovered by V91 (IoU > 0.5), and FP shared (a V91 FP overlapping a V84 FP with IoU > 0.5).
  - Fusion: add V91 kept-candidate instances whose overlap with the union of V84 masks is < 10% of their area, with V84 pixels removed. Instances are added iff `m > τ_f`, where τ_f is cross-fitted on the A/B halves.
  - **CONTINUE iff standalone PQ ≥ 0.45, OR fusion d ≥ +0.008 with P(d > 0) ≥ 0.90 vs V84 fold 0.**
- **G3, 5-fold (707 files):** the arm that passed G2 (standalone or fusion; standalone wins if both pass) must reach d ≥ +0.010 with P ≥ 0.90 vs V84 0.45898. If it passes, build a test CSV (no submission).

## Compute

- Folds 1–4 may be trained in parallel with fold 0 to save wall time. Their held-out predictions are not decoded or scored before the G2 verdict.
- One job per GPU, and only on GPUs whose memory is free.
