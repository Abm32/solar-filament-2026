# V69 literature research: breakthrough candidates (2026-10-10)

Web and reasoning only. No compute was run apart from two CPU checks: a GONG archive directory listing and a date-overlap count on the train/test file names.
Before writing this, I read the project's own record so the list does not repeat closed work:
`v46_research`, `v60_segmentation_first_reframing`, `v60a_refiner_result`, `annotation_sparsity_finding`,
`kaggle_discussion_scan_2026-10-04`, `root_cause_0.39_plateau`, `experiments.md` (V24/V25/V31-V33/V39) and `v68_preregistration`.

## 0. Bottom line

Most of the published "tricks" have already been tried in this repo and failed:
- preprocessing: radial-flatten + CLAHE (V48)
- PU / soft-sampling ignore-loss (V25)
- consensus / union labels (V24)
- Omnipose fields (V31-V33)
- segmentation-first and clDice (V52/V60)
- DenseCRF
- Mask R-CNN / Mask2Former / RF-DETR
- verifier reranking and ensembles

The filament literature is almost entirely **semantic** and reports pixel Dice/IoU, not instance PQ, so it gives no usable instance-level bar.

Two levers are both **untried** and **explicitly allowed by the host**. They come from one fact found during this research. Each training image's COCO `url` points into the GONG archive (`https://gong2.nso.edu/HA/hag/YYYYMM/YYYYMMDD/`). The archive keeps **~3,800 frames per day** at 1-minute cadence across 6 sites (verified: 3,803 files listed for 2014-06-09). Filaments evolve over hours to days, while solar rotation moves the disk centre by only ~9 px/hour at 2048 px. As a result:
1. Every labelled image has dozens of **real neighbouring frames**: different seeing, noise and site, with almost the same geometry. The labels transfer almost for free.
2. There is effectively unlimited **unlabelled same-domain data** for self-training.

The host ruled (kaggle_discussion_scan §D) that unlabelled GONG frames and temporal information are allowed **in training**, and that only the single test image may be used at inference.

Everything else below is second tier, conditional on a zero-GPU ceiling diagnostic (#0).

## 1. What the literature reports (and does not)

| work | task / data | metric reported | relevance |
|---|---|---|---|
| Ahmadzadeh et al. 2024, MAGFiLO, Sci Data 11 ([doi](https://doaj.org/article/13a69e292c4f40f0b27193c2e121b83f)) | 10,244 filaments / 1,593 GONG obs, masks+bbox+spine+chirality | inter-annotator **Kappa 0.66** | Explains the noisy, sparse labels. Note that the public MAGFiLO superset contains the test images, so it cannot be used for training (test GT leak). |
| Solomon et al. 2025, EdgeAttNet ([arXiv 2509.02964](https://arxiv.org/abs/2509.02964)) | MAGFiLO, semantic U-Net + edge-guided attention | mIoU_pairwise 0.645 (pixel) | SOTA on MAGFiLO, but pixel-level. Its preprocessing was already tested in V48 with 0 lift. |
| Diercke et al. 2024, A&A ([arXiv 2402.15407](https://arxiv.org/abs/2402.15407)) | YOLOv5 boxes (ChroTel) -> semi-supervised U-Net on GONG | "92% accuracy" | Decoupled detect-then-segment, semi-supervised on unlabelled GONG. Supports lever #2 in spirit. No PQ. |
| Flat U-Net 2025 ([arXiv 2502.07259](https://arxiv.org/abs/2502.07259)) | BBSO, noisy auto labels | DSC 0.82, recall 0.74 | Pixel-level only. Min-max normalisation, no limb correction. Nothing new for us. |
| MORDEN 2026 ([alphaxiv 2607.24525](https://www.alphaxiv.org/abs/2607.24525)) | semantic, DenseCRF + DBSCAN | abstract only | DenseCRF already failed here. Hype risk. |
| Zhu et al. 2019 ([arXiv 1909.06580](https://arxiv.org/abs/1909.06580)), Liu et al. 2021 (improved U-Nets, ASPP) | BBSO/HSOS semantic | pixel | Old, semantic, small. |
| Mask R-CNN on BBSO, 2019 ([arXiv 1912.02743](https://arxiv.org/abs/1912.02743)) | instance | qualitative | Mask R-CNN already failed here. |
| Reche, ESA SPAICE (DETR + U-Net + tracking on GONG) ([abstract](https://spaice.esa.int/?p=1044)) | instance + tracking | abstract only | Unverifiable. Uses tracking, i.e. temporal structure. |

**Takeaway:** nobody has published an instance-PQ number on MAGFiLO. The credible clean frontier is still the competitor-reported **~0.46 strictly OOF** (tushhaaarrrr) and ~0.48 on a casual split (Spaghetto99). Our 0.46 refined OOF is already at that level, so the remaining gains will be small and must be measured against split noise (~0.012).

Other domains:
- Omnipose ([Nature Methods 2022](https://link.springer.com/10.1038/s41592-022-01639-4)) and Cellpose-SAM ([bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.04.28.651001.full.pdf)) do address elongated instances. This project's oracle reached PQ 0.92 with exact Omnipose fields, but V33 died on optimisation and NaN issues, and bottom-up flows are hurt by sparse labels in the same way the semantic models were.
- HoVer-Net ([arXiv 1812.06499](https://arxiv.org/abs/1812.06499)) is PQ-native, but its offsets point to a centroid and break on elongated objects. This is the same failure mode Omnipose fixed.
- Sartorius-style winners (two-stage: YOLOX detector + UPerNet segmentor, [tascj](https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution)) won mainly through **large same-domain pretraining data** (LIVECell). The closest legal equivalent here is the unlabelled GONG archive.
- Contrails 1st place ([junkoda](https://github.com/junkoda/kaggle_contrails_solution)) used neighbouring time frames as *input*. That is not allowed here at inference, but it shows temporal frames carry signal.

## 2. Ranked experiments

### #0 (zero GPU, do first) Human ceiling: inter-annotator PQ / SQ under the competition scorer

- **Mechanism:** 296 of 707 files have 2+ annotators. Score annotator A's record as the "prediction" against annotator B's record with the exact competition PQ scorer (pixel ownership, IoU>0.5), in both directions. Report PQ, SQ (mean matched IoU), RQ and the IoU histogram of matched pairs.
- **Why:** V24 found 58% of cross-rater instance groups appear in only one rater. If human-vs-human PQ is about 0.45-0.50 and human matched IoU is about 0.65-0.70, then:
  - our 0.46 refined OOF is at the human ceiling;
  - SQ 0.65 is annotator boundary noise, not model error;
  - every mask-refinement lever (#5) is dead;
  - the only real headroom is RQ on high-agreement filaments.
- **Cost:** about 1 CPU-hour. It also bounds every other item below.
- **Falsifies:** if human SQ is ≥0.78, mask quality has real headroom (promote #5). If human PQ is well above 0.6, the FN gap is a model problem (promote #1/#2).

### #1 Temporal label transfer: real-frame augmentation from neighbouring GONG frames (NEW, legal, untried)

- **Mechanism:** for each training file `YYYYMMDDhhmmssXh`, download frames within ±k minutes from the same site, plus co-temporal frames from other sites that observe at the same time. Warp the masks with solar differential rotation (sunpy `solar_rotate_coordinate`, or a plain shift for |Δt| ≤ 10 min, which is under ~1.5 px). Optionally refine the alignment by cross-correlating the images.
- **Result:** the 707 images become about 5-10k *real* labelled images with independent seeing, noise, cloud, JPEG and site differences.
- **Why it fits our error profile:**
  - (a) Faint, small filaments are FN-heavy. With one image per label, the model sees each faint filament under one seeing realisation, while neighbours give the same label under better and worse seeing. This is real, not synthetic, augmentation (synthetic photometric augmentation is already in the recipe).
  - (b) It adds 5-10x mask supervision per filament, which should help matched IoU.
  - (c) Overfitting on 566 images is a likely reason capacity and resolution scaling (v8l@2048, 1536/2048) failed.
- **Evidence:**
  - Zhu et al., CVPR 2019 ([arXiv 1812.01593](https://arxiv.org/abs/1812.01593)): propagating labels to neighbouring video frames as augmentation gave significant gains (SOTA Cityscapes 83.5). Their propagation is *learned* and error-prone. Ours is near-exact, because the geometry is a known rigid or differential rotation over minutes, which is a stronger case.
  - The only caveat they report is that diluting the hand-labelled frames hurt slightly. Fix: sample originals at ≥50%.
- **Expected gain:** speculative, +0.005 to +0.03 direct PQ. There is no direct solar precedent. Also flag that the refiner absorbs ~80% of detector gains.
- **Main risks:**
  1. Annotators labelled what *they saw* in the original frame, so a neighbour with better seeing reveals unlabelled filaments, which re-creates the sparse-label problem. Mitigation: use same-site ±5-10 min first.
  2. **CV leakage.** 50 train days have >1 file and 31 test days coincide with train days. Neighbour frames of a fold-train file can be near-duplicates of a fold-val file from the same day. Group folds by date (±1 day) before any number is trusted. This may also deflate the existing OOF slightly.
  3. Never use frames from test dates for *training* unless the host OKs it. The safe default is to exclude ±1 day of every test date.
- **Cost:**
  - Download: 707 files × ~10 frames × ~1 MB JPEG ≈ 7 GB (CPU, around an hour).
  - Train: fold-0 p2first with the step count matched to `p2first_long` (the 2x-schedule arm already running in V68), giving a clean compute-matched control. About 1 T4-day per arm.
- **First cheap diagnostic (no training, about 1 GPU-hour):**
  - For the 141 fold-0 held-out files, fetch frames at Δt = ±2, ±10 and ±30 min (same site) and transfer the GT.
  - Run the *existing* p2first and compare PQ on neighbours with PQ on the originals.
  - If neighbour PQ ≈ original PQ (within ~0.02) and the transferred-GT overlay matches visually, labels transfer and the experiment is valid.
  - If neighbour PQ drops sharply even at ±2 min, the labels are frame-specific (seeing-driven) and the idea is dead.
  - The spread of PQ across neighbours also gives a free estimate of seeing-induced variance.

### #2 Noisy-student self-training on unlabelled GONG frames (legal, untried)

- **Mechanism:**
  - Teacher: current p2first + 4-flip TTA + refiner.
  - Pseudo-label 2-5k unlabelled GONG frames sampled across 2011-2022 and all 6 sites, excluding every train-fold-val, held-out and test date ±1 day.
  - Keep instances with high score, and mark mid-score instances as *ignore* regions rather than background.
  - Student: same architecture, strong augmentation, trained on labelled + pseudo-labelled data, with labelled samples at ≥50%.
  - Optional: temporal-consensus filtering. Accept a pseudo-instance only if it is detected in ≥3 of 5 frames within ±10 min. This sharply raises pseudo-label precision and needs no extra training.
- **Why:** this is the legal analogue of the LIVECell pretraining that won Sartorius. Zoph et al. 2020 ([arXiv 2006.06882](https://arxiv.org/abs/2006.06882)) report self-training **+1.3 to +3.4 AP on COCO at every data size**, including where pretraining hurt. Diercke et al. 2024 already used semi-supervision on GONG.
- **Skeptic's note:**
  - Union and consensus *label densification* failed here (V24/V25) because test GT is sparse too, so more positives can mean more FPs.
  - Self-training at a high threshold adds *data diversity*, not recall pressure, which is a different mechanism. But it cannot recover the 18.7% of GT that is "invisible to every detector": the teacher never sees those either.
  - Expect the gain to come mainly from SQ and robustness.
- **Expected gain:** +0.005 to +0.015 direct. Treat anything above +0.02 as a red flag for leakage.
- **Cost:** ~3-5 GB download; teacher inference about 2-4 h across 4 T4s; student fold-0 about 1-1.5 T4-days.
- **First cheap diagnostic:** one fold-0 round with 2k pseudo-labelled frames vs `p2first_long` at matched steps. Gate: ≥+0.010 direct and ≥+0.005 refined. Run it on the same date-grouped fold as #1.

### #3 Rotation equivariance: full-angle rotation augmentation + rotation TTA (cheap, untried)

- **Mechanism:** the V21 recipe used `degrees=10` and no flips (flips are being tested now in V68). The solar disk is rotation-symmetric, but filaments carry a weak orientation prior (polar-crown filaments run east-west). Train with `degrees=180`, or rot90 multiples to avoid interpolation blur, and use 8-view dihedral TTA instead of 4-view flips.
- **Why:** this is a cheap extension of the one augmentation axis now under test. TTA averaging over more views smooths thin-mask aliasing (SQ), and the extra views raise consensus recall on small objects.
- **Expected gain:** 0 to +0.01. Low confidence, because it may destroy a useful orientation prior.
- **Cost:** zero for the TTA probe; one T4-day for the training arm.
- **First cheap diagnostic:** add rot90/rot270 views to the existing p2first TTA on fold 0.
  - If PQ drops sharply, the model depends on orientation, and a training arm is needed to tell whether the prior matters.
  - If PQ is flat or up, adopt 8-view TTA immediately.
- Run this *after* the V68 flip arm reports.

### #4 Train-only privileged multi-frame teacher (distillation), a stretch variant of #1

- **Mechanism:**
  - Teacher input: the original frame plus a registered median of ±5 neighbouring frames (higher SNR, "lucky imaging"). Allowed because it is training-only.
  - The teacher sees faint filaments better, and a single-frame student is distilled from its logits and masks.
  - This is learning using privileged information, and it differs from #2 in that the teacher is *better* than anything we could train before.
- **Expected gain:** unknown, ≤+0.01. Do it only if #1's diagnostic shows neighbours carry information and #1 itself gains.
- **Cost:** about 2-3 T4-days.

### #5 SAM2 / HQ-SAM refiner with box+mask prompts (conditional on #0)

- **Mechanism:** replace or augment the compact refiner with a frozen SAM2 Hiera-B+ encoder (cache embeddings for the 707 images at 1024), fine-tune the mask decoder with box + coarse-mask prompts, and keep a "best of old/new" selector (HQTrack style).
- **Evidence:**
  - HQ-SAM ([arXiv 2306.01567](https://arxiv.org/abs/2306.01567)) improves thin structures.
  - SAMRefiner ([arXiv 2502.06756](https://arxiv.org/abs/2502.06756)) shows box-only prompts *lose* to the coarse mask (68.8 vs 81.4 IoU on DAVIS-585) and only box+mask prompts gain.
- **Counter-evidence (strong):** Zhang et al. ([arXiv 2412.04243](https://arxiv.org/abs/2412.04243)) show SAM, SAM2 and HQ-SAM degrade on **tree-like, low-contrast** objects, which describes filament barbs, and that fine-tuning does not fix it.
- **Context:** our refiner already absorbs ~80% of detector gains.
- **Expected gain:** 0 to +0.01. **Flag: hype risk.**
- **Cost:** about 1-2 T4-days; fits in 16 GB with the encoder frozen and embeddings cached.
- **Gate:** run only if #0 shows human SQ well above 0.65.
- **First cheap diagnostic:** zero-shot SAM2 box+mask prompts on 141 OOF matched instances; report the per-instance IoU delta and the fraction of masks damaged.

### #6 PQ-optimal inclusion rule (zero GPU, small)

- **Mechanism:** under per-record PQ, adding prediction *i* helps iff roughly p_match,i · E[IoU_i] > PQ*/2. This generalises Lipton et al.'s result that the F1-optimal threshold is half the optimal F1 ([arXiv 1402.1892](https://arxiv.org/abs/1402.1892)).
- Fit isotonic calibration of p_match and E[IoU] on OOF (by size bin), then rank by their product instead of by raw confidence.
- **Skeptic:** verifier reranking already failed, and a global confidence sweep captures most of this. Expect ≤+0.005.
- It is cheap enough to check in an afternoon, and it guards against size-dependent miscalibration (small filaments, recall 0.44).

## 3. Not recommended / hype flags

- **Full public MAGFiLO corpus (10,244 filaments) for pretraining:** it contains the test images' GT, which is a direct leak. Do not use.
- **DINOv2/v3 or SegFormer backbones:** patch-14/16 ViTs lose 2-5 px filament widths, and Mask2Former (0.154) already showed transformer heads struggle here. Same-domain data (#1/#2) beats generic foundation features at this scale (Zoph 2020).
- **Self-supervised MAE/DINO pretraining on GONG:** multi-day cost on T4s, and Zoph shows pretraining gains shrink with strong augmentation. Self-training (#2) is the cheaper way to use the same data.
- **Omnipose / Cellpose-SAM revival:** the mechanism is right for elongated shapes, but it was closed in V33 for optimisation reasons, and bottom-up flows share the semantic models' sparse-label FP problem. Revisit only with spare GPU.
- **HoVer-Net:** centroid offsets fail on elongated objects.
- **DenseCRF, limb-darkening correction, CLAHE, polar unwrapping:** tried (V48) or no published instance evidence.
- **Leaderboard numbers above ~0.45:** confirmed polluted by public-GT leakage (host). Do not chase them.

## 4. Suggested order on 4 T4s (next ~7 days)

1. Day 0, CPU: #0 human ceiling, #1 GONG download script, and a date-grouped fold split.
2. Day 0-1, 1 GPU: #1 neighbour-frame diagnostic and #3 rot-TTA probe on existing weights.
3. Day 1-3, 3 GPUs: fold-0 arms (a) #1 temporal-transfer, (b) #2 self-training, (c) `p2first_long` control (rerun on the date-grouped fold if V68 used a random split).
4. If either #1 or #2 passes (≥+0.010 direct, P>0.9, beyond seed noise): 5-fold, then a refined-pipeline test, then combine #1+#2 (temporal-consensus pseudo-labels).

## Sources
- MAGFiLO Sci Data 2024: https://doaj.org/article/13a69e292c4f40f0b27193c2e121b83f ; NSF PAR: https://par.nsf.gov/biblio/10427981
- EdgeAttNet: https://arxiv.org/abs/2509.02964
- Diercke et al. 2024: https://arxiv.org/abs/2402.15407 ; https://www.aanda.org/10.1051/0004-6361/202348314
- Flat U-Net: https://arxiv.org/abs/2502.07259
- MORDEN: https://www.alphaxiv.org/abs/2607.24525
- Zhu et al. 2019 filament U-Net: https://arxiv.org/abs/1909.06580 ; Mask R-CNN BBSO: https://arxiv.org/abs/1912.02743
- Chalmers & Ahmadzadeh 2025 (chirality): https://arxiv.org/abs/2509.18214
- Omnipose: https://link.springer.com/10.1038/s41592-022-01639-4 ; Cellpose-SAM: https://www.biorxiv.org/content/10.1101/2025.04.28.651001.full.pdf
- HoVer-Net: https://arxiv.org/abs/1812.06499
- PU detection: https://arxiv.org/abs/2002.04672 ; SparseDet: https://arxiv.org/abs/2201.04620
- Video label propagation: https://arxiv.org/abs/1812.01593
- Self-training: https://arxiv.org/abs/2006.06882
- HQ-SAM: https://arxiv.org/abs/2306.01567 ; SAMRefiner: https://arxiv.org/abs/2502.06756 ; SAM limits on tree-like/low-contrast: https://arxiv.org/abs/2412.04243
- F1-optimal threshold: https://arxiv.org/abs/1402.1892
- Sartorius solution (tascj): https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution ; Contrails 1st: https://github.com/junkoda/kaggle_contrails_solution
- GONG H-alpha archive (verified listing): https://gong2.nso.edu/HA/hag/201406/20140609/
