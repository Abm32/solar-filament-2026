# V89 pre-registration: domain self-supervised pretraining of the crop refiner (2026-10-12, before any V89 number)

## Why
The refiner is the binding component (refined OOF PQ ~0.45 whatever the detector). Every refiner so far starts
from ImageNet ResNet18. V89 asks whether continued self-supervised pretraining on GONG H-alpha pixels gives the
refiner a better init. The architecture (V73 CropUNet, ResNet18 U-Net, 256 crops) and the V84 soft recipe are unchanged.

## Data (fixed, built by `scripts/v89_ssl.py prep` -> `/mnt/v89/imgs.u8`, `meta.npz`)
- **707 train images, pixels only.** No annotation is read.
- **1,356 unlabelled GONG archive frames** in `/mnt/v76/frames` (V76 fetch). Every train and test date ±1 day was excluded.
  Re-asserted in code: every frame is more than 1 h from every test timestamp (measured minimum 28.7 h), and no name is a test file.
  Host ruling (thread 744047): unlabelled GONG frames are allowed for training, excluding test timestamps ±1 h.
- **No test image is used for anything.** No new frames were fetched.
- **Total: 2,063 images** (all are 2048², all readable).
- **Crop sampler (both arms):**
  - Centre: p 0.5 at a "dark-structure" pixel, otherwise uniform on the disk (r < 0.95 R). Dark-structure pixels are
    the top 0.5% of blur(σ30) − blur(σ4) inside r < 0.9 R (4,000 stored per image).
  - Side: log-uniform 96..768 native px, resized to 256. Augmentation uses the v11 ranges (rot90, flip, gamma, gain).
  - **Disclosure:** the dark filter (σ pair and percentile) was chosen from 5 candidates by checking the GT hit rate on 20 train
    images (55% inside the 15 px-dilated GT, against 1.9% at random). It is a 3-number sampling heuristic and does not enter the model directly.

## Arms (two, fixed now). Both start from ImageNet ResNet18 (continued pretraining). Seed 89, fp16 AMP, one GPU each.
- **A `mim`: masked-image modelling of the full U-Net.**
  - Masking: 8×8 grid of 32 px patches, ratio 0.5. Masked patches are filled with the crop mean.
  - Target and loss: per-crop standardised intensity, L1 on masked pixels only.
  - Optimiser: AdamW (encoder lr 1e-4, decoder lr 1e-3, wd 0.05), 1k-step warmup then cosine. **40,000 steps × batch 64** (2.56 M crops).
  - Transfer: everything except the 1×1 `head`.
- **B `moco`: MoCo v2 on the ResNet18 encoder.**
  - Head and contrast: 512-2048-128 MLP head, queue 16,384, T 0.2, key momentum 0.99. Single-GPU shuffle-BN is
    emulated by encoding shuffled keys in 4 chunks.
  - Views: two views per centre (side ×[0.8, 1.25], shift ±0.15 side). Augmentation is stronger: contrast, blur and noise.
  - Optimiser: SGD lr 0.03, momentum 0.9, wd 1e-4, warmup then cosine. **40,000 steps × batch 128**.
  - Transfer: stem and layer1-4 only (the decoder is fresh, as in V73).
- **Step budget:** set from a 400-step smoke run (no PQ seen): about 2.3 h and 3 h on one T4. This is well under the 8 h cap.
  No DDP is used, so the batch and recipe are exactly as stated here.

## Fine-tune (identical to V84 soft)
`scripts/v89_finetune.py` = `train_v82_refiner_targets.py --arm soft`.
- Recipe: stage A 8 ep, 3e-4, on GT crops; stage B 12 ep, 1e-4, on p2first OOF proposals with soft targets; seed 42.
- Same files and same scoring. The only change is that `build_model` loads the SSL weights after the ImageNet build.
- The fold-0 screen trains fold 0 only.

## Controls
- **V84 soft fold 0** (`outputs/v82/soft_fold0_refiner.pt`, seed 42) is the comparator.
- **Retrain noise:** the same recipe from ImageNet with seed 43 (GPU 2) and seed 44 (GPU 3), scored through the identical screen.

## Fold-0 screen (`scripts/v89_screen.py`, CPU)
- **Pipeline:** the full V84 path on the 141 fold-0 files. p2first fold-0 OOF views → flip consensus → refiner (k7) at t0.50 → cut 0.30
  → MIN200 → V84 demote.
- **Demote:** the LightGBM is fit on the V84-soft t0.50 candidates of folds 1-4, and λ is V84's fold-0 λ. It is held
  fixed for all arms; only the fold-0 candidates change.
- **Control check:** the control must reproduce V84's bundle fold-0 per-file counts exactly.
- **Primary:** d = PQ(arm) − PQ(V84 soft), full path, paired file bootstrap 2000, seed 89.
- **Secondary (report only):** t0.50/c0.30 without MIN200/demote, and t0.45/c0.30.
- **Gate:** **d ≥ +0.008 and P(d>0) ≥ 0.90** → 5-fold. If both arms pass, the larger d goes to 5-fold. Otherwise KILL.
  The seed controls' |d| is reported alongside. If an arm passes but |d_noise| ≥ d, that is stated.

## 5-fold (only if the screen passes)
- **Training:** all 5 folds are fine-tuned from the same SSL init. The SSL data has no labels and no test images, so one init is
  leak-free across folds.
- **Extract and bundle:** run the V84 extract and bundle with the new family. The demote is refit cross-fitted on the new OOF, as in V84.
- **Final gate:** d ≥ +0.010 and P ≥ 0.90 on 707 files against V84 at 0.45898.
- **If it passes:** train the all-data refiner and build the test CSV via `scripts/_v74_export.sh` (REFINER = new state, same MASK_T/DEMOTE
  as V84, demote refit on the new OOF). The CSV is validated and **not submitted**.

## Amendment 1 (corrected 2026-10-12 ~05:50 UTC, before any SSL-arm fine-tune or screen number)
- **The first runs were stopped:** at the user's request to use more GPUs, the coordinating session stopped the first single-GPU SSL runs
  (launched 05:01) at about 05:19-05:35. Their partial outputs are kept, **unused**, in `outputs/v89/ssl_*_1gpu_stopped`.
- **DDP was abandoned:** a 4-GPU DDP variant (`scripts/v89_ssl_ddp.py`) was smoke-tested only. It was slower on these T4s, and no DDP output
  was kept or used.
- **The runs were restarted from scratch at about 05:45 UTC with the identical single-GPU recipe** (`scripts/v89_ssl.py`, seed 89, mim 40k
  steps x 64 on GPU 0, moco 40k steps x 128 on GPU 1). There is no DDP and no SyncBN, and nothing in the recipe above changed.
- **Noise controls:** seeds 43 and 44 (this agent, GPU 2) and **45** (launched by the coordinating session on GPU 3, with the user's
  permission). Extra controls only; the gate is unchanged.
