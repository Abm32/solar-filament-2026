# V98 public-board audit (2026-10-25): why nobody clean is above 0.42

Question from the user: 100+ teams sit above our 0.39. Is there a method, a model source
(HuggingFace, Kaggle models), or a submission-window trick we are missing?

Sources read today: full public leaderboard (1171 teams), 8 top-voted / top-scoring Kaggle
kernels pulled with the API, Kaggle datasets and models search, HuggingFace models/datasets/spaces
search, 29 GitHub repos (13 named + `gh search` for everything updated after 2026-10-15),
6 forum threads including the host's Oct 23 remarks. No test labels or leaked CSV were decoded.

## 1. Board composition (2026-10-25, public = ~90 images, 2 decimals)

| rank | score | n | what it is |
|---|---|---:|---|
| 1 | 0.70 | 1 | IONode Online, 18 entries, first seen 10 h ago |
| 2–20 | 0.56–0.64 | 19 | laundered leak (public MAGFiLO 1.0 contains labels for 139/180 test images) |
| 21–96 | 0.55 | 76 | **the "0.55 CSV"**: a hard-coded payload, see §2 |
| 97–98 | 0.53–0.54 | 2 | partial leak |
| 99 | 0.49 | 1 | noisyislands (13 entries, 19 d old; public code is R34 U-Net + seed refiner, medoid-annotator GT; cannot be verified clean) |
| 100 | 0.42 | 1 | Suna Nur Saydam, 20 entries, submitted 17 min before the scan |
| 101–111 | 0.41 | 11 | clean frontier (Spaghetto99, adversarial looper, supersede, Solid-Snake-99 …) |
| 112–151 | 0.40 | 40 | clean |
| 152–171+ | 0.39 | ≥20 | us at rank 162 (V84, OOF 0.459) |

So "100+ above 0.39" = 96 leak + 3 unverifiable + 52 clean teams within 0.01–0.03 of us.
With ~90 images and 2-decimal rounding the per-submission std is ≈0.012, so 0.39 vs 0.41 is
inside noise; V92 already showed public moves with OOF 1:1 for every knob.

## 2. The 0.55 mechanism (proven, not inferred)

Kernel `lamhuy8904/solar-filament-unet-segmentation-0-55` (39 votes, the one Hermengardo pointed
at in "huge gap on the leaderboard"): the "training" cells are a 5-epoch U-Net decoy. The
submission cell contains `CHAMPION_PAYLOAD`, a 180 760-character base85 string decoded with
`lzma.decompress(base64.b85decode(...))` straight into a DataFrame of `filament_id,
segmentation_rle` rows, which is then written out as `submission.csv` per test stem. Comment in
the code: "Pre-compiled high-resolution Native 2048 SOTA inference database (verified 0.55 on
Leaderboard)". 76 teams at exactly 0.55 are copies of that file. Scores 0.56–0.64 are the same
file edited or the public MAGFiLO masks re-exported with small changes.

The July/August "LB 0.70 / 1.00" kernels (Fususu, artkomissar "please fix the leaderboard") are a
different thing: the pre-Aug-12 evaluator gave 1.00 to five empty masks. Those numbers are the old
`instance_wise_dice` board, not PQ, and not comparable.

## 3. Host statements that matter (Oct 23, "Organizers' Remarks")

- "We can easily identify submissions which used shortcuts … clear patterns of bad practices
  emerged. We will clean the leaderboard after the deadline."
- "A PQ greater than 0.35 still has significant merit. What matters most is quality."
- New hard requirement: **model weights must be stored in the repository and loaded in the
  notebook**; otherwise the submission is not reproducible and is not considered for the award.
- Earlier (Sep): "we take a score of 0.41 very seriously."
- Rubric: leaderboard PQ is one item inside the 70 % quantitative block; 30 % is pipeline
  description, morphology, code quality. Private board = the other 50 % of test images.
- External unlabeled H-alpha (KSO, Meudon, BBSO) for self-supervised pretraining is allowed,
  excluding observations within ±1 h of test timestamps.

## 4. HuggingFace / Kaggle model sources

Nothing the leaders use. HF search for `magfilo`, `solar filament`, `filament-segmentation`:
three model repos (`ghostfreak135z/magfilo-segmentation-checkpoints`, `Godseye1311/solar-filaments`,
`Ephyrr/solar_filament_segment`), all 0 downloads, plain U-Net / refiner checkpoints; one HF
dataset (`cuonguyenphu/...-result-0.34`). No filament foundation model exists. Every top kernel
and repo loads ImageNet/COCO encoders through `segmentation_models_pytorch`, `timm`, torchvision
or ultralytics. DINOv3 encoders were tried by two repos and were worse than ImageNet ConvNeXt.
The one Kaggle dataset that matters is `esairlab/magfilo-v1-0-segmentation-of-solar-filaments`
(890 downloads): the public MAGFiLO 1.0 release with `MAGFiLO_1.0_Annotations.json`, i.e. the leak.

## 5. Clean competitors: what they report (GitHub + kernels)

| team | OOF | public | method |
|---|---:|---:|---|
| pragyaangaur | 0.457 | 0.39 | whole-disk ConvNeXt-T U-Net at 1536 + 2048 fine-tune, 8-way TTA, CNN quality head, filament-level fusion of 15 fold members. Their table: OOF 0.44→0.457 all map to public 0.38–0.39. |
| tushhaaarrrr | 0.460 | 0.40 | dense ensemble + Dinkelbach component-tree decision (τ = PQ/2) + UNet++/ConvNeXt refiner |
| Solid-Snake-99 | 0.487 (contaminated) | 0.41 | RF-DETR-Seg-L on 4 quadrants + CropUNet R18 + topology-guarded TTA |
| biconcavelens | 0.461 | 0.40 | Mask R-CNN + YOLO11m + RT-DETR pooled, isotonic calibration, R34 refiner |
| dariushafshar | 0.450 | 0.39 | 10-model U-Net ensemble (R34 + EffV2-S), keep rule by nested CV |
| theainerd | 0.452 | 0.40 | EffB3 U-Net, limb-radial channel, GBM rescorer |
| HeShen-1 | 0.359 | 0.35 | YOLO11m + seed refiner; gap analysis: ~0.02 fold selection bias + ~0.04 test annotator style |
| ours (V84) | 0.459 | 0.39 | p2first YOLO11s-seg + soft CropUNet + Dinkelbach demote |

Every clean team loses 0.05–0.08 OOF→public; none explains it beyond annotator style and the
90-image sample. No clean team exceeds 0.41 except one 0.42 submitted today.

Multi-annotator targets tried by others: random annotator per sample, medoid annotator, densest
annotator, per-record samples, union, soft mean. Nobody reports a gain from any of them, which
matches our V82/V96/V97 results. Nobody predicts several mask sets per image.

Method ideas we have not run and that others report OOF gains for:
1. Whole-disk 1536/2048 ConvNeXt-T U-Net instead of detector crops (pragyaangaur, +0.02–0.03 OOF
   over their crop models, but still 0.39 public).
2. Filament-level fusion of fold members, cluster-mean score (+0.007 OOF).
3. CNN quality head on a 4-channel crop including other candidates' masks (+0.005 OOF).
4. RF-DETR-Seg detector (Solid-Snake-99, 0.41 public, OOF unverifiable).
5. Self-supervised pretraining on KSO/Meudon/BBSO H-alpha (allowed; our V89 GONG SSL was flat).

pragyaangaur's one mechanistic finding (fold-averaged test probability maps are softer than
single-model OOF maps, so a learned keep-head under-scores on test) does not apply to us: V84
exports single all-data models, and V92's fold-0-only probe scored the same 0.39.

## 6. Reading

The board above 0.42 is not a method gap. It is one leaked file plus the public MAGFiLO release,
and the host says it will be removed after the deadline. The clean field is 0.39–0.42 with
per-submission noise of ±0.012; our OOF 0.459 is at the top of the clean pack by self-reported
numbers. Nothing on HuggingFace or Kaggle Models gives anyone an edge.

What can still change our standing:
- **Final evaluation, not the public board.** 30 % of the rubric is report + morphology + code
  quality, and weights must now be inside the repo. The repo is private, unpushed since V66, and
  holds no weights (detector 21 MB, refiner 57 MB, demote 1 MB: 79 MB, fits GitHub without LFS).
- A whole-disk ConvNeXt-T arm (idea 1) is the only untested direction with reported OOF headroom.
  Expect at most +0.01 public if it transfers at the usual 0.85×.
