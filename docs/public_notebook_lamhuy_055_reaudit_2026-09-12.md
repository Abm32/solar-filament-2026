# Lâm Huy `0.55+` notebook re-audit — 2026-09-12

Notebook: <https://www.kaggle.com/code/lamhuy8904/solar-filament-unet-segmentation-0-55>

This document updates and supersedes the provenance details in `public_notebook_lamhuy_055_review_2026-09-05.md`. It does not copy, execute, or use the embedded predictions for modeling or submission.

## Verdict

The notebook does **not** provide a reproducible U-Net method for its claimed `0.55+` result. Its published `submission.csv` is generated entirely from an embedded compressed table named `CHAMPION_PAYLOAD`, plus two fixed fallback squares. The U-Net trained in the notebook is a CPU demonstration, has effectively zero validation Dice, is never called on a test image, and does not generate any row of the submission.

The author's team was recorded at public score **0.55** in the authenticated 6 September leaderboard snapshot. Its last submission timestamp, 2026-09-03 15:14 UTC, is seven minutes after the notebook's 15:07 run. This makes linkage plausible, but public kernel metadata provides no submission receipt or file hash proving that the exact published CSV bytes received 0.55. The exact defensible statements are:

1. the author's team had a 0.55 public result;
2. the published notebook output is exactly reproducible from the embedded payload;
3. the demonstrated U-Net definitively did not produce that output;
4. the origin, training method, model, and validation of the payload cannot be established from public artifacts.

## Current version and change check

Authenticated Kaggle retrieval on 2026-09-12:

- ref: `lamhuy8904/solar-filament-unet-segmentation-0-55`;
- title: `Solar Filament Unet Segmentation | 0.55+`;
- last run: 2026-09-03 15:07:33 UTC;
- status: complete;
- votes: 22;
- execution: CPU only, no GPU/TPU, no internet;
- attached sources: competition input only; no dataset, kernel, or model sources;
- cells: 8 total, 7 code, no stored outputs in the downloadable source.

The current 198,028-byte notebook is byte-for-byte identical to the copies pulled on 4 and 6 September:

`906841cbc0e1f50b89bda6648b1c7f85eff16975e62c3679a120141aa172a687`

There is no new implementation to evaluate.

## Executed data flow

### Demonstration training

The notebook:

- selects the first 60 COCO annotation **records**, representing only 39 physical files;
- uses 51 records for training and 9 for validation;
- rasterizes polygons with integer `cv2.fillPoly`;
- creates binary semantic foreground masks, merging all instances in each record;
- trains a small grayscale U-Net on random 256x256 crops for five CPU epochs;
- saves `best_filament_unet.pth`.

The completed log reports validation Dice `0.0001, 0, 0, 0, 0`. The model is invoked only in training and validation (`model(x)` and `model(vx)`). There is no `model.predict`, test forward pass, checkpoint reload, tiling, connected-component reconstruction, or semantic-to-instance conversion.

### Submission generation

The test path:

1. base85-decodes and LZMA-decompresses `CHAMPION_PAYLOAD`;
2. reads the resulting CSV into `champion_df`;
3. groups precomputed RLE strings by test filename;
4. copies those strings into the final dataframe;
5. adds one fixed central 48x48 mask for each test image absent from the payload;
6. writes `/kaggle/working/submission.csv`.

The reported “Native 2048 inference” takes 0.14 seconds for 180 CPU images because it is dataframe lookup, not neural inference.

## Exact payload and output identity

The payload was decoded by extracting only the literal AST value; no notebook code was executed.

- encoded payload: 168,910 characters;
- encoded SHA-256: `1c080e1b9f0bf9b81191d0199472f7a8191c44812ffc0ee56f3d7414a86f307a`;
- decoded CSV: 388,929 bytes;
- decoded CSV SHA-256: `7e6570382c620ce402bb0048336a319cd1a590bb08facb66bec3729c622de7e4`;
- rows: 1,342;
- populated test images: 178;
- sequential instance IDs: yes;
- within-image overlaps: none;
- two absent images: `20160513133714Th`, `20180507140210Th`;
- fallback mask: fixed `[1000:1048, 1000:1048]`, area 2,304 pixels.

After fallback insertion, the source-reconstructed 1,344-row CSV has SHA-256:

`0ab3932122a563ba96f4d243934d5c00680dd49b9d1165912a4f180907d2c603`

Authenticated `kaggle kernels output` returned a published `submission.csv` with exactly the same hash. This proves the published notebook output follows the payload path. It does not establish the payload's upstream origin.

The published CSV is structurally valid: all 180 images are represented, all RLEs decode, IDs are sequential, minimum area is 172, and masks are disjoint. Structural validity is not provenance or quality evidence.

## Relationship to V17

A strict one-to-one test-mask comparison against V17 found:

- V17 rows: 1,215;
- payload/fallback rows: 1,344;
- matches with IoU >0.5: 866;
- V17-only: 349;
- payload-only: 478;
- payload has more instances on 90 images, fewer on 43, equal on 47.

The payload is therefore not simply V17 with a lower confidence threshold. It is a materially different prediction set. Because test labels are unavailable, neither the unmatched masks nor higher row count can be interpreted as correct recall. The payload must not be used as pseudo-labels, a test-time selector, a target count, or an ensemble component.

## Research comparison

### Flat U-Net

Zhu et al., *Flat U-Net: An Efficient Ultralightweight Model for Solar Filament Segmentation in Full-disk H-alpha Images* (<https://arxiv.org/abs/2502.07259>) report that channel reconstruction/attention improves semantic recall over a classical U-Net with far fewer parameters. Their data are BBSO images with traditional-method binary masks, their metrics are semantic DSC/precision/recall, and their split is not the competition's instance-PQ protocol. The paper supports channel-attention as a semantic feature extractor, not this payload or an instance-separation recipe.

### EdgeAttNet

Solomon et al., *EdgeAttNet: Towards Barb-Aware Filament Segmentation* (<https://arxiv.org/abs/2509.02964>) use radial flattening, disk masking, Gaussian smoothing, CLAHE, and edge-guided bottleneck attention. They report semantic pairwise/multiscale IoU, explicitly excluding unmatched predictions from pairwise IoU. Those metrics do not price false positives and false negatives like PQ. Their complete MAGFILO data/weights must not be imported without proving they exclude competition test observations. A competition-linked EdgeAttNet implementation previously indexed around 0.25-0.26 public, so architecture novelty alone is not score evidence.

### Semantic-to-instance methods

Deep Watershed Transform and related embedding/distance-transform methods can convert semantic maps to instances, but elongated touching filaments are especially vulnerable to accidental splits/merges. This project already measured component splitting as negative and established that plain semantic/boundary improvements do not fix the dominant recognition wall. Any such method would need canonical global PQ and non-overlap evaluation, not Dice or foreground IoU.

## What is transferable

Already implemented here:

- COCO RLE encoding;
- strict disjoint pixel ownership;
- deterministic full test coverage checks;
- U-Net-style crop refinement;
- BCE/Dice-family losses and flip augmentation.

Potentially useful but **not validated by this notebook**:

- radial brightness flattening;
- competition-only CLAHE/local-contrast preprocessing;
- edge-aware or channel-attention features as an auxiliary signal.

The fixed fallback squares are not useful. They add guaranteed predictions with no image evidence and can only be justified by labels, which are unavailable.

## Recommended NEXT experiment

Do not train or submit this notebook. Do not use the payload.

If another GPU experiment is authorized, the only bounded transfer justified by the related papers is a **competition-only detector preprocessing ablation**, not a semantic U-Net submission:

- control: the current robust per-image p2-p98 normalization;
- arm: disk masking + radial-profile flattening + mild Gaussian smoothing + CLAHE, implemented from competition images only;
- unchanged detector architecture, split, annotations, schedule, mask ratio, inference, refiner, thresholds, and ownership;
- evaluate raw proposal recall and matched IoU first, especially the 212/1,134 instances missed by every existing YOLO pool;
- select by canonical tune PQ only and read audit once;
- require at least +0.01 tune PQ over V17 (0.448703), i.e. tune >=0.458703, with audit >=0.470763 before export;
- stop if raw recall does not materially improve; boundary/SQ-only movement is insufficient.

This is a low-prior experiment: Ultralytics already applies rare CLAHE augmentation, V20 used robust normalization and a semantic branch, public semantic U-Nets score poorly, and V22 showed that mask rendering without recognition improvement remains below V17. It is a legitimate final preprocessing test, not a likely explanation of the opaque 0.55 payload.
