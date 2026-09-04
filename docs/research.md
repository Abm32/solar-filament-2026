# Research Notes

Last updated: 2026-09-04.

## Competition Facts

- Competition: Kaggle Solar Filament Segmentation Challenge 2026.
- Data source: GONG H-alpha observations with MAGFiLO filament annotations.
- Submission: one CSV row per predicted filament instance.
- Image/mask size: 2048 x 2048.
- Metric: Panoptic Quality, with additional review of Dice, IoU, one-to-many, and many-to-one relations.
- Final report and public Git repository are required for final evaluation.

## Downloaded Kaggle Data Snapshot

- Train image files: 707.
- Test image files: 180.
- COCO image entries: 1,154.
- Unique COCO file names: 707.
- Annotation entries: 8,199.
- Files with duplicate annotator labels: 296.
- Categories: Left, Right, Unidentifiable, Ambiguous.

## What Matters For This Metric

Panoptic Quality rewards instance-level matching, not just foreground pixels. A high-scoring method needs:

- strong filament foreground detection;
- correct grouping of connected filament structures;
- fewer false small dark-region detections;
- fewer merged neighboring filaments;
- preserved fine structures such as barbs.

Plain semantic segmentation can produce visually plausible masks while losing PQ because connected components can over-split or over-merge instances.

## Public Method Signals

From public Kaggle notebook/search snippets and challenge notes, the strongest visible pattern is:

- detection or proposal stage to identify candidate filament regions;
- U-Net-like segmentation refinement per candidate or tile;
- topology-aware post-processing to repair fragmentation and separate nearby structures;
- CPU/GPU-efficient inference on 2048 x 2048 full-disk images.

Specific public signal found on 2026-09-04:

- Kaggle notebook `ektarr/yolo-u-net-solar-filament-segmentation` reports a 0.69 public score and a 33m35s T4 x2 runtime.
- The organizer's self-evaluation notebook should be pulled and treated as the local scoring reference.

Lower-risk baselines include adaptive thresholding and morphology, but these will likely plateau because background solar texture and limb artifacts produce many dark false positives.

## Initial Conclusion

Build a hybrid pipeline:

1. Classical preprocessing to normalize limb darkening and contrast.
2. Candidate generation from dark elongated structures.
3. YOLOv11 segmentation model trained on polygon annotations at 1024px.
4. Instance reconstruction with connected components, skeleton/geodesic cues, and size/shape filters.
5. Submission encoding with `pycocotools` compressed RLE.

This is more aligned with Panoptic Quality than a single full-image binary U-Net.

## Notebook Code Review

`ektarr/yolo-u-net-solar-filament-segmentation` uses:

- `yolo11s-seg.pt`, `imgsz=1024`, `batch=4`, `epochs=30`, `mask_ratio=2`;
- category-preserving COCO polygon to YOLO-seg conversion;
- one selected annotation set per repeated image, choosing the one with most annotations and largest area;
- `overlap_mask=False`, disabled flips, controlled rotation/scale/value augmentation;
- confidence threshold `0.25` and minimum mask area `20`;
- a crop refiner: ResNet18 encoder U-Net trained on per-filament crops, thresholded at `0.65`.

This looks like the best public starting point because it directly optimizes instance proposals, then improves boundaries locally.

## First Experiments

1. Validate data layout and mask decoding.
2. Submit a classical baseline to verify RLE and instance rows.
3. Train `smp.UnetPlusPlus` or `smp.DeepLabV3Plus` with an EfficientNet encoder on 512 or 768 crops.
4. Add test-time augmentation only after baseline correctness is proven.
5. Add an instance separation pass for touching/nearby filaments.

## Known Constraints

- Current VM reports no available NVIDIA driver via `nvidia-smi`.
- GitHub CLI token is invalid, so remote repository creation/push requires re-authentication.
- Kaggle download requires accepted rules and a valid Kaggle token.

## Sources

- Kaggle overview/evaluation/prizes pages for current rules, dates, submission format, and scoring.
- MAGFiLO Scientific Data paper and MAGFiLO project page for dataset format and scale.
- Kirillov et al., CVPR 2019, for the Panoptic Quality definition.
- Public Kaggle notebook page for YOLO + U-Net score/runtime signal.
