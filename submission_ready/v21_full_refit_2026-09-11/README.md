# V21 full-data refit submission package

V21 is the final all-data refit of the frozen V17 recipe. It uses all approved
competition labels: 707 physical files, 1,154 annotator records, and 8,199
annotations. Since all labels were fitted, this candidate has no held-out local
PQ claim.

## Frozen inference recipe

- Detector: full-data YOLO11s-seg, `mask_ratio=1`
- Detector SHA-256: `5cc803e4f3ca334c471fc1012ef1e27a142b59b085831fc09d5e35d621ca5629`
- Refiner: full-data proposal-centered ResNet18 crop U-Net
- Refiner SHA-256: `26013cec76b7c329f9891fdbf90def19abc60a696d134273837cf5dc3f5f0792`
- Box IoU: 0.3; detector floor: 0.1; retina masks; agnostic NMS
- Four detector views with flip-consensus
- Refinement threshold: 0.45; confidence: 0.30
- Four crop-TTA views; constrained refinement; minimum area 20

## Export

Inference ran as four deterministic 45-image shards, one per Azure T4. All four
finished in 72–77 seconds. Their disjoint filename sets and CSV hashes are pinned
under `shards/`; the merged CSV follows canonical test filename order.

- CSV: `submission_v21.csv`
- Rows: 1,091
- Populated images: 174 of 180
- Median instances per populated image: 6
- Mask area min / median / max: 20 / 1,344 / 22,876 pixels
- CSV SHA-256: `98f58d2e115ebb6ae9c4f270aa4f03e46d75c279ad6a5c5c09afefb9121e9016`

Both the repository validator and an independent full decoder passed: known and
sequential IDs, all 180 images accounted for, every RLE decodes to 2048x2048,
area >=20, and masks are disjoint within each image. `SHA256SUMS` verifies every
package artifact.

## Kaggle result

Submitted once as Kaggle submission **56170647** at 2026-09-11 18:06:29 UTC
(23:36:29 IST). Scoring completed at the 2026-09-11 18:10 UTC observation.

- Public score: **0.38**
- Outcome: below V17's 0.39; V17 remains the public best
- Duplicate uploads: none
- Receipt: `submission_receipt.json`
