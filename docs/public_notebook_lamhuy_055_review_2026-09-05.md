# Public Notebook Review - lamhuy8904 0.55

Reviewed at `2026-09-05 00:19 IST`.

Notebook URL: `https://www.kaggle.com/code/lamhuy8904/solar-filament-unet-segmentation-0-55`

Local research copy (not committed or republished):
`references/notebooks/lamhuy_unet_055/solar-filament-unet-segmentation-0-55.ipynb`.
The review and original source URL are retained; the third-party embedded payload
is not part of our implementation or GitHub backup.

## Summary

The notebook title and markdown describe an end-to-end U-Net segmentation pipeline, but the submitted result is not produced by that trained U-Net.

The notebook trains a small grayscale U-Net demo for only `5` epochs on the first `60` COCO image ids. That model is not used for the final high-scoring submission. The final submission is generated from an embedded compressed CSV named `CHAMPION_PAYLOAD`, decoded with:

```python
raw_bytes = lzma.decompress(base64.b85decode(CHAMPION_PAYLOAD.encode("ascii")))
champion_df = pd.read_csv(io.BytesIO(raw_bytes))
```

The decoded dataframe already contains `filament_id` and `segmentation_rle` rows. The final loop matches those rows to test image stems and writes them to `/kaggle/working/submission.csv`. For test images not present in the payload, it writes a tiny `48 x 48` fallback square mask near the image center.

## Payload Statistics

Decoded payload statistics:

- Rows: `1,342`
- Unique populated images: `178`
- Min instances per populated image: `1`
- Median instances per populated image: `7.0`
- Mean instances per populated image: `7.54`
- Max instances per populated image: `18`
- Columns: `filament_id`, `segmentation_rle`

Mask area statistics for payload rows:

- Mean area: `2,148.8`
- Median area: `1,294`
- P10 area: `478.4`
- P90 area: `4,543.7`
- Min area: `172`
- Max area: `25,088`

For comparison, our V6 submission:

- Rows: `1,094`
- Unique populated images: `176`
- Median instances per populated image: `5.5`
- Mean mask area: `2,171.4`
- Median mask area: `1,429`
- Public score: `0.35`

The 0.55 payload is not winning because masks are dramatically larger or smaller. It is mainly predicting more valid-looking filament instances across more images.

## Comparison Against V6

Payload versus our V6 row counts by image:

- Leader-only populated images: `4`
- V6-only populated images: `2`
- Images where payload predicts more instances: `109`
- Images where V6 predicts more instances: `43`
- Images with the same count: `28`
- Mean per-image count delta, payload minus V6: `+1.38`
- Maximum per-image count delta: `+12`

Largest payload-minus-V6 gaps:

| Stem | Payload Count | V6 Count | Delta |
| --- | ---: | ---: | ---: |
| `20120710063134Lh` | 17 | 5 | +12 |
| `20130310063154Uh` | 16 | 4 | +12 |
| `20120512063234Lh` | 17 | 9 | +8 |
| `20120402105814Th` | 12 | 4 | +8 |
| `20121220110614Th` | 12 | 4 | +8 |
| `20121203063334Lh` | 14 | 6 | +8 |
| `20150602122934Ch` | 13 | 5 | +8 |

## What Is Legitimately Useful

Do not copy or submit the embedded payload. It is a precomputed submission table, not a reproducible model improvement. It may also be based on public test-set overlap or another non-generalizable shortcut. The competition's final review includes method quality and documentation, so copying a payload would be a poor technical and judging strategy.

Useful signals for our own model:

1. V6 is under-recalling relative to a 0.55 public-score file. V6 has only `1,094` rows; the payload has `1,342` real rows.
2. The target mask area distribution is not far from ours. We should avoid area-expanding masks indiscriminately.
3. Candidate selection should prefer a recall-recovering candidate when validation PQ is close to the best candidate.
4. Full test-image coverage may help if missing images contain real filaments, but tiny fallback masks are not a principled fix and can add false positives.
5. The most actionable next move is to use V7's candidate outputs to find a submission with approximately V4/payload-like row count, around `1,250-1,400`, while keeping validation PQ close to best and precision acceptable.

## Next Decision Rule

When V7 completes:

1. Pull outputs to `outputs/kaggle_kernel_v7`.
2. Run `scripts/analyze_kaggle_outputs.py outputs/kaggle_kernel_v7`.
3. Prefer the top validation-PQ candidate only if it is not as conservative as V6.
4. If the top candidate has near-V6 row count, consider a near-best recall candidate with:
   - validation PQ within about `0.02-0.03` of best;
   - meaningfully higher recall;
   - test rows closer to `1,250-1,400`;
   - no extreme false-positive increase.
5. Spend the final daily submission only after comparing V7 candidates against V6's public score baseline of `0.35` and validation PQ baseline of `0.426821`.
