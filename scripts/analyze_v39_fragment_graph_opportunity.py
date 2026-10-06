"""V39 tune-only latent-fragment graph-assembly opportunity audit.

This CPU-only gate measures whether GT instances missed by all six complete
proposal populations are nevertheless represented by multiple pure proposal
fragments that could support TARDIS-style graph assembly. It reads no images,
audit/test population, public Solar data, model, CSV, or submission path.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

try:
    from scripts import analyze_detector_complementarity as complementarity
    from scripts import v31_target_oracle as canonical
except ModuleNotFoundError:  # Direct execution from scripts/.
    import analyze_detector_complementarity as complementarity
    import v31_target_oracle as canonical

ROOT = Path(__file__).resolve().parents[1]
MIN_GT_COVERAGE = 0.05
MIN_PROPOSAL_PURITY = 0.50
MIN_FRAGMENT_MASKS = 2
MIN_FRAGMENT_SOURCES = 2
MIN_UNION_AREA_COVERAGE = 0.65
MIN_UNION_SPINE_COVERAGE = 0.65
MISS_COVERAGE_GATE = 0.65
RECOVERY_IOU = 0.65
PROJECTED_PQ_GATE = 0.50
MEDIAN_COVERAGE_GATE = 0.75
EXPECTED_SOURCE_MATCHES = (841, 794, 826, 884, 843, 893)
EXPECTED_UNION_MATCHES = 922
EXPECTED_UNION_MISSES = 212
V17_COUNTS = (481.00931292355114, 704, 306, 430)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rle_identity(rle: dict) -> tuple[tuple[int, int], bytes]:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    return tuple(int(value) for value in rle["size"]), bytes(counts)


def intersection_from_iou(iou: float, first_area: float, second_area: float) -> float:
    if not 0.0 <= iou <= 1.0 or first_area < 0 or second_area < 0:
        raise ValueError("invalid IoU or area")
    return iou * (first_area + second_area) / (1.0 + iou) if iou else 0.0


def intersection_area(first: dict, second: dict) -> float:
    iou = float(mask_utils.iou([first], [second], [0])[0, 0])
    return intersection_from_iou(
        iou, float(mask_utils.area(first)), float(mask_utils.area(second))
    )


def encode_mask(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))


def analyze_fragments(
    proposals_by_source: dict[str, list[dict]], gt_rle: dict, spine_mask: np.ndarray,
) -> dict:
    """Measure frozen admissible-fragment and oracle-union coverage for one GT."""
    gt_area = float(mask_utils.area(gt_rle))
    spine_rle = encode_mask(spine_mask)
    spine_area = float(mask_utils.area(spine_rle))
    if gt_area <= 0 or spine_area <= 0:
        raise ValueError("V39 requires nonempty GT and projected spine")

    fragments: dict[tuple[tuple[int, int], bytes], dict] = {}
    total_proposals = 0
    for source, rles in proposals_by_source.items():
        total_proposals += len(rles)
        if not rles:
            continue
        proposal_areas = np.asarray(mask_utils.area(rles), dtype=np.float64)
        ious = np.asarray(mask_utils.iou(rles, [gt_rle], [0]), dtype=np.float64)[:, 0]
        for rle, proposal_area, iou in zip(rles, proposal_areas, ious, strict=True):
            if proposal_area <= 0:
                continue
            intersection = intersection_from_iou(float(iou), float(proposal_area), gt_area)
            gt_coverage = intersection / gt_area
            proposal_purity = intersection / float(proposal_area)
            if gt_coverage < MIN_GT_COVERAGE or proposal_purity < MIN_PROPOSAL_PURITY:
                continue
            key = rle_identity(rle)
            if key not in fragments:
                fragments[key] = {
                    "rle": rle,
                    "sources": set(),
                    "area": float(proposal_area),
                    "gt_coverage": gt_coverage,
                    "proposal_purity": proposal_purity,
                    "individual_iou": float(iou),
                }
            fragments[key]["sources"].add(source)

    source_names = sorted({source for row in fragments.values() for source in row["sources"]})
    selected = [row["rle"] for row in fragments.values()]
    if selected:
        union_rle = mask_utils.merge(selected)
        area_coverage = intersection_area(union_rle, gt_rle) / gt_area
        spine_coverage = intersection_area(union_rle, spine_rle) / spine_area
        union_area = float(mask_utils.area(union_rle))
        union_iou = float(mask_utils.iou([union_rle], [gt_rle], [0])[0, 0])
    else:
        area_coverage = spine_coverage = union_area = union_iou = 0.0

    fragment_count = len(fragments)
    source_count = len(source_names)
    addressable = (
        fragment_count >= MIN_FRAGMENT_MASKS
        and source_count >= MIN_FRAGMENT_SOURCES
        and area_coverage >= MIN_UNION_AREA_COVERAGE
        and spine_coverage >= MIN_UNION_SPINE_COVERAGE
    )
    return {
        "total_grouped_proposals": total_proposals,
        "admissible_fragment_masks": fragment_count,
        "admissible_sources": source_count,
        "source_names": source_names,
        "union_area": union_area,
        "union_area_coverage": area_coverage,
        "union_spine_coverage": spine_coverage,
        "union_iou": union_iou,
        "max_individual_iou": max(
            (row["individual_iou"] for row in fragments.values()), default=0.0
        ),
        "max_individual_gt_coverage": max(
            (row["gt_coverage"] for row in fragments.values()), default=0.0
        ),
        "max_individual_purity": max(
            (row["proposal_purity"] for row in fragments.values()), default=0.0
        ),
        "graph_addressable": addressable,
    }


def projected_v17(recovered: int) -> dict:
    iou_sum, tp, fp, fn = V17_COUNTS
    if recovered < 0 or recovered > fn:
        raise ValueError("recovered count is outside the V17 false-negative budget")
    projected_iou = iou_sum + recovered * RECOVERY_IOU
    projected_tp = tp + recovered
    projected_fn = fn - recovered
    denominator = projected_tp + 0.5 * fp + 0.5 * projected_fn
    return {
        "assumed_recovered_iou": RECOVERY_IOU,
        "recovered": recovered,
        "iou_sum": projected_iou,
        "tp": projected_tp,
        "fp": fp,
        "fn": projected_fn,
        "pq": projected_iou / denominator if denominator else 0.0,
    }


def synthetic_checks() -> dict:
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5:15, 4:16] = 1
    left = np.zeros_like(gt)
    left[5:15, 4:11] = 1
    right = np.zeros_like(gt)
    right[5:15, 9:16] = 1
    duplicate = left.copy()
    contaminant = np.ones_like(gt)
    spine = np.zeros_like(gt)
    spine[10, 4:16] = 1
    rows = analyze_fragments(
        {
            "source_a": [encode_mask(left), encode_mask(contaminant)],
            "source_b": [encode_mask(right), encode_mask(duplicate)],
        },
        encode_mask(gt),
        spine,
    )
    checks = {
        "deduplicates_exact_masks": rows["admissible_fragment_masks"] == 2,
        "rejects_impure_contact": rows["total_grouped_proposals"] == 4,
        "requires_multiple_sources": rows["admissible_sources"] == 2,
        "area_union": math.isclose(rows["union_area_coverage"], 1.0),
        "spine_union": math.isclose(rows["union_spine_coverage"], 1.0),
        "addressable": rows["graph_addressable"],
    }
    checks["passed"] = all(checks.values())
    return checks


def grouped_proposals(sources: dict[str, dict[str, Path]], name: str) -> dict[str, list[dict]]:
    output = {}
    for source, paths in sources.items():
        views = complementarity.read_views(paths[name])
        rles, _scores = complementarity.cluster_views(views)
        output[source] = rles
    return output


def summarize_group(rows: list[dict]) -> dict:
    addressable = [row for row in rows if row["graph_addressable"]]
    return {
        "instances": len(rows),
        "graph_addressable": len(addressable),
        "graph_addressable_rate": len(addressable) / len(rows) if rows else 0.0,
        "median_union_area_coverage_addressable": (
            float(np.median([row["union_area_coverage"] for row in addressable]))
            if addressable else 0.0
        ),
        "median_union_spine_coverage_addressable": (
            float(np.median([row["union_spine_coverage"] for row in addressable]))
            if addressable else 0.0
        ),
        "median_admissible_fragments": (
            float(np.median([row["admissible_fragment_masks"] for row in rows]))
            if rows else 0.0
        ),
        "median_admissible_sources": (
            float(np.median([row["admissible_sources"] for row in rows]))
            if rows else 0.0
        ),
    }


def run(
    v16_manifest: Path, v16_cache: Path, v13_cache: Path, v17_cache: Path, output: Path,
) -> dict:
    runtime = canonical.load_population(canonical.CANONICAL_ANNOTATIONS)
    if runtime["populations"]["tune"] != canonical.EXPECTED_TUNE:
        raise RuntimeError("V39 tune population changed")
    manifest = json.loads(v16_manifest.read_text())
    names = sorted(manifest["tune_files"])
    if names != runtime["tune"]:
        raise RuntimeError("V39 manifest tune population differs from canonical tune")

    sources = complementarity.build_sources(ROOT, v16_cache, v13_cache, v17_cache, names)
    for label, paths in sources.items():
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{label}: missing {len(missing)} tune caches, e.g. {missing[:2]}"
            )

    match_rows, labels = complementarity.collect(
        sources, runtime["images"], runtime["annotations"], runtime["by_file"], set(names)
    )
    source_matches = tuple(
        sum(row[label] > complementarity.MATCH_IOU for row in match_rows)
        for label in labels
    )
    union_matches = sum(
        max(row[label] for label in labels) > complementarity.MATCH_IOU
        for row in match_rows
    )
    union_misses = len(match_rows) - union_matches
    regression = {
        "source_matches": source_matches == EXPECTED_SOURCE_MATCHES,
        "union_matches": union_matches == EXPECTED_UNION_MATCHES,
        "union_misses": union_misses == EXPECTED_UNION_MISSES,
    }
    if not all(regression.values()):
        raise RuntimeError(
            "V39 cache regression failed: "
            f"source={source_matches}, union={union_matches}/{union_misses}"
        )

    match_by_key = {
        (str(row["image_id"]), int(row["gt_index"])): row for row in match_rows
    }
    diagnostics = []
    for name in names:
        proposals = grouped_proposals(sources, name)
        for image_id in runtime["by_file"][name]:
            info = runtime["images"][image_id]
            for gt_index, annotation in enumerate(runtime["annotations"][image_id]):
                key = (str(image_id), gt_index)
                if key not in match_by_key:
                    raise RuntimeError(f"V39 match population missing {key}")
                gt_mask = canonical.annotation_mask(info, annotation)
                spine_mask, spine_report = canonical.project_spine(gt_mask, annotation["spine"])
                if not spine_report["valid"]:
                    raise RuntimeError(f"V39 invalid projected spine for {key}")
                metrics = analyze_fragments(
                    proposals, canonical.annotation_rle(info, annotation), spine_mask
                )
                match = match_by_key[key]
                found = max(match[label] for label in labels) > complementarity.MATCH_IOU
                diagnostics.append({
                    "file_name": name,
                    "image_id": str(image_id),
                    "gt_index": gt_index,
                    "detector_group": "found_by_any" if found else "missed_by_all",
                    **metrics,
                })

    if len(diagnostics) != canonical.EXPECTED_TUNE["annotations"]:
        raise RuntimeError(f"V39 diagnostic population changed: {len(diagnostics)}")
    diagnostics.sort(key=lambda row: (row["file_name"], row["image_id"], row["gt_index"]))
    grouped = {
        name: [row for row in diagnostics if row["detector_group"] == name]
        for name in ("missed_by_all", "found_by_any")
    }
    summaries = {name: summarize_group(rows) for name, rows in grouped.items()}
    missed = summaries["missed_by_all"]
    recovered = missed["graph_addressable"]
    projection = projected_v17(recovered)
    synthetic = synthetic_checks()
    finite_values = [
        missed["graph_addressable_rate"],
        missed["median_union_area_coverage_addressable"],
        missed["median_union_spine_coverage_addressable"],
        projection["pq"],
    ]
    gates = {
        "g0_synthetic": synthetic["passed"],
        "g0_cache_regression": all(regression.values()),
        "miss_coverage": missed["graph_addressable_rate"] >= MISS_COVERAGE_GATE,
        "projected_pq": projection["pq"] >= PROJECTED_PQ_GATE,
        "median_area_coverage": (
            missed["median_union_area_coverage_addressable"] >= MEDIAN_COVERAGE_GATE
        ),
        "median_spine_coverage": (
            missed["median_union_spine_coverage_addressable"] >= MEDIAN_COVERAGE_GATE
        ),
        "finite": all(math.isfinite(value) for value in finite_values),
        "firewall": True,
    }
    passed = all(gates.values())
    report = {
        "status": "complete",
        "experiment": "v39_fragment_graph_opportunity",
        "annotation_sha256": canonical.sha256(runtime["path"]),
        "population": runtime["populations"]["tune"],
        "constants": {
            "min_gt_coverage": MIN_GT_COVERAGE,
            "min_proposal_purity": MIN_PROPOSAL_PURITY,
            "min_fragment_masks": MIN_FRAGMENT_MASKS,
            "min_fragment_sources": MIN_FRAGMENT_SOURCES,
            "min_union_area_coverage": MIN_UNION_AREA_COVERAGE,
            "min_union_spine_coverage": MIN_UNION_SPINE_COVERAGE,
            "miss_coverage_gate": MISS_COVERAGE_GATE,
            "recovery_iou": RECOVERY_IOU,
            "projected_pq_gate": PROJECTED_PQ_GATE,
            "median_coverage_gate": MEDIAN_COVERAGE_GATE,
        },
        "synthetic_checks": synthetic,
        "cache_regression": {
            **regression,
            "actual_source_matches": list(source_matches),
            "actual_union_matches": union_matches,
            "actual_union_misses": union_misses,
        },
        "groups": summaries,
        "projected_v17": projection,
        "gates": gates,
        "g1_passed": passed,
        "g2_model_design_authorized": passed,
        "diagnostics": diagnostics,
        "reads_images": False,
        "reads_audit": False,
        "reads_test": False,
        "trains": False,
        "exports": False,
        "submits": False,
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v16-manifest", type=Path,
        default=ROOT / "outputs/kaggle_v16_full/v16_manifest.json",
    )
    parser.add_argument(
        "--v16-cache", type=Path,
        default=ROOT / "outputs/kaggle_v16_full/v16_proposal_cache",
    )
    parser.add_argument(
        "--v13-cache", type=Path,
        default=ROOT / "outputs/kaggle_v13_cache_pull/proposal_cache",
    )
    parser.add_argument(
        "--v17-cache", type=Path,
        default=ROOT / "outputs/kaggle_v17_cache_pull/proposal_cache",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/v39_fragment_graph_opportunity/result.json",
    )
    args = parser.parse_args()
    report = run(
        args.v16_manifest, args.v16_cache, args.v13_cache, args.v17_cache, args.output
    )
    print(json.dumps({
        "g1_passed": report["g1_passed"],
        "groups": report["groups"],
        "projected_v17": report["projected_v17"],
        "gates": report["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
