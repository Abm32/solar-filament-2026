"""V38 tune-only orientation-to-overpass opportunity audit.

This CPU-only gate attributes the 212 GT instances missed by all six complete
V6/V16/V17 proposal populations to frozen local spine geometry. It never opens
image pixels, audit/test populations, trains, exports, or submits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from scripts import analyze_detector_complementarity as complementarity
    from scripts import v31_target_oracle as canonical
except ModuleNotFoundError:  # Direct execution from scripts/.
    import analyze_detector_complementarity as complementarity
    import v31_target_oracle as canonical

ROOT = Path(__file__).resolve().parents[1]
PROXIMITY_RADIUS = 16.0
ORIENTATION_BIN_DEGREES = 30.0
TANGENT_RADIUS = 3
RECOVERY_IOU = 0.65
MISS_COVERAGE_GATE = 0.65
ENRICHMENT_GATE = 1.25
PROJECTED_PQ_GATE = 0.50
EXPECTED_SOURCE_MATCHES = (841, 794, 826, 884, 843, 893)
EXPECTED_UNION_MATCHES = 922
EXPECTED_UNION_MISSES = 212
V17_COUNTS = (481.00931292355114, 704, 306, 430)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def local_tangent(points_xy: np.ndarray, index: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("a tangent needs at least two points")
    for radius in range(TANGENT_RADIUS, len(points)):
        low, high = max(0, index - radius), min(len(points) - 1, index + radius)
        delta = points[high] - points[low]
        norm = float(np.linalg.norm(delta))
        if norm > 0:
            return delta / norm
        if low == 0 and high == len(points) - 1:
            break
    raise ValueError("polyline has no nonzero tangent")


def unoriented_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def pair_geometry(first_xy: np.ndarray, second_xy: np.ndarray) -> dict:
    first = np.asarray(first_xy, dtype=np.float64)
    second = np.asarray(second_xy, dtype=np.float64)
    if len(first) < 2 or len(second) < 2:
        raise ValueError("pair geometry needs two nondegenerate spines")
    distances, indices = cKDTree(second).query(first, k=1)
    first_index = int(np.argmin(distances))
    second_index = int(indices[first_index])
    nearest_distance = float(distances[first_index])
    angle = unoriented_angle_degrees(
        local_tangent(first, first_index), local_tangent(second, second_index)
    )
    endpoint_distance = min(
        float(np.linalg.norm(a - b))
        for a in (first[0], first[-1]) for b in (second[0], second[-1])
    )
    proximity = nearest_distance <= PROXIMITY_RADIUS
    terminus = endpoint_distance <= PROXIMITY_RADIUS
    parallel = proximity and angle < ORIENTATION_BIN_DEGREES
    crossing_like = proximity and angle >= ORIENTATION_BIN_DEGREES
    return {
        "nearest_distance": nearest_distance,
        "angle_degrees": angle,
        "endpoint_distance": endpoint_distance,
        "proximity": proximity,
        "parallel": parallel,
        "crossing_like": crossing_like,
        "terminus": terminus,
        "addressable": proximity or terminus,
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
    horizontal = np.asarray([[x, 10] for x in range(21)], dtype=np.float64)
    parallel = np.asarray([[x, 14] for x in range(21)], dtype=np.float64)
    vertical = np.asarray([[10, y] for y in range(21)], dtype=np.float64)
    far = np.asarray([[x + 100, 10] for x in range(21)], dtype=np.float64)
    same = pair_geometry(horizontal, parallel)
    cross = pair_geometry(horizontal, vertical)
    distant = pair_geometry(horizontal, far)
    reverse = pair_geometry(horizontal[::-1].copy(), parallel[::-1].copy())
    checks = {
        "parallel": same["parallel"] and not same["crossing_like"],
        "crossing": cross["crossing_like"] and not cross["parallel"],
        "far_rejected": not distant["addressable"],
        "order_invariant": (
            math.isclose(same["nearest_distance"], reverse["nearest_distance"])
            and math.isclose(same["angle_degrees"], reverse["angle_degrees"])
            and same["addressable"] == reverse["addressable"]
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def geometry_by_instance(runtime: dict) -> dict[tuple[str, int], dict]:
    output: dict[tuple[str, int], dict] = {}
    for name in runtime["tune"]:
        for image_id in runtime["by_file"][name]:
            entries = runtime["annotations"][image_id]
            spines = [
                canonical.densify_polyline(entry["spine"]).astype(np.float64)
                for entry in entries
            ]
            per_instance = [
                {
                    "addressable": False,
                    "proximity": False,
                    "parallel": False,
                    "crossing_like": False,
                    "terminus": False,
                    "nearest_distance": None,
                    "nearest_angle_degrees": None,
                    "nearest_endpoint_distance": None,
                }
                for _ in entries
            ]
            for first_index in range(len(spines)):
                for second_index in range(first_index + 1, len(spines)):
                    pair = pair_geometry(spines[first_index], spines[second_index])
                    for index in (first_index, second_index):
                        row = per_instance[index]
                        row["addressable"] |= pair["addressable"]
                        for key in ("proximity", "parallel", "crossing_like", "terminus"):
                            row[key] |= pair[key]
                        if (
                            row["nearest_distance"] is None
                            or pair["nearest_distance"] < row["nearest_distance"]
                        ):
                            row["nearest_distance"] = pair["nearest_distance"]
                            row["nearest_angle_degrees"] = pair["angle_degrees"]
                        if (
                            row["nearest_endpoint_distance"] is None
                            or pair["endpoint_distance"] < row["nearest_endpoint_distance"]
                        ):
                            row["nearest_endpoint_distance"] = pair["endpoint_distance"]
            for index, row in enumerate(per_instance):
                output[(str(image_id), index)] = row
    return output


def run(
    v16_manifest: Path, v16_cache: Path, v13_cache: Path, v17_cache: Path, output: Path,
) -> dict:
    runtime = canonical.load_population(canonical.CANONICAL_ANNOTATIONS)
    if runtime["populations"]["tune"] != canonical.EXPECTED_TUNE:
        raise RuntimeError("V38 tune population changed")
    manifest = json.loads(v16_manifest.read_text())
    names = sorted(manifest["tune_files"])
    if names != runtime["tune"]:
        raise RuntimeError("V38 manifest tune population differs from canonical tune")

    images = {key: value for key, value in runtime["images"].items()}
    annotations = runtime["annotations"]
    by_file = runtime["by_file"]
    sources = complementarity.build_sources(
        ROOT, v16_cache, v13_cache, v17_cache, names
    )
    for label, paths in sources.items():
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{label}: missing {len(missing)} tune caches, e.g. {missing[:2]}"
            )

    match_rows, labels = complementarity.collect(
        sources, images, annotations, by_file, set(names)
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
            "V38 cache regression failed: "
            f"source={source_matches}, union={union_matches}/{union_misses}"
        )

    geometry = geometry_by_instance(runtime)
    keys = {(str(row["image_id"]), int(row["gt_index"])) for row in match_rows}
    if keys != set(geometry):
        raise RuntimeError("V38 geometry and detector-match populations do not align")

    groups = {"missed_by_all": [], "found_by_any": []}
    diagnostics = []
    for row in match_rows:
        key = (str(row["image_id"]), int(row["gt_index"]))
        found = max(row[label] for label in labels) > complementarity.MATCH_IOU
        group = "found_by_any" if found else "missed_by_all"
        item = geometry[key]
        groups[group].append(item)
        diagnostics.append({
            "file_name": row["file_name"], "image_id": key[0], "gt_index": key[1],
            "detector_group": group, **item,
        })

    summaries = {}
    for group, rows in groups.items():
        summaries[group] = {
            "instances": len(rows),
            "addressable": sum(row["addressable"] for row in rows),
            "addressable_rate": (
                sum(row["addressable"] for row in rows) / len(rows) if rows else 0.0
            ),
            "proximity": sum(row["proximity"] for row in rows),
            "parallel": sum(row["parallel"] for row in rows),
            "crossing_like": sum(row["crossing_like"] for row in rows),
            "terminus": sum(row["terminus"] for row in rows),
        }
    missed_rate = summaries["missed_by_all"]["addressable_rate"]
    found_rate = summaries["found_by_any"]["addressable_rate"]
    enrichment = (
        missed_rate / found_rate if found_rate else (math.inf if missed_rate else 0.0)
    )
    recovered = summaries["missed_by_all"]["addressable"]
    projection = projected_v17(recovered)
    synthetic = synthetic_checks()
    gates = {
        "g0_synthetic": synthetic["passed"],
        "g0_cache_regression": all(regression.values()),
        "miss_coverage": missed_rate >= MISS_COVERAGE_GATE,
        "enrichment": enrichment >= ENRICHMENT_GATE,
        "projected_pq": projection["pq"] >= PROJECTED_PQ_GATE,
        "finite": all(math.isfinite(value) for value in (
            missed_rate, found_rate, enrichment, projection["pq"]
        )),
        "firewall": True,
    }
    passed = all(gates.values())
    diagnostics.sort(key=lambda row: (row["file_name"], row["image_id"], row["gt_index"]))
    report = {
        "status": "complete", "experiment": "v38_orientation_opportunity",
        "annotation_sha256": canonical.sha256(runtime["path"]),
        "population": runtime["populations"]["tune"],
        "constants": {
            "proximity_radius": PROXIMITY_RADIUS,
            "orientation_bin_degrees": ORIENTATION_BIN_DEGREES,
            "recovery_iou": RECOVERY_IOU,
            "miss_coverage_gate": MISS_COVERAGE_GATE,
            "enrichment_gate": ENRICHMENT_GATE,
            "projected_pq_gate": PROJECTED_PQ_GATE,
        },
        "synthetic_checks": synthetic,
        "cache_regression": {**regression, "actual_source_matches": list(source_matches),
                             "actual_union_matches": union_matches,
                             "actual_union_misses": union_misses},
        "groups": summaries,
        "addressable_enrichment": enrichment,
        "projected_v17": projection,
        "gates": gates,
        "g1_passed": passed,
        "g2_model_design_authorized": passed,
        "diagnostics": diagnostics,
        "reads_images": False, "reads_audit": False, "reads_test": False,
        "trains": False, "exports": False, "submits": False,
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16-manifest", type=Path,
                        default=ROOT / "outputs/kaggle_v16_full/v16_manifest.json")
    parser.add_argument("--v16-cache", type=Path,
                        default=ROOT / "outputs/kaggle_v16_full/v16_proposal_cache")
    parser.add_argument("--v13-cache", type=Path,
                        default=ROOT / "outputs/kaggle_v13_cache_pull/proposal_cache")
    parser.add_argument("--v17-cache", type=Path,
                        default=ROOT / "outputs/kaggle_v17_cache_pull/proposal_cache")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/v38_orientation_opportunity/result.json")
    args = parser.parse_args()
    report = run(args.v16_manifest, args.v16_cache, args.v13_cache, args.v17_cache, args.output)
    print(json.dumps({
        "g1_passed": report["g1_passed"],
        "groups": report["groups"],
        "addressable_enrichment": report["addressable_enrichment"],
        "projected_v17": report["projected_v17"],
        "gates": report["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
