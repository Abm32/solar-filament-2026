"""V32 contamination-safe launcher for the pinned Omnipose v1.0.6 oracle.

The launcher itself has no Torch/Omnipose dependency. It validates canonical
competition inputs and immutable upstream evidence, then starts the worker under
a separately supplied Python interpreter and untouched upstream source tree.
Audit/test/export/submission paths do not exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ANNOTATIONS = (
    ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/"
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)
ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
EXPECTED = {
    "all": {"files": 707, "records": 1154, "annotations": 8199},
    "fit": {"files": 566, "records": 937, "annotations": 6606},
    "tune": {"files": 94, "records": 150, "annotations": 1134},
    "audit": {"files": 47, "records": 67, "annotations": 459},
}
UPSTREAM_COMMIT = "21e4d0199cc03662669c150f6151e3365dc201fd"
UPSTREAM_HASHES = {
    "omnipose/core.py": "300f4cf24ad5256c35726442e024efa9e98cb27b200a3f36a6da01e3839ab332",
    "omnipose/cli.py": "b057bfc0985a483f3a08702f273bb5958d0363c140f668d9425ed8728bead3e8",
    "omnipose/utils.py": "d1b2b98344c6f26d96765548cd4e93b158355c021809b0c70fbb3c622834bea8",
    "dependencies.py": "0383b748e6bbae7808494f4f08b5a9d27caee6b9c42e58d61d9fe6587d3d8bb6",
    "LICENSE": "28b01fce69a552fef6035f0845599e4a53c80dfddcf5d275360565fe0d7d819c",
}
EVIDENCE_HASHES = {
    "mode_selection.json": "ca70e8194cf378af32f15fa96e3d198963d265b7ab07f9610c6586aff5182192",
    "synthetic_parity.json": "7484922c9074ca2587bd03548283376fd999c1447c896b96a7cf2fd94e8ee7ae",
    "upstream_provenance.json": "040df085ac5e830815f2be33131cfedcf2413af8a518a6d84cbe93edbfe287c0",
}
EVIDENCE_ROOT = ROOT / "outputs/v32_upstream_probe"
WORKER = ROOT / "scripts/v32_omnipose_worker.py"
FORBIDDEN = ("public_magfilo", "magfilo-v1-0-segmentation", "test_images", "test/test_images")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_annotation(path: Path) -> Path:
    value = path.expanduser().resolve()
    if any(token in str(value).lower() for token in FORBIDDEN):
        raise RuntimeError(f"V32 firewall rejected annotation path: {value}")
    if value != CANONICAL_ANNOTATIONS.resolve():
        raise RuntimeError("V32 accepts only canonical competition training annotations")
    if not value.is_file() or sha256(value) != ANNOTATION_SHA256:
        raise RuntimeError("canonical annotation identity changed")
    return value


def split_populations(path: Path) -> dict:
    import random
    document = json.loads(path.read_text())
    images = {str(row["id"]): row for row in document["images"]}
    annotations: dict[str, list] = defaultdict(list)
    for row in document["annotations"]:
        annotations[str(row["image_id"])].append(row)
    by_file: dict[str, list] = defaultdict(list)
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    files = sorted(by_file); random.Random(42).shuffle(files)
    groups = {"all": sorted(by_file), "fit": sorted(files[141:]),
              "tune": sorted(files[:94]), "audit": sorted(files[94:141])}
    populations = {}
    for group, names in groups.items():
        ids = [image_id for name in names for image_id in by_file[name]]
        populations[group] = {"files": len(names), "records": len(ids),
                              "annotations": sum(len(annotations[i]) for i in ids)}
    if populations != EXPECTED:
        raise RuntimeError(f"canonical split populations changed: {populations}")
    return populations


def verify_upstream(upstream: Path) -> dict:
    root = upstream.expanduser().resolve()
    for relative, expected in UPSTREAM_HASHES.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"pinned upstream file changed: {relative}")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot verify upstream git commit") from error
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit changed: {commit}")
    return {"root": str(root), "commit": commit, "hashes": UPSTREAM_HASHES}


def verify_evidence() -> dict:
    for name, expected in EVIDENCE_HASHES.items():
        path = EVIDENCE_ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"V32 upstream evidence changed: {name}")
    parity = json.loads((EVIDENCE_ROOT / "synthetic_parity.json").read_text())
    if not parity.get("foreground_equal") or parity.get("recon_labels") != 2:
        raise RuntimeError("upstream synthetic parity is not exact")
    mode = json.loads((EVIDENCE_ROOT / "mode_selection.json").read_text())
    selected = mode["summary"]["auto_adaptive"]
    if selected != {
        "mean_record_pq": 0.8953761538298969,
        "min_record_pq": 0.4039314073894031,
        "tp": 234, "fp": 44, "fn": 2,
        "perfect_records": 11, "mean_niter": 24.96153846153846,
    }:
        raise RuntimeError("published semantic-mode selection changed")
    return {"hashes": EVIDENCE_HASHES, "selected_mode": "auto_adaptive"}


def preflight(annotations: Path, upstream: Path, python: Path) -> dict:
    annotation_path = assert_annotation(annotations)
    populations = split_populations(annotation_path)
    upstream_report = verify_upstream(upstream)
    evidence = verify_evidence()
    interpreter = python.expanduser().resolve()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise RuntimeError(f"invalid V32 parity interpreter: {interpreter}")
    return {"annotation": str(annotation_path), "annotation_sha256": ANNOTATION_SHA256,
            "populations": populations, "upstream": upstream_report,
            "evidence": evidence, "python": str(interpreter)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=CANONICAL_ANNOTATIONS)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    report = preflight(args.annotations, args.upstream, args.python)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preflight_path = args.output.with_suffix(args.output.suffix + ".preflight.json")
    preflight_path.write_text(json.dumps(report, indent=2) + "\n")
    command = [str(args.python), str(WORKER), "--annotations", str(args.annotations),
               "--upstream", str(args.upstream), "--output", str(args.output),
               "--workers", str(args.workers)]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.upstream)
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
