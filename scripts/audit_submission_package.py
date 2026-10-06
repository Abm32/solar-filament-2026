"""Audit a submission package and prevent accidental duplicate uploads."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_submission import validate_submission


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sha256sums(path: Path) -> dict[str, str]:
    """Parse a conventional SHA256SUMS file."""
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Malformed SHA256SUMS line {line_number}: {raw_line!r}")
        digest, name = parts
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(
                f"Malformed SHA256 digest on line {line_number}: {digest!r}"
            ) from exc
        checksums[name.lstrip("*")] = digest.lower()
    if not checksums:
        raise ValueError("SHA256SUMS contains no entries")
    return checksums


def _checksum_for_local_file(checksums: dict[str, str], local_file: str) -> str:
    matches = [digest for name, digest in checksums.items() if Path(name).name == local_file]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one SHA256SUMS entry for {local_file}, found {len(matches)}"
        )
    return matches[0]


def audit_submission_package(
    package_dir: Path,
    test_stems: set[str],
    *,
    shape: tuple[int, int] = (2048, 2048),
    min_area: int = 20,
    expected_submission_ref: str | None = None,
    expected_public_score: float | None = None,
) -> dict:
    """Validate package contents, receipt identity, and upload disposition."""
    package_dir = package_dir.resolve()
    receipt_path = package_dir / "submission_receipt.json"
    sums_path = package_dir / "SHA256SUMS"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Missing submission receipt: {receipt_path}")
    if not sums_path.is_file():
        raise FileNotFoundError(f"Missing checksum manifest: {sums_path}")

    receipt = json.loads(receipt_path.read_text())
    local_file = receipt.get("local_file")
    if not isinstance(local_file, str) or not local_file:
        raise ValueError("Receipt has no valid local_file")
    csv_path = package_dir / local_file
    if not csv_path.is_file():
        raise FileNotFoundError(f"Receipt CSV does not exist: {csv_path}")

    validation = validate_submission(csv_path, test_stems, shape=shape, min_area=min_area)
    actual_sha = sha256_file(csv_path)
    receipt_sha = str(receipt.get("sha256", "")).lower()
    manifest_sha = _checksum_for_local_file(load_sha256sums(sums_path), local_file)
    status = str(receipt.get("status", ""))
    submission_ref = str(receipt.get("submission_ref", ""))
    public_score = receipt.get("public_score")

    checks = {
        "decoded_csv_sha_matches_streamed_sha": validation["sha256"] == actual_sha,
        "receipt_sha_matches": receipt_sha == actual_sha,
        "manifest_sha_matches": manifest_sha == actual_sha,
        "receipt_status_complete": status.upper().endswith("COMPLETE"),
        "receipt_has_submission_ref": bool(submission_ref),
        "expected_submission_ref_matches": (
            expected_submission_ref is None or submission_ref == expected_submission_ref
        ),
        "expected_public_score_matches": (
            expected_public_score is None
            or public_score is not None
            and abs(float(public_score) - expected_public_score) < 1e-12
        ),
    }
    package_valid = all(checks.values())
    already_submitted = checks["receipt_status_complete"] and checks["receipt_has_submission_ref"]

    if not package_valid:
        decision = "blocked_package_integrity_failure"
    elif already_submitted:
        decision = "preserve_existing_submission_do_not_reupload"
    else:
        decision = "validated_not_yet_submitted"

    return {
        "package_dir": str(package_dir),
        "csv": str(csv_path),
        "sha256": actual_sha,
        "submission_ref": submission_ref or None,
        "public_score": public_score,
        "checks": checks,
        "package_valid": package_valid,
        "already_submitted": already_submitted,
        "upload_recommended": package_valid and not already_submitted,
        "decision": decision,
        "validation": validation,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-submission-ref")
    parser.add_argument("--expected-public-score", type=float)
    args = parser.parse_args()

    test_dir = args.test_dir or next((root / "data/raw").rglob("test_images"))
    test_stems = {path.stem for path in test_dir.glob("*.jpeg")}
    if len(test_stems) != 180:
        raise ValueError(f"Expected 180 test images, found {len(test_stems)}")

    report = audit_submission_package(
        args.package_dir,
        test_stems,
        expected_submission_ref=args.expected_submission_ref,
        expected_public_score=args.expected_public_score,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text)
    if not report["package_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
