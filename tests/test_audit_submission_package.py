import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from scripts.audit_submission_package import audit_submission_package


def _make_package(tmp_path: Path) -> tuple[Path, set[str]]:
    package = tmp_path / "submission_ready" / "candidate"
    package.mkdir(parents=True)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))["counts"].decode("utf-8")
    csv_path = package / "submission.csv"
    pd.DataFrame(
        [{"filament_id": "test_1", "segmentation_rle": rle}]
    ).to_csv(csv_path, index=False)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (package / "SHA256SUMS").write_text(
        f"{digest}  submission_ready/candidate/submission.csv\n"
    )
    (package / "submission_receipt.json").write_text(
        json.dumps(
            {
                "submission_ref": "12345",
                "local_file": "submission.csv",
                "sha256": digest,
                "status": "SubmissionStatus.COMPLETE",
                "public_score": 0.39,
            }
        )
    )
    return package, {"test"}


def test_audit_submission_package_preserves_completed_valid_submission(tmp_path):
    package, test_stems = _make_package(tmp_path)

    report = audit_submission_package(
        package,
        test_stems,
        shape=(16, 16),
        expected_submission_ref="12345",
        expected_public_score=0.39,
    )

    assert report["package_valid"] is True
    assert report["already_submitted"] is True
    assert report["upload_recommended"] is False
    assert report["decision"] == "preserve_existing_submission_do_not_reupload"
    assert all(report["checks"].values())


def test_audit_submission_package_blocks_checksum_mismatch(tmp_path):
    package, test_stems = _make_package(tmp_path)
    receipt_path = package / "submission_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))

    report = audit_submission_package(package, test_stems, shape=(16, 16))

    assert report["package_valid"] is False
    assert report["upload_recommended"] is False
    assert report["decision"] == "blocked_package_integrity_failure"
    assert report["checks"]["receipt_sha_matches"] is False


def test_cli_help_runs_by_file_path():
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts/audit_submission_package.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Audit a submission package" in result.stdout
