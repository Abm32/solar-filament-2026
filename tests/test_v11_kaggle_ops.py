"""Protect against the installed Kaggle SDK's quota/version pitfalls."""
from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "v11_kaggle_ops", Path(__file__).resolve().parents[1] / "scripts/v11_kaggle_ops.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


def test_quota_retains_days_and_reservations():
    quota = SimpleNamespace(time_used=timedelta(hours=5), time_reserved=timedelta(minutes=30),
                            total_time_allowed=timedelta(days=1, hours=6))
    result = ops.quota_snapshot(SimpleNamespace(quota_refresh_time=datetime(2026, 9, 12),
                                               gpu_quota=quota, tpu_quota=quota))
    assert result["gpu_quota"]["total_time_allowed"] == 30 * 3600
    assert result["gpu_quota"]["remaining_seconds"] == 24.5 * 3600


@pytest.mark.parametrize("changed", [{"version": 2}, {"sha256": "changed"}, {"private": False},
                                     {"ref": "other/slug"}])
def test_changed_remote_run_is_rejected(changed):
    identity = {"ref": "owner/slug", "version": 1, "sha256": "hash", "private": True}
    identity.update(changed)
    with pytest.raises(ValueError):
        ops.verify_identity(identity, "owner/slug", "hash", {"version": 1})


@pytest.mark.parametrize("returned", [
    "/code/abhimanyu122/solar-filament-v11-refiner-smoke",
    "code/abhimanyu122/solar-filament-v11-refiner-smoke",
    "https://www.kaggle.com/code/abhimanyu122/solar-filament-v11-refiner-smoke",
    "abhimanyu122/solar-filament-v11-refiner-smoke",
])
def test_push_ref_forms_normalize_to_the_bare_ref(returned):
    # The push endpoint returns a site-relative path while every other endpoint
    # returns the bare ref; a successful launch must not be reported as failed.
    assert ops.normalize_ref(returned) == "abhimanyu122/solar-filament-v11-refiner-smoke"


def test_normalize_ref_still_distinguishes_a_different_slug():
    assert ops.normalize_ref("/code/abhimanyu122/other-notebook") != \
        "abhimanyu122/solar-filament-v11-refiner-smoke"


def test_uncertain_push_is_not_automatically_retried(tmp_path, monkeypatch):
    ref = "abhimanyu122/solar-filament-v11-refiner-smoke"
    (tmp_path / "kernel-metadata.json").write_text(json.dumps(
        {"id": ref, "code_file": "run.py", "is_private": True, "enable_gpu": True}))
    (tmp_path / "run.py").write_text("print('smoke')\n")
    quota = SimpleNamespace(time_used=timedelta(), time_reserved=timedelta(),
                            total_time_allowed=timedelta(days=1, hours=6))
    response = SimpleNamespace(quota_refresh_time=datetime(2026, 9, 12),
                               gpu_quota=quota, tpu_quota=quota)
    attempts = []

    def push(*args, **kwargs):
        assert json.loads((tmp_path / "kaggle_launch_receipt.json").read_text())["state"] == "attempt_started"
        attempts.append(True)
        raise RuntimeError("Network disconnected after request; server result unknown")

    api = SimpleNamespace(authenticate=lambda: None, quota_view=lambda: response, kernels_push=push)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended",
                        SimpleNamespace(KaggleApi=lambda: api))
    monkeypatch.setattr(ops, "own_refs", lambda _: set())
    monkeypatch.setattr(sys, "argv", ["ops", "launch", "--stage", str(tmp_path), "--timeout", "300"])
    with pytest.raises(RuntimeError):
        ops.main()
    with pytest.raises(FileExistsError):
        ops.main()
    assert len(attempts) == 1
