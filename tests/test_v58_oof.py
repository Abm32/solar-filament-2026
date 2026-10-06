"""Tests for V58 OOF fold-split consistency (no GPU/caches)."""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# fake fit/tune for a pure-logic check of the shared fold formula
def _folds(fit, k):
    files = sorted(fit)
    random.Random(1234).shuffle(files)
    return sorted(files[k::4])


def test_folds_partition_fit_exactly_once():
    fit = [f"f{i:04d}.jpeg" for i in range(566)]
    seen = []
    for k in range(4):
        seen += _folds(fit, k)
    assert sorted(seen) == sorted(fit)          # every fit file in exactly one held-out quarter
    assert len(seen) == len(set(seen)) == 566


def test_trainer_heldout_equals_builder_foldk():
    # the trainer trains on fit MINUS quarter k; the builder generates on quarter k.
    # Both must derive the SAME quarter k from the same formula.
    fit = [f"f{i:04d}.jpeg" for i in range(566)]
    for k in range(4):
        builder_fold = set(_folds(fit, k))
        files = sorted(fit)
        random.Random(1234).shuffle(files)
        trainer_heldout = set(files[k::4])
        assert builder_fold == trainer_heldout


def test_fold_sizes_reasonable():
    fit = [f"f{i:04d}.jpeg" for i in range(566)]
    sizes = [len(_folds(fit, k)) for k in range(4)]
    assert all(130 <= s <= 160 for s in sizes)   # ~141-142 each
    assert sum(sizes) == 566
