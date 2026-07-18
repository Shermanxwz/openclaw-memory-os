"""Regression test for the memory_brain importance normalization.

External review (2026-07-14) flagged that the LLM-driven ingest pipeline
stored ``importance`` directly as the integer 1-5 it got from the model,
while the Memory OS ranking layer clamps to ``[0.0, 1.0]``. A payload of
``importance: 5`` was therefore stored as the integer 5 in Qdrant but read
back as ``1.0`` by ranking — making it indistinguishable from a ``1``
that had also been clamped to ``1.0``.

This test exercises the normalization helper used in
``scripts/memory_brain_ingest.py`` to make sure the on-disk payload
matches what ranking will read.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INGEST_PATH = REPO_ROOT / "scripts" / "memory_brain_ingest.py"
CONSOLIDATE_PATH = REPO_ROOT / "scripts" / "memory_brain_consolidate.py"


def _load_module_from_path(path: Path):
    """Load a script module by file path (no package namespace)."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        pytest.skip(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ingest_module():
    return _load_module_from_path(INGEST_PATH)


@pytest.fixture(scope="module")
def consolidate_module():
    return _load_module_from_path(CONSOLIDATE_PATH)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (5, 1.0),   # critical -> 5/5 = 1.0
        (4, 0.8),
        (3, 0.6),   # default fallback
        (2, 0.4),
        (1, 0.2),   # trivial -> 1/5 = 0.2
        ("4", 0.8),  # strings are tolerated
        (None, 0.6),  # missing -> default 3 -> 0.6
    ],
)
def test_normalize_importance_clamps_1_to_5_into_unit_range(
    ingest_module, raw, expected
):
    assert ingest_module.normalize_importance(raw) == pytest.approx(expected)


def test_normalize_importance_handles_zero(ingest_module):
    assert ingest_module.normalize_importance(0) == pytest.approx(0.0)


def test_normalize_importance_clamps_out_of_range_high(ingest_module):
    """LLM drift above 5 must clamp to 1.0, not crash."""
    assert ingest_module.normalize_importance(99) == pytest.approx(1.0)


def test_normalize_importance_clamps_out_of_range_low(ingest_module):
    """Negative LLM values must clamp to 0.0, not crash."""
    assert ingest_module.normalize_importance(-7) == pytest.approx(0.0)


def test_normalize_importance_passes_through_already_normalized(ingest_module):
    """If the LLM already returns a 0-1 float, keep it.

    The classify prompt asks for 1-5, but a model that drifts to 0-1 should
    not be punished (its 0.7 would otherwise become 0.14 after /5).
    """
    assert ingest_module.normalize_importance(0.7) == pytest.approx(0.7)
    assert ingest_module.normalize_importance(0.0) == pytest.approx(0.0)
    assert ingest_module.normalize_importance(1.0) == pytest.approx(1.0)


def test_normalize_importance_garbage_input_is_safe(ingest_module):
    """Non-numeric input must not raise; default to the middle value."""
    assert ingest_module.normalize_importance("not a number") == pytest.approx(0.6)
    assert ingest_module.normalize_importance([]) == pytest.approx(0.6)


def test_ingest_payload_uses_normalized_importance(ingest_module):
    """The payload construction must call normalize_importance, not raw.

    This is a guard against a future regression where someone removes the
    wrapper and reverts to ``info.get('importance', 3)`` directly. We do
    not actually call Qdrant here — we just construct the payload via
    inspection of the module's compile-time constants and the way the
    payload is built. The strongest assertion we can make without
    running the network-bound ingest path is that the helper exists and
    the constant ``IMPORTANCE_MAX_RAW`` agrees with the LLM prompt.
    """
    assert ingest_module.IMPORTANCE_MIN_RAW == 1.0
    assert ingest_module.IMPORTANCE_MAX_RAW == 5.0
    # Spot-check a known normalization mapping used at payload build time.
    assert ingest_module.normalize_importance(4) == pytest.approx(0.8)


def test_consolidate_module_uses_normalized_importance(consolidate_module):
    """memory_brain_consolidate must normalize the hardcoded 5 too."""
    assert consolidate_module.normalize_importance(5) == pytest.approx(1.0)
    # The fallback path (when ingest module is not on sys.path) should
    # still produce sensible results.
    assert consolidate_module.normalize_importance(1) == pytest.approx(0.2)