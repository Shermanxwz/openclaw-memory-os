"""Direct unit tests for the analytics helper functions.

These complement the integration coverage in ``test_api.py`` by
exercising the candidate-builder / duplicate-estimator logic in
isolation. They are intentionally small and dependency-light so they
run fast and can be expanded without standing up the full app.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from openclaw_memory_os.analytics import _build_deletion_candidates
from openclaw_memory_os.models import (
    DeletionCandidate,
    Memory,
    MemoryStatus,
    MemoryTier,
)


def _make_memory(
    *,
    id: str,
    tier: MemoryTier,
    importance: float,
    age_days: int = 0,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    review_reason: str | None = None,
    text: str = "x",
) -> Memory:
    created = datetime.now(timezone.utc) - timedelta(days=age_days)
    return Memory(
        id=id,
        text=text,
        tier=tier,
        status=status,
        importance=importance,
        created_at=created,
        review_reason=review_reason,
    )


def test_build_deletion_candidates_keeps_core_and_long():
    memories = [
        _make_memory(id="c1", tier=MemoryTier.CORE, importance=0.1, age_days=400),
        _make_memory(id="l1", tier=MemoryTier.LONG, importance=0.1, age_days=400),
    ]
    assert _build_deletion_candidates(memories) == []


def test_build_deletion_candidates_includes_low_value_expired_working():
    memories = [
        _make_memory(
            id="w1",
            tier=MemoryTier.WORKING,
            importance=0.1,
            age_days=120,
            status=MemoryStatus.EXPIRED,
        ),
    ]
    candidates = _build_deletion_candidates(memories)
    assert len(candidates) == 1
    cand = candidates[0]
    assert isinstance(cand, DeletionCandidate)
    assert cand.id == "w1"
    assert cand.tier == MemoryTier.WORKING
    # The recommended_action is left to the caller (dashboard); the helper
    # should never return ``keep``.
    assert cand.recommended_action != "keep"


def test_build_deletion_candidates_skips_recent_low_value():
    """A 3-day-old working memory is too recent to be a candidate."""
    memories = [
        _make_memory(id="w2", tier=MemoryTier.WORKING, importance=0.1, age_days=3),
    ]
    assert _build_deletion_candidates(memories) == []


def test_build_deletion_candidates_skips_high_importance():
    """Even an old, expired memory with importance >= 0.6 is not a candidate."""
    memories = [
        _make_memory(
            id="w3",
            tier=MemoryTier.WORKING,
            importance=0.7,
            age_days=400,
            status=MemoryStatus.EXPIRED,
        ),
    ]
    assert _build_deletion_candidates(memories) == []


def test_build_deletion_candidates_respects_never_delete_flag():
    memories = [
        _make_memory(
            id="w4",
            tier=MemoryTier.WORKING,
            importance=0.1,
            age_days=400,
            status=MemoryStatus.EXPIRED,
            review_reason="never_delete: pinned by user",
        ),
    ]
    assert _build_deletion_candidates(memories) == []


def test_build_deletion_candidates_only_non_working_short_medium_when_expired():
    """Tier=short/medium are only candidates when status=expired."""
    old_active_short = _make_memory(
        id="s1", tier=MemoryTier.SHORT, importance=0.1, age_days=400,
    )
    old_expired_short = _make_memory(
        id="s2",
        tier=MemoryTier.SHORT,
        importance=0.1,
        age_days=400,
        status=MemoryStatus.EXPIRED,
    )
    candidates = _build_deletion_candidates([old_active_short, old_expired_short])
    ids = [c.id for c in candidates]
    assert ids == ["s2"], f"only expired short/medium should appear; got {ids}"


def test_build_deletion_candidates_handles_naive_datetime():
    """Memory created_at is sometimes naive; helper should not crash."""
    naive_old = Memory(
        id="w5",
        text="x",
        tier=MemoryTier.WORKING,
        importance=0.1,
        created_at=datetime.utcnow() - timedelta(days=200),
        status=MemoryStatus.EXPIRED,
    )
    candidates = _build_deletion_candidates([naive_old])
    assert len(candidates) == 1
    assert candidates[0].id == "w5"


@pytest.mark.parametrize(
    "importance",
    [0.6, 0.7, 1.0],
)
def test_build_deletion_candidates_importance_threshold(importance: float):
    """Importance >= 0.6 is the hard threshold; never a candidate."""
    memories = [
        _make_memory(
            id="w-imp",
            tier=MemoryTier.WORKING,
            importance=importance,
            age_days=400,
            status=MemoryStatus.EXPIRED,
        ),
    ]
    assert _build_deletion_candidates(memories) == []
