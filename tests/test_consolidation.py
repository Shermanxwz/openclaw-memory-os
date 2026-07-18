"""Tests for duplicate memory consolidation."""

from __future__ import annotations

from datetime import datetime, timezone


from openclaw_memory_os.consolidation import consolidate_cluster
from openclaw_memory_os.models import ConsolidationResult, Memory, MemoryStatus, MemoryTier


def _mem(
    *,
    id_: str = "m1",
    text: str = "alpha bravo charlie",
    tier: MemoryTier = MemoryTier.MEDIUM,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    importance: float = 0.5,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    tags: list[str] | None = None,
) -> Memory:
    now = created_at or datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Memory(
        id=id_,
        text=text,
        tier=tier,
        status=status,
        importance=importance,
        tags=tags or [],
        created_at=now,
        updated_at=updated_at or now,
    )


def test_consolidate_single_member():
    m = _mem(id_="only", text="unique")
    result = consolidate_cluster([m])
    assert isinstance(result, ConsolidationResult)
    assert result.consolidated_id == "only"
    assert result.merged_member_ids == ["only"]


def test_consolidate_merge_two():
    a = _mem(id_="a", text="short text", importance=0.3)
    b = _mem(id_="b", text="longer text with more detail", importance=0.8)
    result = consolidate_cluster([a, b], strategy="merge")
    assert result.consolidated_id == "b"  # higher importance
    assert len(result.merged_member_ids) == 2


def test_consolidate_merge_picks_longest_text():
    a = _mem(id_="a", text="short", importance=0.9)
    b = _mem(id_="b", text="a much longer text with many words for testing merge behavior", importance=0.1)
    result = consolidate_cluster([a, b], strategy="merge")
    # Merge picks longer text, but b has lower importance
    # The 'merge' strategy sorts by (updated_at, importance) desc, then picks eligible
    # Since update times are same, importance decides -> a wins
    # But text should be the longest from all eligible
    assert len(result.text) >= len(b.text)


def test_consolidate_keep_newest():
    old = _mem(
        id_="old",
        text="old version",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )
    new = _mem(
        id_="new",
        text="new version",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    result = consolidate_cluster([old, new], strategy="keep_newest")
    assert result.consolidated_id == "new"
    assert result.text == "new version"


def test_consolidate_keep_best():
    low = _mem(id_="low", text="low importance", importance=0.2)
    high = _mem(id_="high", text="high importance", importance=0.95)
    result = consolidate_cluster([low, high], strategy="keep_best")
    assert result.consolidated_id == "high"
    assert result.text == "high importance"


def test_consolidate_merge_unions_tags():
    a = _mem(id_="a", text="text a", tags=["tag1", "tag2"])
    b = _mem(id_="b", text="text b", tags=["tag2", "tag3"])
    result = consolidate_cluster([a, b])
    assert "tag1" in result.preserved_tags
    assert "tag2" in result.preserved_tags
    assert "tag3" in result.preserved_tags


def test_consolidate_core_members_survive():
    core = _mem(id_="core", text="core memory", tier=MemoryTier.CORE, importance=0.9)
    medium = _mem(id_="medium", text="medium memory", tier=MemoryTier.MEDIUM, importance=0.5)
    result = consolidate_cluster([core, medium], strategy="merge")
    # Core member should be in survivors
    assert "core" in result.survivors


def test_consolidate_all_core():
    a = _mem(id_="c1", text="core a", tier=MemoryTier.CORE)
    b = _mem(id_="c2", text="core b", tier=MemoryTier.CORE)
    result = consolidate_cluster([a, b])
    assert result.consolidated_id in ("c1", "c2")


def test_strategy_unknown_defaults_to_merge():
    a = _mem(id_="a", text="alpha", importance=0.5)
    b = _mem(id_="b", text="bravo", importance=0.6)
    result = consolidate_cluster([a, b], strategy="nonexistent_strategy")
    # Should fall back to merge
    assert len(result.merged_member_ids) == 2
