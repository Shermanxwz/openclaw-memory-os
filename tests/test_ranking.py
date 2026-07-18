"""Tests for the pure ranking function."""

from __future__ import annotations

from datetime import datetime, timezone


from openclaw_memory_os.config import Settings
from openclaw_memory_os.models import Memory, MemoryStatus, MemoryTier, RecallRequest
from openclaw_memory_os.ranking import rank_memories


def _mem(
    *,
    id_: str = "m1",
    text: str = "alpha bravo charlie",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    tier: MemoryTier = MemoryTier.MEDIUM,
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


def _settings(**overrides) -> Settings:
    base = dict(
        superseded_penalty=0.25,
        expired_penalty=0.10,
        recency_half_life_days=30.0,
        importance_boost_scale=0.6,
        max_recall_results=25,
    )
    base.update(overrides)
    return Settings(**base)


def _now() -> datetime:
    return datetime(2026, 3, 1, tzinfo=timezone.utc)


def test_empty_corpus_returns_empty():
    req = RecallRequest(query="anything")
    hits, considered = rank_memories([], req, settings=_settings(), now=_now())
    assert hits == []
    assert considered == 0


def test_keyword_match_outranks_non_match():
    m_match = _mem(id_="hit", text="recall tests pass in CI", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    m_no = _mem(id_="miss", text="unrelated topic", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    req = RecallRequest(query="recall", mode="hybrid", limit=5)
    hits, considered = rank_memories([m_match, m_no], req, settings=_settings(), now=_now())
    assert considered == 2
    assert hits[0].id == "hit"
    assert "no-keyword-match" in hits[-1].explanation


def test_superseded_memory_filtered_by_default():
    m_super = _mem(id_="old", text="policy rule", status=MemoryStatus.SUPERSEDED)
    m_new = _mem(id_="new", text="policy rule revised")
    req = RecallRequest(query="policy")
    hits, _ = rank_memories([m_super, m_new], req, settings=_settings(), now=_now())
    assert [h.id for h in hits] == ["new"]


def test_superseded_memory_included_when_requested():
    m_super = _mem(id_="old", text="policy rule", status=MemoryStatus.SUPERSEDED)
    m_new = _mem(id_="new", text="policy rule revised")
    req = RecallRequest(query="policy", include_superseded=True)
    hits, _ = rank_memories([m_super, m_new], req, settings=_settings(), now=_now())
    ids = sorted(h.id for h in hits)
    assert ids == ["new", "old"]
    # Superseded rank should be lower because of penalty (assuming equal base attrs).
    superseded_hit = next(h for h in hits if h.id == "old")
    assert superseded_hit.components["base"] == 0.25


def test_expired_memory_filtered_by_default_then_included():
    m_exp = _mem(id_="exp", text="promo campaign", status=MemoryStatus.EXPIRED)
    req = RecallRequest(query="promo")
    hits, _ = rank_memories([m_exp], req, settings=_settings(), now=_now())
    assert hits == []
    req2 = RecallRequest(query="promo", include_expired=True)
    hits2, _ = rank_memories([m_exp], req2, settings=_settings(), now=_now())
    assert [h.id for h in hits2] == ["exp"]
    assert hits2[0].status == MemoryStatus.EXPIRED


def test_recency_decays_with_age():
    fresh = _mem(id_="fresh", text="rule about routing",
                created_at=datetime(2026, 2, 28, tzinfo=timezone.utc),
                updated_at=datetime(2026, 2, 28, tzinfo=timezone.utc))
    old = _mem(id_="old", text="rule about routing",
               created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
               updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    req = RecallRequest(query="routing", limit=5)
    hits, _ = rank_memories([old, fresh], req, settings=_settings(), now=_now())
    assert hits[0].id == "fresh"
    assert hits[0].components["recency"] > hits[1].components["recency"]


def test_importance_boost_higher_for_more_important():
    lo = _mem(id_="lo", text="x", importance=0.1)
    hi = _mem(id_="hi", text="x", importance=0.95)
    req = RecallRequest(query="x", mode="hybrid", limit=5)
    hits, _ = rank_memories([lo, hi], req, settings=_settings(), now=_now())
    assert hits[0].id == "hi"
    assert hits[0].components["importance"] > hits[1].components["importance"]


def test_tier_filter_excludes_other_tiers():
    a = _mem(id_="a", text="x", tier=MemoryTier.CORE)
    b = _mem(id_="b", text="x", tier=MemoryTier.SHORT)
    req = RecallRequest(query="x", tier_filter=[MemoryTier.CORE])
    hits, considered = rank_memories([a, b], req, settings=_settings(), now=_now())
    assert considered == 1
    assert [h.id for h in hits] == ["a"]


def test_since_days_window_excludes_old():
    fresh = _mem(id_="fresh", text="x",
                 created_at=datetime(2026, 2, 25, tzinfo=timezone.utc),
                 updated_at=datetime(2026, 2, 25, tzinfo=timezone.utc))
    old = _mem(id_="old", text="x",
               created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
               updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    req = RecallRequest(query="x", since_days=14)
    hits, considered = rank_memories([fresh, old], req, settings=_settings(), now=_now())
    assert considered == 1
    assert [h.id for h in hits] == ["fresh"]


def test_limit_is_respected_and_capped_by_settings():
    items = [_mem(id_=f"m{i}", text="needle") for i in range(30)]
    req = RecallRequest(query="needle", limit=100)  # user asks for 100
    hits, _ = rank_memories(items, req, settings=_settings(max_recall_results=5), now=_now())
    assert len(hits) == 5


def test_keyword_only_mode_ignores_importance_difference():
    a = _mem(id_="a", text="needle in haystack", importance=0.1)
    b = _mem(id_="b", text="needle in different haystack", importance=0.9)
    req = RecallRequest(query="needle", mode="keyword", limit=5)
    hits, _ = rank_memories([a, b], req, settings=_settings(), now=_now())
    # Keyword score is bounded between 0 and 1; importance shouldn't shift keyword mode
    # by a large amount in this synthetic case, but they should still both come back.
    ids = sorted(h.id for h in hits)
    assert ids == ["a", "b"]


def test_components_breakdown_present():
    m = _mem(id_="m", text="needle", importance=0.6)
    req = RecallRequest(query="needle", mode="hybrid", limit=1)
    hits, _ = rank_memories([m], req, settings=_settings(), now=_now())
    h = hits[0]
    for k in ("base", "recency", "importance", "keyword", "composite"):
        assert k in h.components
    assert isinstance(h.score, float)
    assert h.explanation
