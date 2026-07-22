"""Tests for the recall fallback strategy.

Fallback contract:

* Default behavior: rank active memories only.
* If the active-only pass yields fewer than
  ``settings.recall_fallback_superseded_min_results`` hits, automatically
  expand the search to include superseded memories as lower-priority
  results.
* ``request.include_superseded=True`` opts in directly and skips the
  fallback logic entirely (the caller wants superseded in the same
  ranking, not as a fallback band).
* The fallback only runs when ``settings.recall_fallback_superseded`` is
  truthy.
* Superseded hits added by the fallback must NOT duplicate active hits
  already returned.
* Superseded hits added by the fallback must sit below every active hit
  in the merged result, so they never outrank live memory.
* The fallback is reported on ``RecallResponse.fallback`` so the
  dashboard / API consumer can surface it.
"""

from __future__ import annotations

from datetime import datetime, timezone


from openclaw_memory_os.config import Settings
from openclaw_memory_os.models import Memory, MemoryStatus, MemoryTier, RecallRequest
from openclaw_memory_os.ranking import build_recall_response


def _mem(
    *,
    id_: str,
    text: str,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    importance: float = 0.5,
) -> Memory:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Memory(
        id=id_,
        text=text,
        tier=MemoryTier.MEDIUM,
        status=status,
        importance=importance,
        created_at=now,
        updated_at=now,
    )


def _settings(**overrides) -> Settings:
    base = dict(
        superseded_penalty=0.25,
        expired_penalty=0.10,
        recency_half_life_days=30.0,
        importance_boost_scale=0.6,
        max_recall_results=25,
        recall_fallback_superseded=True,
        recall_fallback_superseded_min_results=5,
    )
    base.update(overrides)
    return Settings(**base)


def _now() -> datetime:
    return datetime(2026, 3, 1, tzinfo=timezone.utc)


def test_fallback_disabled_by_default_when_enough_active_hits():
    """When the active pass already meets the minimum, no fallback runs."""
    mems = [_mem(id_=f"a{i}", text=f"policy rule {i}") for i in range(10)]
    req = RecallRequest(query="policy", limit=10)
    r = build_recall_response(mems, req, backend_name="sample", settings=_settings(), now=_now())
    assert len(r.hits) == 10
    assert r.fallback.used is False
    assert r.fallback.added == 0
    assert all(h.status == MemoryStatus.ACTIVE for h in r.hits)


def test_fallback_fires_when_active_hits_below_minimum():
    """If active hits < min, superseded memories are appended as fallback."""
    mems = [
        _mem(id_="a1", text="policy rule current"),
        _mem(id_="s1", text="policy rule older", status=MemoryStatus.SUPERSEDED),
        _mem(id_="s2", text="policy rule archived", status=MemoryStatus.SUPERSEDED),
    ]
    req = RecallRequest(query="policy", limit=10)
    r = build_recall_response(mems, req, backend_name="sample", settings=_settings(), now=_now())
    ids = [h.id for h in r.hits]
    assert "a1" in ids
    # Both superseded hits should be surfaced as fallback.
    assert "s1" in ids
    assert "s2" in ids
    assert r.fallback.used is True
    assert r.fallback.added == 2


def test_fallback_does_not_duplicate_active_hits():
    """Fallback must not re-add hits already returned by the active pass."""
    mems = [
        _mem(id_="a1", text="policy rule current"),
        _mem(id_="a2", text="policy rule secondary"),
        _mem(id_="s1", text="policy rule older", status=MemoryStatus.SUPERSEDED),
    ]
    req = RecallRequest(query="policy", limit=10)
    r = build_recall_response(mems, req, backend_name="sample", settings=_settings(), now=_now())
    ids = [h.id for h in r.hits]
    # No duplicate ids in the response.
    assert len(ids) == len(set(ids))
    # And every id from the corpus appears at most once.
    assert ids.count("a1") <= 1
    assert ids.count("a2") <= 1
    assert ids.count("s1") <= 1


def test_fallback_keeps_superseded_below_active_hits():
    """Superseded hits added by the fallback must sit below every active hit."""
    # Active unrelated entries may score moderately; superseded entries
    # that actually match the query would otherwise score higher in the
    # expanded pass — the fallback must clamp them below the active
    # floor so they never outrank live memory.
    mems = [
        _mem(id_="a_unrelated", text="completely different"),
        _mem(id_="s_match", text="policy rule older", status=MemoryStatus.SUPERSEDED),
        _mem(id_="a_match", text="policy rule current"),
    ]
    req = RecallRequest(query="policy", limit=10)
    r = build_recall_response(mems, req, backend_name="sample", settings=_settings(), now=_now())
    active_scores = [h.score for h in r.hits if h.status == MemoryStatus.ACTIVE]
    superseded_scores = [h.score for h in r.hits if h.status == MemoryStatus.SUPERSEDED]
    assert active_scores, "expected at least one active hit"
    if superseded_scores:
        # Every superseded score is strictly below every active score.
        assert max(superseded_scores) < min(active_scores), (
            f"superseded {superseded_scores} should be below active {active_scores}"
        )


def test_fallback_orders_superseded_by_capped_score_descending():
    """Within the fallback band, stronger superseded matches should appear first."""
    mems = [
        _mem(id_="a1", text="policy rule current"),
        _mem(id_="s_strong", text="policy rule exact old", status=MemoryStatus.SUPERSEDED, importance=1.0),
        _mem(id_="s_weak", text="unrelated archived note", status=MemoryStatus.SUPERSEDED, importance=0.0),
    ]
    req = RecallRequest(query="policy", limit=10)
    r = build_recall_response(mems, req, backend_name="sample", settings=_settings(), now=_now())
    superseded_ids = [h.id for h in r.hits if h.status == MemoryStatus.SUPERSEDED]
    assert superseded_ids[:2] == ["s_strong", "s_weak"]


def test_fallback_respects_explicit_include_superseded():
    """When the caller opts in directly, the fallback logic must not engage."""
    mems = [
        _mem(id_="a1", text="policy rule current"),
        _mem(id_="s1", text="policy rule older", status=MemoryStatus.SUPERSEDED),
    ]
    req = RecallRequest(query="policy", limit=10, include_superseded=True)
    r = build_recall_response(mems, req, backend_name="sample", settings=_settings(), now=_now())
    ids = sorted(h.id for h in r.hits)
    assert ids == ["a1", "s1"]
    # The fallback flag must report that the fallback path was NOT used:
    # the caller already opted in.
    assert r.fallback.used is False


def test_fallback_can_be_disabled_via_settings():
    """Disabling fallback returns only active hits, even when below min."""
    mems = [
        _mem(id_="a1", text="policy rule current"),
        _mem(id_="s1", text="policy rule older", status=MemoryStatus.SUPERSEDED),
    ]
    s = _settings(recall_fallback_superseded=False)
    req = RecallRequest(query="policy", limit=10)
    r = build_recall_response(mems, req, backend_name="sample", settings=s, now=_now())
    ids = [h.id for h in r.hits]
    assert "s1" not in ids
    assert r.fallback.enabled is False
    assert r.fallback.used is False


def test_fallback_respects_configurable_min_results():
    """The min_results setting controls when fallback engages."""
    mems = [_mem(id_=f"a{i}", text=f"policy rule {i}") for i in range(4)] + [
        _mem(id_="s1", text="policy rule archived", status=MemoryStatus.SUPERSEDED),
    ]
    # Default min=5 → 4 active hits < 5 → fallback fires.
    r1 = build_recall_response(mems, RecallRequest(query="policy", limit=10),
                                backend_name="sample", settings=_settings(), now=_now())
    assert r1.fallback.used is True
    assert "s1" in [h.id for h in r1.hits]

    # min=3 → 4 active hits >= 3 → fallback does NOT fire.
    s = _settings(recall_fallback_superseded_min_results=3)
    r2 = build_recall_response(mems, RecallRequest(query="policy", limit=10),
                                backend_name="sample", settings=s, now=_now())
    assert r2.fallback.used is False
    assert "s1" not in [h.id for h in r2.hits]


def test_fallback_cap_respects_request_limit():
    """Final result honors request.limit, even after fallback expansion."""
    # 1 active hit + 8 superseded hits. Fallback should fire but the merged
    # result must be capped at limit=4.
    mems = [
        _mem(id_="a1", text="policy rule current"),
    ] + [
        _mem(id_=f"s{i}", text=f"policy rule archived {i}", status=MemoryStatus.SUPERSEDED)
        for i in range(8)
    ]
    req = RecallRequest(query="policy", limit=4)
    r = build_recall_response(mems, req, backend_name="sample", settings=_settings(), now=_now())
    assert len(r.hits) == 4
    # The active hit must survive the cap.
    assert r.hits[0].id == "a1"
    assert r.hits[0].status == MemoryStatus.ACTIVE


def test_fallback_records_min_results_metadata():
    """The response surfaces the configured min_results for diagnostics."""
    s = _settings(recall_fallback_superseded_min_results=7)
    mems = [_mem(id_="a1", text="policy rule current")] + [
        _mem(id_=f"s{i}", text=f"policy rule archived {i}", status=MemoryStatus.SUPERSEDED)
        for i in range(3)
    ]
    r = build_recall_response(mems, RecallRequest(query="policy", limit=10),
                                backend_name="sample", settings=s, now=_now())
    assert r.fallback.min_results == 7
    assert r.fallback.used is True


def test_no_fallback_when_no_superseded_memories_exist():
    """If the corpus has no superseded memories, fallback can't add anything."""
    mems = [_mem(id_=f"a{i}", text=f"policy rule {i}") for i in range(2)]
    req = RecallRequest(query="policy", limit=10)
    r = build_recall_response(mems, req, backend_name="sample", settings=_settings(), now=_now())
    assert r.fallback.used is False
    assert r.fallback.added == 0


def test_fallback_floor_marker_on_components():
    """Superseded hits added by the fallback carry a ``fallback_floor`` marker."""
    mems = [
        _mem(id_="a1", text="policy rule current"),
        _mem(id_="s1", text="policy rule older", status=MemoryStatus.SUPERSEDED),
    ]
    req = RecallRequest(query="policy", limit=10)
    r = build_recall_response(mems, req, backend_name="sample", settings=_settings(), now=_now())
    superseded = [h for h in r.hits if h.status == MemoryStatus.SUPERSEDED]
    assert superseded, "fallback should have added at least one superseded hit"
    for h in superseded:
        assert "fallback_floor" in h.components
        # And the score is at or below the recorded floor.
        assert h.score <= h.components["fallback_floor"] + 1e-3


def test_fallback_default_from_settings_matches_env():
    """``RECALL_FALLBACK_SUPERSEDED_MIN_RESULTS`` env flows into Settings."""
    import os
    from openclaw_memory_os.config import reset_settings_cache
    os.environ["RECALL_FALLBACK_SUPERSEDED_MIN_RESULTS"] = "9"
    reset_settings_cache()
    try:
        from openclaw_memory_os.config import get_settings
        s = get_settings()
        assert s.recall_fallback_superseded_min_results == 9
        assert s.recall_fallback_superseded is True
    finally:
        os.environ.pop("RECALL_FALLBACK_SUPERSEDED_MIN_RESULTS", None)
        reset_settings_cache()


def test_fallback_off_env_disables_strategy():
    """``RECALL_FALLBACK_SUPERSEDED=off`` disables the fallback strategy."""
    import os
    from openclaw_memory_os.config import reset_settings_cache
    os.environ["RECALL_FALLBACK_SUPERSEDED"] = "off"
    reset_settings_cache()
    try:
        from openclaw_memory_os.config import get_settings
        s = get_settings()
        assert s.recall_fallback_superseded is False
    finally:
        os.environ.pop("RECALL_FALLBACK_SUPERSEDED", None)
        reset_settings_cache()
