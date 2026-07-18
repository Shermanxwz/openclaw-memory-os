"""Tests for the v0.3.0 unified RetrievalEngine.

The engine is the only entry point for dense/lexical/hybrid
recall. These tests cover the Active-first / Superseded-fallback
contract, the keyword-only mode, the dense-with-fallback-to-lexical
degradation, and the new query_id / diagnostics fields on the
public response shape.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from openclaw_memory_os.backends import (
    EmbeddingUnavailable,
    MemoryBackend,
)
from openclaw_memory_os.contracts import (
    CandidateStatus,
    CandidateTier,
    ScoredMemoryCandidate,
)
from openclaw_memory_os.models import Memory, MemoryStatus, MemoryTier, RecallRequest
from openclaw_memory_os.policy_store import PolicyStore
from openclaw_memory_os.retrieval_engine import (
    RetrievalEngine,
    _calibrate_dense_scores,
    build_recall_response_v030,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _StubBackend(MemoryBackend):
    """An in-memory backend that pretends to be a Qdrant collection."""

    name = "stub"

    def __init__(self, memories: list[Memory], collection: str = "stub") -> None:
        self._memories = list(memories)
        self._collection = collection

    def list_memories(self) -> list[Memory]:
        return list(self._memories)

    def list_collections(self) -> list[str]:
        return [self._collection]

    def get_memory(self, memory_id: str):
        for m in self._memories:
            if m.id == memory_id:
                return m
        return None

    def dense_search(
        self,
        query: str,
        limit: int = 10,
        status_filter=None,
    ) -> list[ScoredMemoryCandidate]:
        # Toy: rank by naive case-insensitive substring match on
        # the memory's text and emit a synthetic dense score
        # proportional to match position.
        q = (query or "").lower()
        out = []
        for m in self._memories:
            if status_filter and m.status.value.lower() not in {
                s.lower() for s in status_filter
            }:
                continue
            text = (m.text or "").lower()
            if not q or q in text:
                out.append(
                    ScoredMemoryCandidate(
                        collection=self._collection,
                        memory_id=m.id,
                        candidate_key=f"{self._collection}:{m.id}",
                        text=m.text or "",
                        summary=m.summary,
                        source=m.source,
                        tags=list(m.tags or []),
                        status=CandidateStatus(m.status.value),
                        tier=CandidateTier(m.tier.value),
                        importance=m.importance,
                        created_at=m.created_at,
                        updated_at=m.updated_at,
                        supersedes=m.supersedes,
                        superseded_by=m.superseded_by,
                        review_reason=m.review_reason,
                        dense_score=1.0 - (len(out) * 0.05),
                    )
                )
        return out[:limit]

    def lexical_search(self, query, limit=10, status_filter=None):
        # Substring fallback like SampleBackend.
        out = []
        q = (query or "").lower()
        for m in self._memories:
            if status_filter and m.status.value.lower() not in {
                s.lower() for s in status_filter
            }:
                continue
            if not q or q in (m.text or "").lower():
                out.append(m)
        return out[:limit]


def _memory(
    memory_id: str,
    text: str,
    *,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    importance: float = 0.5,
) -> Memory:
    return Memory(
        id=memory_id,
        text=text,
        status=status,
        importance=importance,
        tier=MemoryTier.MEDIUM,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Engine basics
# ---------------------------------------------------------------------------


def test_engine_returns_empty_for_empty_query():
    backend = _StubBackend([])
    store = PolicyStore()
    engine = RetrievalEngine(backend, store)
    result = engine.retrieve("", mode="hybrid")
    assert result.hits == []
    assert result.diagnostics.status == "failed"
    assert result.diagnostics.degraded_reason == "empty_query"


def test_engine_keyword_only_path():
    memories = [
        _memory("a", "the capital of France is Paris"),
        _memory("b", "Memory OS uses BM25 for lexical search"),
        _memory("c", "totally unrelated content here"),
    ]
    backend = _StubBackend(memories)
    store = PolicyStore()
    engine = RetrievalEngine(backend, store)
    result = engine.retrieve("BM25 lexical", mode="keyword")
    # Only lexical path, no dense score on any candidate
    assert result.hits
    for h in result.hits:
        assert h.dense_score is None


def test_engine_dense_only_path():
    memories = [
        _memory("a", "alpha bravo charlie"),
        _memory("b", "delta echo foxtrot"),
    ]
    backend = _StubBackend(memories)
    store = PolicyStore()
    engine = RetrievalEngine(backend, store)
    result = engine.retrieve("alpha", mode="dense")
    assert result.hits
    for h in result.hits:
        assert h.lexical_score is None or h.lexical_score == 0.0


def test_engine_hybrid_path_returns_rrf_scores():
    memories = [
        _memory("a", "alpha bravo charlie"),
        _memory("b", "delta echo foxtrot"),
        _memory("c", "alpha bravo zulu"),
    ]
    backend = _StubBackend(memories)
    store = PolicyStore()
    engine = RetrievalEngine(backend, store)
    result = engine.retrieve("alpha", mode="hybrid")
    assert result.hits
    for h in result.hits:
        assert h.rrf_score is not None
        assert h.score > 0.0


# ---------------------------------------------------------------------------
# Active-first / Superseded fallback
# ---------------------------------------------------------------------------


def test_engine_active_first_skips_superseded_by_default():
    memories = [
        _memory("a1", "alpha bravo charlie", status=MemoryStatus.ACTIVE),
        _memory("s1", "alpha bravo charlie", status=MemoryStatus.SUPERSEDED),
    ]
    backend = _StubBackend(memories)
    store = PolicyStore()
    engine = RetrievalEngine(backend, store)
    result = engine.retrieve("alpha", mode="hybrid")
    statuses = {h.status for h in result.hits}
    assert MemoryStatus.SUPERSEDED not in statuses


def test_engine_falls_back_to_superseded_when_active_insufficient():
    memories = [
        # Only a superseded hit matches the query. Active results
        # are < fallback_min_results (default 3), so the engine
        # should fall back to the superseded pass.
        _memory("s1", "Memory OS uses BM25", status=MemoryStatus.SUPERSEDED),
    ]
    backend = _StubBackend(memories)
    store = PolicyStore()
    engine = RetrievalEngine(backend, store)
    result = engine.retrieve("BM25", mode="hybrid")
    assert result.fallback_used is True
    assert result.fallback_added == 1
    # The display_score of the superseded hit must be below any
    # active hit. There are no active hits, so the constraint
    # is implicit (the fallback is the entire response).
    assert result.hits


def test_engine_superseded_floor_is_below_active_min():
    memories = [
        _memory("a1", "alpha bravo charlie", status=MemoryStatus.ACTIVE, importance=0.9),
        _memory("a2", "alpha delta echo", status=MemoryStatus.ACTIVE, importance=0.8),
        _memory("s1", "alpha bravo charlie", status=MemoryStatus.SUPERSEDED, importance=0.5),
    ]
    backend = _StubBackend(memories)
    store = PolicyStore()
    engine = RetrievalEngine(backend, store)
    result = engine.retrieve("alpha", mode="hybrid")
    # Active hits should all be ahead of superseded
    seen_superseded = False
    for h in result.hits:
        if h.status == MemoryStatus.SUPERSEDED:
            seen_superseded = True
        if seen_superseded and h.status == MemoryStatus.ACTIVE:
            pytest.fail("Active hit appeared after Superseded in ranking")


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def test_calibrate_dense_scores_min_max():
    a = ScoredMemoryCandidate(
        collection="c",
        memory_id="1",
        candidate_key="c:1",
        text="x",
        importance=0.5,
        dense_score=0.9,
    )
    b = ScoredMemoryCandidate(
        collection="c",
        memory_id="2",
        candidate_key="c:2",
        text="y",
        importance=0.5,
        dense_score=0.3,
    )
    out = _calibrate_dense_scores([a, b])
    assert out["c:1"] == pytest.approx(1.0)
    assert out["c:2"] == pytest.approx(0.0)


def test_calibrate_dense_scores_handles_ties():
    a = ScoredMemoryCandidate(
        collection="c",
        memory_id="1",
        candidate_key="c:1",
        text="x",
        importance=0.5,
        dense_score=0.5,
    )
    b = ScoredMemoryCandidate(
        collection="c",
        memory_id="2",
        candidate_key="c:2",
        text="y",
        importance=0.5,
        dense_score=0.5,
    )
    out = _calibrate_dense_scores([a, b])
    assert out["c:1"] == pytest.approx(1.0)
    assert out["c:2"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


def test_build_recall_response_v030_carries_diagnostics():
    memories = [_memory("a1", "alpha bravo charlie")]
    backend = _StubBackend(memories)
    store = PolicyStore()
    engine = RetrievalEngine(backend, store)
    result = engine.retrieve("alpha", mode="hybrid")
    request = RecallRequest(query="alpha", mode="hybrid", limit=10)
    response = build_recall_response_v030(request, result, policy=store.get())
    assert response.backend == "v030"
    assert response.fallback is not None
    assert response.query == "alpha"


# ---------------------------------------------------------------------------
# Embedding degradation
# ---------------------------------------------------------------------------


class _FailingDenseBackend(_StubBackend):
    """A backend whose dense_search always raises EmbeddingUnavailable."""

    def dense_search(self, query, limit=10, status_filter=None):
        raise EmbeddingUnavailable("embedding service is down")


def test_engine_dense_mode_falls_back_to_lexical_on_embedding_failure():
    memories = [_memory("a1", "alpha bravo charlie")]
    backend = _FailingDenseBackend(memories)
    store = PolicyStore()
    engine = RetrievalEngine(backend, store)
    # The user explicitly asked for "dense". When the embedder is
    # down we should silently fall back to lexical, not crash.
    result = engine.retrieve("alpha", mode="dense")
    assert result.diagnostics.dense_available is False
    assert result.diagnostics.degraded_reason == "embedding_unavailable"
    # Hits should still come back via lexical
    assert result.hits
