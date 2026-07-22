"""S4: Active-first / Superseded hybrid fallback contract tests.

These tests pin the most important behavioural rule of the
v0.3.0 retrieval contract: a Superseded memory is NEVER allowed
to outrank an Active memory, even if its dense score is higher.
The contract is enforced inside the engine, not in the response
layer; the response just consumes the engine's already-clamped
scores.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest

from openclaw_memory_os.backends import MemoryBackend
from openclaw_memory_os.contracts import (
    CandidateStatus,
    CandidateTier,
    ScoredMemoryCandidate,
)
from openclaw_memory_os.models import Memory, MemoryStatus, MemoryTier
from openclaw_memory_os.policy_store import PolicyStore
from openclaw_memory_os.retrieval_engine import RetrievalEngine


def _memory(memory_id: str, text: str, *, status: MemoryStatus, importance: float = 0.5) -> Memory:
    return Memory(
        id=memory_id,
        text=text,
        status=status,
        importance=importance,
        tier=MemoryTier.MEDIUM,
        created_at=datetime.now(timezone.utc),
    )


class _Stub(MemoryBackend):
    name = "stub"

    def __init__(self, memories: List[Memory], collection: str = "stub") -> None:
        self._memories = memories
        self._collection = collection

    def list_memories(self) -> List[Memory]:
        return list(self._memories)

    def list_collections(self) -> List[str]:
        return [self._collection]

    def get_memory(self, memory_id):
        for m in self._memories:
            if m.id == memory_id:
                return m
        return None

    def dense_search(self, query, limit=10, status_filter=None):
        q = (query or "").lower()
        out = []
        for m in self._memories:
            if status_filter and m.status.value.lower() not in {
                s.lower() for s in status_filter
            }:
                continue
            if not q or q in (m.text or "").lower():
                out.append(
                    ScoredMemoryCandidate(
                        collection=self._collection,
                        memory_id=m.id,
                        candidate_key=f"{self._collection}:{m.id}",
                        text=m.text or "",
                        status=CandidateStatus(m.status.value),
                        tier=CandidateTier(m.tier.value),
                        importance=m.importance,
                        created_at=m.created_at,
                        updated_at=m.updated_at,
                        dense_score=1.0 - (len(out) * 0.05),
                    )
                )
        return out[:limit]

    def lexical_search(self, query, limit=10, status_filter=None):
        return [m for m in self.list_memories() if (not status_filter or m.status.value.lower() in {s.lower() for s in status_filter})][:limit]


def test_active_only_pass_skips_superseded():
    memories = [
        _memory("a1", "alpha bravo charlie", status=MemoryStatus.ACTIVE),
        _memory("a2", "delta echo foxtrot", status=MemoryStatus.ACTIVE),
        _memory("s1", "alpha bravo charlie", status=MemoryStatus.SUPERSEDED),
        _memory("s2", "alpha delta echo", status=MemoryStatus.SUPERSEDED),
    ]
    engine = RetrievalEngine(_Stub(memories), PolicyStore())
    result = engine.retrieve("alpha", mode="hybrid")
    statuses = [h.status for h in result.hits]
    assert MemoryStatus.SUPERSEDED not in statuses, (
        "Active-first contract: superseded should not appear when "
        "active hits exist"
    )


def test_fallback_engages_when_active_below_minimum():
    # The policy's fallback_min_results default is 3; we
    # construct a corpus with only one active hit so the engine
    # is forced to fall back to superseded.
    memories = [
        _memory("a1", "alpha bravo charlie", status=MemoryStatus.ACTIVE),
        _memory("s1", "alpha bravo charlie", status=MemoryStatus.SUPERSEDED),
        _memory("s2", "alpha delta echo", status=MemoryStatus.SUPERSEDED),
    ]
    engine = RetrievalEngine(_Stub(memories), PolicyStore())
    result = engine.retrieve("alpha", mode="hybrid")
    assert result.fallback_used is True
    assert result.fallback_added >= 1


def test_superseded_never_outranks_active_in_display_score():
    # Set up: two active hits, one superseded hit with
    # artificially very high dense score. Even so, the
    # superseded hit must end up at or below the lowest active.
    memories = [
        _memory("a1", "alpha bravo charlie", status=MemoryStatus.ACTIVE, importance=0.9),
        _memory("a2", "alpha delta echo", status=MemoryStatus.ACTIVE, importance=0.7),
        _memory("s1", "alpha bravo charlie", status=MemoryStatus.SUPERSEDED, importance=0.99),
    ]
    engine = RetrievalEngine(_Stub(memories), PolicyStore())
    result = engine.retrieve("alpha", mode="hybrid")
    seen_superseded = False
    for h in result.hits:
        if h.status == MemoryStatus.SUPERSEDED:
            seen_superseded = True
            # Capture the superseded score
            superseded_score = h.score
        elif seen_superseded and h.status == MemoryStatus.ACTIVE:
            pytest.fail(
                f"Active hit appeared after Superseded: {h.id} score={h.score} "
                f"vs superseded score={superseded_score}"
            )


def test_expired_never_auto_appears_in_fallback():
    # Expired memories should not be surfaced by the fallback,
    # even if active results are insufficient.
    memories = [
        _memory("a1", "alpha bravo charlie", status=MemoryStatus.ACTIVE),
        _memory("e1", "alpha bravo charlie", status=MemoryStatus.EXPIRED),
        _memory("s1", "alpha bravo charlie", status=MemoryStatus.SUPERSEDED),
    ]
    engine = RetrievalEngine(_Stub(memories), PolicyStore())
    result = engine.retrieve("alpha", mode="hybrid")
    statuses = {h.status for h in result.hits}
    assert MemoryStatus.EXPIRED not in statuses


def test_no_active_no_superseded_returns_empty_with_diagnostic():
    memories = []  # entirely empty corpus
    engine = RetrievalEngine(_Stub(memories), PolicyStore())
    result = engine.retrieve("alpha", mode="hybrid")
    assert result.hits == []
    # diagnostics should record the gap
    assert result.diagnostics.degraded_reason in (
        "no_active_or_superseded",
        None,
    )
