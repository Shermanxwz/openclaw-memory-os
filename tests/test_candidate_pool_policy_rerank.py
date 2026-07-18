from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from openclaw_memory_os.backends import MemoryBackend, QdrantBackend
from openclaw_memory_os.candidate_pool import QueryCandidatePool
from openclaw_memory_os.contracts import (
    CandidateStatus, CandidateTier, MemoryRecord, ScoredMemoryCandidate,
)
from openclaw_memory_os.lexical import BM25Index
from openclaw_memory_os.policy_store import Policy, PolicyStore, baseline_policy
from openclaw_memory_os.retrieval_engine import RetrievalEngine, _compute_recency


def _record(mid: str, text: str, *, status=CandidateStatus.ACTIVE, importance=0.5, created_at=None, updated_at=None):
    return MemoryRecord(
        collection="c", memory_id=mid, candidate_key=f"c:{mid}", text=text,
        status=status, tier=CandidateTier.MEDIUM, importance=importance,
        created_at=created_at, updated_at=updated_at,
    )


class CountingBackend(MemoryBackend):
    name = "counting"

    def __init__(self, active, superseded):
        self.active = active
        self.superseded = superseded
        self.dense_calls = 0

    def list_memories(self):
        return []

    def list_collections(self):
        return ["c"]

    def get_memory(self, memory_id):
        return None

    def dense_search(self, query, limit=10, *, status_filter=None):
        self.dense_calls += 1
        wanted = set(status_filter or ["active"])
        source = self.superseded if "superseded" in wanted else self.active
        return [c.model_copy(deep=True) for c in source[:limit]]


def _candidate(record, dense=None, lexical=None):
    return ScoredMemoryCandidate.from_record(
        record, score=float(dense or lexical or 0.0), dense_score=dense, lexical_score=lexical
    )


def _policy(**updates):
    data = dict(baseline_policy)
    data.update(updates)
    return Policy(**data)


def test_twenty_policies_do_not_multiply_backend_or_bm25_calls(monkeypatch):
    active_records = [_record("a", "alpha exact", importance=0.1), _record("b", "beta", importance=0.9)]
    superseded_records = [_record("s", "alpha old", status=CandidateStatus.SUPERSEDED)]
    dense_active = [_candidate(active_records[0], dense=0.9), _candidate(active_records[1], dense=0.8)]
    dense_sup = [_candidate(superseded_records[0], dense=0.7)]
    backend = CountingBackend(dense_active, dense_sup)
    index = BM25Index()
    for rec in active_records + superseded_records:
        index.add(rec)
    lexical_calls = {"count": 0}
    original_search = index.search
    def counted_search(*args, **kwargs):
        lexical_calls["count"] += 1
        return original_search(*args, **kwargs)
    monkeypatch.setattr(index, "search", counted_search)
    engine = RetrievalEngine(backend, PolicyStore(initial=_policy()), lexical_index=index)
    policies = [_policy(version=100 + i, dense_k=40 + i, lexical_k=40 + i) for i in range(20)]
    for policy in policies:
        pool = engine.get_candidate_pool("alpha", channel_limit=200)
        engine.rank_candidate_pool(pool, policy, limit=10)
    assert backend.dense_calls == 2  # Active + Superseded, fixed per query
    assert lexical_calls["count"] == 2  # Active + Superseded, fixed per query


def test_policy_changes_order_on_same_pool_and_active_precedes_superseded():
    now = datetime.now(timezone.utc)
    a = _record("dense", "dense", importance=0.1, created_at=now - timedelta(days=30))
    b = _record("important", "important", importance=1.0, created_at=now)
    s = _record("old", "old", status=CandidateStatus.SUPERSEDED, importance=1.0, created_at=now)
    pool = QueryCandidatePool(
        query="query",
        dense_active=[_candidate(a, dense=1.0), _candidate(b, dense=0.1)],
        lexical_active=[_candidate(b, lexical=1.0), _candidate(a, lexical=0.1)],
        dense_superseded=[_candidate(s, dense=1.0)],
        lexical_superseded=[_candidate(s, lexical=1.0)],
    )
    backend = CountingBackend([], [])
    engine = RetrievalEngine(backend, PolicyStore(initial=_policy()))
    dense_policy = _policy(
        version=201, final_rrf_weight=0.0, final_vector_weight=0.9,
        final_lexical_weight=0.0, importance_weight=0.05, recency_weight=0.05,
        feedback_weight=0.0, fallback_min_results=3,
    )
    importance_policy = _policy(
        version=202, final_rrf_weight=0.0, final_vector_weight=0.0,
        final_lexical_weight=0.05, importance_weight=0.9, recency_weight=0.05,
        feedback_weight=0.0, fallback_min_results=3,
    )
    dense_rank = engine.rank_candidate_pool(pool, dense_policy, limit=10).hits
    important_rank = engine.rank_candidate_pool(pool, importance_policy, limit=10).hits
    assert dense_rank[0].candidate_key == "c:dense"
    assert important_rank[0].candidate_key == "c:important"
    for ranked in (dense_rank, important_rank):
        statuses = [c.status for c in ranked]
        first_sup = statuses.index(CandidateStatus.SUPERSEDED)
        assert all(s == CandidateStatus.ACTIVE for s in statuses[:first_sup])


def test_recency_uses_updated_at_and_missing_is_neutral():
    now = datetime.now(timezone.utc)
    updated = _candidate(
        _record(
            "u", "updated", created_at=now - timedelta(days=365),
            updated_at=now,
        ),
        dense=1.0,
    )
    missing = _candidate(_record("m", "missing"), dense=1.0)
    policy = _policy(recency_half_life_hours=24.0)
    assert _compute_recency(updated, policy) > 0.95
    assert _compute_recency(missing, policy) == 0.5


def test_qdrant_embedding_lru_uses_one_http_call(monkeypatch):
    backend = object.__new__(QdrantBackend)
    backend._embedding_cache = OrderedDict()
    backend._embedding_cache_max = 8
    calls = {"count": 0}
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({"embedding": [0.1, 0.2]}).encode()
    def fake_urlopen(*args, **kwargs):
        calls["count"] += 1
        return Response()
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert backend._embed("same query") == [0.1, 0.2]
    assert backend._embed("same query") == [0.1, 0.2]
    assert calls["count"] == 1
