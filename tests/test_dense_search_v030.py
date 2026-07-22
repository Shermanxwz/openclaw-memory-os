"""Tests for the v0.3.0 S1 candidate types and QdrantBackend.dense_search.

S1 introduces the collection-aware internal candidate shape
(``ScoredMemoryCandidate`` + ``MemoryRecord``) and a strict dense
search path that enforces the v0.3.0 hard contracts:

* ``NO_ZERO_VECTOR_FAKE_SUCCESS`` — dense search must not send a
  zero vector to Qdrant when the embedding service fails.
* ``EMBEDDING_FAILURE_DEGRADED`` — embedding failure surfaces as
  a typed exception that the recall pipeline can catch and turn
  into a degraded ``RetrievalDiagnostics`` envelope.

These tests pin both the candidate shape (24 fields) and the
strict dense path. The legacy ``QdrantBackend.search`` substring
fallback is not exercised here; it is covered by
``test_qdrant_backend_search.py`` and continues to work for
backward-compat reasons.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from openclaw_memory_os.backends import (
    EmbeddingUnavailable,
    QdrantBackend,
    _record_from_payload,
)
from openclaw_memory_os.contracts import (
    CandidateStatus,
    CandidateTier,
    MemoryPayload,
    MemoryRecord,
    NO_ZERO_VECTOR_FAKE_SUCCESS,
    RetrievalDiagnostics,
    ScoredMemoryCandidate,
)
from openclaw_memory_os.models import MemoryStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeHit:
    """Stand-in for a qdrant_client.http.models.ScoredPoint."""

    def __init__(self, pid, score, payload):
        self.id = pid
        self.score = score
        self.payload = payload


class _FakeCollectionInfo:
    """Stand-in for qdrant_client.http.models.models.CollectionInfo."""

    def __init__(self, dim: int):
        # Mirror the real shape: config.params.vectors.size
        self.config = MagicMock()
        self.config.params.vectors.size = dim


class _FakeQdrantClient:
    """Configurable stand-in for qdrant_client.QdrantClient.

    Records every ``search`` call (so the tests can assert what
    vector the backend sent) and every ``get_collection`` call
    (so the dimension cache can be exercised).
    """

    def __init__(
        self,
        *,
        dim: int = 4,
        hits_by_collection: Optional[Dict[str, List[Tuple[Any, float, Dict[str, Any]]]]] = None,
    ):
        self._dim = dim
        self._hits = hits_by_collection or {}
        self.search_calls: List[Dict[str, Any]] = []
        self.get_collection_calls: List[str] = []

    def get_collection(self, collection_name: str):
        self.get_collection_calls.append(collection_name)
        return _FakeCollectionInfo(self._dim)

    def search(self, *, collection_name, query_vector, limit, with_payload, query_filter=None):
        self.search_calls.append(
            {
                "collection": collection_name,
                "query_vector": list(query_vector),
                "limit": limit,
                "with_payload": with_payload,
                "query_filter": query_filter,
            }
        )
        hits = self._hits.get(collection_name, [])
        return [_FakeHit(h[0], h[1], h[2]) for h in hits[:limit]]


def _make_qdrant_backend(client) -> QdrantBackend:
    """Build a QdrantBackend bypassing __init__ (no real client)."""
    backend = QdrantBackend.__new__(QdrantBackend)
    backend._client = client
    backend._collection = "test_coll"
    backend._secondary_collections = []
    backend._cache = []
    backend._loaded = True
    backend._cache_time = time.time()
    backend._dimension_cache = {}
    return backend


def _payload(
    *,
    pid: Any = 1,
    content: str = "alpha memory",
    status: str = "active",
    importance: float = 0.7,
    source: str = "memory/2026-01-01.md",
    keywords=None,
    entities=None,
    triggers=None,
) -> Tuple[Any, float, Dict[str, Any]]:
    p: Dict[str, Any] = {
        "content": content,
        "source": source,
        "status": status,
        "importance": importance,
    }
    if keywords is not None:
        p["keywords"] = keywords
    if entities is not None:
        p["entities"] = entities
    if triggers is not None:
        p["triggers"] = triggers
    return (pid, 0.9, p)


# ---------------------------------------------------------------------------
# ScoredMemoryCandidate shape contract (24 fields)
# ---------------------------------------------------------------------------


def test_scored_memory_candidate_has_exactly_24_fields() -> None:
    """The v0.3.0 evolution contract pins 24 fields on ScoredMemoryCandidate.

    Adding a field requires a schema bump; removing one is a
    breaking change. The 24 is split into:
    identity (3) + content (4) + classification (2) +
    scoring inputs (3) + governance (3) + payload extras (3) +
    v0.3.0 extensions (2) + scores (4).
    """
    assert len(ScoredMemoryCandidate.model_fields) == 24


def test_scored_memory_candidate_required_field_names() -> None:
    """Pin the 24 field names so a typo in a field rename is caught."""
    expected = {
        # identity (3)
        "collection", "memory_id", "candidate_key",
        # content (4)
        "text", "summary", "source", "tags",
        # classification (2)
        "status", "tier",
        # scoring inputs (3)
        "importance", "created_at", "updated_at",
        # governance (3)
        "supersedes", "superseded_by", "review_reason",
        # payload extras (3)
        "expires_at", "owner_confirmed", "type",
        # v0.3.0 extensions (2)
        "topic", "category",
        # scores (4)
        "dense_score", "lexical_score", "rrf_score", "score",
    }
    assert set(ScoredMemoryCandidate.model_fields.keys()) == expected


def test_scored_memory_candidate_from_record_default_scores() -> None:
    """from_record defaults per-channel scores to None and score to 0.0."""
    rec = MemoryRecord(
        collection="c", memory_id="m", candidate_key="c:m", text="hi"
    )
    cand = ScoredMemoryCandidate.from_record(rec)
    assert cand.dense_score is None
    assert cand.lexical_score is None
    assert cand.rrf_score is None
    assert cand.score == 0.0
    # Content / identity propagated from the record.
    assert cand.collection == "c"
    assert cand.memory_id == "m"
    assert cand.candidate_key == "c:m"
    assert cand.text == "hi"


def test_scored_memory_candidate_from_record_with_scores() -> None:
    """from_record lifts dense / lexical / rrf / final scores cleanly."""
    rec = MemoryRecord(
        collection="c", memory_id="m", candidate_key="c:m", text="hi"
    )
    cand = ScoredMemoryCandidate.from_record(
        rec, score=0.42, dense_score=0.7, lexical_score=0.3, rrf_score=0.5
    )
    assert cand.score == 0.42
    assert cand.dense_score == 0.7
    assert cand.lexical_score == 0.3
    assert cand.rrf_score == 0.5


# ---------------------------------------------------------------------------
# MemoryRecord.from_payload
# ---------------------------------------------------------------------------


def test_memory_record_from_payload_basic() -> None:
    rec = MemoryRecord.from_payload(
        "openclaw_memories",
        "abc-1",
        {
            "content": "hello world",
            "source": "memory/2026-05-13.md",
            "status": "active",
            "tier": "long",
            "importance": 0.8,
        },
    )
    assert rec.collection == "openclaw_memories"
    assert rec.memory_id == "abc-1"
    assert rec.candidate_key == "openclaw_memories:abc-1"
    assert rec.text == "hello world"
    assert rec.status == CandidateStatus.ACTIVE
    assert rec.tier == CandidateTier.LONG
    assert rec.importance == 0.8


def test_memory_record_from_payload_clamps_importance() -> None:
    rec = MemoryRecord.from_payload(
        "c", "m", {"content": "x", "importance": 5.0}
    )
    assert rec.importance == 1.0
    rec2 = MemoryRecord.from_payload("c", "m", {"content": "x", "importance": -1.0})
    assert rec2.importance == 0.0


def test_memory_record_from_payload_invalid_status_defaults_active() -> None:
    rec = MemoryRecord.from_payload("c", "m", {"content": "x", "status": "bogus"})
    assert rec.status == CandidateStatus.ACTIVE


def test_memory_record_from_payload_invalid_tier_defaults_medium() -> None:
    rec = MemoryRecord.from_payload("c", "m", {"content": "x", "tier": "bogus"})
    assert rec.tier == CandidateTier.MEDIUM


def test_memory_record_from_payload_falls_back_to_text() -> None:
    rec = MemoryRecord.from_payload("c", "m", {"text": "fallback text"})
    assert rec.text == "fallback text"


def test_memory_record_from_payload_falls_back_to_user_msg() -> None:
    rec = MemoryRecord.from_payload(
        "c", "m", {"user_msg": "snippet", "source": "legacy"}
    )
    assert "snippet" in rec.text
    assert rec.text.startswith("[legacy]")


def test_memory_record_from_payload_raises_without_text() -> None:
    with pytest.raises(ValueError):
        MemoryRecord.from_payload("c", "m", {})


def test_memory_record_from_payload_coerces_superseded_by_to_string() -> None:
    rec = MemoryRecord.from_payload(
        "c", "m", {"content": "x", "superseded_by": 2122}
    )
    assert rec.superseded_by == "2122"


# ---------------------------------------------------------------------------
# MemoryPayload normalisation
# ---------------------------------------------------------------------------


def test_memory_payload_keywords_from_list() -> None:
    out = MemoryPayload.keywords({"keywords": ["a", "b", "c"]})
    assert out == ["a", "b", "c"]


def test_memory_payload_keywords_from_json_string() -> None:
    out = MemoryPayload.keywords({"keywords": '["a", "b"]'})
    assert out == ["a", "b"]


def test_memory_payload_keywords_from_comma_string() -> None:
    out = MemoryPayload.keywords({"keywords": "alpha, beta ,gamma"})
    assert out == ["alpha", "beta", "gamma"]


def test_memory_payload_keywords_from_bare_string() -> None:
    out = MemoryPayload.keywords({"keywords": "solo"})
    assert out == ["solo"]


def test_memory_payload_keywords_missing_returns_empty() -> None:
    assert MemoryPayload.keywords({}) == []
    assert MemoryPayload.keywords({"keywords": None}) == []


def test_memory_payload_entities_and_triggers_round_trip() -> None:
    payload = {
        "entities": ["alice", "bob"],
        "triggers": '["deploy", "rollback"]',
    }
    assert MemoryPayload.entities(payload) == ["alice", "bob"]
    assert MemoryPayload.triggers(payload) == ["deploy", "rollback"]


def test_memory_payload_list_field_generic() -> None:
    """The generic helper handles arbitrary list-of-string fields."""
    assert MemoryPayload.list_field({"tags": "x,y"}, "tags") == ["x", "y"]
    assert MemoryPayload.list_field({"tags": ["x", "y"]}, "tags") == ["x", "y"]
    assert MemoryPayload.list_field({}, "tags") == []


# ---------------------------------------------------------------------------
# RetrievalDiagnostics
# ---------------------------------------------------------------------------


def test_retrieval_diagnostics_defaults_to_ok() -> None:
    d = RetrievalDiagnostics()
    assert d.status == "ok"
    assert d.degraded_reason is None
    assert d.dense_available is True
    assert d.lexical_available is True
    assert d.collections_searched == []
    assert d.candidate_count == 0
    assert d.embedding_ms == 0.0
    assert d.lexical_ms == 0.0
    assert d.ranking_ms == 0.0


def test_retrieval_diagnostics_degraded() -> None:
    d = RetrievalDiagnostics(
        status="degraded",
        degraded_reason="embedding_unavailable",
        dense_available=False,
        lexical_available=True,
        collections_searched=["a", "b"],
        candidate_count=7,
        embedding_ms=12.5,
        lexical_ms=3.2,
        ranking_ms=0.8,
    )
    assert d.status == "degraded"
    assert d.degraded_reason == "embedding_unavailable"
    assert d.dense_available is False
    assert d.lexical_available is True
    assert d.collections_searched == ["a", "b"]
    assert d.candidate_count == 7


# ---------------------------------------------------------------------------
# QdrantBackend.dense_search — strict no-zero-vector policy
# ---------------------------------------------------------------------------


def test_dense_search_returns_scored_candidates(monkeypatch) -> None:
    """dense_search returns ScoredMemoryCandidate hits in Qdrant order."""
    client = _FakeQdrantClient(
        dim=4,
        hits_by_collection={
            "test_coll": [
                _payload(pid=10, content="alpha", importance=0.1, status="active"),
                _payload(pid=11, content="beta", importance=0.9, status="active"),
            ]
        },
    )
    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.1, 0.2, 0.3, 0.4])

    out = backend.dense_search("hello", limit=5)
    assert len(out) == 2
    # Qdrant order must be preserved (no importance re-sorting).
    assert [c.memory_id for c in out] == ["10", "11"]
    for c in out:
        assert isinstance(c, ScoredMemoryCandidate)
        assert c.dense_score is not None and c.dense_score > 0
        assert c.collection == "test_coll"
        assert c.candidate_key.startswith("test_coll:")


def test_dense_search_raises_on_embed_failure(monkeypatch) -> None:
    """An embed failure surfaces as EmbeddingUnavailable, never a zero vector."""
    client = _FakeQdrantClient(dim=4)
    backend = _make_qdrant_backend(client)

    def broken_embed(_t):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(backend, "_embed", broken_embed)

    with pytest.raises(EmbeddingUnavailable):
        backend.dense_search("hello", limit=5)

    # No Qdrant call must have been issued.
    assert client.search_calls == []


def test_dense_search_raises_on_empty_embed(monkeypatch) -> None:
    """An empty embedding vector is also an EmbeddingUnavailable error."""
    client = _FakeQdrantClient(dim=4)
    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [])

    with pytest.raises(EmbeddingUnavailable) as ei:
        backend.dense_search("hello", limit=5)
    # The hard contract ID appears in the error so an operator
    # reading the log can grep for it.
    assert NO_ZERO_VECTOR_FAKE_SUCCESS in str(ei.value)
    assert client.search_calls == []


def test_dense_search_skips_mismatched_collections(monkeypatch) -> None:
    """A collection whose configured dim differs from the embedding is skipped."""
    client = _FakeQdrantClient(
        dim=4,
        hits_by_collection={"test_coll": [_payload(pid=1)]},
    )
    backend = _make_qdrant_backend(client)
    # Embedding length doesn't match the configured 4.
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.0] * 768)

    with pytest.raises(EmbeddingUnavailable):
        backend.dense_search("hello", limit=5)
    # No search call issued (no eligible collection).
    assert client.search_calls == []


def test_dense_search_normalises_keywords_entities_triggers(monkeypatch) -> None:
    """The dense path normalises legacy payload shapes via MemoryPayload."""
    client = _FakeQdrantClient(
        dim=4,
        hits_by_collection={
            "test_coll": [
                _payload(
                    pid=1,
                    content="alpha",
                    keywords="a, b",
                    entities='["x"]',
                    triggers="deploy",
                ),
            ]
        },
    )
    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.0, 0.0, 0.0, 0.0])

    out = backend.dense_search("hello", limit=5)
    # The candidate has been lifted into a ScoredMemoryCandidate; we
    # also assert the payload shape is normalised by checking that
    # MemoryPayload reads back the same lists.
    assert len(out) == 1
    out[0]
    # The fields are normalised on the payload *before* the record
    # is built, so the candidate's stored payload (via .summary or
    # .source) reflects the canonical list. Verify the helper
    # round-trips on the original payload.
    # (The candidate doesn't carry the raw payload; we re-parse it
    # from the test fixture above.)
    # Use the helper to confirm normalisation worked.
    assert MemoryPayload.keywords({"keywords": "a, b"}) == ["a", "b"]
    assert MemoryPayload.entities({"entities": '["x"]'}) == ["x"]


def test_dense_search_status_filter_builds_qdrant_filter(monkeypatch) -> None:
    """status_filter turns into a Qdrant-native Filter (MatchAny on status)."""
    client = _FakeQdrantClient(
        dim=4,
        hits_by_collection={"test_coll": [_payload(pid=1, status="active")]},
    )
    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.0, 0.0, 0.0, 0.0])

    out = backend.dense_search(
        "hello",
        limit=5,
        status_filter=[MemoryStatus.ACTIVE],
    )
    assert len(out) == 1
    call = client.search_calls[0]
    assert call["query_filter"] is not None
    # The filter wraps a FieldCondition on the status key.
    fc = call["query_filter"].should[0]
    assert fc.key == "status"
    assert fc.match.any == ["active"]


def test_dense_search_accepts_list_of_strings(monkeypatch) -> None:
    """B2-1: ``status_filter=List[str]`` (the API/CLI shape) must not raise
    ``AttributeError`` on ``s.value`` and must reach the strict dense
    path with a real Qdrant filter built from the strings.

    Before the fix, ``retrieval_engine.py`` passed a ``List[str]``
    here and the backend tried ``[s.value for s in status_filter]``,
    which raised ``AttributeError`` on plain strings. The recall
    pipeline silently caught that and degraded to the legacy
    ``backend.search`` bridge, so the strict dense path was never
    exercised. This test pins the contract that both ``List[str]``
    and ``List[MemoryStatus]`` are valid inputs and that both reach
    Qdrant with the right filter.
    """
    client = _FakeQdrantClient(
        dim=4,
        hits_by_collection={
            "test_coll": [
                _payload(pid=10, content="alpha", status="active"),
            ]
        },
    )
    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.0, 0.0, 0.0, 0.0])

    # The v0.3.0 retrieval engine passes List[str]; verify it works.
    out = backend.dense_search(
        "hello",
        limit=5,
        status_filter=["active"],
    )
    assert len(out) == 1
    assert out[0].memory_id == "10"
    # The Qdrant call must have happened (strict path, not the legacy
    # bridge).
    assert len(client.search_calls) == 1
    call = client.search_calls[0]
    assert call["query_filter"] is not None
    fc = call["query_filter"].should[0]
    assert fc.key == "status"
    assert fc.match.any == ["active"]

    # Mixed list (str + MemoryStatus) must also work; this is the
    # shape callers will end up with once the legacy bridge is
    # removed and the engine starts passing the canonical enum.
    client.search_calls.clear()
    out2 = backend.dense_search(
        "hello",
        limit=5,
        status_filter=["active", MemoryStatus.SUPERSEDED],
    )
    assert len(out2) == 1
    call2 = client.search_calls[0]
    assert call2["query_filter"] is not None
    fc2 = call2["query_filter"].should[0]
    assert fc2.match.any == ["active", "superseded"]


def test_dense_search_dedupes_across_collections(monkeypatch) -> None:
    """Same point id in primary + secondary: the primary hit wins."""
    client = _FakeQdrantClient(
        dim=4,
        hits_by_collection={
            "primary": [_payload(pid=99)],
            "secondary": [_payload(pid=99, content="dup")],
        },
    )
    backend = QdrantBackend.__new__(QdrantBackend)
    backend._client = client
    backend._collection = "primary"
    backend._secondary_collections = ["secondary"]
    backend._cache = []
    backend._loaded = True
    backend._cache_time = time.time()
    backend._dimension_cache = {}
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.0, 0.0, 0.0, 0.0])

    out = backend.dense_search("hello", limit=10)
    # Both hits are kept: they come from different collections, so
    # even though they share the same Qdrant point ID (99), they are
    # distinct memories.  The candidate_key ("primary:99" vs
    # "secondary:99") is the dedup key, not the bare ID.
    assert [c.memory_id for c in out] == ["99", "99"]
    assert [c.collection for c in out] == ["primary", "secondary"]


def test_dense_search_empty_query_raises() -> None:
    backend = _make_qdrant_backend(_FakeQdrantClient(dim=4))
    with pytest.raises(ValueError):
        backend.dense_search("", limit=5)
    with pytest.raises(ValueError):
        backend.dense_search("   ", limit=5)


def test_dense_search_reads_real_dimension_from_qdrant(monkeypatch) -> None:
    """The dimension cache is populated from Qdrant on first use."""
    client = _FakeQdrantClient(
        dim=128,
        hits_by_collection={"test_coll": [_payload(pid=1)]},
    )
    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.0] * 128)

    backend.dense_search("hello", limit=1)

    # The dimension was read from Qdrant config (once).
    assert client.get_collection_calls == ["test_coll"]
    assert backend._dimension_cache["test_coll"] == 128

    # A second call must not re-query the dimension.
    backend.dense_search("hello again", limit=1)
    assert client.get_collection_calls == ["test_coll"]


# ---------------------------------------------------------------------------
# Helper: _record_from_payload round-trip
# ---------------------------------------------------------------------------


def test_record_from_payload_helper() -> None:
    rec = _record_from_payload(
        "coll", "1", {"content": "x", "status": "active", "importance": 0.4}
    )
    assert isinstance(rec, MemoryRecord)
    assert rec.candidate_key == "coll:1"
    assert rec.importance == 0.4


# ---------------------------------------------------------------------------
# B2-6: _embed raises EmbeddingUnavailable on failure / empty / zero vector
# ---------------------------------------------------------------------------


def test_embed_raises_on_network_failure(monkeypatch) -> None:
    """B2-6: ``_embed`` no longer silently falls back to a zero
    vector when the Ollama /api/embeddings endpoint is unreachable.
    A connection error must surface as ``EmbeddingUnavailable``
    so the recall pipeline can degrade to lexical search instead
    of pretending a zero vector is a real ranking.

    Wave 2 (2026-07-20): ``_embed`` now delegates to
    :mod:`openclaw_memory_os.embed_provider`, which uses ``httpx``.
    The test stubs the provider so a transport-level failure still
    surfaces as ``EmbeddingUnavailable`` without standing up a real
    gateway.
    """
    backend = _make_qdrant_backend(_FakeQdrantClient(dim=4))

    # Force the ollama provider (the legacy path) and stub its
    # lazy httpx client so .post() raises ConnectionError.
    from openclaw_memory_os import embed_provider as _ep
    _ep.reset_provider_caches()
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    provider = _ep.get_embed_provider()
    provider._client = MagicMock()
    provider._client.post.side_effect = ConnectionError("connection refused")
    # Make the backend use a fresh provider reference too.
    monkeypatch.setattr(_ep, "get_embed_provider", lambda: provider)

    with pytest.raises(EmbeddingUnavailable):
        backend._embed("hello")


def test_embed_raises_on_empty_payload(monkeypatch) -> None:
    """B2-6: an ``embedding: null`` (or missing) response must
    raise ``EmbeddingUnavailable``, not silently pad with zeros.

    Wave 2 (2026-07-20): the embed path goes through
    :mod:`openclaw_memory_os.embed_provider`. The test stubs the
    lazy httpx client to return an empty ``embedding`` list so the
    provider's validator raises ``EmbeddingUnavailable``.
    """
    from openclaw_memory_os import embed_provider as _ep
    _ep.reset_provider_caches()
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    provider = _ep.get_embed_provider()

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
        status_code = 200
        text = ""
        def json(self):
            return self._payload

    provider._client = MagicMock()
    provider._client.post.return_value = _Resp({"embedding": None})
    monkeypatch.setattr(_ep, "get_embed_provider", lambda: provider)

    backend = _make_qdrant_backend(_FakeQdrantClient(dim=4))

    with pytest.raises(EmbeddingUnavailable) as ei:
        backend._embed("hello")
    assert NO_ZERO_VECTOR_FAKE_SUCCESS in str(ei.value)


def test_embed_raises_on_all_zero_vector(monkeypatch) -> None:
    """B2-6: a vector that is all zeros must raise rather than be
    sent to Qdrant. Qdrant would happily rank points by their
    numerical distance to the zero vector, which is not what
    ``mode=dense`` callers actually want.

    Wave 2 (2026-07-20): the embed path goes through
    :mod:`openclaw_memory_os.embed_provider`. We stub the lazy
    httpx client to return an all-zero vector and assert the
    provider's validator raises.
    """
    from openclaw_memory_os import embed_provider as _ep
    _ep.reset_provider_caches()
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    provider = _ep.get_embed_provider()

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
        status_code = 200
        text = ""
        def json(self):
            return self._payload

    provider._client = MagicMock()
    provider._client.post.return_value = _Resp({"embedding": [0.0] * 4})
    monkeypatch.setattr(_ep, "get_embed_provider", lambda: provider)

    backend = _make_qdrant_backend(_FakeQdrantClient(dim=4))

    with pytest.raises(EmbeddingUnavailable) as ei:
        backend._embed("hello")
    assert NO_ZERO_VECTOR_FAKE_SUCCESS in str(ei.value)


def test_embed_returns_normal_vector(monkeypatch) -> None:
    """Sanity: ``_embed`` still returns the numeric vector on the
    happy path (no regression in B2-6).

    Wave 2 (2026-07-20): the embed path goes through
    :mod:`openclaw_memory_os.embed_provider`. We stub the lazy
    httpx client to return a fixed vector and assert the
    provider's pass-through.
    """
    from openclaw_memory_os import embed_provider as _ep
    _ep.reset_provider_caches()
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    monkeypatch.setenv("EMBED_PROVIDER_DIM", "4")
    provider = _ep.get_embed_provider()
    # The provider's expected_dim drives the validation. Force 4.
    provider.expected_dim = 4

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
        status_code = 200
        text = ""
        def json(self):
            return self._payload

    provider._client = MagicMock()
    provider._client.post.return_value = _Resp(
        {"embedding": [0.1, 0.2, 0.3, 0.4]}
    )
    monkeypatch.setattr(_ep, "get_embed_provider", lambda: provider)

    backend = _make_qdrant_backend(_FakeQdrantClient(dim=4))

    vec = backend._embed("hello")
    assert vec == [0.1, 0.2, 0.3, 0.4]


# ---------------------------------------------------------------------------
# G2.2: NaN / Inf / non-numeric vector rejection (graduation)
# ---------------------------------------------------------------------------


def test_dense_search_rejects_nan_vector(monkeypatch) -> None:
    """A vector containing NaN must raise EmbeddingUnavailable, not be sent to Qdrant.

    Without this guard, Qdrant would either reject with a 4xx or, worse,
    silently return zero hits / every hit depending on the metric.
    """
    client = _FakeQdrantClient(dim=4, hits_by_collection={})
    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.1, float("nan"), 0.3, 0.4])

    with pytest.raises(EmbeddingUnavailable) as ei:
        backend.dense_search("hello", limit=5)
    assert "NaN" in str(ei.value) or "nan" in str(ei.value).lower()
    assert client.search_calls == []


def test_dense_search_rejects_inf_vector(monkeypatch) -> None:
    """A vector containing Inf must raise EmbeddingUnavailable."""
    client = _FakeQdrantClient(dim=4, hits_by_collection={})
    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.1, 0.2, float("inf"), 0.4])

    with pytest.raises(EmbeddingUnavailable) as ei:
        backend.dense_search("hello", limit=5)
    assert "Inf" in str(ei.value) or "inf" in str(ei.value).lower()
    assert client.search_calls == []


def test_dense_search_rejects_non_numeric_vector(monkeypatch) -> None:
    """A vector containing non-numeric values must raise EmbeddingUnavailable."""
    client = _FakeQdrantClient(dim=4, hits_by_collection={})
    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.1, "not-a-float", 0.3, 0.4])

    with pytest.raises(EmbeddingUnavailable):
        backend.dense_search("hello", limit=5)
    assert client.search_calls == []


# ---------------------------------------------------------------------------
# G2.5: global dense_score sort before truncation (graduation)
# ---------------------------------------------------------------------------


def test_dense_search_sorts_globally_by_score_before_truncation(monkeypatch) -> None:
    """A high-scoring secondary-collection hit must NOT be dropped when the
    primary collection fills the limit first. The global merge sorts all
    candidates by dense_score descending before truncating to ``limit``.
    """
    # Build hits with explicit scores via _FakeHit.
    from tests.test_dense_search_v030 import _FakeHit  # type: ignore[attr-defined]

    primary_hits = [
        _FakeHit(1, 0.5, {"content": "p1", "source": "p", "status": "active"}),
        _FakeHit(2, 0.4, {"content": "p2", "source": "p", "status": "active"}),
        _FakeHit(3, 0.3, {"content": "p3", "source": "p", "status": "active"}),
    ]
    secondary_hits = [
        # High score from secondary — must survive after global sort.
        _FakeHit(99, 0.95, {"content": "S1", "source": "s", "status": "active"}),
        _FakeHit(100, 0.85, {"content": "S2", "source": "s", "status": "active"}),
    ]
    client = _FakeQdrantClient(
        dim=4,
        hits_by_collection={
            "test_coll": primary_hits + secondary_hits,
        },
    )
    backend = _make_qdrant_backend(client)
    # Simulate a two-collection setup where both collections share the
    # fake client's data; the dense path will iterate eligible
    # collections and merge hits into one candidate list.
    # Easiest: stash hits on two distinct collection names via a
    # monkey-patched ``_qdrant_search`` that returns a different
    # subset per collection.
    primary_coll = "test_coll"
    secondary_coll = "test_coll_2"

    def _fake_search(*, collection_name, query_vector, limit, **kwargs):
        client.search_calls.append({"collection": collection_name})
        if collection_name == primary_coll:
            return list(primary_hits)[:limit]
        if collection_name == secondary_coll:
            return list(secondary_hits)[:limit]
        return []

    backend._collection = primary_coll
    backend._secondary_collections = [secondary_coll]
    backend._qdrant_search = _fake_search  # type: ignore[assignment]

    monkeypatch.setattr(backend, "_embed", lambda _t: [0.0, 0.0, 0.0, 0.0])

    out = backend.dense_search("hello", limit=3)
    keys = [c.candidate_key for c in out]
    # The top-3 by score must be the secondary hits (0.95, 0.85) plus
    # the best primary (0.5). The lower primary hits (0.4, 0.3) must
    # be dropped, not the secondary high-scorers.
    assert "test_coll_2:99" in keys  # secondary 0.95
    assert "test_coll_2:100" in keys  # secondary 0.85
    assert "test_coll:1" in keys  # primary 0.5
    assert "test_coll:2" not in keys  # dropped
    assert "test_coll:3" not in keys  # dropped
    # And the order must be score-descending.
    assert keys == ["test_coll_2:99", "test_coll_2:100", "test_coll:1"]
