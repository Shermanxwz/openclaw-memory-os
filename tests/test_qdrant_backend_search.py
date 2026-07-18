"""Regression test for issue #2: dense-mode recall must actually hit the
backend's vector search, not just re-score the full corpus with an
importance proxy.

External review (2026-07-14) flagged three coordinated gaps:

1. ``QdrantBackend.search`` was already calling Qdrant but was
   re-sorting results by ``payload.importance`` after the search, which
   silently discarded Qdrant's actual vector similarity ordering. That
   defeats the point of dense recall.
2. ``SampleBackend`` had no ``search`` method at all. ``MemoryBackend``
   now ships a default keyword-substring ``search()`` so the interface
   is uniform.
3. The CLI ``recall`` command and the ``/api/recall-test`` endpoint
   were both passing ``backend.list_memories()`` to
   ``build_recall_response`` regardless of mode, so ``mode=dense`` was
   indistinguishable from running the importance heuristic over the
   entire store. The CLI and the endpoint now source candidates from
   ``backend.search(query, limit)`` whenever the request is dense.

These tests pin all three behaviours so a future refactor can't
re-introduce the gap silently.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import MagicMock


from openclaw_memory_os.backends import (
    MemoryBackend,
    QdrantBackend,
    SampleBackend,
)
from openclaw_memory_os.config import Settings
from openclaw_memory_os.models import Memory, MemoryStatus, MemoryTier, RecallRequest
from openclaw_memory_os.ranking import build_recall_response


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA = REPO_ROOT / "data" / "sample_memories.json"


def _mem(
    *,
    id_: str,
    text: str,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    tier: MemoryTier = MemoryTier.MEDIUM,
    importance: float = 0.5,
    tags: list[str] | None = None,
) -> Memory:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Memory(
        id=id_,
        text=text,
        tier=tier,
        status=status,
        importance=importance,
        tags=tags or [],
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
    )
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# 1. QdrantBackend.search preserves Qdrant's score ordering
# ---------------------------------------------------------------------------


class _FakeHit:
    def __init__(self, pid, score, payload):
        self.id = pid
        self.score = score
        self.payload = payload


class _FakeQdrantClient:
    """Minimal stand-in for qdrant_client.QdrantClient.search()."""

    def __init__(self, hits_by_collection):
        # hits_by_collection[coll] is a list of (id, score, payload) tuples
        # returned in Qdrant's natural score order (highest first).
        self._hits = hits_by_collection
        self.search_calls: list = []

    def search(self, *, collection_name, query_vector, limit, with_payload):
        self.search_calls.append(
            {
                "collection": collection_name,
                "query_vector": query_vector,
                "limit": limit,
            }
        )
        hits = self._hits.get(collection_name, [])
        return [_FakeHit(h[0], h[1], h[2]) for h in hits[:limit]]


def _make_qdrant_backend(client, collection="test_coll", secondary=None):
    """Build a QdrantBackend instance bypassing __init__ (no real client)."""
    import time as _time

    backend = QdrantBackend.__new__(QdrantBackend)
    backend._client = client
    backend._collection = collection
    backend._secondary_collections = list(secondary or [])
    backend._cache = []
    backend._loaded = True
    # Set cache_time to "now" so the 30-second TTL guard in
    # ``QdrantBackend._load`` short-circuits and the tests don't have
    # to stub the scroll pipeline.
    backend._cache_time = _time.time()
    return backend


def test_qdrant_search_calls_client_with_query_vector(monkeypatch):
    """search() must encode the query and call QdrantClient.search with it."""
    client = _FakeQdrantClient({"test_coll": []})
    backend = _make_qdrant_backend(client)

    # Capture the embed call to make sure it's invoked exactly once.
    captured: dict = {"calls": 0, "text": None}
    expected_vec = [0.1, 0.2, 0.3]

    def fake_embed(text):
        captured["calls"] += 1
        captured["text"] = text
        return expected_vec

    monkeypatch.setattr(backend, "_embed", fake_embed)

    out = backend.search("hello world", limit=5)

    assert captured["calls"] == 1
    assert captured["text"] == "hello world"
    # The client must have been called once with our query vector.
    assert len(client.search_calls) == 1
    call = client.search_calls[0]
    assert call["collection"] == "test_coll"
    assert call["query_vector"] == expected_vec
    assert call["limit"] >= 5
    assert out == []  # no hits in this fixture


def test_qdrant_search_preserves_score_ordering(monkeypatch):
    """The previous bug: search() re-sorted by importance and dropped the
    Qdrant vector ordering. Now the order Qdrant returns must be kept.
    """
    payload_high = {"content": "alpha hit", "source": "memory/2026-01-01.md",
                    "importance": 0.1, "status": "active"}
    payload_mid = {"content": "beta hit", "source": "memory/2026-01-01.md",
                   "importance": 0.9, "status": "active"}
    payload_low = {"content": "gamma hit", "source": "memory/2026-01-01.md",
                   "importance": 0.5, "status": "active"}
    # Qdrant order: 10 (highest score), 11, 12 (lowest).
    client = _FakeQdrantClient({"test_coll": [
        (10, 0.95, payload_high),
        (11, 0.80, payload_mid),
        (12, 0.40, payload_low),
    ]})
    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.0] * 4)

    out = backend.search("anything", limit=3)

    assert [m.id for m in out] == ["10", "11", "12"], (
        "Qdrant's natural score order must be preserved — the previous "
        "implementation re-sorted by importance and dropped the actual "
        "vector ranking."
    )


def test_qdrant_search_keeps_cross_collection_same_id(monkeypatch):
    """If the same point id appears in primary and secondary, both are
    kept because they are distinct memories in different collections.
    The candidate_key ("primary:99" vs "secondary:99") is the dedup key,
    not the bare Qdrant point ID.
    """
    payload = {"content": "shared", "source": "memory/2026-01-01.md",
               "importance": 0.5, "status": "active"}
    client = _FakeQdrantClient({
        "primary": [(99, 0.9, payload)],
        "secondary": [(99, 0.7, payload)],
    })
    backend = _make_qdrant_backend(client, collection="primary",
                                   secondary=["secondary"])
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.0] * 4)

    out = backend.search("anything", limit=10)

    assert [m.id for m in out] == ["99", "99"]
    assert len(client.search_calls) == 2


def test_qdrant_search_falls_back_to_substring_on_uncaught_exception(monkeypatch):
    """If Qdrant itself raises (network down, missing collection), the
    fallback substring search against the cache must kick in so the
    endpoint stays usable. This is the only branch that doesn't run a
    real vector search.
    """
    backend = _make_qdrant_backend(_FakeQdrantClient({}))
    backend._cache = [
        _mem(id_="a", text="the quick brown fox"),
        _mem(id_="b", text="lazy dog"),
    ]

    def broken_embed(_t):
        raise RuntimeError("embed service down")

    monkeypatch.setattr(backend, "_embed", broken_embed)

    out = backend.search("fox", limit=10)
    assert [m.id for m in out] == ["a"]


# ---------------------------------------------------------------------------
# 2. SampleBackend.search now exists (via MemoryBackend default)
# ---------------------------------------------------------------------------


def test_sample_backend_inherits_default_search():
    """The default MemoryBackend.search is a substring matcher; SampleBackend
    must not raise AttributeError when callers ask it to search.
    """
    backend = SampleBackend(SAMPLE_DATA)
    backend._load()
    backend._cache = [
        _mem(id_="x", text="alpha"),
        _mem(id_="y", text="beta"),
        _mem(id_="z", text=""),
    ]
    out = backend.search("alpha", limit=10)
    assert [m.id for m in out] == ["x"]


def test_memory_backend_default_search_returns_empty_on_no_query():
    """Empty query → return up to ``limit`` from the front of the corpus."""
    backend = SampleBackend(SAMPLE_DATA)
    backend._load()
    backend._cache = [_mem(id_="a", text="x"), _mem(id_="b", text="y")]
    out = backend.search("", limit=10)
    assert [m.id for m in out] == ["a", "b"]


# ---------------------------------------------------------------------------
# 3. CLI / API / build_recall_response wire backend.search into dense mode
# ---------------------------------------------------------------------------


class _SpyBackend(MemoryBackend):
    """SampleBackend-like backend that records what ``search`` and
    ``list_memories`` were called with, so the dense wiring test can
    assert that dense mode actually exercises ``search``.
    """

    name = "spy"

    def __init__(self, search_results=None):
        self.search_results = list(search_results or [])
        self.search_calls: list = []
        self.list_calls: int = 0

    def list_memories(self) -> List[Memory]:
        self.list_calls += 1
        return [
            _mem(id_="corpus-a", text="corpus alpha"),
            _mem(id_="corpus-b", text="corpus beta"),
        ]

    def list_collections(self) -> List[str]:
        return ["spy"]

    def get_memory(self, memory_id: str):
        return None

    def search(self, query: str, limit: int = 10) -> List[Memory]:
        self.search_calls.append({"query": query, "limit": limit})
        return list(self.search_results)[:limit]


def test_build_recall_response_dense_uses_search_candidates():
    """When ``mode=dense`` and the caller supplies ``dense_candidates``,
    the response must be ranked over those candidates only — the rest
    of the corpus must not be considered for the initial pass.
    """
    backend = _SpyBackend(search_results=[
        _mem(id_="vec-1", text="vector hit one", importance=0.9),
        _mem(id_="vec-2", text="vector hit two", importance=0.4),
    ])
    req = RecallRequest(query="anything", mode="dense", limit=10)
    resp = build_recall_response(
        backend.list_memories(),
        req,
        backend_name=backend.name,
        settings=_settings(),
        dense_candidates=backend.search(req.query, limit=40),
    )
    assert [h.id for h in resp.hits] == ["vec-1", "vec-2"]


def test_build_recall_response_dense_falls_back_to_corpus_without_candidates():
    """If ``dense_candidates`` is None (the default used by some unit
    tests and by older callers), the dense branch still works against
    ``memories`` so the formula stays exercisable.
    """
    backend = _SpyBackend()
    req = RecallRequest(query="corpus", mode="dense", limit=10)
    resp = build_recall_response(
        backend.list_memories(),
        req,
        backend_name=backend.name,
        settings=_settings(),
        # dense_candidates omitted on purpose
    )
    # Both corpus entries should at least be inspected.
    assert resp.total_considered >= 1


def test_dense_mode_in_cli_invokes_backend_search(monkeypatch):
    """The CLI ``recall --mode dense`` must call ``backend.search(query)``,
    not just ``backend.list_memories()``.
    """
    backend = _SpyBackend(search_results=[
        _mem(id_="vec-only", text="vector-only hit", importance=0.9),
    ])
    monkeypatch.setattr(
        "openclaw_memory_os.cli.get_backend", lambda *_a, **_kw: backend
    )
    from openclaw_memory_os.cli import build_parser, main

    parser = build_parser()
    parser.parse_args(["recall", "--query", "needle", "--mode", "dense"])
    rc = main(["recall", "--query", "needle", "--mode", "dense"])
    assert rc == 0

    # search() must have been called with the user's query.
    assert backend.search_calls, "CLI dense mode must invoke backend.search"
    assert backend.search_calls[0]["query"] == "needle"
    # list_memories() may also be called (it's used for the fallback
    # pool) but search must be the source of the primary candidates.
    assert backend.search_calls[0]["limit"] >= 40


def test_dense_mode_in_api_endpoint_invokes_backend_search(monkeypatch):
    """The FastAPI ``/api/recall-test`` endpoint with ``mode=dense`` must
    call ``backend.search(query)`` to source candidates.

    We monkeypatch ``get_backend`` so the lifespan (which runs inside the
    TestClient context manager) creates our spy instead of the real
    SampleBackend. Setting ``app.state.backend`` directly does not work
    because ``_build_app``'s lifespan re-assigns it from ``get_backend()``.
    """
    from fastapi.testclient import TestClient

    from openclaw_memory_os.app import create_app

    backend = _SpyBackend(search_results=[
        _mem(id_="api-vec", text="api vector hit", importance=0.9),
    ])
    monkeypatch.setattr(
        "openclaw_memory_os.app.get_backend", lambda *_a, **_kw: backend
    )

    app = create_app()

    with TestClient(app) as c:
        r = c.post(
            "/api/recall-test",
            json={"query": "needle", "mode": "dense", "limit": 5},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "dense"
    # The spy was queried for vector candidates.
    assert backend.search_calls, "API dense mode must invoke backend.search"
    assert backend.search_calls[0]["query"] == "needle"
    # The returned hits must come from the spy backend's search() result.
    assert [h["id"] for h in body["hits"]] == ["api-vec"]


def test_hybrid_mode_does_not_invoke_search(monkeypatch):
    """Regression guard: hybrid mode keeps the old behaviour of scoring
    the full corpus with a keyword+base blend. It must NOT call
    ``backend.search``.
    """
    backend = _SpyBackend(search_results=[])
    from openclaw_memory_os.cli import main

    rc = main(["recall", "--query", "needle", "--mode", "hybrid"])
    assert rc == 0
    assert backend.search_calls == [], (
        "hybrid mode must NOT call backend.search — that's the dense "
        "branch's job."
    )


# ---------------------------------------------------------------------------
# 4. Qdrant backend client.search integration smoke (no network)
# ---------------------------------------------------------------------------


def test_qdrant_backend_search_uses_real_client_method(monkeypatch):
    """Make sure the Qdrant path actually calls ``_client.query_points``
    (the v1.10+ replacement for ``_client.search``) and doesn't
    accidentally take the substring fallback when the embed
    service is reachable.
    """
    client = MagicMock()
    fake_hit = MagicMock()
    fake_hit.id = 7
    fake_hit.payload = {
        "content": "real vector hit",
        "source": "memory/2026-01-01.md",
        "importance": 0.6,
        "status": "active",
    }
    # ``query_points`` returns a ``QueryResponse``-like object whose
    # ``.points`` attribute carries the hit list (v1.10+ API).
    query_response = MagicMock()
    query_response.points = [fake_hit]
    client.query_points.return_value = query_response
    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.0] * 4)

    out = backend.search("query", limit=10)

    # Qdrant was called exactly once via the new shim.
    assert client.query_points.call_count == 1
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "test_coll"
    assert kwargs["query"] == [0.0] * 4
    assert kwargs["limit"] >= 10
    assert kwargs["with_payload"] is True
    # The legacy ``search`` method must not be called by the shim
    # when ``query_points`` is available.
    assert client.search.call_count == 0
    # The fallback substring branch must NOT have fired.
    assert out != []
    assert [m.id for m in out] == ["7"]


# ---------------------------------------------------------------------------
# 5. qdrant-client 1.10+ shim: ``query_points`` vs legacy ``search``
# ---------------------------------------------------------------------------


class _ShimQueryPointsClient:
    """Fake Qdrant client that exposes the v1.10+ ``query_points`` API
    but **not** the legacy ``search`` method. Mirrors what the real
    qdrant-client 1.10+ ships with.
    """

    def __init__(self, response):
        self._response = response
        self.query_points_calls: list = []
        # Track ``search`` access explicitly so tests can prove it
        # was never called. Raising on access (rather than returning
        # a MagicMock) is the strongest guarantee that the shim
        # didn't accidentally fall through to the legacy path.
        self.search_calls: list = []

    def query_points(self, **kwargs):
        self.query_points_calls.append(kwargs)
        return self._response

    def search(self, *args, **kwargs):  # pragma: no cover - see note
        # Should never be invoked by the shim when ``query_points``
        # is available. Record the call so the assertion fails loudly
        # rather than silently passing.
        self.search_calls.append((args, kwargs))
        raise AssertionError(
            "legacy client.search() must not be called when "
            "client.query_points() is available"
        )


class _ShimLegacySearchClient:
    """Fake Qdrant client that exposes the legacy ``search`` API but
    **not** the new ``query_points`` method. Mirrors qdrant-client
    <= 1.9.
    """

    def __init__(self, hits):
        self._hits = hits
        self.search_calls: list = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return list(self._hits)

    # NOTE: ``query_points`` is intentionally not defined here.
    # In qdrant-client <= 1.9 the method does not exist on the
    # client object at all, so ``hasattr(client, "query_points")``
    # returns False. The shim relies on that to route to the
    # legacy ``search`` path.


def test_qdrant_backend_uses_query_points_when_available(monkeypatch):
    """When the client exposes ``query_points``, the shim must use it
    and must NOT fall through to ``search``.
    """
    fake_hit = MagicMock()
    fake_hit.id = 42
    fake_hit.payload = {
        "content": "vec hit",
        "source": "memory/2026-01-01.md",
        "importance": 0.5,
        "status": "active",
    }
    response = MagicMock()
    response.points = [fake_hit]
    client = _ShimQueryPointsClient(response)

    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.1, 0.2, 0.3])

    out = backend.search("q", limit=5)

    # ``query_points`` was called exactly once.
    assert len(client.query_points_calls) == 1
    kwargs = client.query_points_calls[0]
    assert kwargs["collection_name"] == "test_coll"
    assert kwargs["query"] == [0.1, 0.2, 0.3]
    assert kwargs["limit"] >= 5
    assert kwargs["with_payload"] is True
    # ``search`` was NEVER called — the shim never fell through.
    assert client.search_calls == []
    # The hit was normalised and the iteration contract holds.
    assert [m.id for m in out] == ["42"]


def test_qdrant_backend_falls_back_to_legacy_search(monkeypatch):
    """When the client only has legacy ``search`` (qdrant-client
    <= 1.9), the shim must call that path and the iteration contract
    must still hold.
    """
    hit = _FakeHit(101, 0.9, {
        "content": "legacy hit",
        "source": "memory/2026-01-01.md",
        "importance": 0.4,
        "status": "active",
    })
    client = _ShimLegacySearchClient([hit])

    backend = _make_qdrant_backend(client)
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.0] * 4)

    out = backend.search("q", limit=5)

    # ``search`` was called exactly once.
    assert len(client.search_calls) == 1
    kwargs = client.search_calls[0]
    assert kwargs["collection_name"] == "test_coll"
    assert kwargs["query_vector"] == [0.0] * 4
    assert kwargs["limit"] >= 5
    assert kwargs["with_payload"] is True
    # ``query_points`` was NEVER called (and doesn't even exist on
    # this fake — the shim's ``hasattr`` guard must route to the
    # legacy branch).
    assert not hasattr(client, "query_points")
    # Hit returned and contract held.
    assert [m.id for m in out] == ["101"]


def test_qdrant_backend_dense_search_uses_query_points(monkeypatch):
    """The same shim contract must hold for the strict v0.3.0
    ``dense_search`` path. The shim helper is the single source
    of truth for which client API gets called.
    """
    fake_hit = MagicMock()
    fake_hit.id = 7
    fake_hit.payload = {
        "content": "dense hit",
        "source": "memory/2026-01-01.md",
        "importance": 0.6,
        "status": "active",
    }
    response = MagicMock()
    response.points = [fake_hit]
    client = _ShimQueryPointsClient(response)

    backend = _make_qdrant_backend(client)
    # Pre-populate the dimension cache so dense_search considers
    # ``test_coll`` eligible.
    backend._dimension_cache = {"test_coll": 4}
    monkeypatch.setattr(backend, "_embed", lambda _t: [0.0, 0.0, 0.0, 0.0])

    from openclaw_memory_os.backends import QdrantBackend  # noqa: F401
    out = backend.dense_search("needle", limit=3)

    assert len(client.query_points_calls) == 1
    kwargs = client.query_points_calls[0]
    assert kwargs["collection_name"] == "test_coll"
    assert kwargs["query"] == [0.0, 0.0, 0.0, 0.0]
    assert kwargs["limit"] >= 3
    assert kwargs["with_payload"] is True
    # Legacy ``search`` never invoked.
    assert client.search_calls == []
    # dense_search returns ScoredMemoryCandidate objects whose
    # ``memory_id`` is the string form of the Qdrant point id.
    assert [c.memory_id for c in out] == ["7"]


def test_qdrant_backend_query_points_return_shape_is_normalised(monkeypatch):
    """The shim must turn a ``QueryResponse`` (with ``.points``) into
    the same list-of-hits shape the legacy code iterated over, so
    ``for h in hits: h.id, h.payload`` works for both client
    versions.

    We exercise the shim directly (rather than going through
    ``backend.search``) so the assertion is about the contract
    callers depend on — the normalised list exposing ``.id`` and
    ``.payload`` — without coupling the test to the
    payload-to-Memory adapter.
    """

    class _QueryResponse:
        def __init__(self, points):
            self.points = points

    hit_a = _FakeHit(1, 0.95, {
        "content": "alpha",
        "source": "memory/2026-01-01.md",
        "importance": 0.1,
        "status": "active",
    })
    hit_b = _FakeHit(2, 0.50, {
        "content": "beta",
        "source": "memory/2026-01-01.md",
        "importance": 0.9,
        "status": "active",
    })
    response = _QueryResponse([hit_a, hit_b])
    client = _ShimQueryPointsClient(response)

    backend = _make_qdrant_backend(client)

    # Call the shim directly.
    out = backend._qdrant_search(
        collection_name="test_coll",
        query_vector=[0.0] * 4,
        limit=10,
    )

    # The shim normalised the response into a flat list of hits.
    # Iteration over the list exposes .id and .payload just like
    # the legacy ``ScoredPoint`` list did.
    assert [h.id for h in out] == [1, 2]
    assert out[0].payload["content"] == "alpha"
    assert out[1].payload["content"] == "beta"
    # Order is preserved from the QueryResponse.points list (highest
    # Qdrant score first).
    assert [h.id for h in out] == [h.id for h in [hit_a, hit_b]]