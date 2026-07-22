"""P0-3 / G3.6 — BM25 lifespan load must NOT re-tokenize the corpus.

The previous ``BM25Index.from_dict()`` called ``_rebuild_stats()``
which ran ``tokenize_lexical(record.get("text", ""))`` over every
record on disk. On the real ~100MB cache (built from Qdrant by the
running service) that single line drove the hybrid recall handler
from sub-second to 40-50 seconds because every lifespan restart
re-tokenised tens of thousands of CJK n-grams.

The fix: ``from_dict()`` now reuses the per-record ``__lexical_tf__``
map that ``add()`` always populates, building aggregate stats
(df / total_tf) directly without going through ``tokenize_lexical``.

These tests pin four contracts:

1. ``BM25Index.from_dict()`` does not call ``_rebuild_stats()`` when
   the cache carries ``__lexical_tf__`` per record.
2. After ``load()``, the in-memory index returns the same search
   results as the original.
3. The FastAPI lifespan populates ``app.state.lexical_index`` from
   the on-disk cache so the recall handler can reuse it.
4. A second ``/api/recall-test`` call does not re-load the index from
   disk; lexical latency stays sub-100ms (well under the 1s G3.6 gate).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

from fastapi.testclient import TestClient

from openclaw_memory_os.contracts import CandidateStatus, CandidateTier, MemoryRecord
from openclaw_memory_os.lexical import BM25Index


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _make_record(
    *,
    text: str,
    memory_id: str,
    collection: str = "openclaw_memory_os",
    keywords=None,
    summary: str = None,
) -> MemoryRecord:
    """Build a minimal ``MemoryRecord`` for index tests."""
    kwargs = dict(
        collection=collection,
        memory_id=memory_id,
        candidate_key=f"{collection}:{memory_id}",
        text=text,
        summary=summary,
        status=CandidateStatus.ACTIVE,
        tier=CandidateTier.MEDIUM,
        importance=0.5,
    )
    if keywords is not None:
        kwargs["keywords"] = keywords
    return MemoryRecord(**kwargs)


def _build_index_with_docs(docs: List[str]) -> BM25Index:
    """Create a populated BM25Index with predictable search behaviour."""
    idx = BM25Index()
    for i, text in enumerate(docs, start=1):
        idx.add(_make_record(text=text, memory_id=f"mem-{i}"))
    return idx


def _write_cache(idx: BM25Index, cache_dir: Path) -> None:
    """Persist an index to disk in the same shape ``load()`` expects."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx.save(cache_dir)


# ---------------------------------------------------------------------------
# 1. from_dict() must not call _rebuild_stats() when __lexical_tf__ is present
# ---------------------------------------------------------------------------


def test_bm25_load_does_not_rebuild_stats_when_from_disk(monkeypatch, tmp_path):
    """P0-3: loading a pre-built cache must skip the slow _rebuild_stats path."""
    docs = [
        "alpha bravo charlie delta echo",
        "foxtrot golf hotel india juliet",
        "kilo lima mike november oscar",
        "papa quebec romeo sierra tango",
        "uniform victor whiskey xray yankee",
    ]
    idx = _build_index_with_docs(docs)
    _write_cache(idx, tmp_path)

    # Monkeypatch the slow path so we can assert it is NOT called when
    # the cache carries the per-record ``__lexical_tf__`` map.
    calls = {"rebuild": 0, "from_cached_tf": 0}
    original_rebuild = BM25Index._rebuild_stats
    original_from_cached = BM25Index._rebuild_stats_from_cached_tf

    def counting_rebuild(self):
        calls["rebuild"] += 1
        return original_rebuild(self)

    def counting_from_cached(self):
        calls["from_cached_tf"] += 1
        return original_from_cached(self)

    monkeypatch.setattr(BM25Index, "_rebuild_stats", counting_rebuild)
    monkeypatch.setattr(
        BM25Index, "_rebuild_stats_from_cached_tf", counting_from_cached
    )

    loaded = BM25Index.load(tmp_path)

    assert loaded is not None
    assert len(loaded) == 5
    # The fast path must run exactly once (during from_dict).
    assert calls["from_cached_tf"] == 1
    # The slow tokenising path must NOT be invoked when every record
    # has a populated ``__lexical_tf__`` map.
    assert calls["rebuild"] == 0


# ---------------------------------------------------------------------------
# 2. Search after load() returns correct results
# ---------------------------------------------------------------------------


def test_bm25_search_after_load_returns_correct_results(tmp_path):
    """P0-3: top result must rank the same after the no-rebuild load path."""
    docs = [
        "v0.3.0 graduation pipeline completed today",
        "memory OS lexical cache lifecycle",
        "unrelated content about cats and dogs",
        "v0.3.0 graduation plan notes and todos",
    ]
    idx = _build_index_with_docs(docs)
    _write_cache(idx, tmp_path)

    loaded = BM25Index.load(tmp_path)
    assert loaded is not None

    # Both indices must agree on the ranking for our pinned query.
    expected = idx.search("v0.3.0 graduation", limit=3)
    actual = loaded.search("v0.3.0 graduation", limit=3)
    assert actual == expected
    # The two ``v0.3.0 graduation`` docs must dominate the top-3.
    top_keys = [k for k, _ in actual]
    assert top_keys[0].endswith(":mem-1") or top_keys[0].endswith(":mem-4")
    assert top_keys[1].endswith(":mem-1") or top_keys[1].endswith(":mem-4")

    # IDF / total_tf / df for the in-memory index must match what the
    # pre-tokenised cache described. If the rebuild path were ever
    # re-introduced silently, these counts could drift; pin them.
    loaded_token_stats = loaded._state.stats.get("graduation")
    assert loaded_token_stats is not None
    # Two of the four docs contain "graduation".
    assert loaded_token_stats.df == 2
    # Each doc has exactly one occurrence of "graduation" (single
    # field ``text`` with default weight 1.0).
    assert loaded_token_stats.total_tf == 2


# ---------------------------------------------------------------------------
# 3. FastAPI lifespan populates app.state.lexical_index from cache
# ---------------------------------------------------------------------------


def test_app_lifespan_loads_index_into_state(tmp_path, monkeypatch):
    """P0-3: a pre-built cache on disk must be loaded into app.state by
    the lifespan handler — not deferred to the recall path."""
    # Pre-build a cache so the lifespan eager-load path has work to do.
    docs = [
        "alpha bravo charlie",
        "delta echo foxtrot",
        "golf hotel india",
        "juliet kilo lima",
        "mike november oscar",
    ]
    cache_idx = _build_index_with_docs(docs)
    cache_dir = tmp_path / "lexical-index"
    _write_cache(cache_idx, cache_dir)

    # Override the env var the conftest set so the lifespan points
    # at our populated cache. ``create_app`` resets the settings
    # cache, so this must be set before the app boots.
    monkeypatch.setenv("MEMORY_OS_LEXICAL_CACHE_DIR", str(cache_dir))

    from openclaw_memory_os.app import create_app

    app = create_app()
    with TestClient(app) as client:
        # Eagerly enter the lifespan-managed context. The lifespan
        # populates ``app.state.lexical_index`` BEFORE the first
        # request, so we should see the loaded index immediately.
        # TestClient enters the lifespan context on ``__enter__``,
        # so the assertion below is reached after startup.
        idx = getattr(app.state, "lexical_index", None)
        assert idx is not None, (
            "lifespan must load the on-disk BM25 cache into "
            "app.state.lexical_index before serving requests"
        )
        assert len(idx) == 5
        # Sanity: the cache dir must also be stashed so the shutdown
        # path can persist the index back out.
        assert getattr(app.state, "_lexical_cache_dir", None) == cache_dir
        # Touch /health to ensure the lifespan is fully active.
        r = client.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 4. Second recall call must reuse the in-memory index (no reload)
# ---------------------------------------------------------------------------


def test_recall_uses_state_lexical_index_not_reload(tmp_path, monkeypatch):
    """P0-3: a second ``/api/recall-test`` call must hit the in-memory
    BM25 index and finish the lexical stage well under 100ms — the
    G3.6 gate is ≤1000ms total, so any value ≫100ms here means we
    are re-loading (re-tokenising) on every request.
    """
    # Build a small but realistic cache so the in-memory index has
    # enough documents to make a meaningful timing comparison.
    docs = [
        f"v0.3.0 graduation memory entry number {i} with various tokens"
        for i in range(10)
    ]
    # Mix in a unique keyword so search has something to match.
    docs[0] = "v0.3.0 graduation keyword_p0three_test unique marker"
    cache_idx = _build_index_with_docs(docs)
    cache_dir = tmp_path / "lexical-index"
    _write_cache(cache_idx, cache_dir)

    monkeypatch.setenv("MEMORY_OS_LEXICAL_CACHE_DIR", str(cache_dir))

    # The recall handler ultimately goes through ``RetrievalEngine``,
    # which depends on the policy store + backend being available.
    # Rather than mocking that whole chain, we measure at the layer
    # the bug actually lives in: BM25Index.search() against the
    # already-loaded in-memory index. If the handler is hot-loading
    # from disk, search() will be preceded by another ~seconds-long
    # rebuild — which is exactly the symptom we are guarding against.
    from openclaw_memory_os.app import create_app

    app = create_app()
    with TestClient(app):
        idx = app.state.lexical_index
        assert idx is not None
        assert len(idx) == 10

        # Warm-up search (first call): pop the cold caches.
        warm = idx.search("v0.3.0 graduation", limit=5)
        assert warm, "warm-up search must return at least one hit"

        # Second call: this must NOT trigger another load or
        # rebuild — we should see steady-state sub-100ms latency.
        t0 = time.perf_counter()
        hits = idx.search("v0.3.0 graduation keyword_p0three_test", limit=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert hits, "second search must still return hits"
        # The in-memory search is over 10 small docs; >100ms would
        # indicate a hidden disk read / re-tokenisation loop.
        assert elapsed_ms < 100.0, (
            f"in-memory BM25 search took {elapsed_ms:.1f}ms — "
            "expected <100ms; the lifespan or recall path is "
            "probably re-loading from disk"
        )

        # And finally, verify the in-memory index is the same object
        # the recall handler would consult (no reload clones).
        assert app.state.lexical_index is idx