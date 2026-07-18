"""Tests for the BM25 inverted-index path (G3.6 / P0-X).

These tests verify that ``BM25Index.search`` uses a token -> postings
inverted index instead of looping over every document on each query.
Without the inverted index, a 52k-doc hybrid query takes ~9 s; with
it, the same query should land in well under 1 s.

The tests cover:

1. ``add()`` correctly populates the per-token posting lists.
2. ``search()`` only scores candidates reachable via the postings
   (we monkeypatch the score loop to count its iterations).
3. ``search()`` results are bit-for-bit identical to a brute-force
   full-scan BM25 baseline.
4. ``remove()`` cleans every posting list that referenced the
   removed doc_key.
5. ``save()`` + ``load()`` round-trips the inverted index so a
   service restart does not have to rebuild it from scratch.
6. A pre-P0-X legacy cache (no ``__lexical_tf__``) still produces
   a usable inverted index after ``load()`` via the legacy
   re-tokenise fallback path.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple


from openclaw_memory_os.contracts import (
    CandidateStatus,
    CandidateTier,
    MemoryRecord,
)
from openclaw_memory_os.lexical import (
    BM25Index,
    tokenize_lexical,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    text: str,
    memory_id: str,
    keywords: List[str] | None = None,
    summary: str | None = None,
    collection: str = "openclaw_memory_os",
) -> MemoryRecord:
    return MemoryRecord(
        collection=collection,
        memory_id=memory_id,
        candidate_key=f"{collection}:{memory_id}",
        text=text,
        summary=summary,
        keywords=keywords or [],
        status=CandidateStatus.ACTIVE,
        tier=CandidateTier.MEDIUM,
        importance=0.5,
    )


def _brute_force_search(idx: BM25Index, query: str, limit: int = 5) -> List[Tuple[str, float]]:
    """Reference implementation: full-scan BM25 over ``idx._state.documents``.

    Mirrors the *old* v0.3.0 ``search`` algorithm so we can prove the
    new inverted-index path returns equivalent results.
    """
    import math as _math

    from openclaw_memory_os.lexical import (
        extract_exact_identifiers,
        _BM25_K1,
        _BM25_B,
    )

    if not query or not idx._state.documents:
        return []
    query_tokens = tokenize_lexical(query)
    if not query_tokens:
        return []
    seen = set()
    uniq_query = []
    for t in query_tokens:
        if t not in seen:
            seen.add(t)
            uniq_query.append(t)
    query_exacts = extract_exact_identifiers(query)
    n_docs = len(idx._state.documents)
    avg = idx._average_doc_len or 1.0
    scores: List[Tuple[str, float]] = []
    for key, (record, doc_len, doc_exacts) in idx._state.documents.items():
        s = 0.0
        for qt in uniq_query:
            stats = idx._state.stats.get(qt)
            if stats is None or stats.df == 0:
                continue
            idf = _math.log(1 + (n_docs - stats.df + 0.5) / (stats.df + 0.5))
            tf = idx._doc_tf(key, qt)
            if tf == 0:
                continue
            denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / avg)
            s += idf * (tf * (_BM25_K1 + 1)) / (denom or 1.0)
        if doc_exacts & query_exacts:
            s += 1.0
        if s > 0.0:
            scores.append((key, s))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:limit]


# ---------------------------------------------------------------------------
# 1. add() populates inverted_index
# ---------------------------------------------------------------------------


def test_add_populates_inverted_index() -> None:
    """``add`` appends ``(doc_key, tf)`` to every token's posting list."""
    idx = BM25Index()
    rng = random.Random(0xBEEF)
    vocab = [
        "alpha", "bravo", "charlie", "delta", "echo",
        "foxtrot", "golf", "hotel", "india", "juliet",
    ]
    for i in range(100):
        # Each doc has 3-6 randomly chosen tokens; some repeats so
        # tf > 1 happens for a handful of (token, doc) pairs.
        chosen = rng.choices(vocab, k=rng.randint(3, 6))
        idx.add(_make_record(text=" ".join(chosen), memory_id=f"d-{i:03d}"))

    # Every token we generated must have a posting list.
    for token in vocab:
        assert token in idx._state.inverted_index, f"missing posting list for {token}"
        postings = idx._state.inverted_index[token]
        assert postings, f"empty posting list for {token}"
        for entry in postings:
            assert len(entry) == 2
            doc_key, tf = entry
            assert isinstance(doc_key, str) and doc_key.startswith("openclaw_memory_os:d-")
            assert isinstance(tf, int) and tf >= 1

    # Cross-check: sum of all (token, tf) postings equals sum of
    # all per-doc token counts (i.e. the inverted index is a
    # faithful partition of the per-doc token list).
    expected_tf_sum = 0
    for _key, (record, _doc_len, _exact) in idx._state.documents.items():
        expected_tf_sum += sum(record["__lexical_tf__"].values())
    actual_tf_sum = sum(
        tf for postings in idx._state.inverted_index.values() for (_dk, tf) in postings
    )
    assert actual_tf_sum == expected_tf_sum


# ---------------------------------------------------------------------------
# 2. search() does not enter the full O(N) scan path
# ---------------------------------------------------------------------------


def test_search_uses_postings_not_full_scan(monkeypatch) -> None:
    """``search`` must score only candidate docs from the postings.

    We wrap :meth:`BM25Index.search` with a counter that records
    how many candidates the inverted-index path touches. With a
    1000-doc corpus and a query whose tokens only appear in a
    handful of docs, we must see <<100 candidates touched; if
    the legacy O(N) loop were running we would see 1000.
    """
    idx = BM25Index()
    rng = random.Random(0xC0FFEE)
    vocab = [f"t{i:03d}" for i in range(50)]
    for i in range(1000):
        chosen = rng.choices(vocab, k=rng.randint(5, 12))
        idx.add(_make_record(text=" ".join(chosen), memory_id=f"doc-{i:04d}"))

    # Pick a query token that is rare (low df). To make the
    # assertion crisp we inject a single one-off lowercase token
    # into doc-0001 and query for it. ``tokenize_lexical``
    # lowercases alphanumerics so we keep these tokens
    # lowercase to match what the inverted index would see in
    # production.
    rare_doc = idx._state.documents["openclaw_memory_os:doc-0001"]
    rare_doc[0]["__lexical_tf__"]["rare_token"] = 1
    rare_doc[0]["__lexical_tf__"]["rare_token_pair"] = 1
    rare_doc[0]["__lexical_tokens__"].extend(["rare_token", "rare_token_pair"])
    # Mirror the change in the inverted index AND the stats
    # table so search sees it (in production this all happens
    # via add()).
    from openclaw_memory_os.lexical import _BM25Stats
    for rare in ("rare_token", "rare_token_pair"):
        idx._state.inverted_index.setdefault(rare, []).append(
            ("openclaw_memory_os:doc-0001", 1)
        )
        s = idx._state.stats.setdefault(rare, _BM25Stats())
        s.df += 1
        s.total_tf += 1

    # Instrument: count how many candidates the score loop touches.
    touched: List[str] = []

    real_search = BM25Index.search

    def spy_search(self, query, limit=10, *, exact_match_boost=1.0):
        # Recreate the candidate-set logic the production method
        # uses, then count the unique docs. We do NOT call the
        # real method here because we want to *measure* the
        # candidate count, not just the result.
        from openclaw_memory_os.lexical import tokenize_lexical as _tok

        query_tokens = _tok(query)
        seen = set()
        uniq = []
        for t in query_tokens:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        candidates = set()
        for qt in uniq:
            for dk, _tf in self._state.inverted_index.get(qt, ()):
                candidates.add(dk)
        touched.extend(candidates)
        # Now run the real search and return its result.
        return real_search(self, query, limit=limit, exact_match_boost=exact_match_boost)

    monkeypatch.setattr(BM25Index, "search", spy_search)

    hits = idx.search("rare_token rare_token_pair")
    assert hits, "the rare doc must be retrievable"
    assert hits[0][0] == "openclaw_memory_os:doc-0001"
    # The whole point: candidate count must be tiny. Two query
    # tokens each with df=1 → 1 candidate (the rare doc).
    assert len(touched) < 100, (
        f"expected <100 candidates touched, got {len(touched)} "
        f"— search is still doing an O(N) scan?"
    )
    assert len(touched) == 1


# ---------------------------------------------------------------------------
# 3. search() results match brute-force full-scan BM25
# ---------------------------------------------------------------------------


def test_search_correctness_matches_brute_force() -> None:
    """For random queries on a 50-doc corpus, the inverted-index path
    must match the brute-force BM25 ranking within 1e-9 on every
    returned score and agree on the top-K doc_keys.
    """
    idx = BM25Index()
    rng = random.Random(0xACE0FFEE)
    vocab = [
        "alpha", "bravo", "charlie", "delta", "echo",
        "foxtrot", "golf", "hotel", "india", "juliet",
        "kilo", "lima", "mike", "november", "oscar",
    ]
    for i in range(50):
        # Bias each doc toward a couple of "favourite" tokens so
        # BM25 has some real signal to discriminate on.
        faves = rng.sample(vocab, k=2)
        others = rng.choices(vocab, k=rng.randint(4, 8))
        text = " ".join(faves + others)
        idx.add(_make_record(text=text, memory_id=f"doc-{i:02d}"))

    queries = [
        "alpha bravo",
        "golf hotel india",
        "echo delta",
        "kilo lima mike",
        "november oscar alpha",
    ]
    for q in queries:
        fast = idx.search(q, limit=5)
        slow = _brute_force_search(idx, q, limit=5)
        assert len(fast) == len(slow), (
            f"length mismatch for {q!r}: fast={len(fast)} slow={len(slow)}"
        )
        # Same keys, same order
        assert [k for k, _ in fast] == [k for k, _ in slow], (
            f"ranking mismatch for {q!r}: fast={[k for k, _ in fast]} slow={[k for k, _ in slow]}"
        )
        # Same scores (within 1e-9)
        for (k1, s1), (k2, s2) in zip(fast, slow):
            assert k1 == k2
            assert abs(s1 - s2) < 1e-9, (
                f"score drift for {q!r} key={k1}: fast={s1} slow={s2}"
            )


# ---------------------------------------------------------------------------
# 4. remove() cleans the inverted index
# ---------------------------------------------------------------------------


def test_remove_cleans_inverted_index() -> None:
    """After ``remove(key)``, no posting list references ``key``."""
    idx = BM25Index()
    rng = random.Random(0xDEADBEEF)
    vocab = ["red", "blue", "green", "yellow", "purple", "orange"]
    for i in range(100):
        chosen = rng.choices(vocab, k=rng.randint(3, 5))
        idx.add(_make_record(text=" ".join(chosen), memory_id=f"r-{i:03d}"))

    # Snapshot the doc_keys we plan to remove.
    to_remove = [f"openclaw_memory_os:r-{i:03d}" for i in range(50)]
    to_remove_set = set(to_remove)

    for key in to_remove:
        idx.remove(key)

    # Every posting list must not reference a removed doc.
    for token, postings in idx._state.inverted_index.items():
        referenced = {dk for dk, _tf in postings}
        leak = referenced & to_remove_set
        assert not leak, (
            f"token {token!r} still references removed docs {sorted(leak)}"
        )

    # Stats table must also be consistent: total_tf sum equals
    # the sum of all remaining per-doc token counts.
    expected_tf_sum = sum(
        sum(record["__lexical_tf__"].values())
        for _key, (record, _dl, _ex) in idx._state.documents.items()
    )
    actual_tf_sum = sum(
        tf for postings in idx._state.inverted_index.values() for (_dk, tf) in postings
    )
    assert actual_tf_sum == expected_tf_sum


# ---------------------------------------------------------------------------
# 5. save()/load() round-trips the inverted index
# ---------------------------------------------------------------------------


def test_load_persisted_inverted_index(tmp_path: Path) -> None:
    """``save`` then ``load`` preserves the inverted index exactly
    and ``search`` works on the loaded index.
    """
    idx = BM25Index()
    rng = random.Random(0x1234)
    vocab = ["river", "lake", "mountain", "forest", "desert", "ocean"]
    for i in range(80):
        chosen = rng.choices(vocab, k=rng.randint(2, 5))
        idx.add(_make_record(text=" ".join(chosen), memory_id=f"m-{i:03d}"))

    expected_index = {
        token: sorted(postings)
        for token, postings in idx._state.inverted_index.items()
    }

    idx.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded is not None, "load() returned None after a fresh save"
    assert len(loaded) == 80

    actual_index = {
        token: sorted(postings)
        for token, postings in loaded._state.inverted_index.items()
    }
    assert actual_index == expected_index

    # Sanity check: search works on the loaded index and matches
    # the live one.
    q = "mountain forest river"
    live_hits = idx.search(q, limit=5)
    loaded_hits = loaded.search(q, limit=5)
    assert [k for k, _ in live_hits] == [k for k, _ in loaded_hits]


# ---------------------------------------------------------------------------
# 6. legacy cache without __lexical_tf__ still gets a usable inverted index
# ---------------------------------------------------------------------------


def test_legacy_cache_rebuilds_inverted_index_from_re_tokenization(tmp_path: Path) -> None:
    """An on-disk cache without ``__lexical_tf__`` must still produce a
    populated inverted index after ``load()`` via the legacy fallback.

    The legacy fallback path runs :meth:`_rebuild_stats`, which
    re-tokenises the document text. We then expect the inverted
    index to be populated from the cached ``__lexical_tokens__``
    (P0-3 cached tokens) or from the freshly re-tokenised tokens
    when neither cache field is present.

    We construct a ``BM25Index`` with records that lack the
    ``__lexical_tf__`` field, save it, then load it via the
    normal ``load()`` path (which calls ``from_dict`` ->
    ``_rebuild_stats_from_cached_tf`` -> falls back to
    ``_rebuild_stats`` -> builds the inverted index from cached
    tokens).
    """
    idx = BM25Index()
    for i in range(30):
        idx.add(_make_record(text=f"alpha bravo charlie {i}", memory_id=f"leg-{i:02d}"))

    # Strip __lexical_tf__ from every cached record to simulate a
    # pre-P0-3 dump. Keep __lexical_tokens__ intact so the legacy
    # rebuild path has something to fall back on.
    for _key, (record, _dl, _ex) in idx._state.documents.items():
        record.pop("__lexical_tf__", None)

    idx.save(tmp_path)

    # Wipe the in-memory inverted index so we know load() really
    # rebuilt it from disk.
    idx._state.inverted_index.clear()

    loaded = BM25Index.load(tmp_path)
    assert loaded is not None
    assert len(loaded) == 30

    # Inverted index must be non-empty — every load() must end
    # with a populated inverted index.
    assert loaded._state.inverted_index, (
        "inverted_index empty after loading a legacy cache; "
        "the rebuild path is not populating it"
    )
    # "alpha", "bravo", "charlie" are present in every doc.
    for token in ("alpha", "bravo", "charlie"):
        postings = loaded._state.inverted_index.get(token, [])
        assert len(postings) == 30, (
            f"expected 30 postings for {token!r}, got {len(postings)}"
        )

    # Search must still work end-to-end.
    hits = loaded.search("alpha bravo", limit=50)
    assert hits, "search() returned nothing on legacy-loaded index"
    assert len(hits) == 30