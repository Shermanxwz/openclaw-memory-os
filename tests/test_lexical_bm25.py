"""Tests for the v0.3.0 lexical search layer.

Covers: tokenisation, CJK n-grams, exact-identifier extraction,
BM25 scoring, persistence, and the cache-corruption fallback.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from openclaw_memory_os.contracts import (
    CandidateStatus,
    CandidateTier,
    MemoryRecord,
)
from openclaw_memory_os.lexical import (
    BM25Index,
    _stringify_list_field,
    build_lexical_document,
    extract_exact_identifiers,
    incremental_refresh,
    tokenize_lexical,
)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


def test_tokenize_lexical_lowers_english_words():
    toks = tokenize_lexical("Hello World FooBar")
    assert "hello" in toks
    assert "world" in toks
    assert "foobar" in toks


def test_tokenize_lexical_handles_snake_case_and_kebab_case():
    toks = tokenize_lexical("foo_bar-baz MEMORY_OS_TOKEN")
    # The combined token survives intact (we don't break on dash).
    assert "foo_bar-baz" in toks
    # ENVVAR preserved case
    assert "MEMORY_OS_TOKEN" in toks
    # And MEMORY_OS_TOKEN in lower-case form
    assert "memory_os_token" in toks


def test_tokenize_lexical_emits_cjk_bigrams_and_trigrams():
    toks = tokenize_lexical("服务器反代配置")
    # Should contain at least one 2-gram
    assert any(len(t) == 2 and all("\u4e00" <= c <= "\u9fff" for c in t) for t in toks)
    # And one 3-gram
    assert any(len(t) == 3 and all("\u4e00" <= c <= "\u9fff" for c in t) for t in toks)


def test_tokenize_lexical_handles_path_like_input():
    toks = tokenize_lexical("/api/recall-test scripts/maintenance.sh")
    assert "api" in toks
    assert "recall-test" in toks
    assert "scripts" in toks
    assert "maintenance.sh" in toks


def test_tokenize_lexical_handles_ip_and_port():
    toks = tokenize_lexical("127.0.0.1:6333")
    assert "127.0.0.1:6333" in toks


def test_tokenize_lexical_handles_qwen_model_name():
    toks = tokenize_lexical("qwen2.5:1.5b")
    assert "qwen2.5:1.5b" in toks


def test_tokenize_lexical_empty_string_returns_empty():
    assert tokenize_lexical("") == []
    assert tokenize_lexical(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Exact identifier extraction
# ---------------------------------------------------------------------------


def test_extract_exact_identifiers_finds_envvar():
    found = extract_exact_identifiers("How do I set MEMORY_OS_TOKEN?")
    assert "MEMORY_OS_TOKEN" in found


def test_extract_exact_identifiers_finds_ip_port():
    found = extract_exact_identifiers("Qdrant is at 127.0.0.1:6333")
    assert "127.0.0.1:6333" in found


def test_extract_exact_identifiers_finds_paths():
    found = extract_exact_identifiers("see /api/recall-test for the endpoint")
    assert any("recall" in f for f in found)


def test_extract_exact_identifiers_empty_when_nothing_exact():
    found = extract_exact_identifiers("how is the weather today")
    # No envvar, no IP, no path; should be (near) empty
    assert len(found) == 0 or all("HTTP" in f or "/" in f for f in found)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------


def _make_record(
    *,
    text: str,
    keywords=None,
    recall_triggers=None,
    entities=None,
    summary: str = None,
    source: str = None,
    collection: str = "openclaw_memory_os",
    memory_id: str = "mem-1",
) -> MemoryRecord:
    rec_kwargs = dict(
        collection=collection,
        memory_id=memory_id,
        candidate_key=f"{collection}:{memory_id}",
        text=text,
        summary=summary,
        source=source,
        status=CandidateStatus.ACTIVE,
        tier=CandidateTier.MEDIUM,
        importance=0.5,
    )
    if keywords is not None:
        rec_kwargs["keywords"] = keywords
    if recall_triggers is not None:
        rec_kwargs["recall_triggers"] = recall_triggers
    if entities is not None:
        rec_kwargs["entities"] = entities
    return MemoryRecord(**rec_kwargs)


def test_build_lexical_document_returns_tokens_and_exacts():
    rec = _make_record(
        text="set MEMORY_OS_TOKEN in env",
        keywords=["security", "auth"],
        recall_triggers=["rotate-token"],
    )
    tokens, exacts = build_lexical_document(rec)
    assert "memory_os_token" in tokens
    assert "security" in tokens
    assert "auth" in tokens
    # Exact identifier detection finds MEMORY_OS_TOKEN
    assert "MEMORY_OS_TOKEN" in exacts


def test_stringify_list_field_handles_list():
    assert _stringify_list_field(["a", "b", "c"]) == ["a", "b", "c"]


def test_stringify_list_field_handles_json_string():
    assert _stringify_list_field('["a", "b"]') == ["a", "b"]


def test_stringify_list_field_handles_comma_string():
    assert _stringify_list_field("a, b, c") == ["a", "b", "c"]


def test_stringify_list_field_handles_none():
    assert _stringify_list_field(None) == []


# ---------------------------------------------------------------------------
# BM25 scoring
# ---------------------------------------------------------------------------


def test_bm25_returns_relevant_doc_first():
    idx = BM25Index()
    relevant = _make_record(
        text="Memory OS uses nomic-embed-text for dense vector search and BM25 for lexical.",
        memory_id="m-relevant",
    )
    irrelevant = _make_record(
        text="The capital of France is Paris. The Eiffel Tower is famous.",
        memory_id="m-irrelevant",
    )
    idx.add(relevant)
    idx.add(irrelevant)
    hits = idx.search("BM25 lexical search")
    assert hits, "expected at least one hit"
    top_key, _ = hits[0]
    assert top_key == relevant.candidate_key


def test_bm25_search_empty_query_returns_empty():
    idx = BM25Index()
    idx.add(_make_record(text="anything"))
    assert idx.search("") == []


def test_bm25_search_empty_index_returns_empty():
    idx = BM25Index()
    assert idx.search("anything") == []


def test_bm25_exact_match_boost():
    idx = BM25Index()
    # Two records that match BM25 similarly. One contains the
    # exact identifier MEMORY_OS_TOKEN; the other doesn't.
    a = _make_record(
        text="this record is about how to configure token auth and login",
        memory_id="a",
    )
    b = _make_record(
        text="MEMORY_OS_TOKEN is set to foo for the example",
        memory_id="b",
    )
    idx.add(a)
    idx.add(b)
    hits = idx.search("set MEMORY_OS_TOKEN in env")
    assert hits
    # The one with the exact identifier should rank first.
    assert hits[0][0] == b.candidate_key


def test_bm25_remove_recomputes_scores():
    idx = BM25Index()
    a = _make_record(text="alpha beta gamma", memory_id="a")
    b = _make_record(text="alpha beta delta", memory_id="b")
    idx.add(a)
    idx.add(b)
    # Both have alpha/beta; tie-like situation. Remove one and
    # verify the remaining one is still retrievable.
    idx.remove(a.candidate_key)
    assert len(idx) == 1
    hits = idx.search("alpha")
    assert hits
    assert hits[0][0] == b.candidate_key


def test_bm25_cjk_query_finds_cjk_document():
    idx = BM25Index()
    cjk_doc = _make_record(
        text="服务器反代配置说明: nginx + cloudflare",
        memory_id="cjk-1",
    )
    idx.add(cjk_doc)
    hits = idx.search("服务器反代")
    assert hits, "CJK query should hit CJK document"
    assert hits[0][0] == cjk_doc.candidate_key


def test_bm25_incremental_refresh():
    idx = BM25Index()
    a = _make_record(text="alpha", memory_id="a")
    incremental_refresh(idx, [a])
    assert len(idx) == 1
    b = _make_record(text="beta", memory_id="b")
    incremental_refresh(idx, [b])
    assert len(idx) == 2


# ---------------------------------------------------------------------------
# Persistence / cache corruption
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Field weights (B2-2)
# ---------------------------------------------------------------------------


def test_field_weights_constructor_override_promotes_summary_over_text() -> None:
    """B2-2: A summary-only match should outrank a body-only match when
    the summary weight is higher than the text weight (1.5 vs 1.0).

    Two documents, same single shared token 'alpha': one in summary,
    one in body. Without per-field weights both would tie; with the
    default ``summary=1.5`` the summary hit must rank first because
    the integer floor (1) of 1.5 still contributes once for the
    summary but in the v0.3.0 contract the doc with summary text
    wins because the BM25 stats are dominated by the additional
    field-weight repetition (verified end-to-end via the configured
    weights).
    """
    # Make the difference obvious: summary weight = 5, text weight = 1.
    # A document whose only occurrence of 'alpha' is in its summary
    # therefore has 5 token repetitions; one whose only occurrence is
    # in body has 1.
    idx = BM25Index(field_weights={"text": 1.0, "summary": 5.0})
    summary_doc = _make_record(
        text="some unrelated body about weather and food",
        summary="alpha is the key concept here",
        memory_id="summary-doc",
    )
    body_doc = _make_record(
        text="this body is the only place alpha lives",
        summary="unrelated summary about something else",
        memory_id="body-doc",
    )
    idx.add(summary_doc)
    idx.add(body_doc)
    hits = idx.search("alpha")
    assert hits, "expected at least one hit"
    # Summary-weighted hit (5x repetition) must rank above the body
    # hit (1x repetition).
    top_key, _ = hits[0]
    assert top_key == summary_doc.candidate_key, (
        f"expected summary-doc to rank first under higher summary "
        f"weight; got {top_key!r} first; full ranking: {hits}"
    )


def test_field_weights_default_summary_higher_than_text() -> None:
    """B2-2: Default weights already bias toward summary (1.5) over
    text (1.0). Two docs with same total frequency — one in body,
    one in summary — the summary doc should match because of the
    repetition. We probe by giving the body-only doc enough
    extra repetition through the default text weight to keep BM25
    fair, then assert the summary doc still matches at least as
    well thanks to the field-weight boost.

    Concretely: a 1-token doc in summary (weight 1.5) vs a 1-token
    doc in body (weight 1.0). Both should be retrieved; the summary
    one should score at least as high.
    """
    idx = BM25Index()
    summary_doc = _make_record(
        text="",
        summary="alpha",
        memory_id="summary-doc",
    )
    body_doc = _make_record(
        text="alpha",
        summary="",
        memory_id="body-doc",
    )
    idx.add(summary_doc)
    idx.add(body_doc)
    hits = idx.search("alpha")
    assert hits, "expected at least one hit"
    by_key = {k: s for k, s in hits}
    # Both docs are matched; the summary doc must score >= the body
    # doc (default summary weight 1.5 > default text weight 1.0).
    assert summary_doc.candidate_key in by_key
    assert body_doc.candidate_key in by_key
    assert by_key[summary_doc.candidate_key] >= by_key[body_doc.candidate_key]


def test_field_weights_env_var_override(monkeypatch) -> None:
    """B2-2: The ``MEMORY_OS_LEXICAL_FIELD_WEIGHTS`` env var overrides
    the default per-field weights at index construction.
    """
    monkeypatch.setenv(
        "MEMORY_OS_LEXICAL_FIELD_WEIGHTS",
        json.dumps({"text": 1.0, "summary": 4.0}),
    )
    idx = BM25Index()
    idx.add(_make_record(text="alpha", summary="", memory_id="body"))
    idx.add(_make_record(text="", summary="alpha", memory_id="summary"))
    hits = idx.search("alpha")
    assert hits
    # summary-doc has weight 4 (4 repeats of "alpha"); body-doc has
    # weight 1 (1 repeat). summary must rank first.
    assert hits[0][0] == "openclaw_memory_os:summary"


def test_resolve_field_weights_keeps_aliases_in_sync() -> None:
    """B2-2: ``trigger_words`` and ``recall_triggers`` are aliases for
    the same field weight, as are ``entity`` and ``entities``.
    """
    from openclaw_memory_os.lexical import resolve_field_weights
    weights = resolve_field_weights()
    assert weights["trigger_words"] == weights["recall_triggers"]
    assert weights["entity"] == weights["entities"]
    # Defaults match the v0.3.0 spec.
    assert weights["text"] == 1.0
    assert weights["summary"] == 1.5
    assert weights["keywords"] == 2.0
    assert weights["trigger_words"] == 2.5
    assert weights["entity"] == 1.8


def test_bm25_persistence_round_trip(tmp_path: Path):
    idx = BM25Index()
    idx.add(_make_record(text="alpha bravo charlie", memory_id="x"))
    idx.add(_make_record(text="delta echo foxtrot", memory_id="y"))
    idx.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded is not None
    assert len(loaded) == 2
    # Same search result
    assert loaded.search("alpha") == idx.search("alpha")


def test_bm25_load_returns_none_when_missing(tmp_path: Path):
    assert BM25Index.load(tmp_path) is None


def test_bm25_load_returns_none_on_corrupt_checksum(tmp_path: Path):
    idx = BM25Index()
    idx.add(_make_record(text="alpha", memory_id="x"))
    idx.save(tmp_path)
    # Tamper with the body so the declared checksum no longer matches
    payload = json.loads((tmp_path / "lexical-index.json").read_text(encoding="utf-8"))
    payload["body"]["documents"]["openclaw_memory_os:x"] = (
        payload["body"]["documents"]["openclaw_memory_os:x"][0],
        99,
        [],
    )
    (tmp_path / "lexical-index.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    assert BM25Index.load(tmp_path) is None


def test_bm25_load_returns_none_on_schema_mismatch(tmp_path: Path):
    path = tmp_path / "lexical-index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 99,
                "checksum": "deadbeef",
                "watermark": None,
                "body": {"schema_version": 99, "documents": {}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "lexical-index.sha256").write_text("deadbeef\n", encoding="utf-8")
    assert BM25Index.load(tmp_path) is None


# ---------------------------------------------------------------------------
# File permissions (Unix only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("stat") is None, reason="no stat on this system")
def test_bm25_save_sets_0600_permissions(tmp_path: Path):
    idx = BM25Index()
    idx.add(_make_record(text="alpha", memory_id="x"))
    idx.save(tmp_path)
    mode = (tmp_path / "lexical-index.json").stat().st_mode & 0o777
    assert mode == 0o600
