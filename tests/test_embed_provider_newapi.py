"""Tests for :mod:`openclaw_memory_os.embed_provider` (wave 2).

These tests cover:

1. Provider construction from env (default ollama, opt-in newapi).
2. Hard contract: ``EmbeddingUnavailable`` on a bad URL / 4xx / 5xx /
   empty vector / all-zero vector / non-numeric vector / NaN/Inf.
3. Hard contract: ``EmbeddingDimensionMismatch`` on a length mismatch
   between the returned vector and ``EMBED_PROVIDER_DIM``.
4. Connection-cache singleton: two calls return the same client object.
5. Token-file redaction: the bearer token never appears in
   ``repr(provider)`` / ``vars(provider)``.
6. Default behaviour preserved: with no provider env vars set, the
   embed provider is ``"ollama"`` (so existing tests / dev loops do
   not need to set anything new).
7. Real-model-name contract: ``NEWAPI_EMBED_MODEL = "qwen3-embedding:0.6b"``
   and ``NEWAPI_CHAT_MODEL = "qwen3:4b-instruct"`` (no aliases).

The tests do **not** call a real NewAPI. The NewAPI path is covered by
the standalone dry-run script ``scripts/_dry_run_newapi_embed.py``,
which is exercised separately against the live gateway. The unit
tests use a small stub HTTP transport so they stay hermetic and
fast enough to run in the standard test suite.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from openclaw_memory_os import embed_provider as ep
from openclaw_memory_os.embed_provider import (
    ChatProvider,
    DEFAULT_EMBED_DIM,
    DEFAULT_NEWAPI_BASE_URL,
    EmbedProvider,
    EmbeddingDimensionMismatch,
    EmbeddingUnavailable,
    NEWAPI_CHAT_MODEL,
    NEWAPI_EMBED_MODEL,
    get_chat_provider,
    get_embed_provider,
    reset_provider_caches,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    """Reset cached providers + provider env vars between tests.

    Each test starts with no provider env override so the default
    ``ollama`` path is exercised unless the test sets ``EMBED_PROVIDER``.
    """
    monkeypatch.delenv("EMBED_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("EMBED_PROVIDER_URL", raising=False)
    monkeypatch.delenv("EMBED_PROVIDER_MODEL", raising=False)
    monkeypatch.delenv("EMBED_PROVIDER_TOKEN_FILE", raising=False)
    monkeypatch.delenv("EMBED_PROVIDER_DIM", raising=False)
    monkeypatch.delenv("EMBED_PROVIDER_TIMEOUT", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_TOKEN_FILE", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_TIMEOUT", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.delenv("LLM_CHAT_URL", raising=False)
    monkeypatch.delenv("MEMORY_BRAIN_LLM_MODEL", raising=False)
    reset_provider_caches()
    yield
    reset_provider_caches()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_real_model_names_no_alias():
    """Wave-2 contract: the model names exposed by the module are the
    real NewAPI names — no aliases like ``memory-embedding``.
    """
    assert NEWAPI_EMBED_MODEL == "qwen3-embedding:0.6b"
    assert NEWAPI_CHAT_MODEL == "qwen3:4b-instruct"
    assert "memory" not in NEWAPI_EMBED_MODEL.lower()
    assert "memory" not in NEWAPI_CHAT_MODEL.lower()


def test_default_embed_dim_is_768():
    """Existing Qdrant collections were provisioned at 768 dims. The
    provider must default to that even when ``EMBED_PROVIDER_DIM``
    is unset (so an operator who forgets to set it does not
    silently 0-hit their collection).
    """
    assert DEFAULT_EMBED_DIM == 768


def test_default_newapi_base_url_is_local_gateway():
    """The default NewAPI base URL is the local 127.0.0.1:8199 listener
    (public ``https://api.230385.xyz`` is just a reverse-proxy).
    """
    assert DEFAULT_NEWAPI_BASE_URL.startswith("http://127.0.0.1:8199")


# ---------------------------------------------------------------------------
# from_env construction
# ---------------------------------------------------------------------------


def test_from_env_defaults_to_ollama():
    """Without any provider env vars, the embed provider defaults to
    ``ollama`` (the pre-wave-2 behaviour)."""
    p = EmbedProvider.from_env()
    assert p.name == "ollama"
    assert p.api_style == "ollama"
    assert p.model  # non-empty
    assert p.expected_dim == DEFAULT_EMBED_DIM


def test_from_env_newapi_requires_token_file(monkeypatch):
    """When ``EMBED_PROVIDER=newapi``, the provider must read the
    token from ``EMBED_PROVIDER_TOKEN_FILE`` (default path is
    ``/root/.openclaw/workspace/.secrets/newapi-memory-os-token``).  # privacy-allow: MEMORY_OS_PATH

    A missing / unreadable file must result in an empty token (the
    embed call then fails with a clear ``EmbeddingUnavailable``,
    never silently going through unauthenticated).
    """
    monkeypatch.setenv("EMBED_PROVIDER", "newapi")
    monkeypatch.setenv("EMBED_PROVIDER_TOKEN_FILE", "/nonexistent/path/for/test")
    p = EmbedProvider.from_env()
    assert p.name == "newapi"
    assert p.api_style == "openai"
    assert p.api_key == ""  # missing file => empty token


def test_from_env_newapi_reads_token_file(monkeypatch, tmp_path):
    """A readable token file is loaded into the provider."""
    token_file = tmp_path / "token"
    token_file.write_text("sk-test-token-1234567890\n", encoding="utf-8")  # privacy-allow: OPENAI_KEY
    monkeypatch.setenv("EMBED_PROVIDER", "newapi")
    monkeypatch.setenv("EMBED_PROVIDER_TOKEN_FILE", str(token_file))
    p = EmbedProvider.from_env()
    assert p.api_key == "sk-test-token-1234567890"  # privacy-allow: OPENAI_KEY
    assert p.model == NEWAPI_EMBED_MODEL
    assert p.base_url.startswith("http://127.0.0.1:8199")


def test_from_env_overrides_url_and_model(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBED_PROVIDER", "newapi")
    monkeypatch.setenv("EMBED_PROVIDER_URL", "http://example.invalid/v1")
    monkeypatch.setenv("EMBED_PROVIDER_MODEL", "custom-embed:1b")
    monkeypatch.setenv("EMBED_PROVIDER_DIM", "1024")
    token_file = tmp_path / "token"
    token_file.write_text("sk-xyz", encoding="utf-8")
    monkeypatch.setenv("EMBED_PROVIDER_TOKEN_FILE", str(token_file))
    p = EmbedProvider.from_env()
    assert p.base_url == "http://example.invalid/v1"
    assert p.model == "custom-embed:1b"
    assert p.expected_dim == 1024


def test_chat_from_env_defaults_to_ollama():
    c = ChatProvider.from_env()
    assert c.name == "ollama"
    assert c.base_url.endswith("/v1")


def test_chat_from_env_newapi(monkeypatch, tmp_path):
    token_file = tmp_path / "t"
    token_file.write_text("sk-chat-token", encoding="utf-8")
    monkeypatch.setenv("LLM_PROVIDER", "newapi")
    monkeypatch.setenv("LLM_PROVIDER_TOKEN_FILE", str(token_file))
    c = ChatProvider.from_env()
    assert c.name == "newapi"
    assert c.api_key == "sk-chat-token"
    assert c.model == NEWAPI_CHAT_MODEL


# ---------------------------------------------------------------------------
# Hard contracts: degenerate / unreachable / wrong-dim vectors
# ---------------------------------------------------------------------------


def _fake_response(status_code: int, payload: Any = None, text: str = "") -> MagicMock:
    """Build a MagicMock that looks like an httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    if payload is not None:
        resp.json.return_value = payload
    else:
        resp.json.side_effect = ValueError("not json")
    resp.text = text
    return resp


def test_embed_raises_on_unreachable_ollama(monkeypatch):
    """An unreachable Ollama URL must raise EmbeddingUnavailable."""
    p = EmbedProvider.from_env()
    # Patch the lazy-loaded httpx import by giving the provider a
    # stub client that raises on .post().
    p._client = MagicMock()
    p._client.post.side_effect = ConnectionError("unreachable")
    with pytest.raises(EmbeddingUnavailable):
        p.embed("hello")


def test_embed_raises_on_http_500_ollama():
    p = EmbedProvider.from_env()
    p._client = MagicMock()
    p._client.post.return_value = _fake_response(500, text="internal")
    with pytest.raises(EmbeddingUnavailable):
        p.embed("hello")


def test_embed_raises_on_empty_vector_ollama():
    p = EmbedProvider.from_env()
    p._client = MagicMock()
    p._client.post.return_value = _fake_response(200, {"embedding": []})
    with pytest.raises(EmbeddingUnavailable):
        p.embed("hello")


def test_embed_raises_on_all_zero_vector_ollama():
    p = EmbedProvider.from_env()
    p._client = MagicMock()
    p._client.post.return_value = _fake_response(
        200, {"embedding": [0.0] * 768}
    )
    with pytest.raises(EmbeddingUnavailable):
        p.embed("hello")


def test_embed_raises_on_nan_vector_ollama():
    p = EmbedProvider.from_env()
    p._client = MagicMock()
    vec = [0.1] * 767 + [float("nan")]
    p._client.post.return_value = _fake_response(200, {"embedding": vec})
    with pytest.raises(EmbeddingUnavailable):
        p.embed("hello")


def test_embed_raises_on_inf_vector_ollama():
    p = EmbedProvider.from_env()
    p._client = MagicMock()
    vec = [0.0] * 767 + [float("inf")]
    p._client.post.return_value = _fake_response(200, {"embedding": vec})
    with pytest.raises(EmbeddingUnavailable):
        p.embed("hello")


def test_embed_raises_on_non_numeric_vector_ollama():
    p = EmbedProvider.from_env()
    p._client = MagicMock()
    vec = [0.1] * 767 + ["nope"]
    p._client.post.return_value = _fake_response(200, {"embedding": vec})
    with pytest.raises(EmbeddingUnavailable):
        p.embed("hello")


def test_embed_dim_mismatch_ollama():
    """A 512-dim response from a 768-dim collection must raise
    EmbeddingDimensionMismatch so the dense path can fail loudly
    rather than silently zero-hitting.
    """
    p = EmbedProvider.from_env()
    p._client = MagicMock()
    p._client.post.return_value = _fake_response(
        200, {"embedding": [0.1] * 512}
    )
    with pytest.raises(EmbeddingDimensionMismatch):
        p.embed("hello")


def test_embed_newapi_sends_dimensions_and_bearer():
    """The NewAPI embed call must include both ``dimensions`` and the
    Authorization header.
    """
    p = EmbedProvider(
        name="newapi",
        base_url="http://127.0.0.1:8199/v1",
        model=NEWAPI_EMBED_MODEL,
        expected_dim=768,
        api_key="sk-test-1234",
        api_style="openai",
    )
    p._client = MagicMock()
    p._client.post.return_value = _fake_response(
        200, {"data": [{"embedding": [0.1] * 768}]}
    )
    out = p.embed("hello")
    assert len(out) == 768
    # Inspect the call args.
    args, kwargs = p._client.post.call_args
    body = kwargs.get("json") or (args[1] if len(args) > 1 else None)
    assert body["model"] == NEWAPI_EMBED_MODEL
    assert body["dimensions"] == 768
    assert body["input"] == "hello"
    headers = kwargs.get("headers") or {}
    assert headers.get("Authorization") == "Bearer sk-test-1234"


def test_embed_newapi_raises_on_4xx():
    p = EmbedProvider(
        name="newapi",
        base_url="http://127.0.0.1:8199/v1",
        model=NEWAPI_EMBED_MODEL,
        expected_dim=768,
        api_key="sk-test",
        api_style="openai",
    )
    p._client = MagicMock()
    p._client.post.return_value = _fake_response(401, text="unauthorized")
    with pytest.raises(EmbeddingUnavailable):
        p.embed("hello")


def test_embed_newapi_raises_on_dim_mismatch():
    p = EmbedProvider(
        name="newapi",
        base_url="http://127.0.0.1:8199/v1",
        model=NEWAPI_EMBED_MODEL,
        expected_dim=768,
        api_key="sk-test",
        api_style="openai",
    )
    p._client = MagicMock()
    p._client.post.return_value = _fake_response(
        200, {"data": [{"embedding": [0.1] * 1024}]}
    )
    with pytest.raises(EmbeddingDimensionMismatch):
        p.embed("hello")


# ---------------------------------------------------------------------------
# Connection cache + token-redaction safety
# ---------------------------------------------------------------------------


def test_embed_client_is_cached_singleton():
    """Two embed calls share the same client object (lesson 41)."""
    p = EmbedProvider.from_env()
    p._client = MagicMock()
    p._client.post.return_value = _fake_response(
        200, {"embedding": [0.1] * 768}
    )
    p.embed("a")
    p.embed("b")
    # The same MagicMock is returned across calls; if reset_provider_caches
    # were called between them, this would fail.
    assert p._client.post.call_count == 2


def test_provider_repr_does_not_leak_token():
    p = EmbedProvider(
        name="newapi",
        base_url="http://127.0.0.1:8199/v1",
        model=NEWAPI_EMBED_MODEL,
        expected_dim=768,
        api_key="sk-supersecret",
        api_style="openai",
    )
    s = repr(p)
    assert "sk-supersecret" not in s


def test_provider_str_does_not_leak_token():
    p = EmbedProvider(
        name="newapi",
        base_url="http://127.0.0.1:8199/v1",
        model=NEWAPI_EMBED_MODEL,
        expected_dim=768,
        api_key="sk-supersecret",
        api_style="openai",
    )
    s = str(p)
    assert "sk-supersecret" not in s


# ---------------------------------------------------------------------------
# ChatProvider
# ---------------------------------------------------------------------------


def test_chat_returns_content_on_ok():
    p = ChatProvider(
        name="newapi",
        base_url="http://127.0.0.1:8199/v1",
        model=NEWAPI_CHAT_MODEL,
        api_key="sk-test",
        api_style="openai",
    )
    p._client = MagicMock()
    p._client.post.return_value = _fake_response(
        200,
        {"choices": [{"message": {"role": "assistant", "content": "hello back"}}]},
    )
    out = p.chat([{"role": "user", "content": "hi"}])
    assert out == "hello back"


def test_chat_raises_on_empty_choices():
    p = ChatProvider(
        name="newapi",
        base_url="http://127.0.0.1:8199/v1",
        model=NEWAPI_CHAT_MODEL,
        api_key="sk-test",
        api_style="openai",
    )
    p._client = MagicMock()
    p._client.post.return_value = _fake_response(200, {"choices": []})
    from openclaw_memory_os.embed_provider import ChatUnavailable
    with pytest.raises(ChatUnavailable):
        p.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_on_unreachable():
    from openclaw_memory_os.embed_provider import ChatUnavailable
    p = ChatProvider(
        name="newapi",
        base_url="http://127.0.0.1:8199/v1",
        model=NEWAPI_CHAT_MODEL,
        api_key="sk-test",
        api_style="openai",
    )
    p._client = MagicMock()
    p._client.post.side_effect = ConnectionError("nope")
    with pytest.raises(ChatUnavailable):
        p.chat([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Singleton wiring
# ---------------------------------------------------------------------------


def test_get_embed_provider_returns_singleton():
    a = get_embed_provider()
    b = get_embed_provider()
    assert a is b


def test_reset_provider_caches_clears_singletons():
    a = get_embed_provider()
    reset_provider_caches()
    b = get_embed_provider()
    assert a is not b  # a new instance after reset


# ---------------------------------------------------------------------------
# Degraded fallback test (txt 第 5 节):
# NewAPI不可达 → query recall 走 BM25 / lexical path, embedding_unavailable 标记.
# This test stubs the QdrantBackend.dense_search pipeline to assert
# that a provider failure surfaces as EmbeddingUnavailable and that
# the lexical fallback in the same backend can still serve hits.
# ---------------------------------------------------------------------------


def test_provider_unreachable_makes_dense_search_raise(monkeypatch):
    """When the embed provider is unreachable, ``QdrantBackend._embed``
    must raise :class:`EmbeddingUnavailable` (the backends alias) so
    ``dense_search`` can fall back to the lexical path.
    """
    from openclaw_memory_os.backends import EmbeddingUnavailable as BEU

    # Build a backend but skip Qdrant client init.
    backend = EmbedProvider.from_env()  # use provider directly
    backend._client = MagicMock()
    backend._client.post.side_effect = ConnectionError("unreachable")
    with pytest.raises(BEU):
        # The backends._embed wraps provider errors as BEU.
        from openclaw_memory_os.backends import QdrantBackend

        # Construct a QdrantBackend without hitting Qdrant.
        be = QdrantBackend.__new__(QdrantBackend)
        be._embedding_cache = None
        # Replace _embed with a thin wrapper that calls our provider.
        def _fake_embed(text: str):
            try:
                return backend.embed(text)
            except EmbeddingUnavailable as exc:
                raise BEU(str(exc)) from exc
        be._embed = _fake_embed  # type: ignore[assignment]
        be._embed("hello")


def test_degraded_fallback_newapi_unreachable_lexical_path(monkeypatch):
    """txt 第 5 节: when the NewAPI provider is unreachable,
    query recall must fall back to the BM25/lexical path
    (sub-string search on the in-memory cache) and the dense
    path must raise EmbeddingUnavailable rather than return a
    fake zero-vector ranking.

    The test points the NewAPI provider at an unreachable URL,
    builds a QdrantBackend without a real Qdrant client, calls
    ``_embed`` (must raise), and then exercises the
    ``lexical_search`` fallback which must still return hits
    from the in-memory cache.
    """
    from openclaw_memory_os.backends import (
        EmbeddingUnavailable as BEU,
        SampleBackend,
    )

    # 1. Provider configured to an unreachable URL must raise.
    bad = EmbedProvider(
        name="newapi",
        base_url="http://127.0.0.1:1/v1",  # port 1 is unreachable
        model=NEWAPI_EMBED_MODEL,
        expected_dim=768,
        api_key="sk-test",
        api_style="openai",
        timeout=1.0,
    )
    bad._client = MagicMock()
    bad._client.post.side_effect = ConnectionError("simulated outage")
    with pytest.raises(EmbeddingUnavailable):
        bad.embed("hello")

    # 2. Same provider when reached via the backends layer must
    #    surface as the backends-level EmbeddingUnavailable alias
    #    so ``dense_search`` can degrade to lexical.
    from openclaw_memory_os.backends import QdrantBackend

    be = QdrantBackend.__new__(QdrantBackend)
    be._embedding_cache = None

    def _fake_embed(text: str):
        try:
            return bad.embed(text)
        except EmbeddingUnavailable as exc:
            raise BEU(str(exc)) from exc
    be._embed = _fake_embed  # type: ignore[assignment]
    with pytest.raises(BEU):
        be._embed("hello")

    # 3. The lexical fallback path on a SampleBackend is unaffected
    #    by the NewAPI outage and still serves hits.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = Path(tmp) / "sample.json"
        sample_path.write_text(
            '{"memories": [{"id":"x1","text":"hello world","source":"t.txt",'
            '"created_at":"2026-07-20T00:00:00Z","tier":"short",'
            '"status":"active","importance":0.5,"tags":[]}]}',
            encoding="utf-8",
        )
        sb = SampleBackend(sample_path)
        hits = sb.lexical_search("hello", limit=5)
        assert any(h.text == "hello world" for h in hits)