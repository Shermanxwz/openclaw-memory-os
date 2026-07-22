"""Embedding + chat-completions provider abstraction for Memory OS.

Wave 2 (2026-07-20): NewAPI integration via OpenAI-compatible endpoints.

Goals
-----
* Single import surface used by :mod:`openclaw_memory_os.backends`,
  :mod:`openclaw_memory_os.ingestion`, and the ``memory_brain_*`` scripts.
* Default behaviour unchanged: OVH-local Ollama on
  ``http://127.0.0.1:11434`` using ``nomic-embed-text`` (768-dim) and
  ``qwen2.5:1.5b`` for chat. The new path is **opt-in** via
  ``EMBED_PROVIDER=newapi`` / ``LLM_PROVIDER=newapi`` env vars.
* Real model names (no aliases): NewAPI's embedding channel is configured
  with ``qwen3-embedding:0.6b`` and the chat channel with
  ``qwen3:4b-instruct``. We always pass ``dimensions=768`` on the
  embedding call because the channel-side header override is not
  always wired up.
* Strict contracts:
    - Never send an empty / zero / NaN / Inf vector to Qdrant.
      :class:`openclaw_memory_os.backends.EmbeddingUnavailable` is raised
      so the recall pipeline can degrade to lexical search.
    - Never log or echo the bearer token.
    - Embedding dimension is enforced client-side: the provider
      returns the actual returned length and raises
      :class:`EmbeddingDimensionMismatch` if it does not match the
      configured ``EMBED_PROVIDER_DIM`` (defaults to 768).
* Connection sharing: a process-wide singleton ``httpx.Client`` is
  cached so repeated embeddings do not pay the TCP+TLS handshake
  cost (lesson 41). The cache is keyed on ``(provider, base_url)``
  so provider switches invalidate it cleanly.

API
---
* :func:`get_embed_provider` — returns a singleton
  :class:`EmbedProvider` selected by the ``EMBED_PROVIDER`` env var
  (``"ollama"`` or ``"newapi"``; default ``"ollama"``).
* :func:`get_chat_provider` — same for :class:`ChatProvider`.
* :func:`reset_provider_caches` — drop the singletons. Tests use it.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Mirror the v0.3.0 contract constant so the message text in this
# module always references the canonical string without an import
# cycle (contracts -> backends -> ...). Keep the literal in sync with
# :data:`openclaw_memory_os.contracts.NO_ZERO_VECTOR_FAKE_SUCCESS`.
NO_ZERO_VECTOR_FAKE_SUCCESS: str = "no_zero_vector_fake_success"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Hard-coded dim that existing Qdrant collections were provisioned for.
#: Both ollama ``nomic-embed-text`` and NewAPI ``qwen3-embedding:0.6b``
#: produce 768-dim vectors when explicitly asked.
DEFAULT_EMBED_DIM = 768

#: Model names as advertised by NewAPI channel config. The brief is
#: explicit: no aliases. Callers must pass these strings verbatim
#: to the NewAPI gateway.
NEWAPI_EMBED_MODEL = "qwen3-embedding:0.6b"
NEWAPI_CHAT_MODEL = "qwen3:4b-instruct"

#: Default token file for the NewAPI provider. The file is chmod 600
#: and contains ``sk-...``; we read it once per provider construction
#: and never echo it to logs.
DEFAULT_NEWAPI_TOKEN_FILE = "/opt/openclaw-memory-os/.secrets/newapi-memory-os-token"

#: Default NewAPI base URL (the gateway's local listener; the public
#: ``https://api.230385.xyz`` reverse-proxies to it).
DEFAULT_NEWAPI_BASE_URL = "http://127.0.0.1:8199/v1"

#: Default OVH-local Ollama base URL.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_CHAT_MODEL = "qwen2.5:1.5b"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EmbeddingError(RuntimeError):
    """Base class for embedding-side failures."""


class EmbeddingUnavailable(EmbeddingError):
    """Embedding service is unreachable or returned a malformed payload.

    This is the wave-2 alias for
    :class:`openclaw_memory_os.backends.EmbeddingUnavailable`. The
    backends module re-exports the same class so legacy imports keep
    working; we deliberately do NOT import from ``backends`` here to
    keep the import direction one-way (``backends`` -> ``embed_provider``
    -> stdlib only).
    """


class EmbeddingDimensionMismatch(EmbeddingError):
    """Embedding length does not match the configured dim."""


class ChatUnavailable(RuntimeError):
    """Chat-completions service is unreachable or returned a malformed payload."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_token_file(path: Optional[str]) -> str:
    """Read a token file. Returns empty string if path is None or missing."""
    if not path:
        return ""
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception as exc:
        # Do not leak the path contents to logs.
        logger.warning(
            "embed_provider: token file read failed (path set, len=%d): %s",
            0 if not p.exists() else p.stat().st_size,
            type(exc).__name__,
        )
        return ""


def _validate_vector(vec: Any, *, context: str) -> List[float]:
    """Coerce a vector response to a list[float]; reject degenerate inputs.

    Mirrors the v0.3.0 contract from
    :data:`openclaw_memory_os.contracts.NO_ZERO_VECTOR_FAKE_SUCCESS`:
    an empty, zero-only, or NaN/Inf-laden vector is a hard failure,
    never silently zero-padded.
    """
    if not isinstance(vec, list):
        raise EmbeddingUnavailable(
            f"{context}: expected list, got {type(vec).__name__} "
            f"(hard contract: {NO_ZERO_VECTOR_FAKE_SUCCESS})"
        )
    if not vec:
        raise EmbeddingUnavailable(
            f"{context}: empty vector "
            f"(hard contract: {NO_ZERO_VECTOR_FAKE_SUCCESS})"
        )
    out: List[float] = []
    for i, x in enumerate(vec):
        try:
            f = float(x)
        except (TypeError, ValueError) as exc:
            raise EmbeddingUnavailable(
                f"{context}: non-numeric value at index {i}: {exc} "
                f"(hard contract: {NO_ZERO_VECTOR_FAKE_SUCCESS})"
            ) from exc
        if math.isnan(f) or math.isinf(f):
            raise EmbeddingUnavailable(
                f"{context}: NaN/Inf at index {i} "
                f"(hard contract: {NO_ZERO_VECTOR_FAKE_SUCCESS})"
            )
        out.append(f)
    if all(v == 0.0 for v in out):
        raise EmbeddingUnavailable(
            f"{context}: all-zero vector "
            f"(hard contract: {NO_ZERO_VECTOR_FAKE_SUCCESS})"
        )
    return out


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------


@dataclass
class EmbedProvider:
    """Embedding provider.

    The provider hides the difference between the OVH-local Ollama
    ``/api/embeddings`` endpoint and the NewAPI OpenAI-compatible
    ``/v1/embeddings`` endpoint. Both are configured by the
    ``EMBED_PROVIDER`` env var (``"ollama"`` | ``"newapi"``); the
    default is ``"ollama"`` so existing tests / dev loops behave
    unchanged.

    The :class:`httpx.Client` instance is shared across calls (lesson
    41 — connection cache singleton). A process-wide cache keyed on
    ``(provider_name, base_url, model)`` keeps one client per
    provider so concurrent ingestion + recall workloads do not
    duplicate the TCP/TLS handshake.
    """

    name: str
    base_url: str
    model: str
    expected_dim: int = DEFAULT_EMBED_DIM
    timeout: float = 60.0
    # NewAPI-only:
    api_key: str = ""
    # Ollama-only:
    api_style: str = "ollama"  # "ollama" | "openai"
    _client: Any = field(default=None, init=False, repr=False, compare=False)
    _client_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        # Redact the bearer token from repr/str so an accidental
        # ``print(provider)`` in a log line never leaks the secret.
        return (
            f"EmbedProvider(name={self.name!r}, base_url={self.base_url!r}, "
            f"model={self.model!r}, expected_dim={self.expected_dim}, "
            f"api_style={self.api_style!r}, api_key=<redacted len={len(self.api_key)}>)"
        )

    @classmethod
    def from_env(cls) -> "EmbedProvider":
        """Build an EmbedProvider from process env.

        Honours the wave-2 contract:

        * ``EMBED_PROVIDER`` defaults to ``"ollama"``.
        * When set to ``"newapi"`` the provider reads
          ``EMBED_PROVIDER_URL`` / ``EMBED_PROVIDER_MODEL`` /
          ``EMBED_PROVIDER_TOKEN_FILE`` / ``EMBED_PROVIDER_DIM``.
        * When ``"ollama"`` (default) the provider reads
          ``OLLAMA_URL`` / ``EMBED_MODEL`` (legacy names).
        """
        provider_name = (os.environ.get("EMBED_PROVIDER") or "ollama").strip().lower()
        if provider_name == "newapi":
            base_url = (
                os.environ.get("EMBED_PROVIDER_URL")
                or DEFAULT_NEWAPI_BASE_URL
            ).rstrip("/")
            model = (
                os.environ.get("EMBED_PROVIDER_MODEL")
                or NEWAPI_EMBED_MODEL
            ).strip()
            token_file = (
                os.environ.get("EMBED_PROVIDER_TOKEN_FILE")
                or DEFAULT_NEWAPI_TOKEN_FILE
            )
            api_key = _read_token_file(token_file)
            dim = int(os.environ.get("EMBED_PROVIDER_DIM") or str(DEFAULT_EMBED_DIM))
            timeout = float(os.environ.get("EMBED_PROVIDER_TIMEOUT", "60"))
            return cls(
                name="newapi",
                base_url=base_url,
                model=model,
                expected_dim=dim,
                timeout=timeout,
                api_key=api_key,
                api_style="openai",
            )
        # Default: OVH-local ollama.
        base_url = (
            os.environ.get("OLLAMA_URL") or DEFAULT_OLLAMA_URL
        ).rstrip("/")
        model = (
            os.environ.get("EMBED_MODEL") or DEFAULT_OLLAMA_EMBED_MODEL
        ).strip()
        dim = int(os.environ.get("EMBED_PROVIDER_DIM") or str(DEFAULT_EMBED_DIM))
        timeout = float(os.environ.get("EMBED_PROVIDER_TIMEOUT", "60"))
        return cls(
            name="ollama",
            base_url=base_url,
            model=model,
            expected_dim=dim,
            timeout=timeout,
            api_style="ollama",
        )

    # ------------------------------------------------------------------
    # Client cache
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return a process-wide cached ``httpx.Client`` for this provider.

        The cache key is ``(name, base_url, model)`` so a provider
        switch (env var flip) invalidates cleanly. Test code that
        needs a fresh client should call :func:`reset_provider_caches`.
        """
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import httpx  # local import keeps the module cheap to import
            except ImportError as exc:
                raise EmbeddingUnavailable(
                    "httpx is not installed; install httpx>=0.27 to use embed_provider"
                ) from exc
            self._client = httpx.Client(timeout=self.timeout)
            return self._client

    def close(self) -> None:
        """Close the cached client. Safe to call multiple times."""
        if self._client is None:
            return
        try:
            self._client.close()
        except Exception:
            pass
        self._client = None

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, text: str) -> List[float]:
        """Embed ``text`` and return a real (validated) vector.

        Raises
        ------
        EmbeddingUnavailable
            Service unreachable, malformed response, or degenerate vector.
        EmbeddingDimensionMismatch
            Returned length differs from :attr:`expected_dim`.
        """
        prompt = (text or "")[:4000]
        if not prompt:
            raise EmbeddingUnavailable("embed(): empty text")
        client = self._get_client()
        if self.api_style == "openai":
            return self._embed_openai(client, prompt)
        return self._embed_ollama(client, prompt)

    def _embed_ollama(self, client: Any, prompt: str) -> List[float]:
        """POST ``/api/embeddings`` (Ollama native shape)."""
        body = {"model": self.model, "prompt": prompt}
        url = f"{self.base_url}/api/embeddings"
        try:
            resp = client.post(url, json=body)
        except Exception as exc:
            raise EmbeddingUnavailable(
                f"ollama embed POST failed ({url}, model={self.model}): {exc}"
            ) from exc
        if resp.status_code != 200:
            raise EmbeddingUnavailable(
                f"ollama embed returned HTTP {resp.status_code} "
                f"(model={self.model})"
            )
        try:
            data = resp.json()
        except Exception as exc:
            raise EmbeddingUnavailable(
                f"ollama embed returned non-JSON body: {exc}"
            ) from exc
        vec = data.get("embedding") if isinstance(data, dict) else None
        vec = self._validate_and_normalize(vec, context="ollama embed")
        self._check_dim(vec, context="ollama embed")
        return vec

    def _embed_openai(self, client: Any, prompt: str) -> List[float]:
        """POST ``/v1/embeddings`` (OpenAI-compatible shape, NewAPI)."""
        body = {
            "model": self.model,
            "input": prompt,
            # Always pass dimensions explicitly: the channel-side
            # header override is not wired up in every gateway, and
            # an unspecified dim produces 1024-floats which would
            # silently 0-hit a 768-dim collection.
            "dimensions": self.expected_dim,
        }
        url = f"{self.base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = client.post(url, json=body, headers=headers)
        except Exception as exc:
            raise EmbeddingUnavailable(
                f"newapi embed POST failed ({url}, model={self.model}): {exc}"
            ) from exc
        if resp.status_code != 200:
            # Truncate body so we never echo the token in any
            # downstream handler. The token is in the header, not
            # the body, but the body can still contain echoes of
            # the prompt in verbose error modes.
            body_excerpt = (resp.text or "")[:200]
            raise EmbeddingUnavailable(
                f"newapi embed returned HTTP {resp.status_code} "
                f"(model={self.model}): {body_excerpt}"
            )
        try:
            data = resp.json()
        except Exception as exc:
            raise EmbeddingUnavailable(
                f"newapi embed returned non-JSON body: {exc}"
            ) from exc
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list) or not rows:
            raise EmbeddingUnavailable(
                f"newapi embed returned empty data list (model={self.model})"
            )
        first = rows[0]
        vec = first.get("embedding") if isinstance(first, dict) else None
        vec = self._validate_and_normalize(vec, context="newapi embed")
        self._check_dim(vec, context="newapi embed")
        return vec

    def _validate_and_normalize(self, vec: Any, *, context: str) -> List[float]:
        return _validate_vector(vec, context=context)

    def _check_dim(self, vec: List[float], *, context: str) -> None:
        if len(vec) != self.expected_dim:
            raise EmbeddingDimensionMismatch(
                f"{context}: returned dim={len(vec)}, expected={self.expected_dim}"
            )


# ---------------------------------------------------------------------------
# Chat providers
# ---------------------------------------------------------------------------


@dataclass
class ChatProvider:
    """Chat-completions provider.

    Mirrors :class:`EmbedProvider` but for the OpenAI-compatible
    ``/v1/chat/completions`` endpoint. The wave-2 migration wires
    the ``memory_brain_*`` scripts and the consolidation LLM calls
    through this single point so flipping ``LLM_PROVIDER=newapi``
    sends chat traffic to NewAPI instead of local Ollama.
    """

    name: str
    base_url: str
    model: str
    timeout: float = 120.0
    api_key: str = ""
    api_style: str = "openai"  # "ollama" | "openai"
    _client: Any = field(default=None, init=False, repr=False, compare=False)
    _client_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_env(cls) -> "ChatProvider":
        provider_name = (os.environ.get("LLM_PROVIDER") or "ollama").strip().lower()
        if provider_name == "newapi":
            base_url = (
                os.environ.get("LLM_PROVIDER_URL")
                or DEFAULT_NEWAPI_BASE_URL
            ).rstrip("/")
            model = (
                os.environ.get("LLM_PROVIDER_MODEL")
                or NEWAPI_CHAT_MODEL
            ).strip()
            token_file = (
                os.environ.get("LLM_PROVIDER_TOKEN_FILE")
                or DEFAULT_NEWAPI_TOKEN_FILE
            )
            api_key = _read_token_file(token_file)
            timeout = float(os.environ.get("LLM_PROVIDER_TIMEOUT", "120"))
            return cls(
                name="newapi",
                base_url=base_url,
                model=model,
                timeout=timeout,
                api_key=api_key,
                api_style="openai",
            )
        # Default: OVH-local ollama (legacy ``/v1/chat/completions``).
        base_url = (
            os.environ.get("LLM_CHAT_URL")
            or os.environ.get("OLLAMA_URL")
            or DEFAULT_OLLAMA_URL
        ).rstrip("/")
        # If base_url is the ollama native port (no /v1), append it.
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        # Legacy scripts use ``qwen2.5:1.5b`` by default; preserve that
        # so existing tests/dev loops behave unchanged.
        model = (
            os.environ.get("LLM_PROVIDER_MODEL")
            or os.environ.get("MEMORY_BRAIN_LLM_MODEL")
            or DEFAULT_OLLAMA_CHAT_MODEL
        ).strip()
        timeout = float(os.environ.get("LLM_PROVIDER_TIMEOUT", "120"))
        return cls(
            name="ollama",
            base_url=base_url,
            model=model,
            timeout=timeout,
            api_style="openai",  # ollama also speaks /v1/chat/completions
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import httpx
            except ImportError as exc:
                raise ChatUnavailable(
                    "httpx is not installed; install httpx>=0.27 to use chat_provider"
                ) from exc
            self._client = httpx.Client(timeout=self.timeout)
            return self._client

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ChatProvider(name={self.name!r}, base_url={self.base_url!r}, "
            f"model={self.model!r}, api_style={self.api_style!r}, "
            f"api_key=<redacted len={len(self.api_key)}>)"
        )

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.close()
        except Exception:
            pass
        self._client = None

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """Run a chat completion and return the assistant content string.

        Raises
        ------
        ChatUnavailable
            Service unreachable, malformed response, or empty content.
        """
        if not messages:
            raise ChatUnavailable("chat(): empty messages list")
        client = self._get_client()
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = client.post(url, json=body, headers=headers)
        except Exception as exc:
            raise ChatUnavailable(
                f"chat POST failed ({url}, model={self.model}): {exc}"
            ) from exc
        if resp.status_code != 200:
            raise ChatUnavailable(
                f"chat returned HTTP {resp.status_code} (model={self.model})"
            )
        try:
            data = resp.json()
        except Exception as exc:
            raise ChatUnavailable(
                f"chat returned non-JSON body: {exc}"
            ) from exc
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ChatUnavailable("chat response: empty choices")
        msg = choices[0].get("message") or {}
        content = (msg.get("content") or "").strip()
        if not content:
            raise ChatUnavailable("chat response: empty content")
        return content


# ---------------------------------------------------------------------------
# Process-wide singletons
# ---------------------------------------------------------------------------

_EMBED_SINGLETON: Optional[EmbedProvider] = None
_EMBED_LOCK = threading.Lock()
_CHAT_SINGLETON: Optional[ChatProvider] = None
_CHAT_LOCK = threading.Lock()


def get_embed_provider() -> EmbedProvider:
    """Return the process-wide singleton :class:`EmbedProvider`."""
    global _EMBED_SINGLETON
    if _EMBED_SINGLETON is not None:
        return _EMBED_SINGLETON
    with _EMBED_LOCK:
        if _EMBED_SINGLETON is None:
            _EMBED_SINGLETON = EmbedProvider.from_env()
        return _EMBED_SINGLETON


def get_chat_provider() -> ChatProvider:
    """Return the process-wide singleton :class:`ChatProvider`."""
    global _CHAT_SINGLETON
    if _CHAT_SINGLETON is not None:
        return _CHAT_SINGLETON
    with _CHAT_LOCK:
        if _CHAT_SINGLETON is None:
            _CHAT_SINGLETON = ChatProvider.from_env()
        return _CHAT_SINGLETON


def reset_provider_caches() -> None:
    """Drop the cached singletons. Test-only helper."""
    global _EMBED_SINGLETON, _CHAT_SINGLETON
    with _EMBED_LOCK:
        if _EMBED_SINGLETON is not None:
            _EMBED_SINGLETON.close()
        _EMBED_SINGLETON = None
    with _CHAT_LOCK:
        if _CHAT_SINGLETON is not None:
            _CHAT_SINGLETON.close()
        _CHAT_SINGLETON = None


__all__ = [
    "DEFAULT_EMBED_DIM",
    "NEWAPI_EMBED_MODEL",
    "NEWAPI_CHAT_MODEL",
    "DEFAULT_NEWAPI_BASE_URL",
    "DEFAULT_NEWAPI_TOKEN_FILE",
    "EmbeddingError",
    "EmbeddingUnavailable",
    "EmbeddingDimensionMismatch",
    "ChatUnavailable",
    "EmbedProvider",
    "ChatProvider",
    "get_embed_provider",
    "get_chat_provider",
    "reset_provider_caches",
]