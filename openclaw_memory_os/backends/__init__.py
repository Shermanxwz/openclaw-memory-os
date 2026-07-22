"""Memory backend abstractions and implementations.

Three layers:

* :class:`SampleBackend` -- reads memories from a JSON file. Used by
  default for offline demos / CI.
* :class:`QdrantBackend` -- a thin adapter around a Qdrant collection.
  Uses cursor-based pagination and a payload-only cache so it scales
  to tens of thousands of points without timing out the dashboard.
* :func:`get_backend` -- factory that picks one at startup based on
  configuration.

Schema adaptation (v0.2.0+):

    OpenClaw memory entries in Qdrant use a minimal payload shape:
    ``{source: str, content: str}``. The OS expects a richer schema
    (``text``, ``tier``, ``status``, ``created_at``, ...). The adapter
    translates the small shape into the larger one with sensible
    defaults so the dashboard can read real OpenClaw data without
    touching the original collection.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from openclaw_memory_os.config import Settings, get_settings
from openclaw_memory_os.contracts import (
    AmbiguousMemoryId,
    MemoryPayload,
    ScoredMemoryCandidate,
    NO_ZERO_VECTOR_FAKE_SUCCESS,
)
from openclaw_memory_os.models import Memory, MemoryStatus, MemoryTier, utcnow

try:
    from qdrant_client import models as qmodels  # for payload schema types
except ImportError:  # pragma: no cover - dev/test without qdrant-client
    qmodels = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# v0.3.0 dense-search exception types
# ---------------------------------------------------------------------------


class EmbeddingUnavailable(RuntimeError):
    """Raised when the embedding service cannot produce a real vector.

    The recall pipeline surfaces this as a degraded response (see
    :class:`openclaw_memory_os.contracts.RetrievalDiagnostics`) and
    falls back to lexical search. The hard contract
    :data:`openclaw_memory_os.contracts.NO_ZERO_VECTOR_FAKE_SUCCESS`
    is what makes this an *error* and not a silent zero-pad.
    """


class EmbeddingDimensionMismatch(RuntimeError):
    """Raised when an embedding's length doesn't match a Qdrant collection.

    This is a configuration / model-drift problem (the embedding
    service is using a different model than the collection was
    provisioned for). Surfacing it loudly is the right behaviour:
    silently zero-hitting a collection is a worse failure mode
    than a loud error.
    """


def _record_from_payload(
    collection: str, memory_id: str, payload: Dict[str, Any]
) -> Any:
    """Build a :class:`MemoryRecord` from a raw Qdrant payload dict.

    Thin wrapper around :meth:`openclaw_memory_os.contracts.MemoryRecord.from_payload`
    that lives in the backends module so the import direction stays
    one-way (``backends`` -> ``contracts``). The wrapper is
    importable in isolation for tests.
    """
    from openclaw_memory_os.contracts import MemoryRecord
    return MemoryRecord.from_payload(collection, memory_id, payload)


def _parse_source_timestamp(source: str):
    """Extract a YYYY-MM-DD date from a source path like ``memory/2026-05-13.md``.

    Returns a timezone-aware datetime at midnight UTC, or ``None`` if no
    date can be parsed.
    """
    if not source:
        return None
    m = re.search(r"(20\d{2})[-_/](\d{2})[-_/](\d{2})", str(source))
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
    except ValueError:
        return None


class MemoryBackend(ABC):
    """Backend interface. Implementations are read-only for the OS view."""

    name: str = "abstract"

    @abstractmethod
    def list_memories(self) -> List[Memory]: ...

    @abstractmethod
    def list_collections(self) -> List[str]: ...

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[Memory]: ...

    def get_memory_in_collection(self, collection: str, memory_id: str) -> Optional[Memory]:
        """Look up a memory by ID within a specific collection.

        Default implementation falls back to :meth:`get_memory`. Backends that
        track collection-qualified identity must override this method.
        """
        return self.get_memory(memory_id)

    def lexical_search(
        self,
        query: str,
        limit: int = 10,
        status_filter: Optional[List[str]] = None,
    ) -> List[Memory]:
        """Default lexical fallback: substring match on the in-memory cache."""
        try:
            corpus = self.list_memories()
        except Exception:
            return []
        if status_filter:
            wanted = {s.lower() for s in status_filter}
            corpus = [m for m in corpus if m.status.value.lower() in wanted]
        q = (query or "").lower().strip()
        if not q:
            return corpus[:limit]
        return [m for m in corpus if q in (m.text or "").lower()][:limit]

    def search(self, query: str, limit: int = 10) -> List[Memory]:
        """Default keyword-substring search against the in-memory cache.

        Backends with real vector search (e.g. :class:`QdrantBackend`)
        override this to embed the query and hit the underlying engine.
        This default keeps :class:`SampleBackend` honest for offline /
        test use: it matches the ``query`` as a case-insensitive
        substring of ``memory.text`` and returns up to ``limit`` hits
        in cache order.

        Callers that want true dense-vector results must use a backend
        whose ``search()`` override actually hits an embedding store.
        """
        try:
            corpus = self.list_memories()
        except Exception:
            return []
        q = (query or "").lower().strip()
        if not q:
            return list(corpus)[:limit]
        return [m for m in corpus if q in (m.text or "").lower()][:limit]


class SampleBackend(MemoryBackend):
    """JSON-file backend used for demos, tests, and offline mode."""

    name = "sample"

    def __init__(self, path: Path):
        self.path = Path(path)
        self._cache: List[Memory] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.path.exists():
            logger.warning("Sample data file %s does not exist; starting empty.", self.path)
            self._cache = []
            self._loaded = True
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        items = raw.get("memories", raw) if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            raise ValueError(f"Sample data file {self.path} must contain a list of memories.")
        self._cache = [Memory.model_validate(item) for item in items]
        self._loaded = True

    def reload(self) -> None:
        self._loaded = False
        self._cache = []
        self._load()

    def list_memories(self) -> List[Memory]:
        self._load()
        return list(self._cache)

    def list_collections(self) -> List[str]:
        return ["sample"]

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        self._load()
        matches = [m for m in self._cache if m.id == memory_id]
        if not matches:
            return None
        # Single-collection backend: no ambiguity possible.
        return matches[0]

    def get_memory_in_collection(self, collection: str, memory_id: str) -> Optional[Memory]:
        """Targeted lookup in the named collection.

        The SampleBackend only carries a single logical collection
        (``"sample"``). We still validate that the caller asked for
        it so a request for a non-existent collection returns
        ``None`` instead of silently cross-matching.
        """
        self._load()
        if collection != "sample":
            return None
        for m in self._cache:
            if m.id == memory_id:
                return m
        return None


class QdrantBackend(MemoryBackend):
    """Read-only adapter around a Qdrant collection.

    Performance notes (v0.2.0):

    * We scroll the collection with cursor pagination, payload-only, in
      512-point pages, and we keep the resulting list in memory. For
      collections up to ~50k points this is fine and keeps the OS
      stateless across requests.
    * Vector search for dense recall is delegated to Qdrant via
      :meth:`search` (used by the recall-test endpoint when ``mode``
      is ``dense`` or ``hybrid``). The keyword fallback uses the
      in-memory payload cache.
    """

    name = "qdrant"

    def __init__(self, url: str, collection: str, api_key: Optional[str] = None,
                 secondary_collections: Optional[List[str]] = None):
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. Install it or unset QDRANT_URL."
            ) from exc
        self._client = QdrantClient(url=url, api_key=api_key, timeout=120)
        self._collection = collection
        self._secondary_collections: List[str] = list(secondary_collections or [])
        self._cache: List[Memory] = []
        self._loaded = False
        self._cache_time: float = 0.0
        # v0.3.0: cache the Qdrant-side vector dimension for each
        # collection so ``dense_search`` can sanity-check the
        # embedding length before sending it to Qdrant. The cache
        # is intentionally a dict (per-collection) because the
        # primary and secondary collections can be configured with
        # different dimensions and an embedding service that
        # produces ``nomic-embed-text``-sized vectors will
        # silently 0-hit a collection configured for a different
        # model.
        self._dimension_cache: Dict[str, Optional[int]] = {}
        # A pool build performs separate Active and Superseded vector
        # searches. Cache the query embedding so both searches share one
        # Ollama request. Mutation is bounded by a tiny LRU.
        self._embedding_cache: "OrderedDict[tuple[str, str, str], List[float]]" = OrderedDict()
        self._embedding_cache_max = 128
        # Ensure payload indexes exist for fields used by external ingestion
        # scripts. Done after construction (when qdrant_client is importable).
        try:
            self.ensure_payload_indexes()
        except Exception as exc:
            logger.warning("QdrantBackend: payload index ensure skipped: %s", exc)

    def _payload_to_memory(self, point_id, payload: dict) -> Optional[Memory]:
        text = payload.get("content") or payload.get("text")
        # Fallback: legacy session-recovery payloads store no content/text but
        # carry a truncated `user_msg` snippet. Use it as the display text so
        # the dashboard / recall can still surface these points. Without this
        # fallback, ~3k points in user_memory are invisible (2026-07-12).
        if not text:
            user_msg = payload.get("user_msg") or ""
            if user_msg:
                text = f"[{payload.get('source', 'legacy')}] {user_msg}"
        if not text:
            return None
        source = payload.get("source") or "qdrant"
        created_at = _parse_source_timestamp(source) or utcnow()
        try:
            tier = MemoryTier(payload.get("tier") or "medium")
        except ValueError:
            tier = MemoryTier.MEDIUM
        try:
            status = MemoryStatus(payload.get("status") or "active")
        except ValueError:
            status = MemoryStatus.ACTIVE
        try:
            importance = float(payload.get("importance") or 0.5)
        except (TypeError, ValueError):
            importance = 0.5
        importance = max(0.0, min(1.0, importance))
        supersedes = payload.get("supersedes")
        superseded_by = payload.get("superseded_by")
        return Memory(
            id=str(point_id),
            text=str(text),
            summary=payload.get("summary"),
            source=str(source),
            created_at=created_at,
            updated_at=None,
            tier=tier,
            status=status,
            importance=importance,
            tags=list(payload.get("tags") or []),
            supersedes=str(supersedes) if supersedes is not None else None,
            superseded_by=str(superseded_by) if superseded_by is not None else None,
            review_reason=payload.get("review_reason"),
        )

    def _load(self) -> None:
        if self._loaded and (time.time() - self._cache_time) < 30:
            return
        self._loaded = True
        self._load_sync()

    def _load_sync(self) -> None:
        import time as _time
        t0 = _time.time()
        items: List[Memory] = []
        collections_to_load = [self._collection] + self._secondary_collections
        for coll in collections_to_load:
            offset = None
            coll_count = 0
            while True:
                points, offset = self._client.scroll(
                    collection_name=coll,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    limit=512,
                )
                for p in points:
                    payload = dict(p.payload or {})
                    mem = self._payload_to_memory(p.id, payload)
                    if mem is None:
                        continue
                    items.append(mem)
                    coll_count += 1
                if offset is None:
                    break
            logger.info("QdrantBackend: loaded %d memories from %s", coll_count, coll)
        self._cache = items
        self._cache_time = _time.time()
        logger.info("QdrantBackend: total %d memories in %.1fs", len(items), self._cache_time - t0)

    def list_memories(self) -> List[Memory]:
        self._load()
        return list(self._cache)

    def list_collections(self) -> List[str]:
        return [self._collection] + self._secondary_collections

    def iter_memories_by_collection(self) -> Iterable[Tuple[str, Memory]]:
        """Yield ``(collection, memory)`` pairs across all configured collections.

        B2-3 fix: callers that need per-collection identity (e.g.
        the lexical-index refresh script) iterate this instead of
        re-stamping every memory with the primary collection name.
        The cache is populated on first call (same semantics as
        :meth:`list_memories`); the order is primary collection
        first, then configured secondaries.
        """
        self._load()
        # Rebuild the per-collection map lazily from the cache.
        # Each Memory in ``self._cache`` has a string ``source``
        # of the form ``memory/2026-05-13.md``; we can't reliably
        # reverse-map ``source`` -> collection, so we re-scroll
        # only on the first call (cheap because the cache has
        # already been loaded once).
        seen_candidate_keys: set = set()
        for coll in self.list_collections():
            offset = None
            while True:
                points, offset = self._client.scroll(
                    collection_name=coll,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    limit=512,
                )
                for p in points:
                    pid = str(p.id)
                    ckey = f"{coll}:{pid}"
                    if ckey in seen_candidate_keys:
                        continue
                    seen_candidate_keys.add(ckey)
                    mem = self._payload_to_memory(p.id, dict(p.payload or {}))
                    if mem is not None:
                        yield coll, mem
                if offset is None:
                    break

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        self._load()
        matches = [m for m in self._cache if m.id == memory_id]
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        # Same memory_id found in multiple collections — ambiguous.
        # The flat cache doesn't carry collection per memory, so we
        # report all configured collections as potentially containing
        # this ID. The caller should use collection:memory_id form.
        raise AmbiguousMemoryId(memory_id, collections=self.list_collections())

    def get_memory_in_collection(self, collection: str, memory_id: str) -> Optional[Memory]:
        """Return one memory from exactly one Qdrant collection.

        A targeted lookup fails closed. Falling back to a bare-ID cache scan
        could return a same-ID point from another collection.
        """
        if collection not in self.list_collections():
            return None
        try:
            points = self._client.retrieve(
                collection_name=collection,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning(
                "QdrantBackend.get_memory_in_collection failed for %s:%s: %s",
                collection, memory_id, exc,
            )
            return None
        if not points:
            return None
        return self._payload_to_memory(
            points[0].id, dict(points[0].payload or {})
        )

    def search(self, query: str, limit: int = 10) -> List[Memory]:
        """Dense vector search across primary + secondary collections.

        Pipeline:

        1. Encode ``query`` via :meth:`_embed` (Ollama
           ``/api/embeddings`` by default; falls back to a zero vector
           if the embed service is unreachable — callers should treat
           that as a degraded offline result, not a real ranking).
        2. Call ``QdrantClient.search`` against the primary collection
           and every configured secondary collection, taking the top
           ``per_collection`` hits from each.
        3. De-duplicate by point id, preserving Qdrant's natural score
           ordering across the merged list (the first hit we see for
           an id wins, so its position from the collection we hit
           first is kept).
        4. Cap the result at ``limit``.

        If Qdrant itself raises (network down, missing collection,
        embed service missing) we fall back to a substring match on
        the in-memory payload cache. This keeps the recall-test
        endpoint useful when the OS is starting up before Qdrant is
        reachable, but the substring match is **not** what callers
        asking for ``mode=dense`` actually want — see
        ``tests/test_qdrant_backend_search.py`` for the contract that
        pins both paths.
        """
        self._load()
        try:
            vec = self._embed(query)
            out: List[Memory] = []
            seen_candidate_keys: set = set()
            collections_to_search = [self._collection] + self._secondary_collections
            if collections_to_search:
                per_collection = max(limit, (limit * 2) // len(collections_to_search))
            else:
                per_collection = limit
            for coll in collections_to_search:
                hits = self._qdrant_search(
                    collection_name=coll,
                    query_vector=vec,
                    limit=per_collection,
                )
                for h in hits:
                    pid = str(h.id)
                    ckey = f"{coll}:{pid}"
                    if ckey in seen_candidate_keys:
                        continue
                    seen_candidate_keys.add(ckey)
                    mem = self._payload_to_memory(h.id, dict(h.payload or {}))
                    if mem is not None:
                        out.append(mem)
            # Preserve Qdrant's vector-score ordering. The earlier
            # implementation re-sorted by payload.importance here,
            # which silently discarded the actual embedding
            # similarity ranking — a regression for dense recall.
            # G2 fix: Memory objects from search() don't carry Qdrant
            # scores, so we keep insertion order (Qdrant returns
            # pre-sorted per collection) but sort across collections
            # when the backend is a QdrantBackend by relying on the
            # per-hit score from _qdrant_search. For the legacy
            # search() path, collection-then-insertion order is kept.
            return out[:limit]
        except Exception:
            # Fallback: keyword substring match against the cache.
            q = (query or "").lower()
            return [m for m in self._cache if q in (m.text or "").lower()][:limit]

    def _embed(self, text: str) -> List[float]:
        """Embed ``text`` via the configured provider (wave-2).

        Default behaviour unchanged: OVH-local Ollama
        ``/api/embeddings`` with ``nomic-embed-text`` (768-dim).
        When ``EMBED_PROVIDER=newapi``, calls NewAPI's
        ``/v1/embeddings`` with the real model name
        ``qwen3-embedding:0.6b`` and ``dimensions=768``.

        Raises
        ------
        EmbeddingUnavailable
            If the embedder is unreachable, returns a malformed /
            empty payload, or returns a degenerate (zero / all-NaN)
            vector. The v0.3.0 contract
            :data:`openclaw_memory_os.contracts.NO_ZERO_VECTOR_FAKE_SUCCESS`
            forbids silent zero-padding: the recall pipeline must
            surface a degraded response and let the engine degrade
            to lexical search rather than pretend a zero vector is a
            real ranking.
        EmbeddingDimensionMismatch
            Returned length differs from the configured
            ``EMBED_PROVIDER_DIM`` (default 768). This is a
            configuration / model-drift problem; surfacing it
            loudly is the right behaviour because the alternative
            (silently 0-hitting a mismatched collection) is a worse
            failure mode than a loud error.
        """
        from openclaw_memory_os.embed_provider import (
            EmbeddingDimensionMismatch as _EDM,
            EmbeddingUnavailable as _EU,
            get_embed_provider,
        )

        prompt = text[:4000]
        # Per-call cache so dense_search's Active + Superseded pool
        # build only pays for one embed (lesson 41). The provider
        # itself also has a long-lived httpx.Client, so a hot
        # recall path avoids both the embed and the TCP/TLS
        # handshake.
        cache = getattr(self, "_embedding_cache", None)
        if cache is not None:
            provider = None
            try:
                provider = get_embed_provider()
            except Exception:
                provider = None
            if provider is not None:
                cache_key = (provider.name, provider.base_url, provider.model, prompt)
                if cache_key in cache:
                    cache.move_to_end(cache_key)
                    return list(cache[cache_key])
        try:
            provider = get_embed_provider()
            vec = provider.embed(prompt)
        except _EDM as exc:
            # Re-raise as the backends-level alias so existing
            # callers (``dense_search``, integration tests) keep
            # catching the v0.3.0 name.
            raise EmbeddingDimensionMismatch(str(exc)) from exc
        except _EU as exc:
            raise EmbeddingUnavailable(str(exc)) from exc
        cache = getattr(self, "_embedding_cache", None)
        if cache is not None:
            try:
                provider = get_embed_provider()
                cache_key = (provider.name, provider.base_url, provider.model, prompt)
            except Exception:
                cache_key = ("unknown", "", "", prompt)
            cache[cache_key] = list(vec)
            cache.move_to_end(cache_key)
            while len(cache) > int(getattr(self, "_embedding_cache_max", 128)):
                cache.popitem(last=False)
        return vec

    # ------------------------------------------------------------------
    # v0.3.0 dense_search
    # ------------------------------------------------------------------

    def _collection_dimension(self, collection_name: str) -> Optional[int]:
        """Return the vector dimension configured on a Qdrant collection.

        Reads ``collection_info.config.params.vectors.size`` and
        caches the result in :attr:`_dimension_cache` so we don't
        round-trip to Qdrant on every dense request. Returns
        ``None`` if the dimension cannot be read (network down,
        collection missing, etc.) so the caller can decide how to
        degrade — the dense path treats ``None`` as 'unknown
        dimension, skip the call'.
        """
        if collection_name in self._dimension_cache:
            return self._dimension_cache[collection_name]
        try:
            info = self._client.get_collection(collection_name)
        except Exception as exc:
            logger.debug(
                "QdrantBackend: get_collection(%s) failed: %s",
                collection_name,
                exc,
            )
            self._dimension_cache[collection_name] = None
            return None
        try:
            size = info.config.params.vectors.size
        except AttributeError:
            size = None
        self._dimension_cache[collection_name] = size
        return size

    def _qdrant_search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int,
        query_filter: Optional[Any] = None,
    ) -> List[Any]:
        """Version-tolerant wrapper around the Qdrant vector-search call.

        ``qdrant-client`` >= 1.10 renamed :py:meth:`QdrantClient.search`
        to :py:meth:`QdrantClient.query_points` and changed the
        response shape from ``list[ScoredPoint]`` to a
        :class:`QueryResponse` whose ``.points`` attribute carries the
        same hits. Older callers (and our existing fixtures) still
        expect a plain list.

        This helper branches on what the installed client exposes
        and normalises the response so callers can keep doing
        ``for h in hits: h.id, h.payload`` regardless of the
        client version.

        Returns
        -------
        list
            A list of ``ScoredPoint``-like objects with ``.id`` and
            ``.payload`` attributes (and ``.score`` if available).
        """
        client = self._client
        if hasattr(client, "query_points"):
            kwargs = {
                "collection_name": collection_name,
                "query": query_vector,
                "limit": limit,
                "with_payload": True,
            }
            if query_filter is not None:
                kwargs["query_filter"] = query_filter
            response = client.query_points(**kwargs)
        else:
            # Legacy (<= 1.9) path. ``search`` may also have been
            # removed in some intermediate 1.x releases; callers
            # should upgrade, but we keep this branch so unit tests
            # that mock ``search`` directly continue to work.
            kwargs = {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "limit": limit,
                "with_payload": True,
            }
            if query_filter is not None:
                kwargs["query_filter"] = query_filter
            response = client.search(**kwargs)
        # Normalise the response shape: ``query_points`` returns a
        # ``QueryResponse`` with ``.points``; legacy ``search`` returns
        # the list directly.
        if hasattr(response, "points"):
            return list(response.points)
        return list(response)

    def dense_search(
        self,
        query: str,
        limit: int = 10,
        *,
        status_filter: Optional[Sequence[Union[str, MemoryStatus]]] = None,
    ) -> List[ScoredMemoryCandidate]:
        """Dense vector search returning :class:`ScoredMemoryCandidate` hits.

        This is the v0.3.0 strict dense path:

        * The query is encoded via :meth:`_embed`. **If the encoder
          raises or returns an empty vector, the call raises
          :class:`EmbeddingUnavailable` — we never send a
          zero-vector to Qdrant.** The hard contract
          :data:`openclaw_memory_os.contracts.NO_ZERO_VECTOR_FAKE_SUCCESS`
          is enforced here.
        * The collection's vector dimension is read from Qdrant
          once and cached. If the embedding length doesn't match,
          the call raises :class:`EmbeddingDimensionMismatch` —
          the alternative (silently zero-hitting a mismatched
          collection) is a worse failure mode than a loud error.
        * Status filtering is applied via Qdrant's native
          ``Filter`` (a ``MatchAny`` on the ``status`` field), so
          the Active-first contract
          (:data:`openclaw_memory_os.contracts.ACTIVE_FIRST`) is
          enforced at the source rather than after the fact.
        * The per-hit dense score is preserved on the
          :class:`ScoredMemoryCandidate.dense_score` field so
          downstream RRF / rerank stages can see what Qdrant
          actually returned.

        Parameters
        ----------
        query:
            Natural-language query to embed and search.
        limit:
            Maximum number of candidates to return. Defaults to
            ``10``; the caller is expected to ask for the per-
            channel budget (e.g. ``policy.dense_k``).
        status_filter:
            Optional whitelist of statuses to restrict the search
            to. Each element may be either a :class:`MemoryStatus`
            enum member or a plain ``str`` (the v0.3.0 engine
            passes ``List[str]`` because the API/CLI shapes the
            status as a string from user input). Both shapes are
            normalised to the lowercase ``status`` strings used
            in Qdrant payloads. ``None`` (default) means *every*
            status — callers that want the Active-first contract
            should pass ``["active"]`` or ``[MemoryStatus.ACTIVE]``.
        """
        from openclaw_memory_os.contracts import (
            EMBEDDING_FAILURE_DEGRADED,
            NO_ZERO_VECTOR_FAKE_SUCCESS,
        )
        from qdrant_client.http import models as _qmodels  # type: ignore

        if not query or not query.strip():
            raise ValueError("dense_search requires a non-empty query")

        # ---- 1. Encode the query (strict: no zero-vector fallback) ----
        try:
            vec = self._embed(query)
        except Exception as exc:
            # Do NOT swallow this and fall back to a zero vector.
            # The hard contract NO_ZERO_VECTOR_FAKE_SUCCESS requires
            # the call to surface as a degraded / failed response.
            logger.warning(
                "QdrantBackend.dense_search: embedding failed (%s); "
                "refusing to send zero vector to Qdrant.",
                exc,
            )
            raise EmbeddingUnavailable(
                f"embedding service unavailable: {exc}"
            ) from exc
        if not vec:
            raise EmbeddingUnavailable(
                "embedding service returned an empty vector; "
                f"refusing to send zero vector to Qdrant "
                f"(hard contract: {NO_ZERO_VECTOR_FAKE_SUCCESS})."
            )

        # G2.2: reject NaN / Inf / non-numeric vectors. Sending a
        # vector with bad values to Qdrant either triggers a 4xx or
        # silently matches every point in the collection — both are
        # worse than a loud EmbeddingUnavailable that lets the
        # caller fall back to lexical-only retrieval.
        import math
        try:
            _has_bad = any(
                (not isinstance(x, (int, float)))
                or math.isnan(x)
                or math.isinf(x)
                for x in vec
            )
        except TypeError as exc:
            raise EmbeddingUnavailable(
                f"embedding vector contains non-numeric values: {exc}"
            ) from exc
        if _has_bad:
            raise EmbeddingUnavailable(
                "embedding vector contains NaN or Inf values; "
                f"refusing to send it to Qdrant "
                f"(hard contract: {NO_ZERO_VECTOR_FAKE_SUCCESS})."
            )

        # ---- 2. Validate dimension against each collection ----
        collections_to_search = [self._collection] + self._secondary_collections
        eligible: List[str] = []
        for coll in collections_to_search:
            dim = self._collection_dimension(coll)
            if dim is None:
                logger.debug(
                    "QdrantBackend.dense_search: skipping %s "
                    "(dimension unknown).",
                    coll,
                )
                continue
            if len(vec) != dim:
                logger.warning(
                    "QdrantBackend.dense_search: skipping %s "
                    "(embedding dim=%d, collection dim=%d).",
                    coll,
                    len(vec),
                    dim,
                )
                continue
            eligible.append(coll)
        if not eligible:
            raise EmbeddingUnavailable(
                f"no collections have a dimension compatible with the "
                f"embedding vector (len={len(vec)}); degraded per "
                f"{EMBEDDING_FAILURE_DEGRADED}."
            )

        # ---- 3. Build the status filter (Qdrant-native) ----
        # Records with no status field ("missing") are treated as active
        # per the v0.3.0.x contract: an untagged record should never be
        # silently excluded from search results.
        qfilter = None
        if status_filter:
            # Accept either MemoryStatus enum members or plain
            # strings (callers like the v0.3.0 retrieval engine
            # pass ``List[str]`` because the API/CLI shapes the
            # status as a string from user input). MemoryStatus
            # values are themselves the lowercase strings we want
            # on the Qdrant payload, so a single ``str`` /
            # ``.value`` coercion handles both.
            statuses: List[str] = []
            for s in status_filter:
                if isinstance(s, MemoryStatus):
                    statuses.append(s.value)
                else:
                    statuses.append(str(s))
            # When "active" is requested, also include records with
            # no status field ("missing") so they are not silently
            # dropped.
            has_active = "active" in statuses
            should_clauses: list = [
                _qmodels.FieldCondition(
                    key="status",
                    match=_qmodels.MatchAny(any=statuses),
                )
            ]
            if has_active:
                should_clauses.append(
                    _qmodels.IsEmptyCondition(
                        is_empty=_qmodels.PayloadField(key="status")
                    )
                )
            qfilter = _qmodels.Filter(should=should_clauses)
        else:
            # No explicit filter: default to active + missing (i.e.
            # everything except explicitly superseded/expired).
            qfilter = _qmodels.Filter(
                should=[
                    _qmodels.FieldCondition(
                        key="status",
                        match=_qmodels.MatchAny(any=["active"]),
                    ),
                    _qmodels.IsEmptyCondition(
                        is_empty=_qmodels.PayloadField(key="status")
                    ),
                ]
            )

        # ---- 4. Run the search across eligible collections ----
        seen_candidate_keys: set = set()
        out: List[ScoredMemoryCandidate] = []
        per_collection = max(limit, (limit * 2) // len(eligible))
        for coll in eligible:
            try:
                hits = self._qdrant_search(
                    collection_name=coll,
                    query_vector=vec,
                    limit=per_collection,
                    query_filter=qfilter,
                )
            except Exception as exc:
                logger.warning(
                    "QdrantBackend.dense_search: search(%s) failed: %s",
                    coll,
                    exc,
                )
                continue
            for h in hits:
                pid = str(h.id)
                ckey = f"{coll}:{pid}"
                if ckey in seen_candidate_keys:
                    continue
                seen_candidate_keys.add(ckey)
                payload = dict(h.payload or {})
                # Apply keyword/entities/triggers normalisation
                # before the candidate is built so downstream code
                # never has to special-case legacy payload shapes.
                payload.setdefault("keywords", MemoryPayload.keywords(payload))
                payload.setdefault("entities", MemoryPayload.entities(payload))
                payload.setdefault("triggers", MemoryPayload.triggers(payload))
                try:
                    record = _record_from_payload(coll, pid, payload)
                except ValueError as exc:
                    logger.debug(
                        "QdrantBackend.dense_search: skipping %s:%s (%s)",
                        coll,
                        pid,
                        exc,
                    )
                    continue
                candidate = ScoredMemoryCandidate.from_record(
                    record,
                    score=float(getattr(h, "score", 0.0) or 0.0),
                    dense_score=float(getattr(h, "score", 0.0) or 0.0),
                )
                out.append(candidate)
        # G2 fix: sort ALL candidates by dense_score descending
        # BEFORE truncating to limit. Without this, a high-scoring hit
        # from a secondary collection can be dropped if the primary
        # collection fills the limit first (collection-order bias).
        # The seen_candidate_keys dedup is unaffected because it
        # already filtered duplicates before this point.
        out.sort(key=lambda c: (-c.dense_score, c.candidate_key))
        return out[:limit]

    def ensure_payload_indexes(self) -> None:
        """Idempotently create payload indexes used by external ingestion
        scripts (memory-brain-ingest, session-recovery-ingest, qdrant-sync).

        Without these indexes, scripts that filter by ``source`` / ``tier`` /
        ``status`` / ``type`` / ``category`` would full-scan the collection.
        Qdrant 1.10+ supports online payload index creation; calling this on
        every backend startup is cheap and idempotent.
        """
        # Map of field -> Qdrant schema type. KEYWORD for strings, INTEGER for
        # numeric ranges, BOOL for true/false filters.
        field_schemas: Dict[str, Any] = {
            "source":          qmodels.PayloadSchemaType.KEYWORD if qmodels else "keyword",
            "tier":            qmodels.PayloadSchemaType.KEYWORD if qmodels else "keyword",
            "status":          qmodels.PayloadSchemaType.KEYWORD if qmodels else "keyword",
            "type":            qmodels.PayloadSchemaType.KEYWORD if qmodels else "keyword",
            "category":        qmodels.PayloadSchemaType.KEYWORD if qmodels else "keyword",
            "topic":           qmodels.PayloadSchemaType.KEYWORD if qmodels else "keyword",
            "owner_confirmed": qmodels.PayloadSchemaType.BOOL     if qmodels else "bool",
            "line_start":      qmodels.PayloadSchemaType.INTEGER  if qmodels else "integer",
            "line_end":        qmodels.PayloadSchemaType.INTEGER  if qmodels else "integer",
        }
        for coll in [self._collection] + self._secondary_collections:
            try:
                info = self._client.get_collection(coll)
                existing = set((info.payload_schema or {}).keys())
            except Exception:
                existing = set()
            for f, schema in field_schemas.items():
                if f in existing:
                    continue
                try:
                    self._client.create_payload_index(
                        collection_name=coll,
                        field_name=f,
                        field_schema=schema,
                    )
                    logger.info("QdrantBackend: created payload index %s.%s (%s)", coll, f, schema)
                except Exception as exc:  # pragma: no cover - network/perm
                    logger.debug("QdrantBackend: index %s.%s skipped: %s", coll, f, exc)


def get_backend(settings: Optional[Settings] = None) -> MemoryBackend:
    """Build the appropriate backend based on configuration."""
    settings = settings or get_settings()
    if settings.qdrant_url:
        try:
            return QdrantBackend(
                url=settings.qdrant_url,
                collection=settings.qdrant_collection,
                api_key=settings.qdrant_api_key,
                secondary_collections=settings.qdrant_secondary_collections,
            )
        except Exception as exc:
            logger.warning(
                "Falling back to sample backend: Qdrant unavailable (%s).", exc
            )
    return SampleBackend(settings.sample_data_path)
