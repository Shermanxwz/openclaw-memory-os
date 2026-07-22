"""v0.3.0 unified retrieval engine.

This module is the single entry point for the recall pipeline.
Both the API and CLI must call :meth:`RetrievalEngine.retrieve`
rather than reaching into the backend or the legacy
:mod:`openclaw_memory_os.ranking` module directly. Centralising
the path here means that dense / lexical / hybrid modes, RRF
fusion, and the Active-first / Superseded-fallback contract are
all enforced in one place.

The engine is intentionally stateless across requests; the
BM25 index and Qdrant client live on the backend object. The
default weights come from the active
:class:`openclaw_memory_os.policy_store.Policy` and are
re-resolved on every request so the engine picks up hot-reload
of the policy file without needing a process restart.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .backends import (
    EmbeddingUnavailable,
    MemoryBackend,
)
from .contracts import (
    CandidateStatus,
    MemoryRecord,
    RetrievalDiagnostics,
    ScoredMemoryCandidate,
)
from .candidate_pool import QueryCandidatePool
from .lexical import BM25Index
from .models import Memory, MemoryStatus, RecallHit, RecallRequest, RecallResponse
from .policy_store import Policy, PolicyStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _memory_to_record(memory: Memory, collection: str) -> MemoryRecord:
    """Map a backend :class:`Memory` into the v0.3.0 internal shape."""
    return MemoryRecord(
        collection=collection,
        memory_id=str(memory.id),
        candidate_key=f"{collection}:{memory.id}",
        text=memory.text or "",
        summary=memory.summary,
        source=memory.source,
        tags=list(memory.tags or []),
        status=CandidateStatus(memory.status.value),
        tier=memory.tier,  # type: ignore[arg-type]
        importance=float(memory.importance or 0.0),
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        supersedes=memory.supersedes,
        superseded_by=memory.superseded_by,
        review_reason=memory.review_reason,
        type=None,
        topic=None,
        category=None,
        keywords=[],
        recall_triggers=[],
        entities=[],
    )


def _candidate_to_record(c: ScoredMemoryCandidate) -> MemoryRecord:
    return MemoryRecord(
        collection=c.collection,
        memory_id=c.memory_id,
        candidate_key=c.candidate_key,
        text=c.text,
        summary=c.summary,
        source=c.source,
        tags=list(c.tags or []),
        status=c.status,
        tier=c.tier,
        importance=float(c.importance or 0.0),
        created_at=c.created_at,
        updated_at=c.updated_at,
        supersedes=c.supersedes,
        superseded_by=c.superseded_by,
        review_reason=c.review_reason,
        type=c.type,
        topic=c.topic,
        category=c.category,
        keywords=list(c.keywords or []),
        recall_triggers=list(c.recall_triggers or []),
        entities=list(c.entities or []),
    )


def _records_from_backend(
    backend: MemoryBackend,
    memories: Sequence[Memory],
) -> List[MemoryRecord]:
    """Map a backend's :class:`Memory` list into v0.3.0 records.

    Memories are stamped with their originating collection by
    reading :meth:`MemoryBackend.list_collections` and matching
    by id. For a single-collection backend this is a no-op.
    """
    collections = backend.list_collections()
    primary = collections[0] if collections else "memory"
    out: List[MemoryRecord] = []
    for m in memories:
        out.append(_memory_to_record(m, primary))
    return out


def _calibrate_dense_scores(
    candidates: List[ScoredMemoryCandidate],
) -> Dict[str, float]:
    """Min-max calibrate dense_score into ``[0, 1]`` per-query.

    Returns a map ``candidate_key -> calibrated``. Candidates
    with no dense score (e.g. lexical-only hits) are omitted.
    """
    scores = [c.dense_score for c in candidates if c.dense_score is not None]
    if not scores:
        return {}
    lo = min(scores)
    hi = max(scores)
    span = hi - lo
    if span <= 1e-9:
        return {c.candidate_key: 1.0 for c in candidates if c.dense_score is not None}
    out: Dict[str, float] = {}
    for c in candidates:
        if c.dense_score is None:
            continue
        out[c.candidate_key] = max(0.0, min(1.0, (c.dense_score - lo) / span))
    return out


def _calibrate_lexical_scores(
    candidates: List[ScoredMemoryCandidate],
) -> Dict[str, float]:
    """Min-max calibrate lexical_score into ``[0, 1]`` per-query."""
    scores = [c.lexical_score for c in candidates if c.lexical_score is not None]
    if not scores:
        return {}
    lo = min(scores)
    hi = max(scores)
    span = hi - lo
    if span <= 1e-9:
        return {c.candidate_key: 1.0 for c in candidates if c.lexical_score is not None}
    out: Dict[str, float] = {}
    for c in candidates:
        if c.lexical_score is None:
            continue
        out[c.candidate_key] = max(0.0, min(1.0, (c.lexical_score - lo) / span))
    return out


def _compute_recency(
    cand: "ScoredMemoryCandidate",
    policy: Policy,
) -> float:
    """G3.5: exponential-decay recency score.

    ``recency = exp(-elapsed_hours / half_life_hours)``

    Returns ``1.0`` when the candidate has no timestamp so a
    record without a date is treated as fully-recent (never
    penalised by recency scoring).

    The half-life is read from ``policy.recency_half_life_hours``;
    if the attribute is missing (older policies serialised before
    the field was added) it falls back to 336 hours = 14 days.

    Negative elapsed time (clock skew, future-dated records) is
    clamped to zero so a future timestamp does not produce a
    score above 1.0.
    """
    ts = getattr(cand, "updated_at", None) or getattr(cand, "created_at", None)
    half_life_hours = float(getattr(policy, "recency_half_life_hours", 336.0) or 336.0)
    if half_life_hours <= 0:
        # Defensive: a misconfigured half-life must not break
        # ranking. 1.0 is the safest non-informative value.
        return 1.0
    if ts is None:
        return 0.5
    try:
        if isinstance(ts, str):
            ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            ts_dt = ts
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        elapsed_h = (datetime.now(timezone.utc) - ts_dt).total_seconds() / 3600.0
        if elapsed_h < 0:
            elapsed_h = 0.0
        return math.exp(-elapsed_h / half_life_hours)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """Internal engine result, used by the API and CLI for response shaping."""

    hits: List[ScoredMemoryCandidate]
    diagnostics: RetrievalDiagnostics
    active_count: int
    fallback_used: bool
    fallback_added: int


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RetrievalEngine:
    """Single entry point for dense / lexical / hybrid recall.

    The engine is constructed with a backend, a policy store
    (for weights and thresholds), and an optional BM25 index
    (built from the same corpus). For most use-cases the
    engine is constructed once per process and reused across
    requests.

    Example
    -------

    >>> engine = RetrievalEngine(backend, policy_store)
    >>> result = engine.retrieve("how do I set MEMORY_OS_TOKEN?")
    >>> for hit in result.hits:
    ...     print(hit.candidate_key, hit.score, hit.explanation)
    """

    def __init__(
        self,
        backend: MemoryBackend,
        policy_store: PolicyStore,
        *,
        lexical_index: Optional[BM25Index] = None,
    ) -> None:
        self.backend = backend
        self.policy_store = policy_store
        self._lexical_index = lexical_index
        # Lazy: if no explicit index was passed, attempt to use the
        # backend's own index if it provides one. Otherwise we
        # build a one-shot in-memory index for the duration of the
        # request from the backend's payload cache.
        self._owns_lexical_index = lexical_index is not None
        # Offline evaluation asks many policies to rank the same query.
        # Cache raw channel pools by (query, channel_limit) so policy count
        # never multiplies Qdrant, BM25 or embedding work.
        self._candidate_pool_cache: Dict[Tuple[str, int], QueryCandidatePool] = {}

    # -- corpus snapshot (G5.5) ----------------------------------------------

    def compute_snapshot_id(self) -> Optional[str]:
        """Fingerprint the current corpus as seen by this engine.

        Thin wrapper around :func:`openclaw_memory_os.recall_feedback.
        compute_corpus_snapshot_id` that always uses the engine's own
        backend. Callers that want to attach the snapshot id to the
        :class:`RetrievalDiagnostics` envelope should call this once
        per request; the resulting string is stable across requests
        when the corpus is unchanged.

        Returns ``None`` when the corpus cannot be fingerprinted at
        all (defensive — callers should treat ``None`` as "snapshot
        unavailable, comparison invalid" rather than as a sentinel
        "empty corpus").
        """
        from .recall_feedback import compute_corpus_snapshot_id
        try:
            return compute_corpus_snapshot_id(self.backend)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "RetrievalEngine.compute_snapshot_id: failed (%s); "
                "returning None",
                exc,
            )
            return None

    # -- reusable candidate pools -------------------------------------------

    def build_candidate_pool(
        self, query: str, *, channel_limit: int = 200
    ) -> QueryCandidatePool:
        """Fetch raw Dense/BM25 Active/Superseded channels once."""
        if not query or not query.strip():
            raise ValueError("build_candidate_pool requires a non-empty query")
        budget = max(1, int(channel_limit))
        dense_active: List[ScoredMemoryCandidate] = []
        dense_superseded: List[ScoredMemoryCandidate] = []
        lexical_active: List[ScoredMemoryCandidate] = []
        lexical_superseded: List[ScoredMemoryCandidate] = []
        dense_available = True
        lexical_available = True
        degraded_reason: Optional[str] = None

        try:
            dense_active = self.backend.dense_search(
                query, limit=budget, status_filter=["active"]
            )
            dense_superseded = self.backend.dense_search(
                query, limit=budget, status_filter=["superseded"]
            )
        except EmbeddingUnavailable:
            dense_available = False
            degraded_reason = "embedding_unavailable"
        except AttributeError:
            # Legacy/sample backends have no strict dense channel.  Keep the
            # pool lexical-only rather than manufacturing rank scores.
            dense_available = False
            degraded_reason = "dense_unavailable"
        except Exception as exc:
            dense_available = False
            degraded_reason = "dense_unavailable"
            logger.warning("candidate pool dense build failed: %s", exc)

        try:
            lexical_active = self._lexical_search(query, budget, ["active"])
            lexical_superseded = self._lexical_search(
                query, budget, ["superseded"]
            )
        except Exception as exc:
            lexical_available = False
            if degraded_reason is None:
                degraded_reason = "lexical_unavailable"
            logger.warning("candidate pool lexical build failed: %s", exc)

        return QueryCandidatePool(
            query=query,
            dense_active=[c.model_copy(deep=True) for c in dense_active],
            lexical_active=[c.model_copy(deep=True) for c in lexical_active],
            dense_superseded=[c.model_copy(deep=True) for c in dense_superseded],
            lexical_superseded=[c.model_copy(deep=True) for c in lexical_superseded],
            corpus_snapshot_id=self.compute_snapshot_id(),
            dense_available=dense_available,
            lexical_available=lexical_available,
            degraded_reason=degraded_reason,
        )

    def get_candidate_pool(
        self, query: str, *, channel_limit: int = 200
    ) -> QueryCandidatePool:
        key = (str(query), max(1, int(channel_limit)))
        cached = self._candidate_pool_cache.get(key)
        if cached is None:
            cached = self.build_candidate_pool(key[0], channel_limit=key[1])
            self._candidate_pool_cache[key] = cached
        return cached

    def clear_candidate_pool_cache(self) -> None:
        self._candidate_pool_cache.clear()

    @staticmethod
    def _exact_match_feature(query: str, candidate: ScoredMemoryCandidate) -> float:
        q = (query or "").strip().casefold()
        if not q:
            return 0.0
        fields = [candidate.text, candidate.summary or "", candidate.source or ""]
        fields.extend(candidate.tags or [])
        return 1.0 if any(q in str(value).casefold() for value in fields) else 0.0

    def _rank_pool_band(
        self,
        dense: List[ScoredMemoryCandidate],
        lexical: List[ScoredMemoryCandidate],
        query: str,
        policy: Policy,
    ) -> List[ScoredMemoryCandidate]:
        dense = [c.model_copy(deep=True) for c in dense[: max(1, policy.dense_k)]]
        lexical = [c.model_copy(deep=True) for c in lexical[: max(1, policy.lexical_k)]]
        dense_rank = {c.candidate_key: i for i, c in enumerate(dense, start=1)}
        lexical_rank = {c.candidate_key: i for i, c in enumerate(lexical, start=1)}
        keys = set(dense_rank) | set(lexical_rank)
        if not keys:
            return []
        dense_cal = _calibrate_dense_scores(dense)
        lexical_cal = _calibrate_lexical_scores(lexical)
        base: Dict[str, ScoredMemoryCandidate] = {}
        for candidate in dense:
            base[candidate.candidate_key] = candidate
        for candidate in lexical:
            base.setdefault(candidate.candidate_key, candidate)
        max_rrf = (
            float(policy.rrf_dense_weight) + float(policy.rrf_lexical_weight)
        ) / float(max(1, policy.rrf_k + 1))
        out: List[ScoredMemoryCandidate] = []
        for key in keys:
            candidate = base[key].model_copy(deep=True)
            raw_rrf = 0.0
            if key in dense_rank:
                raw_rrf += policy.rrf_dense_weight / (policy.rrf_k + dense_rank[key])
            if key in lexical_rank:
                raw_rrf += policy.rrf_lexical_weight / (policy.rrf_k + lexical_rank[key])
            rrf = raw_rrf / max_rrf if max_rrf > 0 else 0.0
            vector = dense_cal.get(key, 0.0)
            lexical_score = lexical_cal.get(key, 0.0)
            exact = self._exact_match_feature(query, candidate)
            if exact:
                lexical_score = min(1.0, lexical_score + 0.02 * float(policy.exact_match_boost))
            recency = _compute_recency(candidate, policy)
            feedback = 0.0
            candidate.dense_score = vector
            candidate.lexical_score = lexical_score
            candidate.rrf_score = rrf
            candidate.score = (
                policy.final_rrf_weight * rrf
                + policy.final_vector_weight * vector
                + policy.final_lexical_weight * lexical_score
                + policy.importance_weight * float(candidate.importance or 0.0)
                + policy.recency_weight * recency
                + policy.feedback_weight * feedback
            )
            try:
                object.__setattr__(
                    candidate,
                    "_v030_explanation",
                    f"pool_rrf_raw={raw_rrf:.6f}; pool_rrf={rrf:.4f}; "
                    f"vector={vector:.4f}; lexical={lexical_score:.4f}; "
                    f"recency={recency:.4f}; exact={int(exact)}",
                )
            except Exception:
                pass
            out.append(candidate)
        out.sort(key=lambda c: (-c.score, c.candidate_key))
        return out

    def rank_candidate_pool(
        self, pool: QueryCandidatePool, policy: Policy, *, limit: int = 50
    ) -> RetrievalResult:
        """Pure policy reranking; performs no backend or BM25 calls."""
        started = time.perf_counter()
        active = self._rank_pool_band(
            pool.dense_active, pool.lexical_active, pool.query, policy
        )
        fallback_used = len(active) < int(policy.fallback_min_results)
        superseded: List[ScoredMemoryCandidate] = []
        if fallback_used:
            superseded = self._rank_pool_band(
                pool.dense_superseded,
                pool.lexical_superseded,
                pool.query,
                policy,
            )
            if active and superseded:
                floor = min(c.score for c in active) - 1e-3
                for candidate in superseded:
                    candidate.score = min(candidate.score, floor)
            superseded.sort(key=lambda c: (-c.score, c.candidate_key))
        # Concatenation, not a global re-sort, enforces Active-before-Superseded.
        ranked = active + superseded
        ranked = ranked[: max(1, int(limit))]
        diagnostics = RetrievalDiagnostics(
            status=(
                "degraded"
                if not pool.dense_available or not pool.lexical_available
                else "ok"
            ),
            degraded_reason=pool.degraded_reason,
            dense_available=pool.dense_available,
            lexical_available=pool.lexical_available,
            collections_searched=sorted({c.collection for c in pool}),
            candidate_count=len(ranked),
            ranking_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
        return RetrievalResult(
            hits=ranked,
            diagnostics=diagnostics,
            active_count=len(active),
            fallback_used=fallback_used,
            fallback_added=len(superseded),
        )

    # -- mode dispatch ------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        limit: int = 10,
        status_filter: Optional[List[str]] = None,
        policy: Optional[Policy] = None,
    ) -> RetrievalResult:
        """Run dense / lexical / hybrid recall and return a result.

        ``mode`` is one of ``"keyword"`` (lexical only), ``"dense"``
        (vector only, with graceful degradation to lexical on
        embedding failure), or ``"hybrid"`` (RRF + feature rerank).
        ``status_filter`` restricts the first pass to a subset of
        memory statuses; the second pass (Superseded fallback) is
        driven by the engine.

        ``policy`` is an optional :class:`Policy` override. When
        provided, the engine uses the policy's ``dense_k``,
        ``lexical_k``, ``rrf_k``, ``rrf_dense_weight``,
        ``rrf_lexical_weight``, ``final_rrf_weight``,
        ``final_vector_weight``, ``final_lexical_weight``,
        ``importance_weight``, ``recency_weight``, and
        ``feedback_weight`` instead of the defaults from the
        policy store. This allows the evolution loop to test
        candidate policies against the engine without mutating
        the store.
        """
        resolved_policy = policy if policy is not None else self.policy_store.get()
        policy = resolved_policy
        mode = (mode or "hybrid").lower()
        started = time.perf_counter()
        diagnostics = RetrievalDiagnostics(
            status="ok",
            degraded_reason=None,
            dense_available=True,
            lexical_available=self._lexical_index is not None,
            collections_searched=[],
            candidate_count=0,
            embedding_ms=0.0,
            lexical_ms=0.0,
            ranking_ms=0.0,
        )
        if not query:
            diagnostics.status = "failed"
            diagnostics.degraded_reason = "empty_query"
            return RetrievalResult([], diagnostics, 0, False, 0)

        # ---- first pass: status filter (default Active-only) ----
        first_status = status_filter or ["active"]
        candidates, diagnostics = self._first_pass(
            query, mode, policy, first_status, diagnostics
        )

        # ---- Active-first / Superseded fallback contract ----
        active_count = sum(1 for c in candidates if c.status == CandidateStatus.ACTIVE)
        fallback_used = False
        fallback_added = 0
        if (
            active_count < policy.fallback_min_results
            and "superseded" not in [s.lower() for s in first_status]
            and self._has_superseded(backend=self.backend)
        ):
            extra, _ = self._first_pass(
                query, mode, policy, ["superseded"], diagnostics
            )
            # The Superseded pass uses the same engine; we cap the
            # inner display_score at one-epsilon below the lowest
            # active display_score so Superseded can never outrank
            # an active hit. (Apparent display_score = composite
            # minus 0.0, but the public display_score is set later
            # by the response layer based on this floor.)
            if candidates and extra:
                min_active = min(c.score for c in candidates)
                floor = min_active - 1e-3
                for c in extra:
                    c.score = min(c.score, floor)
                    prev = getattr(c, "_v030_explanation", "") or ""
                    new = (prev + " | fallback=superseded").strip(" |")
                    try:
                        object.__setattr__(c, "_v030_explanation", new)
                    except Exception:
                        pass
            elif extra:
                # No active hits at all; the superseded results
                # become the entire response.
                candidates = list(extra)
            else:
                # No active, no superseded; leave candidates as-is
                # (could be the empty set).
                pass
            fallback_used = True
            fallback_added = len(extra)
        elif active_count < policy.fallback_min_results and not self._has_superseded(
            backend=self.backend
        ):
            # No superseded candidates exist; we don't fabricate
            # any. Only set the diagnostic if no prior reason has
            # been recorded (e.g. an embedding failure that
            # degraded dense to lexical would otherwise be
            # masked).
            if diagnostics.degraded_reason is None:
                diagnostics.degraded_reason = "no_active_or_superseded"

        # Cap to limit
        candidates.sort(key=lambda c: (-c.score, c.candidate_key))
        candidates = candidates[: max(1, limit)]

        # Time the ranking pass
        diagnostics.ranking_ms = round((time.perf_counter() - started) * 1000.0, 3)
        diagnostics.candidate_count = len(candidates)
        return RetrievalResult(candidates, diagnostics, active_count, fallback_used, fallback_added)

    # -- internals ----------------------------------------------------------

    def _has_superseded(self, *, backend: MemoryBackend) -> bool:
        try:
            for m in backend.list_memories():
                if m.status == MemoryStatus.SUPERSEDED:
                    return True
        except Exception:
            return False
        return False

    def _first_pass(
        self,
        query: str,
        mode: str,
        policy: Policy,
        status_filter: List[str],
        diagnostics: RetrievalDiagnostics,
    ) -> Tuple[List[ScoredMemoryCandidate], RetrievalDiagnostics]:
        dense_cands: List[ScoredMemoryCandidate] = []
        lex_cands: List[ScoredMemoryCandidate] = []
        t0 = time.perf_counter()
        if mode in ("dense", "hybrid"):
            try:
                dense_cands = self.backend.dense_search(
                    query,
                    limit=policy.dense_k,
                    status_filter=status_filter,
                )
                diagnostics.dense_available = True
            except EmbeddingUnavailable:
                diagnostics.dense_available = False
                diagnostics.degraded_reason = "embedding_unavailable"
                if mode == "dense":
                    # dense mode with no embedding falls back to
                    # lexical; mark degraded so the dashboard knows.
                    mode = "keyword"
            except AttributeError:
                # Backends that pre-date v0.3.0 may not implement
                # ``dense_search`` but can still expose the legacy
                # ``search(query, limit)`` hook. Use that as a
                # compatibility bridge so API/CLI dense-mode callers
                # still exercise the backend's vector-ish search path
                # instead of silently degrading to a full-corpus
                # lexical scan. The diagnostic makes the degraded
                # contract explicit: this was *not* the strict
                # ScoredMemoryCandidate dense path.
                dense_cands = self._legacy_search_as_dense_candidates(
                    query,
                    limit=max(policy.dense_k, 40),
                    status_filter=status_filter,
                )
                diagnostics.dense_available = bool(dense_cands)
                if diagnostics.degraded_reason is None:
                    diagnostics.degraded_reason = "legacy_search_dense_bridge"
            diagnostics.embedding_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        t1 = time.perf_counter()
        if mode in ("keyword", "lexical", "hybrid"):
            try:
                lex_cands = self._lexical_search(
                    query, policy.lexical_k, status_filter
                )
                diagnostics.lexical_available = True
            except Exception as exc:
                diagnostics.lexical_available = False
                diagnostics.degraded_reason = "lexical_unavailable"
                logger.warning("lexical search failed: %s", exc)
        diagnostics.lexical_ms = round((time.perf_counter() - t1) * 1000.0, 3)

        if mode == "keyword" or mode == "lexical":
            return lex_cands, diagnostics
        if mode == "dense":
            return dense_cands, diagnostics
        # hybrid: RRF + feature rerank
        return self._hybrid_merge(dense_cands, lex_cands, query, policy), diagnostics

    def _legacy_search_as_dense_candidates(
        self,
        query: str,
        *,
        limit: int,
        status_filter: List[str],
    ) -> List[ScoredMemoryCandidate]:
        """Bridge old ``backend.search`` results into v0.3.0 candidates.

        ``MemoryBackend.search`` returns public ``Memory`` objects and has no
        channel score. For compatibility we preserve the returned ordering and
        assign a monotonic rank-derived dense score. Real Qdrant backends use
        ``dense_search`` and never hit this bridge.
        """
        try:
            memories = self.backend.search(query, limit=max(limit, 1))
        except Exception as exc:
            logger.warning("legacy backend.search dense bridge failed: %s", exc)
            return []
        wanted = {s.lower() for s in status_filter or []}
        collection = (self.backend.list_collections() or ["memory"])[0]
        out: List[ScoredMemoryCandidate] = []
        for rank, mem in enumerate(memories, start=1):
            if wanted and mem.status.value.lower() not in wanted:
                continue
            rec = _memory_to_record(mem, collection)
            score = 1.0 / rank
            out.append(
                ScoredMemoryCandidate.from_record(
                    rec,
                    score=score,
                    dense_score=score,
                )
            )
        return out[: max(limit, 1)]

    def _lexical_search(
        self,
        query: str,
        limit: int,
        status_filter: List[str],
    ) -> List[ScoredMemoryCandidate]:
        """Run lexical search, either via the persistent BM25 index
        or by building a one-shot index from the backend's payload
        cache (used by SampleBackend and other read-only backends)."""
        if self._lexical_index is not None and len(self._lexical_index) > 0:
            hits = self._lexical_index.search(
                query, limit=max(limit, 1), exact_match_boost=2.0
            )
            # Look up records for each candidate_key
            out: List[ScoredMemoryCandidate] = []
            for cand_key, score in hits:
                # Prefer the full record cached inside BM25Index.
                # Falling back to backend.get_memory_in_collection()
                # costs a Qdrant lookup per hit and made keyword p95
                # blow past the strict concurrency-5 gate even though
                # BM25 search itself was sub-100ms.
                rec = self._lexical_index.get_record(cand_key)
                if rec is None:
                    rec = self._lookup_record(cand_key)
                if rec is None:
                    continue
                if status_filter and rec.status.value.lower() not in {
                    s.lower() for s in status_filter
                }:
                    continue
                cand = ScoredMemoryCandidate(
                    collection=rec.collection,
                    memory_id=rec.memory_id,
                    candidate_key=rec.candidate_key,
                    text=rec.text,
                    summary=rec.summary,
                    source=rec.source,
                    tags=list(rec.tags or []),
                    status=rec.status,
                    tier=rec.tier,
                    importance=rec.importance,
                    created_at=rec.created_at,
                    updated_at=rec.updated_at,
                    supersedes=rec.supersedes,
                    superseded_by=rec.superseded_by,
                    review_reason=rec.review_reason,
                    expires_at=rec.expires_at,
                    owner_confirmed=rec.owner_confirmed,
                    type=rec.type,
                    topic=rec.topic,
                    category=rec.category,
                    keywords=list(rec.keywords or []),
                    recall_triggers=list(rec.recall_triggers or []),
                    entities=list(rec.entities or []),
                    lexical_score=float(score),
                    score=float(score),
                )
                out.append(cand)
            return out
        # Fall back to building a transient index from the backend
        records = _records_from_backend(self.backend, self.backend.list_memories())
        transient = BM25Index()
        for r in records:
            transient.add(r)
        hits = transient.search(query, limit=max(limit, 1), exact_match_boost=2.0)
        out = []
        for cand_key, score in hits:
            rec = next((r for r in records if r.candidate_key == cand_key), None)
            if rec is None:
                continue
            if status_filter and rec.status.value.lower() not in {
                s.lower() for s in status_filter
            }:
                continue
            cand = ScoredMemoryCandidate(
                collection=rec.collection,
                memory_id=rec.memory_id,
                candidate_key=rec.candidate_key,
                text=rec.text,
                summary=rec.summary,
                source=rec.source,
                tags=list(rec.tags or []),
                status=rec.status,
                tier=rec.tier,
                importance=rec.importance,
                created_at=rec.created_at,
                updated_at=rec.updated_at,
                supersedes=rec.supersedes,
                superseded_by=rec.superseded_by,
                review_reason=rec.review_reason,
                expires_at=rec.expires_at,
                owner_confirmed=rec.owner_confirmed,
                type=rec.type,
                topic=rec.topic,
                category=rec.category,
                keywords=list(rec.keywords or []),
                recall_triggers=list(rec.recall_triggers or []),
                entities=list(rec.entities or []),
                lexical_score=float(score),
                score=float(score),
            )
            out.append(cand)
        return out

    def _lookup_record(self, candidate_key: str) -> Optional[MemoryRecord]:
        """Look up a record by ``collection:memory_id`` in the backend.

        Uses :py:meth:`MemoryBackend.get_memory_in_collection` for a
        targeted, unambiguous lookup so this is safe even when the
        same ``memory_id`` exists in multiple configured collections
        (the v0.3.0 cross-collection dedup contract). Bare
        ``memory_id`` keys (no ``:``) are still accepted and treated
        as a fallback single-collection lookup — the caller is
        expected to know what it's doing.
        """
        if ":" not in candidate_key:
            # No collection qualifier — fall back to a bare-id
            # lookup. The backend may raise AmbiguousMemoryId if
            # the id appears in multiple collections, which is the
            # right behaviour: callers that want to be safe MUST
            # use ``collection:memory_id`` form.
            mem = self.backend.get_memory(candidate_key)
            if mem is None:
                return None
            return _memory_to_record(mem, "")
        collection, memory_id = candidate_key.split(":", 1)
        # Targeted lookup: never ambiguous regardless of how
        # many collections share this ``memory_id``.
        mem = self.backend.get_memory_in_collection(collection, memory_id)
        if mem is None:
            return None
        rec = _memory_to_record(mem, collection)
        return rec

    def _hybrid_merge(
        self,
        dense_cands: List[ScoredMemoryCandidate],
        lex_cands: List[ScoredMemoryCandidate],
        query: str,
        policy: Policy,
    ) -> List[ScoredMemoryCandidate]:
        """Merge dense + lexical with Weighted RRF and per-feature rerank.

        The merge produces one :class:`ScoredMemoryCandidate` per
        unique ``candidate_key``, with the calibrated per-signal
        scores attached. The final ``score`` is the per-policy
        weighted sum of (rrf, vector, lexical, importance, recency,
        feedback); the policy is responsible for the exact
        weights.
        """
        # Build per-key rank maps for RRF
        dense_rank: Dict[str, int] = {
            c.candidate_key: i for i, c in enumerate(dense_cands, start=1)
        }
        lex_rank: Dict[str, int] = {
            c.candidate_key: i for i, c in enumerate(lex_cands, start=1)
        }
        all_keys: Set[str] = set(dense_rank) | set(lex_rank)
        if not all_keys:
            return []

        # Calibrate dense + lexical to [0, 1] for the feature
        # rerank phase.
        dense_calib = _calibrate_dense_scores(dense_cands)
        lex_calib = _calibrate_lexical_scores(lex_cands)

        # Resolve a base record for each candidate_key. We
        # prefer the dense candidate (richer payload) and fall
        # back to the lexical candidate.
        base: Dict[str, ScoredMemoryCandidate] = {}
        for c in dense_cands:
            base[c.candidate_key] = c
        for c in lex_cands:
            base.setdefault(c.candidate_key, c)

        out: List[ScoredMemoryCandidate] = []
        for key in all_keys:
            cand = base[key].model_copy(deep=True)
            d_rank = dense_rank.get(key, None)
            l_rank = lex_rank.get(key, None)
            rrf = 0.0
            if d_rank is not None:
                rrf += policy.rrf_dense_weight / (policy.rrf_k + d_rank)
            if l_rank is not None:
                rrf += policy.rrf_lexical_weight / (policy.rrf_k + l_rank)
            cand.rrf_score = rrf
            # Feature rerank: weighted sum of calibrated scores
            v = dense_calib.get(key, 0.0)
            lex_calib_score = lex_calib.get(key, 0.0)
            importance = float(cand.importance or 0.0)
            # G3.5: recency uses exponential decay.  ``elapsed_hours``
            # is derived from the candidate's ``created_at`` (or
            # ``updated_at``); ``half_life_hours`` is configurable
            # via :attr:`Policy.recency_half_life_hours` (default
            # 336 h = 14 days).  When the timestamp is missing we
            # default to 1.0 so a record without a date is treated
            # as fully-recent (never penalised by recency).
            recency = _compute_recency(cand, policy)
            feedback = 0.0  # feedback weight; injected by caller
            cand.dense_score = cand.dense_score if cand.dense_score is not None else v
            cand.lexical_score = (
                cand.lexical_score if cand.lexical_score is not None else lex_calib_score
            )
            # The feature rerank: weights pinned by the v0.3.0 contract.
            final = (
                policy.final_rrf_weight * rrf
                + policy.final_vector_weight * float(cand.dense_score or 0.0)
                + policy.final_lexical_weight * float(cand.lexical_score or 0.0)
                + policy.importance_weight * importance
                + policy.recency_weight * recency
                + policy.feedback_weight * feedback
            )
            cand.score = final
            # Per-candidate explanation is generated by the
            # response layer from the public component dict. The
            # internal ScoredMemoryCandidate does not have an
            # ``explanation`` field (24-field budget is pinned by
            # the v0.3.0 contract), so we attach the debug bits
            # on the candidate's components dict via a side
            # channel: the response shape is responsible for
            # translating them to a human-readable string.
            extra_bits = []
            if d_rank is not None:
                extra_bits.append(f"d_rank={d_rank}")
            if l_rank is not None:
                extra_bits.append(f"l_rank={l_rank}")
            extra_bits.append(f"rrf={rrf:.3f}")
            extra_bits.append(f"final={final:.3f}")
            # We stash the explanation in a per-attempt tag that
            # the response layer reads; pydantic extra='ignore' will
            # discard it if the model is locked down.
            try:
                object.__setattr__(cand, "_v030_explanation", "; ".join(extra_bits))
            except Exception:
                pass
            out.append(cand)
        return out


def build_recall_response_v030(
    request: RecallRequest,
    result: RetrievalResult,
    *,
    policy: Policy,
    started_ms: float = 0.0,
) -> RecallResponse:
    """Shape an engine :class:`RetrievalResult` into the public
    :class:`RecallResponse` for the API and CLI.

    This is the v0.3.0 equivalent of the legacy
    :func:`openclaw_memory_os.ranking.build_recall_response`. The
    difference is that this function consumes an engine result
    with diagnostics already attached, rather than re-deriving
    the candidate set from ``list_memories``.
    """
    from .models import RecallFallbackInfo  # local import to avoid cycles

    elapsed_ms = (time.perf_counter() - max(started_ms, 1e-9)) * 1000.0 if started_ms else 0.0
    hits: List[RecallHit] = []
    for c in result.hits:
        Memory(
            id=c.memory_id,
            text=c.text,
            summary=c.summary,
            source=c.source,
            tags=list(c.tags or []),
            status=MemoryStatus(c.status.value),
            tier=c.tier,  # type: ignore[arg-type]
            importance=c.importance,
            created_at=c.created_at or __import__("datetime").datetime.utcnow(),
            updated_at=c.updated_at,
            supersedes=c.supersedes,
            superseded_by=c.superseded_by,
            review_reason=c.review_reason,
        )
        v030_explanation = getattr(c, "_v030_explanation", None) or ""
        hits.append(
            RecallHit(
                id=c.memory_id,
                text=c.text,
                summary=c.summary,
                tier=c.tier,  # type: ignore[arg-type]
                status=MemoryStatus(c.status.value),
                importance=c.importance,
                collection=c.collection,
                candidate_key=c.candidate_key,
                score=round(c.score, 4),
                components={
                    "rrf": round(c.rrf_score or 0.0, 4),
                    "vector": round(c.dense_score or 0.0, 4),
                    "lexical": round(c.lexical_score or 0.0, 4),
                    "importance": round(c.importance, 4),
                },
                explanation=v030_explanation,
            )
        )
    return RecallResponse(
        query=request.query,
        mode=request.mode or "hybrid",
        took_ms=round(elapsed_ms, 3) if elapsed_ms else 0.0,
        backend="v030",
        total_considered=len(result.hits),
        hits=hits,
        query_id=str(uuid.uuid4()),
        fallback=RecallFallbackInfo(
            enabled=True,
            min_results=policy.fallback_min_results,
            used=result.fallback_used,
            added=result.fallback_added,
        ),
    )
