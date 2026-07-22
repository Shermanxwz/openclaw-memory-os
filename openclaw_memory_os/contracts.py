"""Hard contracts and identity model for the Memory OS.

This module is the single source of truth for:

* The canonical memory identity (``collection:memory_id``).
* The hard contract invariants that the OS **must** never violate —
  these are documented as runtime constants so accidental violations
  are visible in code review and can be unit-tested.
* The shape of cross-module candidates (see :class:`MemoryRef`,
  :class:`MemoryRecord`, :class:`ScoredMemoryCandidate`,
  :class:`RecallHit`).

Why this file exists
====================

Before v0.3.0 the system used a flat ``Memory.id`` string as the unique
key for a memory across Qdrant collections. That worked while there was
one collection, but the evolution / hybrid-search design needs to:

1. Treat memories from different Qdrant collections as first-class
   entities (per-collection dense search, per-collection diagnostic
   stats).
2. Identify a candidate *across* modules (dense search, lexical
   search, reranker, evaluator) without losing the collection
   context — which is exactly what gets lost if you keep using
   ``memory.id`` alone.

So the OS now identifies every memory as ``MemoryRef(collection, memory_id)``
and propagates a ``candidate_key`` of ``"{collection}:{memory_id}"``
throughout the system. Backends, scoring, evaluation, and feedback
all key on this composite identity.

Hard contracts (invariants)
===========================

These are pinned by tests in ``tests/test_contracts.py``. They are
intentionally not configurable — the OS's design philosophy is that
some decisions should be hard to evolve away from, and the cost of
getting them wrong is high (silent data loss, surprising
re-rankings, etc.).

* ``ACTIVE_FIRST`` — Active memories are always searched before
  Superseded memories. There is no admin toggle to flip this.
* ``SUPERSEDED_FALLBACK_ONLY`` — Superseded memories are surfaced
  only when the Active pass produced too few hits (active-first
  fallback). They are not added as "more results" in a flat list.
* ``SUPERSEDED_BELOW_ACTIVE`` — When the fallback adds Superseded
  hits, every such hit is clamped below the lowest Active score so
  that the scoreboard keeps Superseded below Active.
* ``NO_PHYSICAL_DELETION`` — The OS never deletes memories; it only
  supersedes, expires, archives, or marks for review. There is no
  config flag to enable physical deletion. Memory Brain never deletes
  memories; the previous ``MEMORY_BRAIN_ALLOW_DELETE`` opt-in
  (which lived outside the OS package) has been removed, so the
  "never deletes" contract is now enforced end-to-end.
* ``EMBEDDING_FAILURE_DEGRADED`` — If an embedding call fails, the
  OS returns a degraded response (``degraded=true``,
  ``reason="embedding_unavailable"``) and explicitly falls back to
  lexical search. It does **not** send a zero-vector to Qdrant and
  pretend the result is real.
* ``NO_ZERO_VECTOR_FAKE_SUCCESS`` — Never pad a missing embedding
  with zeros. If we don't have a real vector, we don't search.
* ``UNJUDGED_IS_NOT_NEGATIVE`` — An unjudged recall hit is not a
  negative training signal. Negative labels require explicit
  negative feedback (downvote / reject) from the operator.
* ``QWEN_NOT_IN_ONLINE_PATH`` — The ``qwen`` LLM is never used in
  online query serving or automated policy decisions. It is only
  invoked for offline batch jobs (cold-start ingestion validation,
  candidate-pool labelling). Any code path that would route a
  recall request through qwen is a bug.

Identity model
==============

::

    MemoryRef                    -- {collection, memory_id}
        |                          carries a stable ``key`` property
        v
    MemoryRecord                 -- MemoryRef + the full payload
        |                          (text, tier, status, importance, ...)
        v
    ScoredMemoryCandidate        -- MemoryRecord + score + per-signal
        |                          score breakdown (dense, lexical, rrf)
        v
    RecallHit                    -- public DTO returned by the API;
                                   carries the same identity via
                                   ``collection`` + ``memory_id``.

Every public type includes the ``collection`` field so the full
identity is preserved end-to-end. ``candidate_key`` is the canonical
string form (``"{collection}:{memory_id}"``) used for:

* Set membership (e.g. ``active_ids`` exclusion in the fallback).
* Cache keys (lexical index, BM25 corpus).
* Audit log entries.
* Feedback rows (so a feedback entry can refer to a memory in any
  collection without colliding with a same-id memory in another).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Hard-contract constants
# ---------------------------------------------------------------------------

#: Active memories are searched before Superseded memories; there is no
#: admin toggle to flip this. Pinning it as a constant makes the
#: invariant grep-able and unit-testable.
ACTIVE_FIRST: str = "active_first"

#: Superseded memories are surfaced only as a fallback when the Active
#: pass yielded too few hits. Never add them as "more results" in a
#: flat list.
SUPERSEDED_FALLBACK_ONLY: str = "superseded_fallback_only"

#: When the fallback adds Superseded hits, each is clamped below the
#: lowest Active score so that Superseded stays visibly below Active.
SUPERSEDED_BELOW_ACTIVE: str = "superseded_below_active"

#: The OS never physically deletes memories. Review-only flows surface
#: candidates; humans decide what to do with them.
NO_PHYSICAL_DELETION: str = "no_physical_deletion"

#: If embedding fails, surface ``degraded=true`` with an explicit
#: reason and fall back to lexical search. Do NOT send a zero-vector
#: to Qdrant and pretend the result is real.
EMBEDDING_FAILURE_DEGRADED: str = "embedding_failure_degraded"

#: Never pad a missing embedding with zeros. If we don't have a real
#: vector, we don't search.
NO_ZERO_VECTOR_FAKE_SUCCESS: str = "no_zero_vector_fake_success"

#: An unjudged recall hit is **not** a negative label. Negative
#: feedback must come from an explicit operator signal (downvote,
#: reject, supersede).
UNJUDGED_IS_NOT_NEGATIVE: str = "unjudged_is_not_negative"

#: The ``qwen`` LLM is not part of the online query path or the
#: evolution decision loop. It is only allowed for offline batch jobs.
QWEN_NOT_IN_ONLINE_PATH: str = "qwen_not_in_online_path"


HARD_CONTRACTS: Tuple[str, ...] = (
    ACTIVE_FIRST,
    SUPERSEDED_FALLBACK_ONLY,
    SUPERSEDED_BELOW_ACTIVE,
    NO_PHYSICAL_DELETION,
    EMBEDDING_FAILURE_DEGRADED,
    NO_ZERO_VECTOR_FAKE_SUCCESS,
    UNJUDGED_IS_NOT_NEGATIVE,
    QWEN_NOT_IN_ONLINE_PATH,
)


# ---------------------------------------------------------------------------
# Identity model
# ---------------------------------------------------------------------------


class AmbiguousMemoryId(Exception):
    """Raised when a bare ``memory_id`` matches records in multiple collections.

    The v0.3.0 identity model requires ``(collection, memory_id)`` as the
    unique key.  When a caller requests a memory by bare ID alone and the
    same ID exists in more than one collection, the OS cannot silently
    pick one — that would violate the cross-collection identity contract.
    Instead it raises this exception so the caller can disambiguate.
    """

    def __init__(self, memory_id: str, collections: List[str]) -> None:
        self.memory_id = memory_id
        self.collections = collections
        super().__init__(
            f"memory_id {memory_id!r} is ambiguous: found in collections "
            f"{collections!r}. Use collection:memory_id form instead."
        )


class MemoryRef(BaseModel):
    """Canonical identity of a memory: ``(collection, memory_id)``.

    The OS uses this pair as the unique key across all modules. The
    ``memory_id`` alone is **not** sufficient: two collections can
    legitimately hold memories with the same point id (Qdrant point
    ids are per-collection), and the OS now treats each collection
    as a first-class entity.

    Use :attr:`key` (string form ``"{collection}:{memory_id}"``) when
    you need a hashable / serialisable handle, e.g. for set
    membership, log lines, or audit entries.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    collection: str = Field(..., min_length=1, description="Qdrant collection name (or 'sample' for the JSON backend).")
    memory_id: str = Field(..., min_length=1, description="Per-collection point id / record key.")

    @property
    def key(self) -> str:
        """Canonical ``"collection:memory_id"`` string handle."""
        return f"{self.collection}:{self.memory_id}"

    @classmethod
    def from_key(cls, key: str) -> "MemoryRef":
        """Parse a ``"collection:memory_id"`` string back into a :class:`MemoryRef`.

        Splits on the **first** ``:`` only, so memory_ids that
        themselves contain ``:`` survive intact. Raises :class:`ValueError`
        for malformed keys.
        """
        if not isinstance(key, str) or not key:
            raise ValueError("MemoryRef.from_key requires a non-empty string")
        if ":" not in key:
            raise ValueError(
                f"MemoryRef key must be 'collection:memory_id'; got {key!r}"
            )
        collection, _, memory_id = key.partition(":")
        if not collection or not memory_id:
            raise ValueError(
                f"MemoryRef key must have non-empty collection and memory_id; got {key!r}"
            )
        return cls(collection=collection, memory_id=memory_id)


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def normalize_string_list(value, *, field_name: str = "value") -> list:
    """Coerce legacy payload values into a ``List[str]``.

    Qdrant payloads (and historical ingestion scripts) have stored
    ``keywords`` / ``entities`` / ``triggers`` as one of:

    * a ``list[str]`` (canonical),
    * a JSON-encoded string of a list,
    * a comma-separated string,
    * a single bare string,
    * ``None``.

    This helper is permissive: anything truthy becomes a list of
    strings; anything falsy (``None`` / empty list / empty string)
    becomes ``[]``. The function never raises — legacy payloads must
    not crash the recall pipeline.

    Parameters
    ----------
    value:
        The raw payload value. May be any type.
    field_name:
        Used only for the log line on parse failures (so we know
        which field gave up without raising).

    Returns
    -------
    list[str]
        A list of stripped, non-empty strings.
    """
    import json
    import logging

    logger = logging.getLogger(__name__)

    if value is None:
        return []
    if isinstance(value, list):
        out: list = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # Try JSON first (covers `["a", "b"]` and `"a"`).
        try:
            parsed = json.loads(s)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
        if isinstance(parsed, str):
            return [parsed.strip()] if parsed.strip() else []
        # Fallback: comma-separated.
        if "," in s:
            return [part.strip() for part in s.split(",") if part.strip()]
        return [s]
    # Unknown type — coerce to string.
    try:
        return [str(value)] if str(value).strip() else []
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("normalize_string_list: %s coerce failed: %s", field_name, exc)
        return []


def candidate_key(collection: Optional[str], memory_id: Optional[str]) -> str:
    """Build the canonical ``"collection:memory_id"`` handle.

    Mirrors :meth:`MemoryRef.key` but accepts ``None`` for either side
    — callers that only have an id (e.g. legacy audit entries) get a
    best-effort handle with the empty collection replaced by
    ``"unknown"``. Prefer :class:`MemoryRef` when both sides are
    known; this helper exists for log lines and audit messages.
    """
    c = (collection or "").strip() or "unknown"
    m = (memory_id or "").strip()
    if not m:
        raise ValueError("candidate_key requires a non-empty memory_id")
    return f"{c}:{m}"


# ---------------------------------------------------------------------------
# Lifecycle status (mirrors ``models.MemoryStatus`` but lives here so
# contracts have no upward dependency on ``models``). The recall
# pipeline only ever needs the four-way split.
# ---------------------------------------------------------------------------


class CandidateStatus(str, Enum):
    """Lifecycle status carried on a candidate.

    Mirrors :class:`openclaw_memory_os.models.MemoryStatus` but lives
    in the contracts module so the candidate types below do not pull
    in the full ``models`` module (which would create a circular
    import in the v0.3.0 wiring: ``models`` already imports from
    ``contracts`` for the hard-contract constants).
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    NEEDS_REVIEW = "needs_review"


class CandidateTier(str, Enum):
    """Tier classification carried on a candidate.

    Mirrors :class:`openclaw_memory_os.models.MemoryTier` for the
    same reason as :class:`CandidateStatus` — avoid a circular
    import while keeping the contracts module self-contained.
    """

    CORE = "core"
    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"
    WORKING = "working"


# ---------------------------------------------------------------------------
# MemoryRecord + ScoredMemoryCandidate + RetrievalDiagnostics
# ---------------------------------------------------------------------------


class MemoryRecord(BaseModel):
    """A candidate's full payload, with collection-aware identity.

    This is the *internal* shape used by the recall pipeline once a
    candidate has been fetched from a backend. It carries the same
    fields as :class:`openclaw_memory_os.models.Memory` plus an
    explicit ``collection`` (since the same payload might live in
    multiple Qdrant collections) and ``candidate_key`` (the
    canonical string identity).

    Use :class:`ScoredMemoryCandidate` for everything that needs a
    score. Use :class:`MemoryRecord` when the score has not been
    computed yet (e.g. while scanning a payload cache).
    """

    model_config = ConfigDict(extra="ignore")

    # --- identity (3) ------------------------------------------------------
    collection: str = Field(..., min_length=1)
    memory_id: str = Field(..., min_length=1)
    candidate_key: str = Field(..., min_length=1)

    # --- content (4) -------------------------------------------------------
    text: str = Field(...)
    summary: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    # --- classification (2) ------------------------------------------------
    status: CandidateStatus = CandidateStatus.ACTIVE
    tier: CandidateTier = CandidateTier.MEDIUM

    # --- scoring inputs (3) ------------------------------------------------
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # --- provenance / governance (3) --------------------------------------
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    review_reason: Optional[str] = None

    # --- payload extras (3) ------------------------------------------------
    # Legacy ingestion has shipped these as both required and optional
    # across releases. We keep all three optional and ``None``-tolerant
    # so an old payload that omits them still deserialises cleanly.
    expires_at: Optional[datetime] = None
    owner_confirmed: bool = False
    type: Optional[str] = None

    # --- v0.3.0 extensions (2) ---------------------------------------------
    # These are newer payload fields that older records won't have.
    # They default to ``None`` / empty so the type stays backward-
    # compatible with v0.2.x payloads.
    topic: Optional[str] = None
    category: Optional[str] = None

    # --- v0.3.0 lexical search fields (3) ---------------------------------
    # These are populated by ``memory_brain_ingest.py`` after the
    # LLM-driven classification pass. The lexical index relies on
    # them for the field-weighted BM25 document. Optional because
    # legacy records may not have them.
    keywords: List[str] = Field(default_factory=list)
    recall_triggers: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)

    @classmethod
    def from_payload(
        cls,
        collection: str,
        memory_id: str,
        payload: Dict[str, Any],
        *,
        text_fallback_key: str = "content",
    ) -> "MemoryRecord":
        """Build a :class:`MemoryRecord` from a raw Qdrant payload dict.

        The candidate's ``text`` is read from ``text_fallback_key``
        (``"content"`` by default — the canonical ingestion field)
        and falls back to a ``"text"`` key, then a ``"user_msg"``
        snippet (legacy session-recovery payloads). If neither is
        present the constructor raises :class:`ValueError` so a
        garbage payload cannot silently turn into an empty
        candidate.
        """
        text = (
            payload.get(text_fallback_key)
            or payload.get("text")
            or (payload.get("user_msg") or "").strip()
        )
        if not text:
            raise ValueError(
                f"MemoryRecord.from_payload: no text/ content for "
                f"{collection}:{memory_id} (keys={sorted(payload.keys())})"
            )
        text_str = str(text)
        if not (payload.get(text_fallback_key) or payload.get("text")):
            text_str = f"[{payload.get('source', 'legacy')}] {text_str}"

        # Status / tier come back as raw strings; we coerce to enums
        # and fall back to the safe defaults on any parse failure.
        try:
            status = CandidateStatus(payload.get("status") or "active")
        except ValueError:
            status = CandidateStatus.ACTIVE
        try:
            tier = CandidateTier(payload.get("tier") or "medium")
        except ValueError:
            tier = CandidateTier.MEDIUM

        try:
            importance = float(payload.get("importance") or 0.5)
        except (TypeError, ValueError):
            importance = 0.5
        importance = max(0.0, min(1.0, importance))

        supersedes = payload.get("supersedes")
        superseded_by = payload.get("superseded_by")
        return cls(
            collection=str(collection),
            memory_id=str(memory_id),
            candidate_key=f"{collection}:{memory_id}",
            text=text_str,
            summary=payload.get("summary"),
            source=str(payload.get("source") or "qdrant"),
            tags=list(payload.get("tags") or []),
            status=status,
            tier=tier,
            importance=importance,
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            supersedes=str(supersedes) if supersedes is not None else None,
            superseded_by=(
                str(superseded_by) if superseded_by is not None else None
            ),
            review_reason=payload.get("review_reason"),
            expires_at=payload.get("expires_at"),
            owner_confirmed=bool(payload.get("owner_confirmed") or False),
            type=payload.get("type"),
            topic=payload.get("topic"),
            category=payload.get("category"),
        )


class ScoredMemoryCandidate(BaseModel):
    """A :class:`MemoryRecord` with a score and per-signal breakdown.

    This is the canonical *internal* candidate type produced by the
    recall pipeline (dense_search → lexical_search → RRF merge →
    feature rerank) and consumed by the public DTO. The 24 fields
    below are pinned by the v0.3.0 evolution contract — adding a
    field requires a schema bump; removing one is a breaking
    change.

    Field budget
    ------------

    * Identity (3): ``collection``, ``memory_id``, ``candidate_key``
    * Content (4): ``text``, ``summary``, ``source``, ``tags``
    * Classification (2): ``status``, ``tier``
    * Scoring inputs (3): ``importance``, ``created_at``, ``updated_at``
    * Governance (3): ``supersedes``, ``superseded_by``, ``review_reason``
    * Payload extras (3): ``expires_at``, ``owner_confirmed``, ``type``
    * v0.3.0 extensions (2): ``topic``, ``category``
    * Scores (3): ``dense_score``, ``lexical_score``, ``rrf_score``
    * Final (1): ``score``

    Total: 24.
    """

    model_config = ConfigDict(extra="ignore")

    # --- identity (3) ------------------------------------------------------
    collection: str
    memory_id: str
    candidate_key: str

    # --- content (4) -------------------------------------------------------
    text: str
    summary: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    # --- classification (2) ------------------------------------------------
    status: CandidateStatus = CandidateStatus.ACTIVE
    tier: CandidateTier = CandidateTier.MEDIUM

    # --- scoring inputs (3) ------------------------------------------------
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # --- governance (3) ----------------------------------------------------
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    review_reason: Optional[str] = None

    # --- payload extras (3) ------------------------------------------------
    expires_at: Optional[datetime] = None
    owner_confirmed: bool = False
    type: Optional[str] = None

    # --- v0.3.0 extensions (2) ---------------------------------------------
    topic: Optional[str] = None
    category: Optional[str] = None

    # --- scores (4) --------------------------------------------------------
    # The per-channel scores (dense / lexical / rrf) and the final
    # composite. ``None`` for a per-channel score means "this
    # channel did not contribute to this candidate's score" (e.g.
    # a pure lexical hit has no dense_score).
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None
    rrf_score: Optional[float] = None
    score: float = Field(default=0.0)

    @classmethod
    def from_record(
        cls,
        record: MemoryRecord,
        *,
        score: float = 0.0,
        dense_score: Optional[float] = None,
        lexical_score: Optional[float] = None,
        rrf_score: Optional[float] = None,
    ) -> "ScoredMemoryCandidate":
        """Lift a :class:`MemoryRecord` into a :class:`ScoredMemoryCandidate`.

        Per-channel scores default to ``None`` (channel did not
        contribute); the final ``score`` defaults to ``0.0`` and is
        expected to be filled in by the rerank / RRF stages.
        """
        return cls(
            collection=record.collection,
            memory_id=record.memory_id,
            candidate_key=record.candidate_key,
            text=record.text,
            summary=record.summary,
            source=record.source,
            tags=list(record.tags),
            status=record.status,
            tier=record.tier,
            importance=record.importance,
            created_at=record.created_at,
            updated_at=record.updated_at,
            supersedes=record.supersedes,
            superseded_by=record.superseded_by,
            review_reason=record.review_reason,
            expires_at=record.expires_at,
            owner_confirmed=record.owner_confirmed,
            type=record.type,
            topic=record.topic,
            category=record.category,
            score=float(score),
            dense_score=dense_score,
            lexical_score=lexical_score,
            rrf_score=rrf_score,
        )


class RetrievalDiagnostics(BaseModel):
    """Diagnostic envelope attached to every recall response.

    Captures *what actually happened* on the request: which channels
    were available, which collections were searched, how many
    candidates came out of each stage, and the wall-clock cost of
    each stage. The :attr:`status` field is the top-level signal
    the caller should switch on:

    * ``"ok"`` — every channel succeeded; hits were ranked normally.
    * ``"degraded"`` — at least one channel was unavailable but the
      pipeline still produced a result (e.g. embedding failed so we
      fell back to lexical-only).
    * ``"failed"`` — no channel could serve the request; the
      :attr:`hits` list is empty and :attr:`degraded_reason`
      explains why.

    The ``*_ms`` fields are measured *inside* the pipeline. They are
    best-effort (some backends don't expose the breakdown) and
    default to ``0.0`` when the measurement isn't available.
    """

    model_config = ConfigDict(extra="ignore")

    status: str = Field(default="ok", description="ok | degraded | failed")
    degraded_reason: Optional[str] = Field(
        default=None,
        description=(
            "When status != 'ok', the human-readable reason (e.g. "
            "'embedding_unavailable', 'no_collections_configured')."
        ),
    )
    dense_available: bool = Field(default=True)
    lexical_available: bool = Field(default=True)
    collections_searched: List[str] = Field(default_factory=list)
    candidate_count: int = Field(default=0, ge=0)
    embedding_ms: float = Field(default=0.0, ge=0.0)
    lexical_ms: float = Field(default=0.0, ge=0.0)
    ranking_ms: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# MemoryPayload normalisation
# ---------------------------------------------------------------------------


class MemoryPayload:
    """Normalisation helpers for Qdrant payload fields.

    Qdrant payloads (and historical ingestion scripts) have stored
    ``keywords`` / ``entities`` / ``triggers`` as one of:

    * a ``list[str]`` (canonical),
    * a JSON-encoded string of a list,
    * a JSON-encoded string of a bare string,
    * a comma-separated string,
    * a single bare string,
    * ``None`` / empty.

    :func:`normalize_string_list` (top-level) already handles the
    general case; this class wraps it in a stable, field-keyed
    surface so callers can do ``MemoryPayload.keywords(payload)``
    without knowing which of the legacy shapes a given field
    happened to be stored in. The methods are also how the v0.3.0
    candidate construction pinpoints normalisation: every
    collection of strings coming out of a Qdrant payload passes
    through here so the upstream code never has to special-case
    the input shape.
    """

    #: Canonical payload field names for list-of-string fields.
    LIST_FIELDS: Tuple[str, ...] = ("keywords", "entities", "triggers")

    @classmethod
    def keywords(cls, payload: Dict[str, Any]) -> List[str]:
        return normalize_string_list(payload.get("keywords"), field_name="keywords")

    @classmethod
    def entities(cls, payload: Dict[str, Any]) -> List[str]:
        return normalize_string_list(payload.get("entities"), field_name="entities")

    @classmethod
    def triggers(cls, payload: Dict[str, Any]) -> List[str]:
        return normalize_string_list(payload.get("triggers"), field_name="triggers")

    @classmethod
    def list_field(cls, payload: Dict[str, Any], name: str) -> List[str]:
        """Normalise an arbitrary list-of-string field by name.

        ``name`` is used in the log line if normalisation gives up
        (the helper never raises, so the log is the only signal).
        """
        return normalize_string_list(payload.get(name), field_name=name)


__all__ = [
    "AmbiguousMemoryId",
    "MemoryRef",
    "MemoryRecord",
    "ScoredMemoryCandidate",
    "RetrievalDiagnostics",
    "MemoryPayload",
    "CandidateStatus",
    "CandidateTier",
    "normalize_string_list",
    "candidate_key",
    "ACTIVE_FIRST",
    "SUPERSEDED_FALLBACK_ONLY",
    "SUPERSEDED_BELOW_ACTIVE",
    "NO_PHYSICAL_DELETION",
    "EMBEDDING_FAILURE_DEGRADED",
    "NO_ZERO_VECTOR_FAKE_SUCCESS",
    "UNJUDGED_IS_NOT_NEGATIVE",
    "QWEN_NOT_IN_ONLINE_PATH",
    "HARD_CONTRACTS",
]