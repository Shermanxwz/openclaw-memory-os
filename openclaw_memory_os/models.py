"""Pydantic models for memories, recall hits, and health summaries.

These types are deliberately decoupled from any backend so the same
shapes can be sourced from Qdrant payloads, JSON files, or a future
SQLite index.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python 3.9+ ships zoneinfo
    ZoneInfo = None  # type: ignore[assignment]

# Asia/Shanghai is the operator timezone for the contractually-fixed weekly
# schedule. Kept as a module-level constant so the dashboard computation is
# not silently re-resolved on every request.
_OPERATOR_TZ = ZoneInfo("Asia/Shanghai") if ZoneInfo is not None else timezone.utc


class MemoryTier(str, Enum):
    """Tier classification for a memory.

    Tiers describe how durable / foundational a memory is, NOT how
    important it is in a single query. Importance is an orthogonal
    numeric score on :class:`Memory`.
    """

    CORE = "core"            # identity / long-lived rules
    LONG = "long"            # project-level durable notes
    MEDIUM = "medium"        # topic-level notes
    SHORT = "short"          # ephemeral notes (CI state, scratch)
    WORKING = "working"      # session-scoped, often superseded


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory entry."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    NEEDS_REVIEW = "needs_review"


class Memory(BaseModel):
    """A single memory entry.

    ``id`` is opaque (string) and treated as canonical. Payloads must
    include the fields defined here; the Qdrant adapter maps Qdrant
    payload dicts into this shape.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    text: str = Field(..., description="Human-readable memory content.")
    summary: Optional[str] = Field(default=None, description="Optional short summary.")
    tier: MemoryTier = MemoryTier.MEDIUM
    status: MemoryStatus = MemoryStatus.ACTIVE
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = Field(default=None, description="Free-form source label.")
    created_at: datetime
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    supersedes: Optional[str] = Field(default=None, description="ID of memory this one replaces.")
    superseded_by: Optional[str] = Field(default=None, description="ID of memory that replaces this one.")
    review_reason: Optional[str] = Field(default=None, description="Why this is a deletion candidate.")
    embedding: Optional[List[float]] = Field(default=None, exclude=True)

    # v0.2.2: Extended ingestion fields
    owner_confirmed: bool = Field(default=False, description="Whether the memory owner has confirmed this entry.")
    line_start: Optional[int] = Field(default=None, ge=0, description="Source file line start.")
    line_end: Optional[int] = Field(default=None, ge=0, description="Source file line end.")
    type: Optional[str] = Field(default=None, description="Memory type classification (e.g. rule, lesson, note, context).")
    topic: Optional[str] = Field(default=None, description="Topic label for grouping related memories.")

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if self.expires_at is None:
            return False
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires < now


class RecallHit(BaseModel):
    """A single ranked recall-test result."""

    id: str
    text: str
    summary: Optional[str] = None
    tier: MemoryTier
    status: MemoryStatus
    importance: float
    score: float = Field(..., description="Final composite score used for ranking.")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of the score: base, recency, importance, penalties, keyword.",
    )
    explanation: str = Field(..., description="Human-readable explanation of why this hit ranked.")
    # v0.3.0: collection-aware identity so feedback / evaluation can link
    # back to a specific (collection, memory_id) pair even when multiple
    # collections hold the same point id. Empty string for callers that
    # only know the memory_id (legacy path).
    collection: str = Field(
        default="",
        description="v0.3.0: Qdrant collection this hit came from (or 'sample' for the JSON backend).",
    )
    candidate_key: str = Field(
        default="",
        description="v0.3.0: canonical 'collection:memory_id' handle for feedback linkage.",
    )


class RecallRequest(BaseModel):
    """Request body for ``POST /api/recall-test``."""

    query: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field(default="hybrid", description="``keyword`` | ``hybrid`` (default) | ``dense``")
    since_days: Optional[int] = Field(default=None, ge=0, description="Optional recency window.")
    include_superseded: bool = Field(default=False)
    include_expired: bool = Field(default=False)
    tier_filter: Optional[List[MemoryTier]] = None
    limit: int = Field(default=10, ge=1, le=100)


class RecallFallbackInfo(BaseModel):
    """Metadata about the recall fallback strategy.

    Surfaced on :class:`RecallResponse.fallback` so dashboards can show
    when a superseded memory was added because the active-only pass
    came up short. Defaults keep the field backward-compatible for
    older clients that ignore extra keys.
    """

    enabled: bool = True
    min_results: int = 5
    used: bool = Field(default=False, description="Did the fallback fire for this response?")
    added: int = Field(default=0, description="How many superseded hits were appended.")


class RecallResponse(BaseModel):
    """v0.3.0: adds ``query_id``, ``policy_version``, and ``diagnostics``."""
    query: str
    mode: str
    took_ms: float
    backend: str
    total_considered: int
    hits: List[RecallHit]
    query_id: str = Field(default="", description="UUID for feedback / evaluation linkage.")
    policy_version: str = Field(default="", description="Active policy version at recall time.")
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    fallback: RecallFallbackInfo = Field(
        default_factory=lambda: RecallFallbackInfo(),
        description="Recall fallback metadata.",
    )


class TierCount(BaseModel):
    tier: MemoryTier
    count: int


class StatusCount(BaseModel):
    status: MemoryStatus
    count: int


class MonthCount(BaseModel):
    month: str = Field(..., description="YYYY-MM")
    count: int


class DuplicateCluster(BaseModel):
    representative_id: str
    member_ids: List[str]
    score: float = Field(..., description="Heuristic cluster score (higher = more likely duplicate).")
    rationale: str


class ConsolidationResult(BaseModel):
    """Result of consolidating a duplicate cluster into a single memory."""
    consolidated_id: str
    text: str
    merged_member_ids: List[str]
    preserved_tags: List[str]
    survivors: List[str] = Field(default_factory=list, description="IDs that were not merged (e.g. tier=core).")


class DeletionCandidate(BaseModel):
    id: str
    text: str
    tier: MemoryTier
    status: MemoryStatus
    reason: str
    recommended_action: str = Field(..., description="Always ``review`` in this OS.")


def _compute_next_run(schedule: str, *, now: Optional[datetime] = None) -> Optional[str]:
    """Compute the next scheduled run timestamp from a human schedule string.

    The contract for the weekly content-governance job is fixed:
    ``Tue 04:01 Asia/Shanghai``. This helper supports that fixed string as
    well as the obvious variants (``Tue 4:01 ...``) and falls back to
    ``None`` when the schedule cannot be parsed deterministically.

    The computation is intentionally narrow: the OS must never introspect
    cron tables or read host timezone configuration. If the contract string
    ever changes, update this parser and the contract test together.
    """
    if not schedule:
        return None
    parts = schedule.strip().split()
    if len(parts) < 2:
        return None
    weekday_token, time_token = parts[0].lower(), parts[1]

    weekday_map = {
        "mon": 0, "monday": 0,
        "tue": 1, "tues": 1, "tuesday": 1,
        "wed": 2, "weds": 2, "wednesday": 2,
        "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
        "fri": 4, "friday": 4,
        "sat": 5, "saturday": 5,
        "sun": 6, "sunday": 6,
    }
    target_weekday = weekday_map.get(weekday_token)
    if target_weekday is None:
        return None

    try:
        hour_str, minute_str = time_token.split(":", 1)
        target_hour = int(hour_str)
        target_minute = int(minute_str)
    except (ValueError, AttributeError):
        return None

    if ZoneInfo is None:  # pragma: no cover - environment guard only
        return None

    now_local = (now or datetime.now(_OPERATOR_TZ))
    if now_local.tzinfo is None:
        now_local = now_local.replace(tzinfo=_OPERATOR_TZ)
    else:
        now_local = now_local.astimezone(_OPERATOR_TZ)

    days_ahead = (target_weekday - now_local.weekday()) % 7
    candidate = now_local.replace(
        hour=target_hour, minute=target_minute, second=0, microsecond=0
    ) + timedelta(days=days_ahead)
    # If we landed on today's slot but it has already passed, push one week.
    if candidate <= now_local:
        candidate = candidate + timedelta(days=7)
    return candidate.isoformat()


class AutonomousGovernanceJob(BaseModel):
    """Compact operational status block for the weekly autonomous governance job.

    Designed for the operator dashboard: it surfaces **when the job last ran**,
    **when it will run next**, and **what the result was**, with the schedule,
    mode, and scope collapsed into a compact subtitle.

    Status fields (``last_run`` / ``last_result``) are intentionally ``None``
    by default — the OS has no authoritative source for run history in this
    environment (no cron introspection, no log scraping). The dashboard
    renders them as ``unknown`` so operators see honest state, not fabricated
    values. ``next_run`` is computed from the fixed schedule contract so the
    card always answers "when's the next run?" deterministically.

    Hard contract:

      * Job name:           ``weekly-memory-autonomous-content-governance``
      * Schedule:           Tuesday 04:01 Asia/Shanghai
      * Mode:               ``FORCE_CONTENT_SUPERSEDE=1``
      * Scope:              ``memory-content``
      * Allowed actions:    supersede / expire / archive / dedupe / promote
      * Hard boundary:      NEVER physically delete; never touch repo / system /
                            config / secrets / external targets
    """

    name: str = Field(
        default="weekly-memory-autonomous-content-governance",
        description="Cron-style identifier of the autonomous governance job.",
    )
    schedule: str = Field(
        default="Tue 04:01 Asia/Shanghai",
        description="Human-readable schedule expression in the operator timezone.",
    )
    mode: str = Field(
        default="FORCE_CONTENT_SUPERSEDE=1",
        description="Mode flag forced on for this run (content supersede deep audit).",
    )
    scope: str = Field(
        default="memory-content",
        description="Scope of the job. Memory content governance only.",
    )
    safety_boundary: str = Field(
        default=(
            "Never physically delete; never modify repo, system, config, secrets, "
            "personal taxonomy, or external services. Only supersede, expire, "
            "archive, dedupe, or promote memory content."
        ),
        description="Hard safety boundary statement rendered on the dashboard.",
    )
    allowed_actions: List[str] = Field(
        default_factory=lambda: [
            "supersede",
            "expire",
            "archive",
            "dedupe",
            "promote",
        ],
        description="Allowed memory-content lifecycle actions for this job.",
    )
    last_run: Optional[str] = Field(
        default=None,
        description=(
            "ISO-8601 timestamp of the last successful run, or ``None`` when "
            "unknown. The OS does not introspect cron or scrape logs to "
            "synthesise this value; it is reserved for an upstream status "
            "source to populate."
        ),
    )
    last_result: Optional[str] = Field(
        default=None,
        description=(
            "Short status token (e.g. ``ok`` / ``failed`` / ``running`` / ``pending``) "
            "from the most recent run, or ``None`` when unknown."
        ),
    )
    last_summary: Optional[str] = Field(
        default=None,
        description="Short redacted summary of the most recent governance run.",
    )
    next_run: Optional[str] = Field(
        default=None,
        description=(
            "ISO-8601 timestamp of the next scheduled run, computed from the "
            "fixed schedule contract (see ``schedule``)."
        ),
    )

    @classmethod
    def for_dashboard(
        cls,
        *,
        now: Optional[datetime] = None,
        last_run: Optional[str] = None,
        last_result: Optional[str] = None,
        last_summary: Optional[str] = None,
    ) -> "AutonomousGovernanceJob":
        """Build a default dashboard instance with ``next_run`` pre-computed.

        ``last_run`` / ``last_result`` default to ``None`` (= unknown) so the
        UI renders honest empty state until an upstream status source
        (future cron report / status file) populates them.
        """
        # Materialise the default descriptor first so the schedule contract
        # is the single source of truth for the next-run calculation.
        base = cls()
        return cls(
            name=base.name,
            schedule=base.schedule,
            mode=base.mode,
            scope=base.scope,
            safety_boundary=base.safety_boundary,
            allowed_actions=list(base.allowed_actions),
            last_run=last_run,
            last_result=last_result,
            last_summary=last_summary,
            next_run=_compute_next_run(base.schedule, now=now),
        )


class MemoryBrainStatus(BaseModel):
    """Status files emitted by optional Memory Brain ingest/consolidate scripts."""
    ingest: Dict[str, Any] = Field(default_factory=dict, description="Latest structured-memory ingest status.")
    consolidate: Dict[str, Any] = Field(default_factory=dict, description="Latest memory consolidation status.")


class MaintenanceHealth(BaseModel):
    """Structured maintenance health for the dashboard."""
    enabled: bool = Field(default=True, description="Is there an expected cron/timer entry.")
    lock_present: bool = Field(default=False, description="Is the flock lock file currently present.")
    last_run: Optional[str] = Field(default=None, description="ISO-8601 UTC of last log write.")
    last_ok: Optional[str] = Field(default=None, description="ISO-8601 UTC of last successful run.")
    log_lines: int = Field(default=0, description="Approximate line count of log file.")
    log_path: Optional[str] = Field(default=None, description="Resolved log file path.")



class LastMaintenanceSummary(BaseModel):
    """Summary of the most recent maintenance run."""
    ingested_total: int = Field(default=0, description="Total chunks processed in last ingestion (legacy alias for chunks_scanned).")
    ingested_new: int = Field(default=0, description="Newly ingested chunks.")
    chunks_scanned: int = Field(default=0, description="Total chunks scanned this run (== ingested_total). Clarifies 'new/scanned' on dashboard.")
    expired_count: int = Field(default=0, description="Expired candidates.")
    superseded_links: int = Field(default=0, description="Supersede links applied.")
    snapshot_name: Optional[str] = Field(default=None, description="Most recent snapshot name.")
    snapshot_size_bytes: int = Field(default=0, description="Snapshot size in bytes.")
    collections: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Per-collection maintenance summary.")
    totals: Dict[str, Any] = Field(default_factory=dict, description="Multi-collection aggregate maintenance counters.")


class HealthSummary(BaseModel):
    backend: str
    total_memories: int
    active: int
    superseded: int
    expired: int
    needs_review: int
    duplicates_estimate: int
    deletion_candidate_count: int
    never_delete: int = 0
    last_maintenance: Optional[str] = None
    maintenance_health: MaintenanceHealth = Field(default_factory=lambda: MaintenanceHealth())
    last_maintenance_summary: LastMaintenanceSummary = Field(default_factory=lambda: LastMaintenanceSummary())
    memory_brain: MemoryBrainStatus = Field(default_factory=lambda: MemoryBrainStatus())
    autonomous_governance: AutonomousGovernanceJob = Field(
        default_factory=lambda: AutonomousGovernanceJob(),
        description="Static descriptor of the weekly autonomous content-governance job.",
    )
    tier_distribution: List[TierCount]
    status_distribution: List[StatusCount]
    monthly_counts: List[MonthCount]
    # v0.2.2 graduation — surface legacy/default points so the tier chart
    # doesn't lie about historical payloads that never carried a tier field.
    legacy_default_count: int = 0
    importance_distribution: List["ImportanceBucket"] = Field(default_factory=list)
    generated_at: datetime
    collections: List[str] = []


class ImportanceBucket(BaseModel):
    """Histogram bucket for memory importance scores."""
    label: str
    count: int
    min_importance: float
    max_importance: float


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_dict(model: BaseModel) -> Dict[str, Any]:
    """Helper for serializing Pydantic models into JSON-friendly dicts."""
    return model.model_dump(mode="json")


class FeedbackEntry(BaseModel):
    """User feedback on a recall hit (useful / not useful).

    v0.3.0: new fields ``query_id`` and ``candidate_key`` replace the
    legacy ``query`` / ``memory_id`` pair for the structured feedback
    path. The old ``memory_id`` / ``query`` / ``note`` fields are still
    accepted for backward-compatibility but will be ignored when
    ``query_id`` is present.
    """
    memory_id: str = ""
    query: str = ""
    useful: bool
    query_id: str = Field(default="", description="v0.3.0: UUID from recall run.")
    candidate_key: str = Field(default="", description="v0.3.0: collection:memory_id.")
    feedback_at: datetime = Field(default_factory=utcnow)
    note: Optional[str] = Field(default=None, max_length=500)


class AuditLogEntry(BaseModel):
    """An entry in the SQLite audit log."""
    id: int = Field(default=0, description="Auto-increment primary key.")
    timestamp: datetime = Field(default_factory=utcnow)
    action: str = Field(..., description="e.g. ingest, recall, feedback, consolidate")
    actor: Optional[str] = Field(default=None, description="Who/what performed the action.")
    memory_id: Optional[str] = Field(default=None)
    detail: Optional[str] = Field(default=None, max_length=2000)


class IngestProgress(BaseModel):
    """Persisted ingestion progress state for checkpoint/resume."""
    checkpoint_id: str = Field(default="", description="Timestamp-based checkpoint key.")
    started_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    total_files: int = 0
    total_chunks: int = 0
    written: int = 0
    failed: int = 0
    current_chunk: Optional[str] = None
    status: str = "running"  # running | completed | failed
    source_files: List[str] = Field(default_factory=list)
    completed_chunk_ids: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class ConsolidationRequest(BaseModel):
    """Request body for POST /api/consolidate-duplicates."""
    cluster_ids: List[str] = Field(..., min_length=1, description="List of memory IDs to consolidate.")
    strategy: str = Field(default="merge", description="merge | keep_newest | keep_best")


class ReclassifyRequest(BaseModel):
    """Request body for POST /api/maintenance/reclassify."""
    collections: Optional[List[str]] = Field(
        default=None,
        description="Optional list of Qdrant collections to reclassify. Defaults to backend's configured collections.",
    )
    dry_run: bool = Field(
        default=False,
        description="If true, only preview changes without writing to Qdrant.",
    )
