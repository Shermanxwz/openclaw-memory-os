"""v0.3.0 offline evaluation for the recall pipeline.

Constructs evaluation cases from structured feedback in the
SQLite store, splits them by time (60/20/20), and computes
standard information-retrieval metrics: Recall@k, MRR@k,
nDCG@k, useful-at-1/5, explicit-negative-at-5, fallback rate,
degraded rate, and latency percentiles.

Key design decisions
--------------------

* **Only user-judged queries are evaluation cases.** A query that
  never received any feedback is skipped, because we cannot know
  whether the user was satisfied or simply didn't interact.

* **Positive = ``useful=true``; negative = ``useful=false``.** The
  absence of feedback is NOT negative.

* **CandidatePool.** Each query's recall is run *once* with a
  large ``dense_k / lexical_k`` to build a pool of candidates.
  All candidate policies then re-rank this pool rather than
  re-running Qdrant. This keeps evaluation practical on a
  50k-point corpus.

* **No model calls during evaluation.** qwen2.5:1.5b is not
  invoked; the embedding cache is consulted but the embedder is
  not called for query vectorisation (the embedding was already
  computed during the retrieval that generated the CandidatePool).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple


if TYPE_CHECKING:
    from .backends import MemoryBackend

logger = logging.getLogger(__name__)

# Maximum query candidates per evaluation run. Keeps resource use
# bounded on a 50k-point corpus.
_MAX_CASES = 500


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class _QrelEntry:
    """Relevance judgement for one (query, candidate) pair."""

    candidate_key: str
    relevant: bool  # True = useful, False = not useful


@dataclass
class _EvaluationCase:
    """One judged query, with its positive/negative doc ids."""

    query_id: str
    query_text: str
    positives: Set[str]
    negatives: Set[str]


@dataclass
class _RankedResult:
    """One candidate in a ranked list for a query."""

    candidate_key: str
    score: float


@dataclass
class _RankFnOutcome:
    """Optional structured return type for ``rank_fn`` in :func:`evaluate`.

    The legacy contract is that ``rank_fn(query_text, query_id)``
    returns ``List[str]``. v0.3.0.x adds the option of returning an
    instance of this dataclass so the evaluator can attribute
    per-case metrics for the **degraded** path (e.g. dense retrieval
    timed out and the engine fell back to lexical-only) and the
    **superseded-fallback** path (active-only pass yielded fewer
    than the configured minimum and superseded candidates were
    appended as a fallback band).

    Attributes
    ----------
    ranked: List[str]
        Candidate keys in descending order of relevance.
    degraded: bool
        ``True`` when the engine ran in degraded mode for this
        query (e.g. dense back-end unreachable, lexical-only
        fallback). Drives :attr:`EvalResult.degraded_rate`.
    fallback_used: bool
        ``True`` when the superseded-fallback band was activated.
        Drives :attr:`EvalResult.fallback_useful_rate` together
        with ``fallback_keys`` and the case's judged positives.
    fallback_keys: List[str]
        Candidate keys contributed specifically by the
        superseded-fallback band. Must be disjoint from ``ranked``.
    """

    ranked: List[str]
    degraded: bool = False
    fallback_used: bool = False
    fallback_keys: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.fallback_keys is None:
            self.fallback_keys = []


# ---------------------------------------------------------------------------
# Query normalisation (G5.2)
# ---------------------------------------------------------------------------


def normalize_query(query: str) -> str:
    """Normalise a query string so near-duplicates collapse.

    Steps:

      1. NFC unicode normalisation (so visually-identical glyphs match).
      2. NFKC + strip zero-width / format chars (BOM, ZWJ, etc.).
      3. Case-fold (so "Foo" and "foo" merge).
      4. Strip whitespace + collapse internal whitespace.
      5. Lowercase.

    The output is a stable canonical form used for dedup in
    :func:`split_cases`. Two queries that differ only in
    capitalisation, surrounding whitespace, fullwidth / CJK-width
    glyphs, or invisible format characters will produce the same
    string so they collapse into a single eval set entry.

    Parameters
    ----------
    query : str
        The raw query string. ``None`` is treated as an empty string.

    Returns
    -------
    str
        The canonical, normalised form. Empty input round-trips to
        the empty string.
    """
    if not query:
        return ""
    # NFKC normalises compatibility decompositions (e.g. fullwidth
    # ASCII → ASCII) before we do anything else.
    q = unicodedata.normalize("NFKC", query)
    # Strip format / combining / control / surrogate chars so
    # zero-width joiners, BOMs, and other invisibles don't break
    # equality. Whitespace-like control chars (\t, \n, \r) are
    # converted to a regular space so they collapse alongside
    # regular spaces in the final ``split()`` step. Everything
    # else in categories Cf/Cs/Co/Mn/Me is dropped.
    q = "".join(
        ch
        for ch in q
        if not (
            unicodedata.category(ch) in ("Cf", "Cs", "Co", "Mn", "Me")
            or (unicodedata.category(ch).startswith("C") and ch not in (" ", "\t", "\n", "\r"))
        )
    )
    # Normalise tabs / newlines to spaces so ``split()`` below
    # collapses them uniformly.
    q = q.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    # Case-fold (Unicode-aware, stronger than ``.lower()``).
    q = q.casefold().strip()
    # Collapse internal whitespace to single spaces.
    q = " ".join(q.split())
    return q


@dataclass
class CandidatePool:
    """Deterministic container for a per-query candidate pool.

    The offline evaluation pipeline runs the **expensive** part
    (dense + lexical retrieval) once per query to produce a pool of
    candidates, then re-ranks that pool for each candidate policy.
    This keeps evaluation practical on a 50k-point corpus while
    still letting us score multiple policies against the same
    pool.

    The pool is fully **deterministic** and **side-effect free**:
    it is constructed from provided ranked IDs / candidate dicts
    (e.g. results from a one-shot recall test) and exposes
    attribute-style accessors for the dense, lexical, and
    superseded sub-pools. The container itself never reaches out
    to Qdrant or any other live backend.

    Attributes
    ----------
    query_id: str
        Stable identifier for the query the pool belongs to.
    query_text: str
        Original query string (kept for traceability).
    dense_active: List[Dict[str, Any]]
        Candidates returned by the dense (vector) channel with
        status ``active``. Each dict carries at minimum
        ``candidate_key``, ``score`` and any optional metadata
        (``collection``, ``memory_id``, ``channel``).
    lexical_active: List[Dict[str, Any]]
        Candidates returned by the lexical (BM25) channel with
        status ``active``.
    superseded: List[Dict[str, Any]]
        Candidates with status ``superseded`` from either channel.
    corpus_snapshot_id: Optional[str]
        Optional identifier of the corpus snapshot the pool was
        computed against. ``None`` when unavailable.
    extra: Dict[str, Any]
        Free-form metadata (e.g. diagnostics fields from the
        originating engine call). Never used for scoring; kept
        for the dashboard / debug tooling.

    Notes
    -----
    The pool is intentionally permissive: it accepts whatever the
    caller provides and exposes deterministic ordering helpers
    (``deduped_active_keys``, ``superseded_keys``, etc.). It does
    **not** fabricate data when the caller supplies nothing — if
    a channel is empty the corresponding list stays empty.
    """

    query_id: str
    query_text: str = ""
    dense_active: List[Dict[str, Any]] = field(default_factory=list)
    lexical_active: List[Dict[str, Any]] = field(default_factory=list)
    superseded: List[Dict[str, Any]] = field(default_factory=list)
    corpus_snapshot_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_ranked(
        cls,
        query_id: str,
        query_text: str,
        ranked_candidates: Sequence[Any],
        *,
        corpus_snapshot_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "CandidatePool":
        """Build a pool from a flat sequence of candidate records.

        Each record is expected to expose either ``candidate_key``
        (string) or an ``id`` attribute / key, plus an optional
        ``status`` (one of ``"active"``, ``"superseded"``,
        ``"expired"``) and an optional ``channel`` (``"dense"``
        / ``"lexical"``). Records missing a channel are routed
        to ``dense_active`` by default for backward compatibility
        with existing recall responses.
        """
        dense_active: List[Dict[str, Any]] = []
        lexical_active: List[Dict[str, Any]] = []
        superseded: List[Dict[str, Any]] = []
        for c in ranked_candidates:
            entry = cls._normalise_record(c)
            status = str(entry.get("status") or "active").lower()
            channel = str(entry.get("channel") or "dense").lower()
            if status == "superseded":
                superseded.append(entry)
                continue
            if status not in ("active",):
                # expired / unknown → treat as non-active fallback material
                # so it surfaces under superseded when the engine reports it
                superseded.append(entry)
                continue
            if channel == "lexical":
                lexical_active.append(entry)
            else:
                dense_active.append(entry)
        return cls(
            query_id=query_id,
            query_text=query_text or "",
            dense_active=dense_active,
            lexical_active=lexical_active,
            superseded=superseded,
            corpus_snapshot_id=corpus_snapshot_id,
            extra=dict(extra or {}),
        )

    @classmethod
    def empty(
        cls,
        query_id: str,
        query_text: str = "",
        *,
        corpus_snapshot_id: Optional[str] = None,
    ) -> "CandidatePool":
        """Return an empty pool. Useful as a default / no-result state."""
        return cls(
            query_id=query_id,
            query_text=query_text or "",
            corpus_snapshot_id=corpus_snapshot_id,
        )

    # -- introspection helpers -------------------------------------------

    @staticmethod
    def _normalise_record(record: Any) -> Dict[str, Any]:
        """Convert a recall hit / dict / dataclass into a plain dict.

        Always returns a dict with at least ``candidate_key`` and
        ``score`` (defaulting to ``""`` and ``0.0``). Unknown
        attributes / keys are kept verbatim so the pool can carry
        extra metadata without losing it.
        """
        if isinstance(record, dict):
            ck = record.get("candidate_key") or record.get("id") or ""
            score = record.get("score", 0.0)
            entry = dict(record)
            entry.setdefault("candidate_key", str(ck))
            entry.setdefault("score", float(score) if score is not None else 0.0)
            return entry
        # dataclass / pydantic-like object
        entry: Dict[str, Any] = {}
        ck = getattr(record, "candidate_key", None) or getattr(record, "id", None) or ""
        entry["candidate_key"] = str(ck)
        entry["score"] = float(getattr(record, "score", 0.0) or 0.0)
        for attr in (
            "memory_id",
            "collection",
            "status",
            "channel",
            "rank",
            "explanation",
        ):
            if hasattr(record, attr):
                value = getattr(record, attr)
                if value is not None:
                    entry[attr] = value
        return entry

    # -- accessors --------------------------------------------------------

    @property
    def dense_active_keys(self) -> List[str]:
        """Active candidate keys in dense-channel order (deduplicated)."""
        return self._deduped_keys([c["candidate_key"] for c in self.dense_active])

    @property
    def lexical_active_keys(self) -> List[str]:
        """Active candidate keys in lexical-channel order (deduplicated)."""
        return self._deduped_keys([c["candidate_key"] for c in self.lexical_active])

    @property
    def superseded_keys(self) -> List[str]:
        """Superseded candidate keys in original order (deduplicated)."""
        return self._deduped_keys([c["candidate_key"] for c in self.superseded])

    @property
    def active_keys(self) -> List[str]:
        """Union of dense + lexical active keys, dense-first then lexical.

        Deduplicated while preserving the dense channel's order so
        re-rankers that simply concatenate channels see a stable
        ranking.
        """
        return self._deduped_keys(
            [c["candidate_key"] for c in self.dense_active]
            + [c["candidate_key"] for c in self.lexical_active]
        )

    @staticmethod
    def _deduped_keys(keys: Sequence[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for k in keys:
            if not k:
                continue
            if k in seen:
                continue
            seen.add(k)
            out.append(k)
        return out

    # -- summary ----------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Compact summary suitable for dashboards / API responses."""
        return {
            "query_id": self.query_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "dense_active_count": len(self.dense_active),
            "lexical_active_count": len(self.lexical_active),
            "superseded_count": len(self.superseded),
            "active_unique": len(self.active_keys),
            "superseded_unique": len(self.superseded_keys),
        }


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


def snapshot_corpus(
    backend: "MemoryBackend",
    path: str,
) -> str:
    """Save a JSON snapshot of all memories to a file.

    This allows reproducible evaluation by capturing the exact
    corpus state at a point in time. The snapshot includes every
    memory's full payload (id, text, status, tier, importance,
    tags, source, timestamps, etc.).

    Parameters
    ----------
    backend:
        A :class:`MemoryBackend` instance to read memories from.
    path:
        Filesystem path to write the JSON snapshot to.

    Returns
    -------
    str
        The path written.
    """

    memories = backend.list_memories()
    records = []
    for m in memories:
        rec = {
            "id": str(m.id),
            "text": m.text or "",
            "summary": m.summary,
            "source": m.source,
            "tags": list(m.tags or []),
            "status": m.status.value if m.status else "active",
            "tier": m.tier.value if m.tier else "medium",
            "importance": float(m.importance or 0.5),
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            "supersedes": m.supersedes,
            "superseded_by": m.superseded_by,
            "review_reason": m.review_reason,
        }
        records.append(rec)
    snapshot = {
        "version": "1",
        "count": len(records),
        "memories": records,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("snapshot_corpus: wrote %d memories to %s", len(records), p)
    return str(p)


def _get_db() -> sqlite3.Connection:
    # Reuse the authoritative DB opener so migrations, WAL, FK policy and
    # 0600 permissions cannot diverge between evaluation and feedback writes.
    from .recall_feedback import _get_db as _feedback_db
    return _feedback_db()


def _load_cases(limit: int = _MAX_CASES) -> List[_EvaluationCase]:
    """Load evaluation cases from the feedback DB, newest first.

    Only queries that have at least one ``useful=true`` entry are
    included. Cases are ordered by descending ``created_at`` and
    capped at ``limit``.
    """
    conn = _get_db()
    try:
        rows = conn.execute(
            """
            SELECT fe.query_id, rr.query_text, fe.candidate_key, fe.useful
            FROM feedback_events fe
            JOIN recall_runs rr ON fe.query_id = rr.query_id
            WHERE fe.migration_status IS NULL
               OR (fe.migration_status = 'migrated:audit'
                   AND fe.resolution_status = 'migrated:verified')
            ORDER BY fe.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return []
    # Group by query_id
    cases_map: Dict[str, _EvaluationCase] = {}
    for r in rows:
        qid = r["query_id"]
        if qid not in cases_map:
            cases_map[qid] = _EvaluationCase(
                query_id=qid,
                query_text=r["query_text"] or "",
                positives=set(),
                negatives=set(),
            )
        ck = r["candidate_key"]
        if r["useful"] == 1:
            cases_map[qid].positives.add(ck)
        else:
            cases_map[qid].negatives.add(ck)
    # Transform to list, sorted by newest query_id (naive: query_id
    # as string — in practice the IDs are UUIDs / timestamps)
    cases = list(cases_map.values())
    # Only keep queries with at least one positive
    cases = [c for c in cases if c.positives]
    return cases


# ---------------------------------------------------------------------------
# Time split
# ---------------------------------------------------------------------------


@dataclass
class EvalSplit:
    train: List[_EvaluationCase]
    validation: List[_EvaluationCase]
    test: List[_EvaluationCase]


def time_split(
    cases: List[_EvaluationCase],
    train_pct: float = 0.6,
    validation_pct: float = 0.2,
) -> EvalSplit:
    """Split cases by their query_id (assumed to sort chronologically).

    ``train_pct`` of the oldest cases go to train, the next
    ``validation_pct`` go to validation, the remainder go to test.
    """
    n = len(cases)
    if n == 0:
        return EvalSplit([], [], [])
    train_end = max(1, int(n * train_pct))
    val_end = max(train_end + 1, int(n * (train_pct + validation_pct)))
    return EvalSplit(
        train=cases[:train_end],
        validation=cases[train_end:val_end],
        test=cases[val_end:],
    )


# ---------------------------------------------------------------------------
# Time-based split with near-duplicate dedup (G5.2)
# ---------------------------------------------------------------------------


def _case_query_text(case: Any) -> str:
    """Return ``case.query_text`` whether ``case`` is a dict or an object."""
    if isinstance(case, dict):
        return str(case.get("query_text") or "")
    return str(getattr(case, "query_text", "") or "")


def _case_created_at(case: Any) -> str:
    """Return ``case.created_at`` whether ``case`` is a dict or an object.

    Falls back to the empty string when the field is missing; callers
    sort missing-timestamp cases to the end via the ``"\uffff"`` sentinel
    in :func:`split_cases`.
    """
    if isinstance(case, dict):
        return str(case.get("created_at") or "")
    return str(getattr(case, "created_at", "") or "")


def _sort_key_for_timestamp(ts: str):
    """Sort key for a ``created_at`` string.

    Prefers parsed ISO 8601 (so "2026-07-15" < "2026-07-16" works
    the way you'd expect, *and* "1" < "2" < "10" works for plain
    integer-shaped timestamps used in tests). Falls back to the raw
    string when parsing fails, with an empty / missing timestamp
    sorting to the very end via the ``"\uffff"`` sentinel.
    """
    if not ts:
        return (1, "\uffff")
    # Plain-integer timestamps (common in tests): parse as ordinal.
    if ts.isdigit():
        try:
            return (0, int(ts))
        except ValueError:
            pass
    try:
        # ``datetime.fromisoformat`` accepts the "YYYY-MM-DD" and
        # "YYYY-MM-DD HH:MM:SS" shapes that ``recall_runs.created_at``
        # produces via SQLite's ``datetime('now')``.
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return (0, ts)
    return (0, parsed)


def split_cases(
    cases: List[Any],
    *,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,  # noqa: ARG001 — kept for backward-compatible signature
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Time-based 60/20/20 split with near-duplicate dedup (G5.2).

    The legacy ``split_cases`` in :mod:`openclaw_memory_os.evolution`
    (kept unchanged because Wave B/C will refactor it) splits by
    ``query_id`` hash with a fixed seed. This implementation is the
    v0.3.0.x Runbook-v2 version: it splits by ``created_at`` so the
    eval set is chronologically faithful, and it collapses near-
    duplicate queries via :func:`normalize_query` so the same logical
    query in different forms (case differences, fullwidth / CJK-width
    glyphs, extra whitespace) doesn't fragment the eval set.

    Algorithm
    ---------
    1. Group cases by ``normalize_query(case.query_text)``. Each
       group picks its **earliest** ``created_at`` as the
       representative; only that representative participates in the
       60/20/20 split.
    2. Sort representatives by ``created_at`` ascending.
    3. Take the first ``train_ratio`` fraction as train, the next
       ``val_ratio`` as validation, the remainder as test.
    4. Cases whose representative ended up in val / test still need
       to appear in *some* bucket so they aren't dropped from the
       eval set; they are routed to train (the largest, most
       forgiving bucket) so the offline evaluator has data to score
       for every query the user has interacted with.

    Parameters
    ----------
    cases : list
        Any iterable of case-like objects. Each case must expose
        ``query_text`` (str) and ``created_at`` (ISO 8601 timestamp
        string, or anything that sorts lexicographically the way
        you want — the function does NOT parse the timestamp).
        Dict-style cases (``{"query_text": ..., "created_at": ...}``)
        and attribute-style cases (e.g. :class:`_EvaluationCase`)
        are both supported.
    train_ratio, val_ratio, test_ratio : float
        Fractions that should sum to 1.0. Defaults reproduce the
        60 / 20 / 20 split specified in the Runbook.
    seed : int
        Legacy parameter accepted for backward compatibility with
        the ``evolution.split_cases(cases, seed=...)`` call sites.
        The time-based split is deterministic from ``created_at``
        alone, so ``seed`` is intentionally ignored.

    Returns
    -------
    tuple of three lists
        ``(train, validation, test)``. Together they contain every
        input case exactly once (cases from the same normalised
        query group are routed to the same bucket as their
        representative).
    """
    if not cases:
        return [], [], []

    # 1) Group by normalised query_text; each group keeps the case
    #    with the earliest created_at as the "representative".
    groups: Dict[str, List[Any]] = {}
    group_order: List[str] = []  # preserve insertion order for stability
    for case in cases:
        norm = normalize_query(_case_query_text(case))
        if norm not in groups:
            groups[norm] = []
            group_order.append(norm)
        groups[norm].append(case)

    # Representatives: for each group, pick the earliest-created_at
    # case. Tie-break by original input order so the function is
    # fully deterministic when timestamps collide (e.g. all-empty).
    representatives: List[Any] = []
    for norm in group_order:
        members = groups[norm]
        # Empty timestamps sort to the end via the "(1, '\uffff')" key.
        rep = min(
            members,
            key=lambda m: _sort_key_for_timestamp(_case_created_at(m)),
        )
        representatives.append(rep)

    # 2) Sort representatives by created_at ascending. Empty
    # timestamps sort to the end so they don't steal early slots.
    representatives.sort(
        key=lambda c: _sort_key_for_timestamp(_case_created_at(c))
    )

    n = len(representatives)
    # Compute split indices; guard against degenerate splits.
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        # Caller asked for an impossible split; fall back to all-train.
        train_end = n
        val_end = n
    else:
        train_end = max(1, int(round(n * train_ratio)))
        # Make sure val has at least one slot when n is large enough
        # for a meaningful split; otherwise push everything into train.
        if n >= 3:
            val_end = max(train_end + 1, int(round(n * (train_ratio + val_ratio))))
        else:
            val_end = train_end
        # Clamp so test is non-empty when n permits.
        val_end = min(val_end, n)

    rep_train = set(id(r) for r in representatives[:train_end])
    rep_val = set(id(r) for r in representatives[train_end:val_end])
    rep_test = set(id(r) for r in representatives[val_end:])

    # 3) Bucket every case. Representatives land in their bucket;
    #    non-representative siblings of a group follow their
    #    representative. Anything that doesn't have a normalised
    #    query (empty query_text) lands in train as a defensive
    #    fallback so it isn't lost.
    train: List[Any] = []
    val: List[Any] = []
    test: List[Any] = []
    for norm in group_order:
        members = groups[norm]
        rep = None
        for m in members:
            if id(m) in rep_train or id(m) in rep_val or id(m) in rep_test:
                rep = m
                break
        if rep is None:
            # Defensive: empty group (shouldn't happen, but guard).
            for m in members:
                train.append(m)
            continue
        if id(rep) in rep_train:
            for m in members:
                train.append(m)
        elif id(rep) in rep_val:
            for m in members:
                val.append(m)
        elif id(rep) in rep_test:
            for m in members:
                test.append(m)
        else:
            for m in members:
                train.append(m)

    return train, val, test


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """Recall@k = fraction of relevant docs in top-k."""
    if not relevant:
        return 0.0
    top_k = set(ranked[:k])
    hit = len(top_k & relevant)
    return hit / len(relevant)


def _mrr(ranked: Sequence[str], relevant: Set[str]) -> float:
    """Mean reciprocal rank — first relevant result."""
    if not relevant:
        return 0.0
    for i, doc in enumerate(ranked):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int, neg: Set[str]) -> float:
    """nDCG@k with binary relevance (1 for relevant, 0 for not, ignore
    unjudged — they contribute 0 gain). Negative items are treated as
    non-relevant (gain 0)."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, doc in enumerate(ranked[:k]):
        if doc in relevant:
            dcg += 1.0 / math.log2(i + 2)
        # Negative items gain 0.
    # Ideal DCG: all relevant at top.
    num_rel = min(len(relevant), k)
    if num_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_rel))
    return dcg / idcg if idcg > 0 else 0.0


def _useful_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-k that are relevant."""
    if not ranked or k == 0:
        return 0.0
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in relevant) / min(len(top_k), k)


def _explicit_negative_at_k(ranked: Sequence[str], neg: Set[str], k: int) -> float:
    """Fraction of top-k that are explicitly not-useful."""
    if not ranked or k == 0:
        return 0.0
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in neg) / min(len(top_k), k)


def _judged_ndcg_at_10(
    judged: Sequence[Tuple[str, int]],
    k: int = 10,
) -> Optional[float]:
    """nDCG@k computed from graded relevance judgements.

    ``judged`` is a sequence of ``(candidate_key, grade)`` pairs in
    ranked order. Grades are integer relevance labels (>=0). The
    function returns ``None`` when ``judged`` is empty — callers
    are expected to surface ``None`` (not 0.0) so the dashboard
    can distinguish "no judgement available" from "judged and
    scored zero".

    Negative grades are clamped to 0 (we do not penalise items
    for being judged-irrelevant; we simply exclude them from the
    ideal ranking). Items without a positive grade contribute 0
    gain but can appear in the ideal ordering as 0-gain items.
    """
    if not judged:
        return None
    # DCG
    dcg = 0.0
    grades: List[int] = []
    for i, (_ck, grade) in enumerate(judged[:k]):
        g = max(0, int(grade))
        if g > 0:
            dcg += (2 ** g - 1) / math.log2(i + 2)
        grades.append(g)
    # Ideal DCG: sort by grade desc, then take top-k with the
    # standard 2^g - 1 gain.
    ideal = sorted((g for g in grades if g > 0), reverse=True)
    if not ideal:
        return 0.0
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def _useful_superseded_fallback_rate(
    *,
    active_hits: Sequence[str],
    fallback_hits: Sequence[str],
    judged_positives: Set[str],
) -> Optional[float]:
    """Fraction of fallback expansions that actually surfaced a useful hit.

    A "fallback expansion" happens when the active-only pass yields
    fewer than the configured minimum results and the engine pulls
    in superseded candidates as a fallback band. This metric
    measures how often that fallback band contributed *new*
    positives (judged-useful docs the active pass missed).

    Parameters
    ----------
    active_hits:
        Candidate keys returned by the active-only pass.
    fallback_hits:
        Additional candidate keys appended by the superseded
        fallback (must not overlap with ``active_hits``).
    judged_positives:
        The set of candidate keys the user has marked useful.

    Returns
    -------
    Optional[float]
        ``useful / total_expansions`` rounded to [0, 1], or
        ``None`` when there were no fallback expansions to score.
        Callers should surface ``None`` (not 0.0) so the
        dashboard can distinguish "never fell back" from "fell
        back and was always useless".
    """
    if not judged_positives:
        return None
    added = [h for h in fallback_hits if h and h not in set(active_hits)]
    if not added:
        return None
    useful = sum(1 for h in added if h in judged_positives)
    return useful / len(added)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Aggregate metrics for one policy evaluated on a split.

    New in v0.3.0.x
    ----------------
    * ``judged_ndcg_at_10`` is ``None`` when no graded judgements
      are available, otherwise the average graded nDCG@10 across
      the judged cases. The dashboard surfaces ``None`` as
      ``status="unavailable"`` so operators can distinguish
      "no judgement yet" from "judged and scored zero".
    * ``useful_superseded_fallback_rate`` is ``None`` when the
      candidate never had to fall back to the superseded band,
      otherwise the fraction of fallback expansions that actually
      surfaced a judged-useful hit.
    * ``corpus_snapshot_id`` is the identifier of the corpus
      snapshot the pool was computed against, or ``None`` when
      unavailable at runtime.
    * ``useful_at_5`` is the fraction of cases where at least
      one of the top-5 hits is in the case's positive set.
    * ``explicit_negative_at_5`` is the fraction of *judged*
      cases where at least one of the top-5 hits is in the
      case's negative set. Unjudged cases are excluded from
      both numerator and denominator so the metric is not
      biased by absence-of-feedback (absence is NOT negative).
    * ``no_result_rate`` is the fraction of cases where the
      ranker returned an empty list.
    * ``degraded_rate`` is the fraction of cases where the
      ranker reported ``degraded=True`` (engine ran in
      degraded mode — e.g. dense retrieval failed and the
      engine fell back to lexical-only). The metric is
      derived from the optional :class:`_RankFnOutcome`
      return type; legacy ``rank_fn`` callables that return
      ``List[str]`` keep this metric at ``0.0``.
    * ``fallback_useful_rate`` is the fraction of cases where
      the ranker used the superseded-fallback band AND at
      least one of the fallback keys matched a judged
      positive. Like :attr:`degraded_rate`, it requires the
      :class:`_RankFnOutcome` return type to populate.
    * ``p50_latency`` / ``p95_latency`` are derived from the
      wall-clock latency the evaluator observed around each
      ``rank_fn`` invocation. They are reported in
      milliseconds.
    """

    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    useful_at_1: float = 0.0
    useful_at_5: float = 0.0
    explicit_negative_at_5: float = 0.0
    no_result_rate: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    degraded_rate: float = 0.0
    fallback_rate: float = 0.0
    # v0.3.0.x (G5.4): "fraction of cases where Superseded-fallback hit
    # was useful". Stays at 0.0 when the ranker does not return a
    # structured :class:`_RankFnOutcome`. See ``evaluate()``.
    fallback_useful_rate: float = 0.0
    num_cases: int = 0
    # v0.3.0.x: graded judgements + superseded-fallback metric.
    # Both default to ``None`` so legacy callers / older code paths
    # keep working unchanged; the dashboard / API surface the
    # ``None`` value as ``status="unavailable"`` rather than zero.
    judged_ndcg_at_10: Optional[float] = None
    useful_superseded_fallback_rate: Optional[float] = None
    num_judged_cases: int = 0
    # v0.3.0.x: corpus snapshot identifier (forwarded from the
    # pool when available). ``None`` when the offline pipeline has
    # not yet recorded a snapshot id.
    corpus_snapshot_id: Optional[str] = None
    # v0.3.0.x: explicit availability flags for the new metrics
    # so the API can render a stable ``status`` envelope.
    judged_ndcg_status: str = "unavailable"
    fallback_rate_status: str = "unavailable"

    @property
    def positive_hit_at_5(self) -> float:
        """Runbook v2 G6.6 alias for :attr:`useful_at_5`.

        The runbook lists ``positive_hit@5`` as a separate metric on
        the promotion gate even though it is computed from the same
        underlying ``useful_at_k`` helper. Exposing it as a property
        alias lets the promotion gate enforce it as a distinct
        check without duplicating storage on the dataclass.

        Returns the same value as :attr:`useful_at_5`.
        """
        return self.useful_at_5

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the result into a JSON-friendly dict.

        ``Optional[float]`` fields are emitted as ``null`` when
        not available — callers must not coerce ``None`` to ``0.0``
        before returning to the dashboard.
        """
        return {
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr_at_10": self.mrr_at_10,
            "ndcg_at_10": self.ndcg_at_10,
            "useful_at_1": self.useful_at_1,
            "useful_at_5": self.useful_at_5,
            "explicit_negative_at_5": self.explicit_negative_at_5,
            "positive_hit_at_5": self.positive_hit_at_5,
            "no_result_rate": self.no_result_rate,
            "p50_latency": self.p50_latency,
            "p95_latency": self.p95_latency,
            "degraded_rate": self.degraded_rate,
            "fallback_rate": self.fallback_rate,
            "fallback_useful_rate": self.fallback_useful_rate,
            "num_cases": self.num_cases,
            "judged_ndcg_at_10": self.judged_ndcg_at_10,
            "useful_superseded_fallback_rate": self.useful_superseded_fallback_rate,
            "num_judged_cases": self.num_judged_cases,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "judged_ndcg_status": self.judged_ndcg_status,
            "fallback_rate_status": self.fallback_rate_status,
        }


def _coerce_rank_result(value: Any) -> Tuple[List[str], bool, bool, List[str]]:
    """Normalise the return value of a ``rank_fn`` callable.

    Legacy ``rank_fn`` callables return ``List[str]`` of ranked
    candidate keys; v0.3.0.x ``rank_fn`` callables may return a
    :class:`_RankFnOutcome` to expose per-case degraded /
    superseded-fallback metadata. This helper accepts either
    shape and returns a 4-tuple
    ``(ranked_keys, degraded, fallback_used, fallback_keys)``
    so the rest of :func:`evaluate` can stay agnostic.
    """
    if isinstance(value, _RankFnOutcome):
        return (
            list(value.ranked or []),
            bool(value.degraded),
            bool(value.fallback_used),
            list(value.fallback_keys or []),
        )
    if isinstance(value, dict):
        ranked = value.get("ranked") or value.get("keys") or []
        degraded = bool(value.get("degraded", False))
        fallback_used = bool(value.get("fallback_used", False))
        fallback_keys = list(value.get("fallback_keys") or [])
        return list(ranked), degraded, fallback_used, fallback_keys
    # List[str] — legacy contract.
    try:
        keys = list(value)  # type: ignore[arg-type]
    except TypeError:
        keys = []
    return keys, False, False, []


def evaluate(
    cases: List[_EvaluationCase],
    rank_fn,
    *,
    limit: int = 10,
) -> EvalResult:
    """Evaluate ``rank_fn`` on ``cases``.

    ``rank_fn`` is a callable ``rank_fn(query_text, query_id) -> X``
    that returns either:

    * ``List[str]`` of candidate_keys in descending order of
      relevance (legacy contract), or
    * a :class:`_RankFnOutcome` so the evaluator can attribute
      per-case metrics for the **degraded** path and the
      **superseded-fallback** path.

    An empty list / empty :attr:`_RankFnOutcome.ranked` means
    "no results" and contributes to :attr:`EvalResult.no_result_rate`.
    """
    n = len(cases)
    if n == 0:
        return EvalResult(num_cases=0)

    r1 = r5 = r10 = 0.0
    mrrs = 0.0
    ndcg = 0.0
    u1_cases = 0.0
    u5_cases = 0.0
    en5_cases = 0.0
    no_results = 0
    degraded_count = 0
    fallback_useful_count = 0
    latencies: List[float] = []
    # Counters for the "explicit-negative-at-5 over judged cases" denominator.
    judged_case_count = 0
    judged_case_with_negative = 0

    for case in cases:
        t0 = time.perf_counter()
        raw = rank_fn(case.query_text, case.query_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        ranked, degraded, fallback_used, fallback_keys = _coerce_rank_result(raw)
        if degraded:
            degraded_count += 1

        if not ranked:
            no_results += 1
            continue

        r1 += _recall_at_k(ranked, case.positives, 1)
        r5 += _recall_at_k(ranked, case.positives, 5)
        r10 += _recall_at_k(ranked, case.positives, 10)
        mrrs += _mrr(ranked, case.positives)
        ndcg += _ndcg_at_k(ranked, case.positives, 10, case.negatives)
        u1_cases += 1.0 if any(d in case.positives for d in ranked[:1]) else 0.0
        u5_cases += 1.0 if any(d in case.positives for d in ranked[:5]) else 0.0

        # explicit_negative_at_5 = fraction of *judged* cases where at
        # least one of the top-5 hits is in the case's negative set.
        # Unjudged cases (no positives and no negatives) are excluded
        # from both numerator and denominator so that absence-of-feedback
        # does not bias the metric.
        is_judged = bool(case.positives or case.negatives)
        if is_judged:
            judged_case_count += 1
            top5 = ranked[:5]
            if top5 and any(d in case.negatives for d in top5):
                judged_case_with_negative += 1
                en5_cases += 1.0
        # (the "fallback_useful" attribution only counts when both
        # fallback metadata AND judged positives are present, mirroring
        # the _useful_superseded_fallback_rate helper above.)

        if fallback_used and case.positives:
            # A fallback hit is "useful" if it matches a judged positive.
            # The fallback band itself might overlap with the active
            # ranking, so we consider the union of active + fallback keys
            # minus the already-returned active hits — i.e. only the
            # NEW keys that the fallback added.
            active_set = set(ranked)
            added = [k for k in fallback_keys if k and k not in active_set]
            if added and any(k in case.positives for k in added):
                fallback_useful_count += 1

    latencies.sort()
    p50 = latencies[(len(latencies) * 0 // 2)] if latencies else 0.0  # floor
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

    explicit_negative_rate = (
        judged_case_with_negative / judged_case_count
        if judged_case_count > 0
        else 0.0
    )

    return EvalResult(
        recall_at_1=r1 / n,
        recall_at_5=r5 / n,
        recall_at_10=r10 / n,
        mrr_at_10=mrrs / n,
        ndcg_at_10=ndcg / n,
        useful_at_1=u1_cases / n,
        useful_at_5=u5_cases / n,
        explicit_negative_at_5=explicit_negative_rate,
        no_result_rate=no_results / n,
        p50_latency=p50,
        p95_latency=p95,
        degraded_rate=degraded_count / n,
        fallback_useful_rate=fallback_useful_count / n,
        num_cases=n,
    )


def evaluate_candidate(
    baseline_result: EvalResult,
    candidate_result: EvalResult,
    *,
    strict_validation: bool = True,
) -> Tuple[bool, List[str]]:
    """Compare candidate vs baseline and return ``(passed, reasons)``.

    The candidate must **not degrade** any of the hard metrics, and
    must **improve** nDCG@10 by at least 1%. When ``strict_validation``
    is false, the nDCG improvement delta is ignored (used for shadow
    analysis where we compare means instead).
    """
    reasons: List[str] = []
    ok = True

    checks = [
        ("Recall@5 not lower", candidate_result.recall_at_5 >= baseline_result.recall_at_5 - 1e-6),
        ("MRR@10 not lower by >0.5%", candidate_result.mrr_at_10 >= baseline_result.mrr_at_10 - 0.005),
        ("nDCG@10 improved by ≥1%", candidate_result.ndcg_at_10 >= baseline_result.ndcg_at_10 * 1.01 if strict_validation else True),
        ("useful@1 not lower", candidate_result.useful_at_1 >= baseline_result.useful_at_1 - 1e-6),
        ("negative@5 not higher", candidate_result.explicit_negative_at_5 <= baseline_result.explicit_negative_at_5 + 1e-6),
        ("no-result rate not worse", candidate_result.no_result_rate <= baseline_result.no_result_rate + 0.01),
    ]
    for name, passed in checks:
        if not passed:
            reasons.append(name)

    ok = len(reasons) == 0
    return ok, reasons
