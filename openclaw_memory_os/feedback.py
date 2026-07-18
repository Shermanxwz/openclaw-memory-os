"""Recall feedback tracking.

Captures useful/not-useful signals from the dashboard on recall results.
v0.3.0: structured feedback is now stored in dedicated SQLite tables
(``feedback_events``, ``recall_runs``, ``recall_results``) rather than
as generic audit_log entries. The legacy ``audit_log`` write is preserved
for backward compatibility (read-only compat), but new structured data
flows through the new tables.

Key design decisions:
* ``query_text`` is stored in full (not hashed) so evaluation can
  reconstruct context.
* ``recall_runs`` older than 180 days are cleaned up automatically.
* Old ``audit_log`` entries remain read-compatible; new data goes
  to the new tables.
"""

from __future__ import annotations

import logging
from typing import Optional

from .audit import get_audit_store
from .models import FeedbackEntry

logger = logging.getLogger(__name__)


def record_feedback(
    memory_id: str,
    query: str,
    useful: bool,
    *,
    actor: Optional[str] = None,
    note: Optional[str] = None,
    query_id: str = "",
    candidate_key: str = "",
) -> int:
    """Record a useful/not-useful feedback entry.

    Writes to both the legacy audit_log (backward compat) and the
    new structured feedback_events table (if query_id/candidate_key
    are provided).

    Returns the audit log row ID.
    """
    store = get_audit_store()

    # Legacy audit_log write (always, for backward compat)
    detail = f"query={query[:200]!r} useful={useful}"
    if note:
        detail += f" note={note[:200]!r}"
    if query_id:
        detail += f" query_id={query_id}"
    if candidate_key:
        detail += f" candidate_key={candidate_key}"
    row_id = store.log(
        "feedback",
        actor=actor,
        memory_id=memory_id,
        detail=detail,
    )

    # Structured feedback_events write (v0.3.0 path)
    if query_id and candidate_key:
        store.record_feedback_event(
            query_id=query_id,
            candidate_key=candidate_key,
            useful=useful,
            query_text=query,
            note=note,
            actor=actor,
        )
    elif candidate_key:
        # candidate_key but no query_id — still write to feedback_events
        store.record_feedback_event(
            query_id=query_id,
            candidate_key=candidate_key,
            useful=useful,
            query_text=query,
            note=note,
            actor=actor,
        )

    logger.info(
        "Feedback recorded: memory=%s useful=%s row=%d query_id=%s candidate_key=%s",
        memory_id, useful, row_id, query_id, candidate_key,
    )
    return row_id


def encode_feedback_body(memory_id: str, query: str, useful: bool, note: Optional[str] = None) -> FeedbackEntry:
    """Build a :class:`FeedbackEntry` without immediately persisting it."""
    return FeedbackEntry(
        memory_id=memory_id,
        query=query,
        useful=useful,
        note=note,
    )
