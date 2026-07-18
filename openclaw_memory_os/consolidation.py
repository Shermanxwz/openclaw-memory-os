"""Duplicate memory consolidation.

Given a cluster of near-duplicate memories (identified by the analytics
module), this module merges them into a single representative entry,
preserving the best attributes from each member.

This module does NOT alter any backend storage. It returns a
:class:`ConsolidationResult` describing what a hypothetical merge would
look like. Actual storage mutation is delegated to the caller (ingestion
script, API endpoint, or CLI).
"""

from __future__ import annotations

import logging
from typing import Sequence

from .models import ConsolidationResult, Memory, MemoryTier

logger = logging.getLogger(__name__)


def consolidate_cluster(
    members: Sequence[Memory],
    *,
    strategy: str = "merge",
) -> ConsolidationResult:
    """Produce a consolidated memory from a group of near-duplicates.

    Args:
        members: The memory entries in one duplicate cluster.
        strategy: One of ``merge`` (default), ``keep_newest``, ``keep_best``.

    Returns:
        A :class:`ConsolidationResult` describing the merge.
    """
    if len(members) < 2:
        return ConsolidationResult(
            consolidated_id=members[0].id if members else "",
            text=members[0].text if members else "",
            merged_member_ids=[m.id for m in members],
            preserved_tags=list(members[0].tags) if members else [],
        )

    if strategy == "keep_newest":
        return _keep_newest(members)
    elif strategy == "keep_best":
        return _keep_best(members)
    else:
        return _merge(members)


def _merge(members: Sequence[Memory]) -> ConsolidationResult:
    """Merge all members into one: longest text, highest importance, union of tags."""
    # Sort by update time descending to pick the "best" base
    sorted_mems = sorted(
        members,
        key=lambda m: (m.updated_at or m.created_at, m.importance),
        reverse=True,
    )

    # Pick the most recent/important as the base; skip tier=core members
    core_members = [m for m in members if m.tier == MemoryTier.CORE]
    eligible = [m for m in sorted_mems if m.tier != MemoryTier.CORE]
    if not eligible:
        # All core: just return the newest
        base = sorted_mems[0]
        return ConsolidationResult(
            consolidated_id=base.id,
            text=base.text,
            merged_member_ids=[m.id for m in members],
            preserved_tags=list(base.tags),
        )

    base = eligible[0]

    # Pick the longest text (or base's if it's already longest)
    candidate_texts = [base.text] + [m.text for m in eligible]
    merged_text = max(candidate_texts, key=len)

    # Merge tags (union)
    merged_tags = list(set(t for m in members for t in (m.tags or [])))

    # Highest importance
    max(m.importance for m in members)

    member_ids = [m.id for m in members]
    survivor_ids = [m.id for m in core_members if m.id != base.id]

    return ConsolidationResult(
        consolidated_id=base.id,
        text=merged_text,
        merged_member_ids=member_ids,
        preserved_tags=merged_tags,
        survivors=survivor_ids,
    )


def _keep_newest(members: Sequence[Memory]) -> ConsolidationResult:
    newest = max(members, key=lambda m: m.updated_at or m.created_at)
    return ConsolidationResult(
        consolidated_id=newest.id,
        text=newest.text,
        merged_member_ids=[m.id for m in members],
        preserved_tags=list(newest.tags),
    )


def _keep_best(members: Sequence[Memory]) -> ConsolidationResult:
    best = max(members, key=lambda m: m.importance)
    return ConsolidationResult(
        consolidated_id=best.id,
        text=best.text,
        merged_member_ids=[m.id for m in members],
        preserved_tags=list(best.tags),
    )
