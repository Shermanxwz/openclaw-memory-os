"""Reusable raw retrieval pools for offline policy evaluation.

The expensive retrieval channels are executed once per query.  Candidate
policies then perform deterministic in-memory truncation, RRF and feature
reranking over exactly the same channel results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, TYPE_CHECKING

from .contracts import ScoredMemoryCandidate

if TYPE_CHECKING:  # pragma: no cover
    from .policy_store import Policy
    from .retrieval_engine import RetrievalEngine, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class QueryCandidatePool:
    """Raw channel results for one query.

    Active and Superseded records stay in separate bands.  Pool construction
    never deletes or mutates memories; Superseded remains available only for
    the second-stage fallback enforced by ``rank_candidate_pool``.
    """

    query: str
    dense_active: List[ScoredMemoryCandidate] = field(default_factory=list)
    lexical_active: List[ScoredMemoryCandidate] = field(default_factory=list)
    dense_superseded: List[ScoredMemoryCandidate] = field(default_factory=list)
    lexical_superseded: List[ScoredMemoryCandidate] = field(default_factory=list)
    corpus_snapshot_id: Optional[str] = None
    dense_available: bool = True
    lexical_available: bool = True
    degraded_reason: Optional[str] = None

    def unique_candidates(self) -> List[ScoredMemoryCandidate]:
        seen: set[str] = set()
        out: List[ScoredMemoryCandidate] = []
        for channel in (
            self.dense_active,
            self.lexical_active,
            self.dense_superseded,
            self.lexical_superseded,
        ):
            for candidate in channel:
                if candidate.candidate_key in seen:
                    continue
                seen.add(candidate.candidate_key)
                out.append(candidate)
        return out

    def __len__(self) -> int:
        return len(self.unique_candidates())

    def __iter__(self) -> Iterator[ScoredMemoryCandidate]:
        return iter(self.unique_candidates())


class CandidatePool:
    """Build and cache one :class:`QueryCandidatePool` per query."""

    def __init__(
        self,
        engine: "RetrievalEngine",
        queries: Iterable[str],
        limit: int = 200,
    ) -> None:
        self.engine = engine
        self.limit = max(1, int(limit))
        self._pool: Dict[str, QueryCandidatePool] = {}
        for query in queries:
            q = str(query)
            try:
                getter = getattr(engine, "get_candidate_pool", None)
                if callable(getter):
                    self._pool[q] = getter(q, channel_limit=self.limit)
                else:
                    self._pool[q] = engine.build_candidate_pool(
                        q, channel_limit=self.limit
                    )
            except Exception as exc:
                logger.warning("CandidatePool: failed to build pool for %r: %s", q, exc)
                self._pool[q] = QueryCandidatePool(
                    query=q,
                    dense_available=False,
                    lexical_available=False,
                    degraded_reason="candidate_pool_unavailable",
                )

    def get(self, query: str) -> QueryCandidatePool:
        return self._pool.get(str(query), QueryCandidatePool(query=str(query)))

    def get_hits(self, query: str) -> List[ScoredMemoryCandidate]:
        """Backward-compatible flat view, without a policy ranking."""
        return self.get(query).unique_candidates()

    def rank(
        self,
        query: str,
        policy: "Policy",
        *,
        limit: int = 50,
    ) -> "RetrievalResult":
        return self.engine.rank_candidate_pool(self.get(query), policy, limit=limit)


__all__ = ["CandidatePool", "QueryCandidatePool"]
