"""Ranking logic for the recall-test endpoint.

This module is intentionally pure and synchronous: given a query and a list
of :class:`Memory` objects, return ranked :class:`RecallHit` objects with
explanations. Keeping the function pure makes it trivial to unit-test and
lets us reuse it for both the API and the CLI ``recall-test`` command.

Design notes (full version in ``docs/recall-ranking.md``):

* We score each candidate with:
    - a base score derived from ``status`` (active=1.0, superseded/expired
      configurable penalty, needs_review=0.4),
    - a recency boost using exponential decay with a configurable
      half-life (``recency_half_life_days``),
    - an importance boost scaled by ``importance_boost_scale``,
    - a simple keyword overlap score in ``keyword``/``hybrid`` modes.

* In ``hybrid`` mode (default) we sum the keyword score and the base
  score. A future iteration will swap the base score for a dense
  embedding similarity and add a cross-encoder rerank step. The shape
  of :class:`RecallHit` is designed to surface the breakdown so that
  future scoring models can be debugged end-to-end.
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from .config import Settings, get_settings
from .models import Memory, MemoryStatus, RecallHit, RecallRequest


# ---------------------------------------------------------------------------
# Feedback-weight helpers
# ---------------------------------------------------------------------------

_FEEDBACK_WEIGHTS_PATH: Optional[str] = None
"""Override path for loading feedback weights (used by tests)."""


def set_feedback_weights_path(path: Optional[str]) -> None:
    """Set a non-default path for loading feedback weights."""
    global _FEEDBACK_WEIGHTS_PATH
    _FEEDBACK_WEIGHTS_PATH = path


def _load_feedback_weights() -> Optional[dict]:
    """Read the feedback-weights.json snapshot, or return None."""
    import json
    import os
    from pathlib import Path

    path: Optional[str] = _FEEDBACK_WEIGHTS_PATH
    if path is None:
        state_home = os.environ.get(
            "XDG_STATE_HOME",
            os.path.expanduser("~/.local/state"),
        )
        p = Path(state_home) / "openclaw-memory-os" / "feedback-weights.json"
        path = str(p)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _feedback_weight_scale(
    feedback_weights: Optional[dict],
) -> float:
    """Derive an additive scale factor from the feedback weight snapshot.

    Returns a multiplier in [0.8, 1.2] that is applied to the
    ``importance_boost`` coefficient in ``rank_memories``. The logic:

    * If ``feedback_weights`` is None, return 1.0 (no change).
    * If the 7-day useful ratio is available and >= 0.7, boost to 1.2
      (the user finds results useful, so amplify importance).
    * If the 7-day useful ratio is available and < 0.3, reduce to 0.8
      (the user does not find results useful, so de-emphasise importance).
    * If no 7-day ratio, fall back to 30-day, then 24h, then 1.0.
    """
    if feedback_weights is None:
        return 1.0
    # Try 7d first, then 30d, then 24h
    for key in ("ratio_7d", "ratio_30d", "ratio_24h"):
        val = feedback_weights.get(key)
        if val is not None:
            if val >= 0.7:
                return 1.2
            elif val < 0.3:
                return 0.8
            else:
                return 1.0
    return 1.0


_TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _recency_boost(updated_at: Optional[datetime], now: datetime, half_life_days: float) -> float:
    if updated_at is None and now is None:
        return 0.0
    ts = updated_at
    if ts is None:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta_days = max((now - ts).total_seconds() / 86400.0, 0.0)
    if half_life_days <= 0:
        return 0.0
    # exp(- ln2 * days / half_life) is in (0, 1] and decreases with age.
    return math.exp(-math.log(2.0) * delta_days / half_life_days)


def _keyword_score(query_tokens: Sequence[str], memory: Memory) -> Tuple[float, Set[str]]:
    if not query_tokens:
        return 0.0, set()
    haystack_tokens: Set[str] = set()
    for field in (memory.text, memory.summary or "", " ".join(memory.tags or [])):
        haystack_tokens.update(_tokenize(field))
    if not haystack_tokens:
        return 0.0, set()
    matched = {t for t in query_tokens if t in haystack_tokens}
    if not matched:
        return 0.0, set()
    coverage = len(matched) / len(set(query_tokens))
    density = len(matched) / max(len(haystack_tokens), 1)
    return coverage * 0.7 + density * 0.3, matched


def _base_score(memory: Memory, settings: Settings) -> Tuple[float, str]:
    status = memory.status
    if status == MemoryStatus.ACTIVE:
        return 1.0, "status:active"
    if status == MemoryStatus.SUPERSEDED:
        return settings.superseded_penalty, f"status:superseded*{settings.superseded_penalty:.2f}"
    if status == MemoryStatus.EXPIRED:
        return settings.expired_penalty, f"status:expired*{settings.expired_penalty:.2f}"
    if status == MemoryStatus.NEEDS_REVIEW:
        return 0.4, "status:needs_review*0.40"
    return 0.5, "status:unknown*0.50"


def _within_window(memory: Memory, since_days: Optional[int], now: datetime) -> bool:
    if since_days is None:
        return True
    ts = memory.updated_at or memory.created_at
    if ts is None:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age_days = (now - ts).total_seconds() / 86400.0
    return age_days <= since_days


def _passes_filters(
    memory: Memory,
    req: RecallRequest,
    settings: Settings,
    now: datetime,
) -> Tuple[bool, Optional[str]]:
    if memory.status == MemoryStatus.SUPERSEDED and not req.include_superseded:
        return False, "filtered:superseded"
    if memory.status == MemoryStatus.EXPIRED and not req.include_expired:
        return False, "filtered:expired"
    if not _within_window(memory, req.since_days, now):
        return False, f"filtered:since_days>{req.since_days}"
    if req.tier_filter and memory.tier not in req.tier_filter:
        return False, f"filtered:tier={memory.tier.value}"
    return True, None


def rank_memories(
    memories: Iterable[Memory],
    request: RecallRequest,
    *,
    settings: Optional[Settings] = None,
    now: Optional[datetime] = None,
    feedback_weights: Optional[dict] = None,
) -> Tuple[List[RecallHit], int]:
    """Rank ``memories`` against ``request`` and return hits + considered count.

    ``feedback_weights`` is an optional dict from the weight snapshot
    (see ``scripts/replay_feedback.py``). When present, its useful ratio
    is used to additively scale the ``importance_boost`` coefficient:
    a high useful ratio (>0.7) amplifies importance by 1.2x, a low ratio
    (<0.3) reduces it to 0.8x. The scale is a simple multiplier on
    ``settings.importance_boost_scale`` and lives in [0.8, 1.2].

    The returned list is sorted by descending composite score and capped at
    ``request.limit``. The second tuple element is the number of memories
    inspected *after* filters, useful for the response envelope.
    """

    settings = settings or get_settings()
    now = now or datetime.now(timezone.utc)
    query_tokens = _tokenize(request.query)

    # ---- Feedback learning: additive scale on importance boost ----------
    _fb = _feedback_weight_scale(feedback_weights)

    hits: List[RecallHit] = []
    considered = 0

    for mem in memories:
        ok, reason = _passes_filters(mem, request, settings, now)
        if not ok:
            continue
        considered += 1

        base, base_label = _base_score(mem, settings)
        recency = _recency_boost(mem.updated_at or mem.created_at, now, settings.recency_half_life_days)
        importance_boost = settings.importance_boost_scale * float(mem.importance or 0.0) * _fb
        keyword, matched = _keyword_score(query_tokens, mem)

        # Mode-dependent composition
        mode = (request.mode or "hybrid").lower()
        if mode == "keyword":
            composite = keyword
            keyword_component = keyword
        elif mode == "dense":
            # No embeddings in the sample backend; we approximate dense
            # score with importance + recency so the mode is testable.
            composite = 0.5 * base + recency + importance_boost
            keyword_component = 0.0
        else:  # hybrid (default)
            composite = base * (0.4 + 0.6 * recency) + importance_boost + 0.5 * keyword
            keyword_component = keyword

        components = {
            "base": round(base, 4),
            "recency": round(recency, 4),
            "importance": round(importance_boost, 4),
            "keyword": round(keyword_component, 4),
            "composite": round(composite, 4),
        }

        why_bits = [base_label]
        if recency > 0.5:
            why_bits.append(f"recent({recency:.2f})")
        elif recency > 0.1:
            why_bits.append(f"aging({recency:.2f})")
        else:
            why_bits.append(f"old({recency:.2f})")
        if importance_boost > 0.05:
            why_bits.append(f"importance={mem.importance:.2f}")
        if matched:
            preview = ",".join(sorted(matched)[:5])
            why_bits.append(f"matched[{preview}]")
        else:
            why_bits.append("no-keyword-match")
        if mem.status != MemoryStatus.ACTIVE:
            why_bits.append(f"status={mem.status.value}")
        explanation = "; ".join(why_bits)

        hits.append(
            RecallHit(
                id=mem.id,
                text=mem.text,
                summary=mem.summary,
                tier=mem.tier,
                status=mem.status,
                importance=mem.importance,
                score=round(composite, 4),
                components=components,
                explanation=explanation,
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[: max(1, min(request.limit, settings.max_recall_results))], considered


def build_recall_response(
    memories: Sequence[Memory],
    request: RecallRequest,
    *,
    backend_name: str,
    settings: Optional[Settings] = None,
    now: Optional[datetime] = None,
    dense_candidates: Optional[Sequence[Memory]] = None,
    feedback_weights: Optional[dict] = None,
):
    """Convenience wrapper that returns a :class:`RecallResponse` with timing.

    Behavior:

    * Ranks memories with :func:`rank_memories` using the request's filters.
    * When ``request.mode == "dense"`` and ``dense_candidates`` is
      supplied, that candidate list is used instead of ``memories``.
      The caller is responsible for sourcing those candidates via
      ``backend.search(query, limit)`` so the recall-test endpoint
      actually exercises vector search rather than re-scoring the
      full corpus against an importance-only heuristic. With
      ``dense_candidates=None`` (the default, used by the pure-Python
      tests) the dense branch falls back to ``memories`` so the
      scoring formula stays exercisable in isolation.
    * If the request did **not** opt in to superseded memories
      (``include_superseded=False``) and the active-only pass returns
      fewer than ``settings.recall_fallback_superseded_min_results`` hits,
      automatically re-ranks with superseded memories included and uses
      the union of both passes — active hits first (preserving their
      rank order), then any superseded hits that weren't already
      surfaced. The fallback only runs when
      ``settings.recall_fallback_superseded`` is true; setting it off
      (or the request flag ``include_superseded=True``) disables it.
    * The response ``total_considered`` reflects the active-pass count so
      dashboards don't suddenly report a different corpus size when the
      fallback engages.
    """
    from .models import RecallResponse  # local import to avoid cycles

    settings = settings or get_settings()
    now = now or datetime.now(timezone.utc)
    started = time.perf_counter()
    # In dense mode, the caller (CLI / API) is expected to have already
    # narrowed the candidate set via ``backend.search(query, limit)``.
    # When ``dense_candidates`` is supplied, use it instead of the full
    # ``memories`` corpus so the dense branch ranks true vector hits
    # rather than re-scoring the whole store with an importance proxy.
    is_dense = (request.mode or "").lower() == "dense"
    if is_dense and dense_candidates is not None:
        corpus = list(dense_candidates)
    else:
        corpus = list(memories)
    hits, considered = rank_memories(corpus, request, settings=settings, now=now, feedback_weights=feedback_weights)

    fallback_used = False
    fallback_added = 0
    if (
        not request.include_superseded
        and settings.recall_fallback_superseded
        and len(hits) < max(1, settings.recall_fallback_superseded_min_results)
        and any(m.status == MemoryStatus.SUPERSEDED for m in memories)
    ):
        expanded_req = request.model_copy(update={"include_superseded": True})
        expanded_hits, _ = rank_memories(
            memories, expanded_req, settings=settings, now=now, feedback_weights=feedback_weights,
        )
        active_ids = {h.id for h in hits}
        # Re-score superseded hits with a stricter "below-active" floor so
        # the fallback never promotes a superseded memory above an active
        # one. The active-pass score distribution sets the floor; if the
        # active hits already span [min_active, max_active], we use the
        # lower end of that band (or a small positive constant if no
        # active hit matched) as the floor. Superseded hits stay sorted
        # by their composite score, but capped at ``floor - epsilon`` so
        # they're visibly/score-wise lower than any active hit.
        active_scores = [h.score for h in hits] if hits else [0.0]
        min_active = min(active_scores)
        floor = min_active - 1e-3 if hits else 0.0
        extra: List = []
        for h in expanded_hits:
            if h.id in active_ids:
                continue
            # Force the score below the floor. We rebuild a copy of the
            # hit with the adjusted score so we don't mutate pydantic
            # fields and the rescaled score stays scoped to the
            # fallback path.
            capped_score = min(h.score, floor)
            new_components = dict(h.components or {})
            if new_components:
                new_components["fallback_floor"] = round(floor, 4)
            extra.append(
                h.model_copy(
                    update={
                        "score": capped_score,
                        "components": new_components,
                    }
                )
            )
        # Stable order: highest capped score first (so more-relevant
        # superseded entries still come first within the fallback band),
        # then by id for determinism.
        extra.sort(key=lambda h: (-h.score, h.id))
        if extra:
            fallback_used = True
            fallback_added = len(extra)
            cap = max(1, min(request.limit, settings.max_recall_results))
            hits = (hits + extra)[:cap]

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    # v0.3.0: Generate query_id, attach policy_version, compute diagnostics.
    import uuid as _uuid
    from .policy_store import PolicyStore as _PolicyStore
    _qid = str(_uuid.uuid4())
    try:
        _pv = "v" + str(_PolicyStore().get().version)
    except Exception:
        _pv = "v1"
    _diag = {
        "active_hits": len(hits),
        "fallback_used": fallback_used,
        "fallback_added": fallback_added,
        "considered": considered,
    }
    return RecallResponse(
        query=request.query,
        mode=request.mode,
        took_ms=round(elapsed_ms, 3),
        backend=backend_name,
        total_considered=considered,
        hits=hits,
        query_id=_qid,
        policy_version=_pv,
        diagnostics=_diag,
        fallback={
            "enabled": settings.recall_fallback_superseded,
            "min_results": settings.recall_fallback_superseded_min_results,
            "used": fallback_used,
            "added": fallback_added,
        },
    )
