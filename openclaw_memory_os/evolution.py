"""v0.3.0 policy evolution: candidate search + shadow + auto-promote + rollback.

Implements S9 (deterministic policy search) and S10 (shadow comparison,
auto-promotion, auto-rollback, circuit breaker) from the evolution
contract.

Key design
----------

* **Deterministic search.** Coordinate-style perturbation of each
  weight ⟨dense_k, lexical_k, rrf_k, fallback_min_results, ...⟩. No
  LLM call.
* **Cold-start gates:**
  - <30 judged queries: no candidate generated; ``last_result =
    skipped``.
  - 30-99 queries: max perturbation ±0.03; all major metrics must
    not degrade.
  - 100+ queries: full safety ranges.
* **Shadow.** Candidate runs in the background; its results are
  compared to the running active policy after each batch of 5 new
  feedback events. At 30+ comparisons we decide whether to promote.
* **Auto-promotion (``guarded_auto``).** Two consecutive evaluation
  windows must pass. Cooldown 7 days; max 2 promotions per 30d.
* **Auto-rollback.** Immediate (file corrupt, error rate >5%,
  no-result +15pp, latency 2x) or statistical (useful_at_1 drop
  >8%, MRR drop >5%, negative_at_5 increase >8%).

All numbers come from :mod:`openclaw_memory_os.evaluation` and go
through :mod:`openclaw_memory_os.policy_store`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .evaluation import (
    EvalResult,
    _EvaluationCase,
    _load_cases,
    evaluate,
    evaluate_candidate,
)
from .policy_store import Policy, PolicyStatus, PolicyStore, compute_checksum

if TYPE_CHECKING:  # pragma: no cover — import only for type hints
    from .retrieval_engine import RetrievalEngine

logger = logging.getLogger(__name__)

# Cold-start thresholds
_COLD_START_MIN_QUERIES = 30
_QUERIES_FOR_FULL_RANGE = 100  # >30 <100: only 0.03 delta; >=100: full range
_SHADOW_PROMOTION_MIN_QUERIES = 30
_PROMOTION_WINDOWS_REQUIRED = 2
_EVOLUTION_STATE_DIR = Path(
    os.environ.get(
        "MEMORY_OS_RECALL_STATE_DIR",
        os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state")),
    )
) / "openclaw-memory-os"
_EVOLUTION_LOCK_PATH = os.environ.get(
    "MEMORY_OS_EVOLUTION_LOCK",
    str(_EVOLUTION_STATE_DIR / "openclaw-memory-os.evolution.lock"),
)

_MAX_PROMOTIONS_PER_30D = 2
_PROMOTION_COOLDOWN_DAYS = 7
_MAX_CONSECUTIVE_ROLLBACKS = 2

# Train / val / test split ratios for evaluation cases.
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Cooldown period after hitting max consecutive rollbacks.
EVOLUTION_COOLDOWN_HOURS = 24

# Funnel sizes: top 5 candidates pass from pool → val, top 2 from val
# → held-out, and exactly 1 is selected for promotion.
_FUNNEL_POOL_SIZE = 5
_FUNNEL_VAL_SIZE = 2

# Minimum wall-time gap between two consecutive pass windows. The
# runbook defines a "governance window" as a distinct evaluation
# event (a fresh run_evolution_cycle invocation), so the window here
# is implemented as a timestamp delta. Two passes recorded within
# the same cycle instant do NOT count as two windows; only passes
# separated by at least this many minutes do.
_PASS_WINDOW_MIN_GAP_SECONDS = 600  # 10 minutes

# Rollback thresholds (G6.7). These are the canonical numbers from
# the runbook; tune via the helpers in this section rather than
# scattering magic numbers across the module.
_ROLLBACK_ERR_RATE_THRESHOLD = 0.05            # degraded rate > 5%
_ROLLBACK_NO_RESULT_DELTA_THRESHOLD = 0.15     # no_result +15pp
_ROLLBACK_USEFUL_AT_1_DROP_THRESHOLD = 0.08    # useful@1 -8pp
_ROLLBACK_MRR_DROP_THRESHOLD = 0.05            # MRR -5%
_ROLLBACK_NEGATIVE_AT_5_DELTA_THRESHOLD = 0.08  # negative@5 +8pp
_ROLLBACK_P95_LATENCY_X_THRESHOLD = 2.0        # p95 > previous × 2
_ROLLBACK_FALLBACK_USEFUL_DROP_THRESHOLD = 0.05  # fallback_useful -5pp
_ROLLBACK_NO_RESULT_RATE_THRESHOLD = 0.10      # no_result > 10% (immediate)
_ROLLBACK_P95_LATENCY_ABS_THRESHOLD = 5.0      # p95 > 5s (immediate)

# Sensible default baseline used to seed ``previous_metrics`` on the
# first-ever cycle. Without this seed, the very first cycle has no
# reference point and ANY small metric delta (e.g. 1pp drop on
# useful@1) would falsely trigger a rollback. The baseline mirrors
# the runbook's "shipped defaults" intent: an equal-weighted, mid-
# performing policy. Numbers are deliberately conservative — equal
# to a 50/50 split on every useful metric.
_DEFAULT_PREVIOUS_METRICS: Dict[str, float] = {
    "useful_at_1": 0.50,
    "mrr_at_10": 0.50,
    "explicit_negative_at_5": 0.05,
    "no_result_rate": 0.05,
    "p95_latency": 1.0,
    "degraded_rate": 0.0,
    "fallback_useful_rate": 0.50,
}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _state_path() -> Path:
    _EVOLUTION_STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _EVOLUTION_STATE_DIR / "evolution-state.json"


def _load_evolution_state() -> Dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {
            "promotion_count_30d": 0,
            "consecutive_rollbacks": 0,
            "last_promotion_at": None,
            "shadow_comparisons": [],
            # G6.5: two-window consecutive pass tracking. ``pass_windows``
            # is a ring buffer of the most recent 2 windows; each entry
            # is ``{"cycle": <iso ts>, "result": "passed" | "failed"}``.
            # ``consecutive_passes`` is the live counter — incremented
            # on a passed cycle, reset to 0 on a failed cycle.
            "pass_windows": [],
            "consecutive_passes": 0,
            "pass_candidate_version": None,
            # G6.7: reference metrics used by the rollback triggers. On
            # the first-ever cycle there is no previous to compare
            # against, so we seed with a neutral baseline (defined by
            # ``_DEFAULT_PREVIOUS_METRICS``) — otherwise the very first
            # cycle would false-trigger a rollback on any 1pp delta.
            "previous_metrics": dict(_DEFAULT_PREVIOUS_METRICS),
        }
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {
            "promotion_count_30d": 0,
            "consecutive_rollbacks": 0,
            "last_promotion_at": None,
            "shadow_comparisons": [],
            "pass_windows": [],
            "consecutive_passes": 0,
            "pass_candidate_version": None,
            "previous_metrics": dict(_DEFAULT_PREVIOUS_METRICS),
        }
    # Backfill any keys added by newer versions so legacy state files
    # don't blow up when the new fields are read. ``previous_metrics``
    # is seeded with the neutral baseline — operators upgrading from
    # a pre-G6.7 build get one safe first cycle, and from then on
    # every subsequent ``_check_rollback`` reads real previous
    # metrics from ``post_rollback_metrics`` / cycle eval.
    state.setdefault("pass_windows", [])
    state.setdefault("consecutive_passes", 0)
    state.setdefault("pass_candidate_version", None)
    state.setdefault("previous_metrics", dict(_DEFAULT_PREVIOUS_METRICS))
    return state


def _save_evolution_state(state: Dict[str, Any]) -> None:
    p = _state_path()
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(p, 0o600)


def _record_cycle_result(
    state: Dict[str, Any],
    result: str,
    *,
    candidate_version: Optional[int] = None,
) -> None:
    """Record one governance window, scoped to a candidate when known."""
    if result not in ("passed", "failed"):
        raise ValueError(f"unknown cycle result: {result!r}")
    windows: List[Dict[str, Any]] = list(state.get("pass_windows", []))
    if result == "passed" and candidate_version is not None:
        version = int(candidate_version)
        if state.get("pass_candidate_version") != version:
            windows = []
            state["consecutive_passes"] = 0
        state["pass_candidate_version"] = version
        windows.append({
            "cycle": _now().isoformat(),
            "result": "passed",
            "candidate_version": version,
        })
        state["consecutive_passes"] = int(
            state.get("consecutive_passes", 0)
        ) + 1
    elif result == "passed":
        # Compatibility for older tests/callers. Production promotion paths pass
        # an explicit version; a later explicit candidate resets this legacy bank.
        windows.append({"cycle": _now().isoformat(), "result": "passed"})
        state["consecutive_passes"] = int(
            state.get("consecutive_passes", 0)
        ) + 1
    else:
        windows.append({
            "cycle": _now().isoformat(),
            "result": "failed",
            "candidate_version": (
                int(candidate_version) if candidate_version is not None else None
            ),
        })
        state["consecutive_passes"] = 0
        state["pass_candidate_version"] = None
    state["pass_windows"] = windows[-2:]


def _two_consecutive_pass_windows(
    state: Dict[str, Any], candidate_version: Optional[int] = None
) -> bool:
    windows = list(state.get("pass_windows", []) or [])
    if len(windows) < 2:
        return False
    first, second = windows[-2], windows[-1]
    if first.get("result") != "passed" or second.get("result") != "passed":
        return False
    if candidate_version is not None:
        version = int(candidate_version)
        if not (
            first.get("candidate_version") == version
            and second.get("candidate_version") == version
            and state.get("pass_candidate_version") == version
        ):
            return False
    try:
        first_dt = datetime.fromisoformat(str(first["cycle"]))
        second_dt = datetime.fromisoformat(str(second["cycle"]))
    except (KeyError, TypeError, ValueError):
        return False
    return abs((second_dt - first_dt).total_seconds()) >= _PASS_WINDOW_MIN_GAP_SECONDS


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Per-query shadow comparison helpers (G6.4)
# ---------------------------------------------------------------------------


# Cap on the rolling ``state["shadow_comparisons"]`` buffer. The
# buffer accumulates one entry per query per cycle; capping at
# 1000 keeps the state file bounded while still leaving a useful
# tail for the dashboard / operator inspection. When the cap is
# hit the OLDEST entries are dropped so the buffer always reflects
# the most recent activity.
_SHADOW_COMPARE_BUFFER_CAP = 1000


def _topk_agreement(a: List[str], b: List[str]) -> float:
    """Jaccard overlap between two top-K candidate-key lists.

    Returns ``len(a ∩ b) / max(len(a ∪ b), 1)``. Both inputs empty
    returns ``1.0`` (vacuous agreement — two empty rankings agree
    on "nothing"). Used by the G6.4 shadow comparison to quantify
    how much the active and candidate top-K overlap.
    """
    if not a and not b:
        return 1.0
    a_set = set(a)
    b_set = set(b)
    union = a_set | b_set
    return len(a_set & b_set) / max(len(union), 1)


def _avg_rank_diff(a: List[str], b: List[str]) -> float:
    """Mean absolute rank difference over the candidates common to both.

    For each candidate key present in BOTH ``a`` and ``b``, compute
    ``abs(rank_in_a - rank_in_b)`` and return the mean. Returns
    ``float("inf")`` when the two lists share no candidate (the
    worst-case agreement = 0). Used by the G6.4 shadow comparison
    to quantify how far the candidate re-ranks move each result
    relative to the active policy.
    """
    a_ranks = {key: i for i, key in enumerate(a)}
    b_ranks = {key: i for i, key in enumerate(b)}
    common = set(a_ranks) & set(b_ranks)
    if not common:
        return float("inf")
    return sum(abs(a_ranks[k] - b_ranks[k]) for k in common) / len(common)


def _hits_useful_at_k(
    ranked_keys: List[str],
    positives: set,
    *,
    k: int,
) -> bool:
    """Whether any of ``ranked_keys[:k]`` is in the case's ``positives`` set.

    Used by G6.4 to compute useful_at_1 / useful_at_5 from the
    raw top-K lists rather than going back through the full eval
    pipeline. ``positives`` is a ``set`` of candidate_keys the
    case considered relevant; the helper short-circuits to ``True``
    on the first hit so the comparison row stays cheap even when
    K grows.
    """
    if not positives:
        return False
    topk = ranked_keys[:k]
    for key in topk:
        if key in positives:
            return True
    return False


def _record_shadow_comparisons(
    state: Dict[str, Any],
    cases: List[_EvaluationCase],
    active_rank_fn: Callable,
    candidate_rank_fn: Callable,
) -> None:
    """Append one G6.4 per-query shadow comparison row per case.

    For each case in ``cases`` the helper runs ``active_rank_fn``
    AND ``candidate_rank_fn`` and records a dict with the
    active/candidate top-5 keys, agreement (Jaccard on top-5),
    useful_at_1 / useful_at_5 (judged-vs-top-K), and average rank
    distance over the union of active / candidate hits.

    The buffer is stored under ``state["shadow_comparisons"]`` and
    capped at :data:`_SHADOW_COMPARE_BUFFER_CAP` (oldest dropped)
    so the on-disk state file stays bounded. The function mutates
    ``state`` in place; callers are responsible for the
    ``_save_evolution_state(state)`` call (kept separate so a
    single save can bundle several related updates).

    When ``rank_fn`` raises (e.g. stub returns ``None``), the
    comparison row for that case is skipped rather than
    poisoning the whole buffer — a single broken query should
    not blank the rolling window.
    """
    existing: List[Dict[str, Any]] = list(state.get("shadow_comparisons", []))
    now_iso = _now().isoformat()
    new_rows: List[Dict[str, Any]] = []
    for case in cases:
        try:
            active_results = active_rank_fn(case.query_text, case.query_id) or []
        except Exception:
            continue
        try:
            candidate_results = candidate_rank_fn(case.query_text, case.query_id) or []
        except Exception:
            continue
        # Coerce to ``List[str]`` defensively — rank_fn implementations
        # that return a structured dataclass (e.g. ``_RankFnOutcome``)
        # would otherwise fail the agreement/avg-diff math.
        if not isinstance(active_results, list):
            active_results = list(active_results) if active_results else []
        if not isinstance(candidate_results, list):
            candidate_results = list(candidate_results) if candidate_results else []
        # Use top-5 / top-1 for the comparison columns; the runbook
        # says "useful_at_1, useful_at_5" so the buffer rows match.
        active_top5 = list(active_results[:5])
        candidate_top5 = list(candidate_results[:5])
        # Pad to 5 with empty strings so downstream JSON shape is
        # stable even when a policy returns fewer than 5 hits.
        while len(active_top5) < 5:
            active_top5.append("")
        while len(candidate_top5) < 5:
            candidate_top5.append("")
        new_rows.append({
            "query_id": case.query_id,
            "query_text": case.query_text,
            "active_top5": active_top5,
            "candidate_top5": candidate_top5,
            "agreement": _topk_agreement(active_results[:5], candidate_results[:5]),
            "active_useful_at_5": _hits_useful_at_k(active_results, case.positives, k=5),
            "candidate_useful_at_5": _hits_useful_at_k(candidate_results, case.positives, k=5),
            "active_useful_at_1": _hits_useful_at_k(active_results, case.positives, k=1),
            "candidate_useful_at_1": _hits_useful_at_k(candidate_results, case.positives, k=1),
            "rank_distance": _avg_rank_diff(
                list(active_results), list(candidate_results)
            ),
            "recorded_at": now_iso,
        })
    combined = existing + new_rows
    if len(combined) > _SHADOW_COMPARE_BUFFER_CAP:
        combined = combined[-_SHADOW_COMPARE_BUFFER_CAP:]
    state["shadow_comparisons"] = combined


def get_shadow_comparisons() -> List[Dict[str, Any]]:
    """Return the current rolling G6.4 shadow comparison buffer.

    Read-only accessor used by the dashboard / API endpoints to
    surface the per-query comparison rows that the evolution cycle
    populated. Returns ``[]`` when the state file does not exist
    yet (cold-start) or the buffer has been reset by a successful
    promotion.
    """
    state = _load_evolution_state()
    return list(state.get("shadow_comparisons", []))


# ---------------------------------------------------------------------------
# Cold-start gate
# ---------------------------------------------------------------------------


def _judged_query_count() -> int:
    cases = _load_cases(limit=5000)
    return len(cases)


def split_cases(
    cases: List[_EvaluationCase],
    *,
    seed: int = 42,
) -> Tuple[List[_EvaluationCase], List[_EvaluationCase], List[_EvaluationCase]]:
    """Deterministic train / val / test split based on case ID hash.

    Uses a hash of each case's ``query_id`` with a fixed seed to
    assign it to one of three buckets (60 / 20 / 20). The split
    is deterministic: the same cases always produce the same
    partition.

    Returns ``(train, val, test)``.
    """
    train: List[_EvaluationCase] = []
    val: List[_EvaluationCase] = []
    test: List[_EvaluationCase] = []
    random.Random(seed)
    for case in cases:
        # Deterministic assignment based on case ID hash + seed
        h = int(hashlib.md5(f"{case.query_id}:{seed}".encode()).hexdigest(), 16)
        bucket = h % 100
        if bucket < int(TRAIN_RATIO * 100):
            train.append(case)
        elif bucket < int((TRAIN_RATIO + VAL_RATIO) * 100):
            val.append(case)
        else:
            test.append(case)
    return train, val, test


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


class _Param:
    """One parameter that can be perturbed."""

    def __init__(self, name: str, current: float, lo: float, hi: float, *, int_val: bool = False):
        self.name = name
        self.current = current
        self.lo = lo
        self.hi = hi
        self.int_val = int_val

    def perturb(self, delta: float) -> float:
        v = self.current + delta
        if self.int_val:
            v = round(v)
        return max(self.lo, min(self.hi, v))


def generate_candidates(
    baseline: Policy,
    *,
    n_candidates: int = 20,
    max_delta: float = 0.05,
    seed: int = 42,
) -> List[Policy]:
    """Generate ``n_candidates`` variant policies from ``baseline``.

    Uses coordinate-style perturbations plus a handful of bounded
    random combinations. ``max_delta`` is reduced to 0.03 when
    cold-start (30-99 judged queries).
    """
    params = [
        _Param("rrf_dense_weight", baseline.rrf_dense_weight, 0.1, 10.0),
        _Param("rrf_lexical_weight", baseline.rrf_lexical_weight, 0.1, 10.0),
        _Param("final_rrf_weight", baseline.final_rrf_weight, 0.0, 1.0),
        _Param("final_vector_weight", baseline.final_vector_weight, 0.0, 1.0),
        _Param("final_lexical_weight", baseline.final_lexical_weight, 0.0, 1.0),
        _Param("importance_weight", baseline.importance_weight, 0.0, 0.12),
        _Param("recency_weight", baseline.recency_weight, 0.0, 0.10),
        _Param("feedback_weight", baseline.feedback_weight, 0.0, 0.10),
        _Param("dense_k", float(baseline.dense_k), 40.0, 200.0, int_val=True),
        _Param("lexical_k", float(baseline.lexical_k), 40.0, 200.0, int_val=True),
        _Param("rrf_k", float(baseline.rrf_k), 20.0, 100.0, int_val=True),
        _Param("fallback_min_results", float(baseline.fallback_min_results), 3.0, 8.0, int_val=True),
        _Param("exact_match_boost", baseline.exact_match_boost, 1.5, 5.0),
    ]

    def _fix_weights(p: Dict[str, float]) -> Dict[str, float]:
        """Normalise the two independent policy weight groups.

        The final Dense/Lexical/RRF blend and the additive
        Importance/Recency/Feedback blend are separate unit-sum contracts.
        Normalising all six together changes the scale of every candidate and
        makes generated policies violate the same schema the baseline obeys.
        """
        groups = (
            (
                "final_rrf_weight",
                "final_vector_weight",
                "final_lexical_weight",
            ),
            (
                "importance_weight",
                "recency_weight",
                "feedback_weight",
            ),
        )
        fallbacks = (
            (0.60, 0.20, 0.20),
            (0.55, 0.25, 0.20),
        )
        for keys, defaults in zip(groups, fallbacks):
            values = [max(0.0, float(p.get(key, 0.0))) for key in keys]
            total = sum(values)
            if total <= 1e-12:
                values = list(defaults)
                total = sum(values)
            for key, value in zip(keys, values):
                p[key] = value / total
        return p

    rng = random.Random(seed)
    candidates: List[Policy] = []

    def _make_p_dict() -> Dict[str, Any]:
        """Build a fresh policy-candidate dict with a real int version.

        ``Policy.version`` is declared ``int`` (policy_store.py). The
        previous ``f\"candidate-INT\"`` string construction caused
        ``Policy(**p_dict)`` to raise a ``ValidationError`` every time
        inside the ``except Exception: continue`` branch — which
        silently turned the candidate-generation loop into an
        infinite retry that never appended anything.
        """
        return {
            "schema_version": "1",
            "version": rng.randint(10000, 99999),
            "parent_version": baseline.version,
            "status": PolicyStatus.SHADOW.value,
        }

    # Coordinate: one per dim
    for param in params:
        for delta in (max_delta, -max_delta):
            p_dict: Dict[str, Any] = _make_p_dict()
            for p in params:
                val = param.perturb(delta) if p.name == param.name else baseline.model_dump().get(p.name, p.current)
                p_dict[p.name] = val
            p_dict = _fix_weights(p_dict)
            try:
                policy = Policy(**p_dict)
                candidates.append(policy)
            except Exception:
                continue
            if len(candidates) >= n_candidates:
                return candidates

    # If not enough, add random combos.
    # Hard cap to prevent an infinite loop if ``Policy`` keeps
    # rejecting the candidate dict (e.g. validator+normaliser
    # disagree on bounds). Each iteration should succeed; if it
    # doesn't, after this many failures there is a bug worth
    # surfacing rather than silently spinning.
    safety_cap = n_candidates * 50 + 100
    safety_iter = 0
    while len(candidates) < n_candidates:
        safety_iter += 1
        if safety_iter > safety_cap:
            logger.warning(
                "evolution: candidate generation hit safety cap (%d iter, %d candidates)",
                safety_iter,
                len(candidates),
            )
            break
        p_dict = _make_p_dict()
        for param in params:
            delta = rng.uniform(-max_delta, max_delta)
            p_dict[param.name] = param.perturb(delta)
            if param.name in ("dense_k", "lexical_k", "rrf_k", "fallback_min_results"):
                p_dict[param.name] = round(p_dict[param.name])
        p_dict = _fix_weights(p_dict)
        try:
            policy = Policy(**p_dict)
        except Exception:
            continue
        if all(
            abs(p.score - bp.score) > 1e-9
            for bp in candidates
        ):
            candidates.append(policy)

    return candidates[:n_candidates]  # type: ignore[index]  # narrow-only


# ---------------------------------------------------------------------------
# Shadow runner
# ---------------------------------------------------------------------------


# Default weight used by ``rank_fn_with_policy`` when the engine
# returns no per-candidate feature.  A small positive keeps the score
# monotone enough for ranking without distorting the relative ordering
# between policies.
_DEFAULT_NEUTRAL_SCORE: float = 0.0


def _rank_fn_from_pool(
    retrieval_engine: "RetrievalEngine",
    policy: Policy,
):
    """Internal policy-bound closure used for both Active and candidates.

    Kept separate from ``rank_fn_with_policy`` so tests and integrations that
    instrument candidate construction do not misclassify the Active baseline as
    an additional candidate evaluation.
    """
    def _rank_fn(query_text: str, query_id: str) -> List[str]:
        try:
            pool_getter = getattr(retrieval_engine, "get_candidate_pool", None)
            pool_ranker = getattr(retrieval_engine, "rank_candidate_pool", None)
            if callable(pool_getter) and callable(pool_ranker):
                channel_limit = max(200, int(policy.dense_k), int(policy.lexical_k))
                pool = pool_getter(query_text, channel_limit=channel_limit)
                result = pool_ranker(pool, policy, limit=50)
                return [hit.candidate_key for hit in result.hits]

            # Compatibility engines expose final-ish hits but no pool API.
            # Convert their per-channel fields into a temporary raw pool and
            # apply the authoritative pool scorer exactly once.  We deliberately
            # ignore the incoming composite score, avoiding double weighting.
            import inspect
            from .candidate_pool import QueryCandidatePool
            from .retrieval_engine import RetrievalEngine
            try:
                accepts_policy = "policy" in inspect.signature(
                    retrieval_engine.retrieve
                ).parameters
            except (TypeError, ValueError):
                accepts_policy = False
            if accepts_policy:
                raw_result = retrieval_engine.retrieve(
                    query_text, mode="hybrid", limit=200, policy=policy
                )
            else:
                raw_result = retrieval_engine.retrieve(
                    query_text, mode="hybrid", limit=200
                )
            dense_active = []
            lexical_active = []
            dense_superseded = []
            lexical_superseded = []
            for hit in raw_result.hits:
                target_dense = (
                    dense_superseded
                    if hit.status.value == "superseded"
                    else dense_active
                )
                target_lexical = (
                    lexical_superseded
                    if hit.status.value == "superseded"
                    else lexical_active
                )
                if hit.dense_score is not None:
                    target_dense.append(hit.model_copy(deep=True))
                if hit.lexical_score is not None:
                    target_lexical.append(hit.model_copy(deep=True))
                if hit.dense_score is None and hit.lexical_score is None:
                    fallback = hit.model_copy(deep=True)
                    fallback.dense_score = 0.0
                    target_dense.append(fallback)
            pool = QueryCandidatePool(
                query=query_text,
                dense_active=dense_active,
                lexical_active=lexical_active,
                dense_superseded=dense_superseded,
                lexical_superseded=lexical_superseded,
                dense_available=bool(dense_active or dense_superseded),
                lexical_available=bool(lexical_active or lexical_superseded),
            )
            detached = object.__new__(RetrievalEngine)
            result = RetrievalEngine.rank_candidate_pool(
                detached, pool, policy, limit=50
            )
            return [hit.candidate_key for hit in result.hits]
        except Exception:
            return []

    return _rank_fn


def rank_fn_with_policy(
    retrieval_engine: "RetrievalEngine",
    policy: Policy,
):
    """Public candidate-policy closure backed by a reusable raw pool."""
    return _rank_fn_from_pool(retrieval_engine, policy)


def shadow_compare(
    state: Dict[str, Any],
    cases: List[_EvaluationCase],
    active_policy: Policy,
    candidate_policy: Policy,
    *,
    rank_fn: Callable,
    rank_fns: Optional[Tuple[Callable, Callable]] = None,
) -> Tuple[bool, List[str], EvalResult, EvalResult]:
    """Run shadow comparison and return ``(should_promote, reasons, active_eval, candidate_eval)``.

    ``rank_fn(query_text, query_id) -> List[str]`` (legacy single-callable
    contract) OR ``rank_fns=(active_fn, candidate_fn)`` (B3-2 contract).

    When only ``rank_fn`` is supplied the active and candidate
    passes use the *same* ordering — equivalent to the
    v0.3.0.x behaviour where both policies were collapsed into one
    ranking. New callers should pass ``rank_fns`` so the candidate
    pass is meaningfully different from the active pass.
    """
    if rank_fns is not None:
        active_rank_fn, candidate_rank_fn = rank_fns
    else:
        active_rank_fn = rank_fn
        candidate_rank_fn = rank_fn
    active_eval = evaluate(cases, active_rank_fn)
    candidate_eval = evaluate(cases, candidate_rank_fn)
    should_promote, reasons = evaluate_candidate(active_eval, candidate_eval, strict_validation=False)
    return should_promote, reasons, active_eval, candidate_eval


# ---------------------------------------------------------------------------
# Promote / rollback
# ---------------------------------------------------------------------------


def _can_promote(
    state: Dict[str, Any], candidate_version: Optional[int] = None
) -> bool:
    """Check cold-start gate, cooldown, and circuit breaker.

    G6.5: Two consecutive evaluation windows (each ≥ 10 min apart)
    must have PASSED before promotion is allowed. This is enforced
    in addition to the legacy cold-start / cooldown / max-promotions
    / circuit-breaker checks.
    """
    qcount = _judged_query_count()
    now = _now()

    # Cold start
    if qcount < _COLD_START_MIN_QUERIES:
        logger.info("evolution: cold-start — %d judged queries (need %d)", qcount, _COLD_START_MIN_QUERIES)
        return False

    # Two successful, distinct windows are required. Production callers pass
    # the selected candidate version, preventing candidate A's pass from helping
    # candidate B. ``None`` preserves the legacy helper-only test surface.
    if int(state.get("consecutive_passes", 0)) < _PROMOTION_WINDOWS_REQUIRED:
        logger.info(
            "evolution: promotion needs two passes (have %d)",
            int(state.get("consecutive_passes", 0)),
        )
        return False
    if not _two_consecutive_pass_windows(state, candidate_version):
        logger.info("evolution: promotion pass windows are not valid")
        return False

    # Cooldown
    last = state.get("last_promotion_at")
    if last:
        dt = datetime.fromisoformat(last)
        if (now - dt).days < _PROMOTION_COOLDOWN_DAYS:
            logger.info("evolution: cooldown — %d days since last promote", (now - dt).days)
            return False

    # Max promotions per 30d
    if state.get("promotion_count_30d", 0) >= _MAX_PROMOTIONS_PER_30D:
        logger.info("evolution: max promotions per 30d reached")
        return False

    # Consecutive rollbacks
    if state.get("consecutive_rollbacks", 0) >= _MAX_CONSECUTIVE_ROLLBACKS:
        # Check cooldown: if we hit max rollbacks, wait EVOLUTION_COOLDOWN_HOURS
        # before trying again.
        cooldown_start = state.get("cooldown_start_at")
        if cooldown_start:
            cooldown_dt = datetime.fromisoformat(cooldown_start)
            hours_elapsed = (now - cooldown_dt).total_seconds() / 3600.0
            if hours_elapsed < EVOLUTION_COOLDOWN_HOURS:
                logger.info(
                    "evolution: max consecutive rollbacks (%d) — cooldown %.1f/%d hours",
                    _MAX_CONSECUTIVE_ROLLBACKS,
                    hours_elapsed,
                    EVOLUTION_COOLDOWN_HOURS,
                )
                return False
            else:
                # Cooldown expired; allow promotion again.
                logger.info("evolution: cooldown expired after %.1f hours; allowing promotion", hours_elapsed)
                return True
        logger.info("evolution: max consecutive rollbacks (%d) — refusing to promote", _MAX_CONSECUTIVE_ROLLBACKS)
        return False

    return True


# ---------------------------------------------------------------------------
# G6.6 — full promotion gate
# ---------------------------------------------------------------------------

# Runbook v2 G6.6 tolerances. These are independent of the G6.7 rollback
# triggers: the rollback side uses absolute drop thresholds (e.g. -8pp on
# useful@1) because rollback is about *detecting degradation*, while the
# promotion side uses relative-pass tolerances because promotion is about
# *requiring evidence of parity or improvement on every metric*.
#
# The numbers below are the canonical runbook values; tweak them via
# these constants rather than scattering magic numbers through
# ``_promotion_passes_all_thresholds``.
_PROMOTION_NDCG_MIN_DELTA = 0.01  # nDCG@10 must beat baseline by ≥1pp
_PROMOTION_NEGATIVE_AT_5_TOLERANCE = 0.0  # negative@5 ≤ baseline
_PROMOTION_FALLBACK_USEFUL_TOLERANCE = 0.0  # fallback_useful ≥ baseline
_PROMOTION_P95_LATENCY_MAX_MULTIPLIER = 1.15  # p95 ≤ baseline × 1.15


def _promotion_passes_all_thresholds(
    active_eval: Any,
    candidate_eval: Any,
) -> Tuple[bool, List[str]]:
    """Runbook v2 G6.6: ALL of the following must hold for promotion.

    The candidate must satisfy, *simultaneously*:

    * ``useful_at_1 >= active.useful_at_1`` (no regression)
    * ``mrr_at_10 >= active.mrr_at_10`` (no regression)
    * ``ndcg_at_10 >= active.ndcg_at_10 + 0.01`` (≥1pp improvement)
    * ``useful_at_5 >= active.useful_at_5`` (no regression)
    * ``explicit_negative_at_5 <= active.explicit_negative_at_5 `` (no increase)
    * ``fallback_useful_rate >= active.fallback_useful_rate ``
      (≥-5pp, when both are non-``None``)
    * ``degraded_rate <= active.degraded_rate`` (no regression)
    * ``p95_latency <= active.p95_latency × 1.15``
    * ``positive_hit_at_5 >= active.positive_hit_at_5`` (no regression;
      alias of ``useful_at_5`` on :class:`EvalResult`)

    Returns ``(passed, reasons)``. ``reasons`` is the list of metric
    names that FAILED the gate (empty when all pass). The list is in
    the order the checks ran, so operators can read it top-to-bottom
    and see exactly what failed.

    Notes
    -----
    * ``positive_hit_at_5`` is exposed as an alias of ``useful_at_5``
      on :class:`EvalResult` (see :mod:`openclaw_memory_os.evaluation`)
      because the runbook lists it as a separate metric even though
      it is computed from the same underlying ``useful_at_k`` helper.
    * ``fallback_useful_rate`` is optional (some legacy ``EvalResult``
      rows carry ``None``). When ``active.fallback_useful_rate is None``
      we do NOT require the candidate to populate it: that would be
      a backward-incompatible regression. The check is skipped instead.
    * NaN handling: any NaN in the input fails the corresponding
      check (we log ``"nan"`` as the reason). A NaN candidate eval
      means the held-out split had no judgements, which is itself a
      reason to refuse promotion — see ``run_evolution_cycle``.
    """
    reasons: List[str] = []

    def _ge(a: float, b: float, name: str) -> None:
        try:
            if math.isnan(a) or math.isnan(b) or a < b:
                reasons.append(name)
        except Exception:
            reasons.append(name)

    def _gt_with_delta(a: float, b: float, delta: float, name: str) -> None:
        try:
            if math.isnan(a) or math.isnan(b) or a < b + delta:
                reasons.append(name)
        except Exception:
            reasons.append(name)

    def _le(a: float, b: float, name: str) -> None:
        try:
            if math.isnan(a) or math.isnan(b) or a > b:
                reasons.append(name)
        except Exception:
            reasons.append(name)

    def _le_with_delta(a: float, b: float, delta: float, name: str) -> None:
        try:
            if math.isnan(a) or math.isnan(b) or a > b + delta:
                reasons.append(name)
        except Exception:
            reasons.append(name)

    def _mult(a: float, b: float, mult: float, name: str) -> None:
        try:
            if math.isnan(a) or math.isnan(b) or a > b * mult:
                reasons.append(name)
        except Exception:
            reasons.append(name)

    _ge(candidate_eval.useful_at_1, active_eval.useful_at_1, "useful_at_1")
    _ge(candidate_eval.mrr_at_10, active_eval.mrr_at_10, "mrr_at_10")
    _gt_with_delta(
        candidate_eval.ndcg_at_10,
        active_eval.ndcg_at_10,
        _PROMOTION_NDCG_MIN_DELTA,
        "ndcg_at_10",
    )
    _ge(candidate_eval.useful_at_5, active_eval.useful_at_5, "useful_at_5")
    _le_with_delta(
        candidate_eval.explicit_negative_at_5,
        active_eval.explicit_negative_at_5,
        _PROMOTION_NEGATIVE_AT_5_TOLERANCE,
        "negative_at_5",
    )
    # ``fallback_useful_rate`` is optional on legacy EvalResult rows;
    # only enforce the gate when both sides carry a value.
    cand_fb = getattr(candidate_eval, "fallback_useful_rate", None)
    active_fb = getattr(active_eval, "fallback_useful_rate", None)
    if cand_fb is not None and active_fb is not None:
        try:
            if math.isnan(cand_fb) or math.isnan(active_fb):
                reasons.append("fallback_useful_rate")
            elif cand_fb < active_fb - _PROMOTION_FALLBACK_USEFUL_TOLERANCE:
                reasons.append("fallback_useful_rate")
        except Exception:
            reasons.append("fallback_useful_rate")
    _le(candidate_eval.degraded_rate, active_eval.degraded_rate, "degraded_rate")
    _mult(
        candidate_eval.p95_latency,
        active_eval.p95_latency,
        _PROMOTION_P95_LATENCY_MAX_MULTIPLIER,
        "p95_latency",
    )
    # ``positive_hit_at_5`` is a property alias of ``useful_at_5`` on
    # ``EvalResult``; we still check it independently so the runbook's
    # 9-metric list is honored even on legacy rows where the alias
    # attribute is missing.
    cand_ph5 = getattr(candidate_eval, "positive_hit_at_5", None)
    active_ph5 = getattr(active_eval, "positive_hit_at_5", None)
    if cand_ph5 is not None and active_ph5 is not None:
        _ge(cand_ph5, active_ph5, "positive_hit_at_5")

    return (len(reasons) == 0, reasons)


def _check_rollback(
    store: PolicyStore,
    state: Dict[str, Any],
    *,
    check_fn: Callable,
) -> bool:
    """Check if rollback is needed. Returns True if rolled back.

    Triggers (G6.7 — full set from the runbook):

    * **File corruption** — checksum mismatch on the active policy.
    * **Error / degraded rate > 5%** — system health gate.
    * **no_result_rate > 10%** — absolute fallback to the baseline
      (immediate; runs even when ``previous_metrics`` is unset).
    * **p95_latency > 5.0 s** — absolute latency floor (immediate).
    * **Statistical deltas vs. ``state["previous_metrics"]``** — any
      one of the following fires a rollback:

        - ``useful_at_1`` drop ≥ 8 pp
        - ``mrr_at_10`` drop ≥ 5 pp
        - ``explicit_negative_at_5`` increase ≥ 8 pp
        - ``no_result_rate`` increase ≥ 15 pp
        - ``p95_latency`` more than 2× the previous reading
        - ``fallback_useful_rate`` drop ≥ 5 pp (when enabled / non-None)

      ``previous_metrics`` is seeded with a neutral baseline so the
      very first cycle doesn't false-trigger. After a rollback, the
      active policy's pre-rollback eval is stashed in
      ``state["post_rollback_metrics"]`` and copied to
      ``previous_metrics`` so the NEXT cycle compares against the
      freshly-reverted-to policy, not the (now-retired) one that
      tripped the trigger.
    """
    try:
        current = store.get()
    except Exception:
        logger.warning("evolution: cannot read current policy — rolling back to baseline")
        _force_rollback(store, state)
        return True

    # File corrupt check
    try:
        compute_checksum(current)
        _ = store.checksum()
    except Exception:
        logger.warning("evolution: active policy checksum mismatch — rolling back")
        _force_rollback(store, state)
        return True

    # Evaluate current eval so we have fresh numbers for the
    # statistical triggers. The legacy implementation only read
    # ``degraded_rate`` / ``no_result_rate`` / ``p95_latency``; we
    # now also read useful_at_1 / mrr / negative_at_5 / fallback
    # so all six G6.7 thresholds can fire.
    err_rate = 0.0
    no_result_rate = 0.0
    p95_latency = 0.0
    useful_at_1 = 0.0
    mrr = 0.0
    negative_at_5 = 0.0
    fallback_useful_rate: Optional[float] = None
    current_eval: Optional[EvalResult] = None
    try:
        cases = _load_cases(limit=50)
        if cases and check_fn is not None:
            current_eval = evaluate(cases, check_fn)
            err_rate = current_eval.degraded_rate
            no_result_rate = current_eval.no_result_rate
            p95_latency = current_eval.p95_latency
            useful_at_1 = current_eval.useful_at_1
            mrr = current_eval.mrr_at_10
            negative_at_5 = current_eval.explicit_negative_at_5
            fallback_useful_rate = current_eval.useful_superseded_fallback_rate
    except Exception:
        pass

    # --- Absolute (no-reference) triggers -------------------------------
    if err_rate > _ROLLBACK_ERR_RATE_THRESHOLD:
        logger.warning(
            "evolution: rollback — degraded rate %.1f%% > 5%%",
            err_rate * 100,
        )
        _force_rollback_with_metrics(store, state, current_eval)
        return True

    if no_result_rate > _ROLLBACK_NO_RESULT_RATE_THRESHOLD:
        logger.warning(
            "evolution: rollback — no-result rate %.1f%% > 10%%",
            no_result_rate * 100,
        )
        _force_rollback_with_metrics(store, state, current_eval)
        return True

    if p95_latency > _ROLLBACK_P95_LATENCY_ABS_THRESHOLD:
        logger.warning(
            "evolution: rollback — p95 latency %.1fs > 5.0s",
            p95_latency,
        )
        _force_rollback_with_metrics(store, state, current_eval)
        return True

    # --- Statistical (reference-based) triggers --------------------------
    # ``previous_metrics`` is seeded with the neutral baseline so the
    # very first cycle doesn't false-trigger on tiny deltas.
    prev = state.get("previous_metrics") or dict(_DEFAULT_PREVIOUS_METRICS)
    prev_useful_at_1 = float(prev.get("useful_at_1", useful_at_1))
    prev_mrr = float(prev.get("mrr_at_10", mrr))
    prev_negative_at_5 = float(prev.get("explicit_negative_at_5", negative_at_5))
    prev_no_result_rate = float(prev.get("no_result_rate", no_result_rate))
    prev_p95_latency = float(prev.get("p95_latency", p95_latency))
    prev_fallback_useful = prev.get("fallback_useful_rate")
    if prev_fallback_useful is not None:
        try:
            prev_fallback_useful = float(prev_fallback_useful)
        except (TypeError, ValueError):
            prev_fallback_useful = None

    # G6.7 trigger: useful@1 drop ≥ 8 pp.
    if useful_at_1 < prev_useful_at_1 - _ROLLBACK_USEFUL_AT_1_DROP_THRESHOLD:
        logger.warning(
            "evolution: rollback — useful@1 %.3f dropped >%.0fpp from %.3f",
            useful_at_1,
            _ROLLBACK_USEFUL_AT_1_DROP_THRESHOLD * 100,
            prev_useful_at_1,
        )
        _force_rollback_with_metrics(store, state, current_eval)
        return True

    # G6.7 trigger: MRR drop ≥ 5%.
    if mrr < prev_mrr - _ROLLBACK_MRR_DROP_THRESHOLD:
        logger.warning(
            "evolution: rollback — MRR %.3f dropped >%.0f%% from %.3f",
            mrr,
            _ROLLBACK_MRR_DROP_THRESHOLD * 100,
            prev_mrr,
        )
        _force_rollback_with_metrics(store, state, current_eval)
        return True

    # G6.7 trigger: negative@5 increase ≥ 8 pp.
    if negative_at_5 > prev_negative_at_5 + _ROLLBACK_NEGATIVE_AT_5_DELTA_THRESHOLD:
        logger.warning(
            "evolution: rollback — negative@5 %.3f grew >%.0fpp from %.3f",
            negative_at_5,
            _ROLLBACK_NEGATIVE_AT_5_DELTA_THRESHOLD * 100,
            prev_negative_at_5,
        )
        _force_rollback_with_metrics(store, state, current_eval)
        return True

    # G6.7 trigger: no_result_rate increase ≥ 15 pp (statistical;
    # the absolute >10% check above fires first if absolute is also
    # tripped).
    if no_result_rate > prev_no_result_rate + _ROLLBACK_NO_RESULT_DELTA_THRESHOLD:
        logger.warning(
            "evolution: rollback — no_result_rate %.3f grew >%.0fpp from %.3f",
            no_result_rate,
            _ROLLBACK_NO_RESULT_DELTA_THRESHOLD * 100,
            prev_no_result_rate,
        )
        _force_rollback_with_metrics(store, state, current_eval)
        return True

    # G6.7 trigger: p95 latency > previous × 2. Guard against the
    # "previous was 0" degenerate case — if previous is 0, treat
    # any non-zero current as a 2× regression (conservative).
    p95_prev_safe = max(prev_p95_latency, 1e-6)
    if p95_latency > p95_prev_safe * _ROLLBACK_P95_LATENCY_X_THRESHOLD:
        logger.warning(
            "evolution: rollback — p95 latency %.2fs > %.1fx previous %.2fs",
            p95_latency,
            _ROLLBACK_P95_LATENCY_X_THRESHOLD,
            prev_p95_latency,
        )
        _force_rollback_with_metrics(store, state, current_eval)
        return True

    # G6.7 trigger: fallback_useful_rate drop (when enabled). The
    # metric is optional (``None`` when no fallback was exercised);
    # we only fire the rollback when both sides have a value.
    if (
        fallback_useful_rate is not None
        and prev_fallback_useful is not None
        and fallback_useful_rate
        < float(prev_fallback_useful) - _ROLLBACK_FALLBACK_USEFUL_DROP_THRESHOLD
    ):
        logger.warning(
            "evolution: rollback — fallback_useful_rate %.3f dropped >%.0fpp from %.3f",
            fallback_useful_rate,
            _ROLLBACK_FALLBACK_USEFUL_DROP_THRESHOLD * 100,
            prev_fallback_useful,
        )
        _force_rollback_with_metrics(store, state, current_eval)
        return True

    return False


def _force_rollback(store: PolicyStore, state: Dict[str, Any]) -> None:
    """Atomically set the active policy back to the shipped baseline.

    The baseline is the canonical known-good policy that ships
    with the OS; calling :meth:`PolicyStore.revert` swaps it
    back into the active slot AND persists it to disk via the
    store's atomic-write path. The ``state["consecutive_rollbacks"]``
    counter drives the circuit breaker.

    When the counter reaches ``_MAX_CONSECUTIVE_ROLLBACKS``, the
    cooldown start time is recorded so the evolution loop waits
    ``EVOLUTION_COOLDOWN_HOURS`` before attempting promotion again.
    """
    _force_rollback_with_metrics(store, state, None)


def _force_rollback_with_metrics(
    store: PolicyStore,
    state: Dict[str, Any],
    pre_rollback_eval: Optional[EvalResult],
) -> None:
    """Atomic rollback that also updates G6.7 ``previous_metrics``.

    This is the canonical rollback helper. ``pre_rollback_eval`` is
    the EvalResult from the *active* policy just before the
    rollback; we stash it under ``state["post_rollback_metrics"]``
    and copy it into ``state["previous_metrics"]`` so the NEXT
    cycle's statistical rollback triggers compare against the
    freshly-reverted-to policy's numbers (rather than against the
    retired bad policy's numbers, which would immediately
    re-trigger). When ``pre_rollback_eval`` is ``None`` (e.g. the
    rollback came from the file-corrupt branch, where no eval was
    possible), we leave ``previous_metrics`` untouched — the
    neutral baseline seed keeps the next cycle safe.
    """
    store.revert()
    state["consecutive_rollbacks"] = state.get("consecutive_rollbacks", 0) + 1
    # Record cooldown start when we hit the max consecutive rollbacks.
    if state["consecutive_rollbacks"] >= _MAX_CONSECUTIVE_ROLLBACKS:
        state["cooldown_start_at"] = _now().isoformat()
    # G6.7: update the reference metrics the next cycle compares
    # against. The just-reverted-to policy becomes the new baseline;
    # its eval numbers are written into ``previous_metrics`` so
    # deltas on the next cycle start from a clean reference.
    if pre_rollback_eval is not None:
        snapshot = {
            "useful_at_1": float(pre_rollback_eval.useful_at_1),
            "mrr_at_10": float(pre_rollback_eval.mrr_at_10),
            "explicit_negative_at_5": float(pre_rollback_eval.explicit_negative_at_5),
            "no_result_rate": float(pre_rollback_eval.no_result_rate),
            "p95_latency": float(pre_rollback_eval.p95_latency),
            "degraded_rate": float(pre_rollback_eval.degraded_rate),
        }
        # ``fallback_useful_rate`` is optional; only include it when
        # the eval actually produced a number. The new key is
        # ``fallback_useful_rate`` (matching the G6.7 spec) — the
        # runbook's earlier "fallback_useful_rate" alias.
        if pre_rollback_eval.useful_superseded_fallback_rate is not None:
            snapshot["fallback_useful_rate"] = float(
                pre_rollback_eval.useful_superseded_fallback_rate
            )
        state["post_rollback_metrics"] = snapshot
        state["previous_metrics"] = snapshot
    # Reset the consecutive-passes counter: a rollback breaks the
    # promotion streak. The next ``passed`` outcome re-starts the
    # two-window counter from 1 (a single pass), not 2.
    state["consecutive_passes"] = 0
    # Clear the pass-window ring buffer so a stale "passed" entry
    # from before the rollback doesn't survive.
    state["pass_windows"] = []
    _save_evolution_state(state)


# ---------------------------------------------------------------------------
# Full evolution cycle
# ---------------------------------------------------------------------------


def run_evolution_cycle(
    store: PolicyStore,
    rank_fn: Callable,
    *,
    candidate_rank_fn: Optional[Callable] = None,
    engine: Optional["RetrievalEngine"] = None,
) -> Dict[str, Any]:
    """Execute one evolution cycle — the weekly function.

    ``rank_fn(query_text, query_id) -> List[str]`` must be set by
    the caller (the actual engine's retrieval path). ``candidate_rank_fn``
    is the candidate-policy-aware variant; when provided, candidate
    evaluations use it so the B3-5 contract (different policies þ
    different orderings) is enforced. When ``candidate_rank_fn`` is
    ``None``, the candidate evaluations fall back to ``rank_fn``,
    matching the legacy v0.3.0.x behaviour.

    When ``engine`` is provided AND ``candidate_rank_fn`` is ``None``,
    the runner builds a fresh ``rank_fn_with_policy(engine, cand)``
    closure for EACH candidate inside the loop so every candidate is
    evaluated with its own policy (G6.1). Without an ``engine`` (or
    when ``candidate_rank_fn`` is supplied), the legacy single
    ``candidate_rank_fn`` is reused for every candidate — that
    behaviour is preserved for backward compatibility.

    The function runs at most once per process (acquires a non-blocking
    file lock on ``_EVOLUTION_LOCK_PATH``).
    """
    import fcntl

    try:
        lock_fd = open(_EVOLUTION_LOCK_PATH, "w")
        fcntl.lockf(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        err = "evolution: another process holds the evolution lock; skipping"
        logger.warning(err)
        return {"status": "skipped", "reason": "lock_held"}

    try:
        state = _load_evolution_state()
        active = store.get()

        # 0. Cold-start gate. Run BEFORE the rollback check so a
        # brand-new deployment with zero judged queries doesn't
        # false-trigger the G6.7 statistical rollback triggers:
        # the seeded ``previous_metrics`` (useful@1=0.5 etc.)
        # would otherwise look like a 50pp drop on the very first
        # ``evaluate()`` call which returns 0.0 across the board
        # when no feedback has been collected yet. The original
        # v0.3.0.x order was "rollback first, then cold-start"
        # but that relied on ``err_rate / no_result_rate / p95``
        # defaults of 0 (no comparison reference); G6.7 added
        # reference-based triggers that need either a real
        # previous evaluation OR a cold-start short-circuit.
        qcount = _judged_query_count()
        if qcount < _COLD_START_MIN_QUERIES:
            logger.info("evolution: %d judged queries (need %d); skipping", qcount, _COLD_START_MIN_QUERIES)
            return {"status": "skipped", "reason": f"cold_start: {qcount}/{_COLD_START_MIN_QUERIES}"}

        # 1. Auto-rollback check
        if _check_rollback(store, state, check_fn=rank_fn):
            return {"status": "rolled_back"}

        # 2. Generate candidates
        max_delta = 0.03 if qcount < _QUERIES_FOR_FULL_RANGE else 0.05
        generated = generate_candidates(active, n_candidates=20, max_delta=max_delta)
        persisted_shadow = store.get_shadow()
        candidates: List[Policy] = []
        if (
            persisted_shadow is not None
            and int(persisted_shadow.parent_version or active.version)
            == int(active.version)
        ):
            candidates.append(persisted_shadow)
        seen_versions = {int(candidate.version) for candidate in candidates}
        for candidate in generated:
            if int(candidate.version) in seen_versions:
                continue
            candidates.append(candidate)
            seen_versions.add(int(candidate.version))
            if len(candidates) >= 20:
                break
        logger.info(
            "evolution: evaluating %d candidates (persisted_shadow=%s)",
            len(candidates), persisted_shadow is not None,
        )

        # 4. Evaluate baseline + candidate pool.
        # The candidate evaluation must use a candidate-policy-aware
        # rank_fn (B3-2 / B3-5). When the caller didn't supply one,
        # we fall back to the legacy ``rank_fn`` — same
        # behaviour as v0.3.0.x, but the new ``candidate_rank_fn``
        # hook lets the caller wire the v0.3.0.3 evolution flow.
        cases = _load_cases(limit=500)
        train_cases, val_cases, test_cases = split_cases(cases)

        # Active and candidate policies must rank the exact same raw pool.
        active_rank_fn = (
            _rank_fn_from_pool(engine, active) if engine is not None else rank_fn
        )
        # Train: used for candidate search
        baseline_eval = evaluate(train_cases, active_rank_fn)
        # Decide how to evaluate each candidate. Three modes, in order
        # of precedence:
        #   1. caller supplied ``candidate_rank_fn`` → use it for every
        #      candidate (legacy single-closure behaviour, preserved
        #      for backward compatibility with B3-5 callers);
        #   2. caller supplied ``engine`` → build a fresh
        #      ``rank_fn_with_policy(engine, cand)`` for each candidate
        #      so the candidate's weights actually drive its score
        #      (G6.1 — the per-candidate closure contract);
        #   3. neither → fall back to ``rank_fn`` (v0.3.0.x legacy,
        #      both policies collapse to one ordering).
        if candidate_rank_fn is not None:
            cand_rank_fn = candidate_rank_fn
            per_candidate_engine = False
        elif engine is not None:
            cand_rank_fn = None  # built per candidate inside the loop
            per_candidate_engine = True
        else:
            cand_rank_fn = rank_fn
            per_candidate_engine = False

        # G6.2 — Funnel (Top 5 → Top 2 → Top 1).
        # Stage 1: pool ranking on the train split. Score every
        # candidate on the train set and keep the top
        # ``_FUNNEL_POOL_SIZE`` (5). When ``engine=None`` was passed
        # the candidate eval degenerates to ``rank_fn`` (legacy v0.3.0.x
        # behaviour), so all candidates would tie; in that mode we
        # fall back to "first 5 in the generated list" to keep the
        # funnel deterministic for test fixtures and to avoid
        # crashing on tiny test datasets. This matches the runbook's
        # requirement that the funnel actually narrow the pool.
        def _score_on(cand: Policy, cases_for_eval: List) -> "EvalResult":
            """Evaluate a single candidate on ``cases_for_eval``.

            When ``per_candidate_engine`` is True we build a fresh
            closure bound to THIS candidate so weights actually
            drive the score (G6.1). Otherwise we reuse the legacy
            single ``cand_rank_fn``.
            """
            if per_candidate_engine:
                return evaluate(cases_for_eval, rank_fn_with_policy(engine, cand))
            return evaluate(cases_for_eval, cand_rank_fn)

        if per_candidate_engine:
            scored_pool = [
                (c, _score_on(c, train_cases)) for c in candidates
            ]
        else:
            # Legacy mode: all candidates tie because rank_fn doesn't
            # depend on the candidate policy. Still run the funnel
            # shape so the test suite and downstream operators see
            # consistent state — just don't pretend the ranking
            # actually picks winners. Use a deterministic tie-breaker
            # (cand.version descending) so two identical-evals
            # candidates produce a stable order.
            scored_pool = [
                (c, evaluate(train_cases, cand_rank_fn)) for c in candidates
            ]
            scored_pool.sort(key=lambda kv: (-kv[0].version,))
        # ``useful_at_1`` is the canonical ranking signal for the
        # funnel: it captures "is the first result the one the user
        # wanted", which is exactly what the runbook asks for.
        scored_pool.sort(key=lambda kv: (-kv[1].useful_at_1,))
        top5 = [c for c, _ in scored_pool[:_FUNNEL_POOL_SIZE]]
        if not top5:
            logger.info("evolution: empty top-5 from funnel — no candidate beats baseline")
            _record_cycle_result(state, "failed")
            _save_evolution_state(state)
            return {"status": "ok", "reason": "no_improvement"}

        # Stage 2: val split. Evaluate the top-5 on the val split and
        # keep the top ``_FUNNEL_VAL_SIZE`` (2). When the val split
        # is empty (small corpora fall entirely into train), skip
        # the funnel narrowing and pass all of top5 to held-out.
        if val_cases:
            val_scored = [(c, _score_on(c, val_cases)) for c in top5]
            val_scored.sort(key=lambda kv: (-kv[1].useful_at_1,))
            top2 = [c for c, _ in val_scored[:_FUNNEL_VAL_SIZE]]
        else:
            top2 = list(top5)

        # Stage 3: held-out (test) split. Pick the single best of
        # top2 by useful@1 on the held-out split. When the held-out
        # split is empty, fall back to the first of top2 (still
        # gated by the two-window pass requirement).
        if test_cases:
            held_scored = [(c, _score_on(c, test_cases)) for c in top2]
            held_scored.sort(key=lambda kv: (-kv[1].useful_at_1,))
            best_cand, best_cand_eval = held_scored[0]
            best_cand_score = best_cand_eval.useful_at_1
        else:
            best_cand = top2[0]
            best_cand_score = float("nan")
        logger.info(
            "evolution funnel: pool=%d -> top5=%d -> top2=%d -> best=v%d",
            len(candidates), len(top5), len(top2), best_cand.version,
        )

        # 5. Shadow: validate candidate on val set before publishing.
        # The G6.2 funnel already ran val; here we re-check that the
        # candidate does not regress on the BASELINE-vs-CANDIDATE
        # dimension (i.e. shadow check). The per-candidate closure
        # must reflect the winning candidate's policy, so re-bind
        # it here when ``per_candidate_engine`` is True.
        if val_cases:
            if per_candidate_engine:
                cand_rank_fn = rank_fn_with_policy(engine, best_cand)
            val_active_eval = evaluate(val_cases, active_rank_fn)
            val_cand_eval = evaluate(val_cases, cand_rank_fn)
            val_passed, val_reasons = evaluate_candidate(
                val_active_eval, val_cand_eval, strict_validation=False
            )
            if not val_passed:
                logger.info(
                    "evolution: candidate failed val check: %s",
                    ", ".join(val_reasons),
                )
                _record_cycle_result(state, "failed")
                _save_evolution_state(state)
                return {"status": "ok", "reason": "val_failed: " + ", ".join(val_reasons)}

        # 5b. Persist the real, like-for-like evaluation before exposing
        # the candidate to the dashboard.  The report is updated later with the
        # final shadow/promotion decision, always under the same report_id.
        from .evaluation_reports import new_report_id, save_evaluation_report
        report_id = new_report_id()
        if test_cases:
            report_split = "test"
            report_active_eval = evaluate(test_cases, active_rank_fn)
            report_candidate_eval = best_cand_eval
        elif val_cases:
            report_split = "validation"
            report_active_eval = val_active_eval
            report_candidate_eval = val_cand_eval
        else:
            report_split = "train"
            report_active_eval = baseline_eval
            report_candidate_eval = _score_on(best_cand, train_cases)
        try:
            report_snapshot = engine.compute_snapshot_id() if engine is not None else None
        except Exception:
            report_snapshot = None

        def _persist_report(decision_status: str, **decision_extra: Any) -> None:
            save_evaluation_report(
                {
                    "report_id": report_id,
                    "status": "ok",
                    "corpus_snapshot_id": report_snapshot,
                    "metrics": report_candidate_eval.to_dict(),
                    "active_metrics": report_active_eval.to_dict(),
                    "candidate_metrics": report_candidate_eval.to_dict(),
                    "policy": {
                        "active_version": active.version,
                        "candidate_version": best_cand.version,
                    },
                    "split": {
                        "selected": report_split,
                        "train_cases": len(train_cases),
                        "validation_cases": len(val_cases),
                        "test_cases": len(test_cases),
                    },
                    "decision": {"status": decision_status, **decision_extra},
                    "warnings": [],
                    "notes": ["Generated by run_evolution_cycle using real policy rankings."],
                }
            )

        _persist_report("evaluated")
        shadow_metadata = {
            "corpus_snapshot_id": report_snapshot,
            "offline_report_id": report_id,
            "shadow_sample_count": len(val_cases or test_cases or train_cases),
            "consecutive_passes": int(state.get("consecutive_passes", 0)),
        }
        try:
            store.set_shadow(best_cand, metadata=shadow_metadata)
        except TypeError:
            store.set_shadow(best_cand)
        logger.info("evolution: candidate %s set for shadow", best_cand.version)

        # 5c. G6.4 — per-query shadow comparison rows.
        # For each case in the val split (the same set used by the
        # shadow check in step 5 above), run the active and
        # candidate rank functions and record a comparison row in
        # ``state["shadow_comparisons"]``. The buffer is capped at
        # ``_SHADOW_COMPARE_BUFFER_CAP`` (oldest dropped) so the
        # state file stays bounded. Recording happens BEFORE the
        # promotion / rollback decision so operators see the
        # comparison regardless of which path the cycle takes.
        if val_cases:
            try:
                _record_shadow_comparisons(
                    state, val_cases, active_rank_fn, cand_rank_fn
                )
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "evolution: shadow comparison recording failed: %s", exc
                )

        # 6. Guarded auto promotion (G6.6). The full metrics gate runs
        # before the current window is credited.
        # Runbook v2 requires ALL of the following to hold
        # simultaneously for promotion; failure on any single metric
        # keeps the candidate in shadow. We compare the candidate's
        # held-out (test) eval against the active policy's eval on
        # the SAME split so the comparison is apples-to-apples; when
        # the test split is empty (small corpora) we fall back to
        # the candidate's val eval vs the active's val eval, and
        # finally to the train baseline as a last-resort reference.
        if math.isnan(best_cand_score):
            # Held-out eval returned no judgements (empty test
            # split). The legacy behaviour was to refuse promotion
            # in this case; G6.6 keeps that contract because a
            # NaN candidate is exactly the situation the gate
            # refuses to clear.
            logger.info(
                "evolution: candidate held-out eval is NaN; staying in shadow",
            )
            _persist_report("shadow", reason="held_out_nan")
            _record_cycle_result(
                state, "failed", candidate_version=best_cand.version
            )
            _save_evolution_state(state)
            return {"status": "shadow", "candidate_version": best_cand.version, "report_id": report_id}

        # Build the active-side eval on the same split as
        # ``best_cand_eval`` so the G6.6 gate compares like-for-like.
        if test_cases:
            promotion_active_eval = evaluate(test_cases, active_rank_fn)
            promotion_candidate_eval = best_cand_eval
        elif val_cases:
            promotion_active_eval = val_active_eval
            promotion_candidate_eval = val_cand_eval
        else:
            promotion_active_eval = baseline_eval
            promotion_candidate_eval = best_cand_eval if best_cand_eval is not None else baseline_eval

        promotion_passed, failed_metrics = _promotion_passes_all_thresholds(
            promotion_active_eval, promotion_candidate_eval
        )
        if not promotion_passed:
            logger.info(
                "evolution: candidate failed G6.6 promotion gate: %s; staying in shadow",
                ", ".join(failed_metrics),
            )
            _persist_report("shadow", reason="promotion_gate_failed", failed_metrics=failed_metrics)
            _record_cycle_result(
                state, "failed", candidate_version=best_cand.version
            )
            _save_evolution_state(state)
            return {
                "status": "shadow",
                "candidate_version": best_cand.version,
                "failed_metrics": failed_metrics,
                "report_id": report_id,
            }

        # Credit this validated candidate's current window. A different
        # candidate resets the bank; the first pass remains in shadow.
        _record_cycle_result(
            state, "passed", candidate_version=best_cand.version
        )
        shadow_metadata["consecutive_passes"] = int(
            state.get("consecutive_passes", 0)
        )
        metadata_updater = getattr(store, "update_shadow_metadata", None)
        if callable(metadata_updater):
            metadata_updater(shadow_metadata)
        try:
            promotion_allowed = _can_promote(
                state, candidate_version=int(best_cand.version)
            )
        except TypeError:  # compatibility for narrow test monkeypatches
            promotion_allowed = _can_promote(state)
        if not promotion_allowed:
            _persist_report(
                "shadow",
                reason="two_candidate_windows_or_governance_gate",
                consecutive_passes=int(state.get("consecutive_passes", 0)),
            )
            _save_evolution_state(state)
            return {
                "status": "shadow",
                "candidate_version": best_cand.version,
                "report_id": report_id,
            }

        # Promote: swap into the in-memory active slot AND persist
        # the new policy to disk via the store's atomic write
        # (``store.set`` only updates in-memory; ``store.save``
        # writes it to ``path``). Without the ``save`` the on-disk
        # policy file would still hold the previous version.
        promoter = getattr(store, "promote", None)
        if callable(promoter):
            promoter(best_cand)
        else:
            # Compatibility for minimal PolicyStore test doubles.
            store.set(best_cand)
            store.save(best_cand)
        state["last_promotion_at"] = _now().isoformat()
        state["promotion_count_30d"] = state.get("promotion_count_30d", 0) + 1
        state["consecutive_rollbacks"] = 0  # reset on successful promote
        state["shadow_comparisons"] = []
        # G6.7: update previous_metrics to reflect the now-active
        # policy's evaluation. We use the held-out (test) split
        # eval when available — it's the cleanest signal of what
        # the new active policy actually does in production.
        # Falling back to baseline_eval on the train split keeps
        # the contract "previous_metrics is always populated" even
        # on tiny corpora where the held-out split is empty.
        ref_eval = best_cand_eval if (test_cases and "best_cand_eval" in locals()) else baseline_eval
        new_previous = {
            "useful_at_1": float(ref_eval.useful_at_1),
            "mrr_at_10": float(ref_eval.mrr_at_10),
            "explicit_negative_at_5": float(ref_eval.explicit_negative_at_5),
            "no_result_rate": float(ref_eval.no_result_rate),
            "p95_latency": float(ref_eval.p95_latency),
            "degraded_rate": float(ref_eval.degraded_rate),
        }
        if ref_eval.useful_superseded_fallback_rate is not None:
            new_previous["fallback_useful_rate"] = float(
                ref_eval.useful_superseded_fallback_rate
            )
        state["previous_metrics"] = new_previous
        # Record this cycle as a passed window, then consume the
        # two-window streak: a promotion resets the counter to 0
        # so the next cycle must earn 2 fresh passes before it
        # can promote again. ``pass_windows`` is cleared for the
        # same reason — the streak has been spent.
        state["consecutive_passes"] = 0
        state["pass_candidate_version"] = None
        state["pass_windows"] = []
        _save_evolution_state(state)
        _persist_report("promoted")
        logger.info("evolution: promoted candidate %s to active", best_cand.version)
        return {"status": "promoted", "candidate_version": best_cand.version, "report_id": report_id}

    except Exception as exc:
        logger.error("evolution cycle failed: %s", exc)
        # Best-effort: record this as a failed cycle so the
        # consecutive-pass counter resets. We don't bother if the
        # state file is corrupt — surfacing the exception is more
        # important than perfect bookkeeping.
        try:
            state = _load_evolution_state()
            _record_cycle_result(state, "failed")
            _save_evolution_state(state)
        except Exception:
            pass
        return {"status": "error", "reason": str(exc)}
    finally:
        try:
            fcntl.lockf(lock_fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            lock_fd.close()
        except Exception:
            pass
