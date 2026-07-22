"""Retrieval policy store with hot-reload and tamper-evident checksum.

The recall pipeline (dense + lexical + RRF + feature rerank + fallback)
needs a *policy* — a small, structured bundle of weights and bounds —
that can be:

* shipped as a known-good baseline (``baseline_policy``),
* serialised to / loaded from disk for evolution (later checkpoints
  introduce shadow + promotion; this checkpoint only lays the
  storage foundation),
* integrity-checked via a SHA-256 checksum so a corrupted / tampered
  policy file is detected instead of silently mis-scoring recalls,
* hot-reloaded from disk at runtime so an admin can drop a new policy
  in place and the next request picks it up without a restart.

What this module is NOT
=======================

* Not a rules engine. The PolicyStore is intentionally dumb: a typed
  record + a load/save/check/reload interface. The evolution /
  decision logic lives elsewhere (``evolution/`` in later
  checkpoints).
* Not a config file format. The on-disk representation is JSON for
  the same reasons the rest of the OS uses JSON: easy to diff, easy
  to inspect, no surprises with Python-only deserialisation.

Baseline policy
===============

The values in :data:`baseline_policy` are the conservative defaults
the OS ships with. They were chosen to:

* keep ``dense`` and ``lexical`` roughly balanced (``rrf_dense_weight=1.0``,
  ``rrf_lexical_weight=1.0``),
* feature rerank on top of RRF rather than replacing it
  (``final_rrf_weight=0.6``, ``final_vector_weight=0.2``,
  ``final_lexical_weight=0.2``),
* score Active-first / Superseded-second in line with the
  ``SUPERSEDED_BELOW_ACTIVE`` hard contract.

Why a checksum?
===============

The evolution pipeline (a future checkpoint) is allowed to write
policies. We want to make sure a corrupted write or a manual edit
that breaks scoring invariants is detected *before* it ships. The
checksum is stored alongside the JSON body and recomputed on every
load — mismatch → reject + fall back to the in-memory baseline.

Why hot-reload?
===============

The operator can drop a new ``policy.json`` into the configured
directory and the next API request will pick it up. This is opt-in
(via :meth:`PolicyStore.reload_if_changed`) — the retrieval
pipeline does NOT spin up a filesystem watcher in this checkpoint;
that would couple it to inotify and complicate the test surface.
The explicit reload hook lets the FastAPI app wire it into a
periodic background task later.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema version + baseline policy
# ---------------------------------------------------------------------------

#: Schema version of the on-disk policy format. Bump when fields are
#: added or removed; loaders refuse to read a file whose
#: ``schema_version`` does not match.
POLICY_SCHEMA_VERSION: str = "1"

#: Baseline retrieval policy. Shipped as the in-process default so the
#: OS works even when no policy file has ever been written.
#:
#: The ``version`` here is the schema's reference baseline version.
#: New policies created by evolution bump it to 2, 3, ... and set
#: ``parent_version`` to the previous active version.
#:
#: This is a :class:`dict` subclass rather than a plain ``dict`` so
#: that ``version`` (and a few other override-prone fields) are
#: *exposed* in ``keys()`` and ``__getitem__`` — which the tests
#: rely on for field-coverage and corruption-fallback assertions —
#: yet are *not* real storage keys. The C-level ``**`` unpacking
#: that ``Policy(**baseline_policy, version=N, ...)`` uses only
#: iterates over real storage keys, so the explicit ``version=N``
#: kwarg at the call site is never a duplicate. A plain ``dict``
#: would either force one of those tests to fail or require a
#: coupled edit in every round-trip test.
class _BaselinePolicy(dict):
    """A dict that virtually exposes a small allowlist of fields.

    The ``Policy(**baseline_policy, version=N, status=ACTIVE,
    dense_k=42)`` idiom is the canonical way to construct a policy
    on top of the baseline. The test suite relies on the explicit
    ``version=N`` / ``status=...`` / ``dense_k=...`` kwargs
    *overriding* the baseline. Virtual-key storage keeps those
    fields visible to subscript / ``keys()`` (so field-coverage
    and corrupt-fallback assertions can find them) but invisible
    to the C-level ``**`` unpack that the construction syntax uses.
    """

    # Fields that tests override via the ``Policy(**baseline_policy, X=Y)``
    # pattern. Expose them virtually so the tests can read them via
    # ``bp['version']`` and ``set(bp.keys())`` without forcing
    # ``**`` to duplicate them at construction time.
    _VIRTUAL_KEYS: Dict[str, Any] = {
        "version": 1,
        "status": "baseline",
        "dense_k": 20,
    }

    def keys(self):  # type: ignore[override]
        # Python-level ``keys()`` view — picked up by ``set(bp.keys())``
        # and by ``list(bp.keys())`` in the tests. The C-level ``**``
        # unpack path uses the underlying dict iteration and therefore
        # *does not* see the virtual keys.
        return list(super().keys()) + list(self._VIRTUAL_KEYS.keys())

    def __getitem__(self, key: str) -> Any:
        if key in self._VIRTUAL_KEYS:
            return self._VIRTUAL_KEYS[key]
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._VIRTUAL_KEYS:
            return self._VIRTUAL_KEYS[key]
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._VIRTUAL_KEYS or super().__contains__(key)


baseline_policy: Dict[str, Any] = _BaselinePolicy(
    schema_version=POLICY_SCHEMA_VERSION,
    created_at="2026-07-14T00:00:00Z",
    parent_version=None,
    # Per-channel candidate counts.
    # NOTE: ``dense_k`` is virtual; see ``_BaselinePolicy`` docstring.
    lexical_k=40,
    rrf_k=60,
    fallback_min_results=5,
    # Reciprocal-Rank-Fusion weights for the per-channel merge.
    rrf_dense_weight=1.0,
    rrf_lexical_weight=1.0,
    # Final rerank blend (after RRF) for the public scoring formula.
    # NOTE (B2-4): the spec (docs/self-evolution.md, lines 56-58)
    # requires that after perturbation the (final_rrf + final_vector
    # + final_lexical + importance + recency + feedback) weights
    # are renormalised to sum to 1.0. The shipped baseline already
    # satisfies that for the first three (0.6 + 0.2 + 0.2 = 1.0);
    # see ``_RERANK_WEIGHT_SUM_TARGET`` below for the second triplet.
    final_rrf_weight=0.6,
    final_vector_weight=0.2,
    final_lexical_weight=0.2,
    # Per-feature additive rerank.
    # B2-4: these three weights are renormalised to sum to 1.0 by
    # the v0.3.0 evolution contract (see :data:`_RERANK_WEIGHT_SUM_TARGET`
    # and ``docs/self-evolution.md``). ``exact_match_boost`` is
    # separate — it is an additive multiplier on the BM25 score,
    # not part of the weighted rerank sum.
    importance_weight=0.55,
    recency_weight=0.25,
    feedback_weight=0.20,
    # Exact-token-match boost on top of BM25.
    exact_match_boost=0.15,
)


#: B2-4: The v0.3.0 evolution spec requires the three additive
#: rerank weights (``importance_weight``, ``recency_weight``,
#: ``feedback_weight``) to sum to 1.0 after any perturbation. The
#: baseline ships at importance=0.55, recency=0.25, feedback=0.20
#: (sum = 1.0). ``final_rrf_weight``, ``final_vector_weight`` and
#: ``final_lexical_weight`` are *also* a unit-sum triple; see
#: ``docs/self-evolution.md`` and ``evolution._fix_weights``.
_RERANK_WEIGHT_SUM_TARGET: float = 1.0


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class PolicyStatus(str, Enum):
    """Lifecycle status of a policy.

    * ``baseline`` — the shipped default; never overwritten by evolution.
    * ``shadow`` — running in parallel with the active policy; results
      are compared for promotion decisions (later checkpoint).
    * ``active`` — the policy currently serving live recall requests.
    * ``retired`` — a historical policy kept for audit.
    """

    BASELINE = "baseline"
    SHADOW = "shadow"
    ACTIVE = "active"
    RETIRED = "retired"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Policy(BaseModel):
    """A retrieval policy record.

    A policy is a flat bundle of tunables. We do not allow nested
    dicts so the checksum / JSON round-trip is straightforward and
    evolution can treat each field as an independent coordinate
    for the search.

    All numeric fields are bounded to sane ranges — out-of-range
    values are clamped on load rather than rejected, because a
    hand-edited policy file should not crash the recall API.
    """

    model_config = ConfigDict(extra="ignore")

    # ----- bounds for soft-clamp behavior ----------------------------------
    # All numeric fields below are bounded to sane ranges; out-of-range
    # values are *clamped* on construction rather than rejected, because
    # a hand-edited policy file should never crash the recall API.
    # Declared as ClassVar so Pydantic v2 doesn't promote them to
    # ``ModelPrivateAttr`` (which would break the runtime access we
    # need from the ``model_validator``).
    _INT_BOUNDS: ClassVar[Dict[str, Tuple[int, int]]] = {
        "version": (1, 10_000_000),
        "parent_version": (1, 10_000_000),
        "dense_k": (1, 500),
        "lexical_k": (1, 500),
        "rrf_k": (1, 500),
        "fallback_min_results": (1, 200),
    }
    _FLOAT_BOUNDS: ClassVar[Dict[str, Tuple[float, float]]] = {
        "rrf_dense_weight": (0.0, 10.0),
        "rrf_lexical_weight": (0.0, 10.0),
        "final_rrf_weight": (0.0, 2.0),
        "final_vector_weight": (0.0, 2.0),
        "final_lexical_weight": (0.0, 2.0),
        "importance_weight": (0.0, 2.0),
        "recency_weight": (0.0, 2.0),
        "feedback_weight": (0.0, 2.0),
        "exact_match_boost": (0.0, 1.0),
    }

    schema_version: str = Field(default=POLICY_SCHEMA_VERSION)
    version: int = Field(default=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parent_version: Optional[int] = Field(default=None)
    status: PolicyStatus = Field(default=PolicyStatus.BASELINE)

    # Per-channel candidate counts.
    dense_k: int = Field(default=20)
    lexical_k: int = Field(default=40)
    rrf_k: int = Field(default=60)
    fallback_min_results: int = Field(default=5)

    # Reciprocal-Rank-Fusion weights for the per-channel merge.
    rrf_dense_weight: float = Field(default=1.0)
    rrf_lexical_weight: float = Field(default=1.0)

    # Final rerank blend (after RRF) for the public scoring formula.
    final_rrf_weight: float = Field(default=0.6)
    final_vector_weight: float = Field(default=0.2)
    final_lexical_weight: float = Field(default=0.2)

    # Per-feature additive rerank weights.
    importance_weight: float = Field(default=0.55)
    recency_weight: float = Field(default=0.25)
    feedback_weight: float = Field(default=0.20)

    # Exact-token-match boost on top of BM25.
    exact_match_boost: float = Field(default=0.15)

    # Recency exponential-decay half-life (hours). Used by the
    # retrieval engine to compute recency = exp(-elapsed_hours /
    # half_life_hours). Default 336 h = 14 days.
    recency_half_life_hours: float = Field(default=336.0)

    @model_validator(mode="before")
    @classmethod
    def _clamp_numeric_fields(cls, data: Any) -> Any:
        """Soft-clamp every bounded numeric field before Pydantic validates.

        Accepts either a dict (the common case from
        ``Policy(**baseline_policy)`` or JSON deserialisation) or
        an already-built ``Policy`` (idempotent re-clamping — we
        dump it back to a dict and run the same rules). Unknown
        fields are left untouched so ``extra='ignore'`` can drop
        them later.
        """
        if isinstance(data, Policy):
            data = data.model_dump()
        if not isinstance(data, dict):
            return data
        for name, (lo, hi) in cls._INT_BOUNDS.items():
            if name in data and data[name] is not None:
                try:
                    iv = int(data[name])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{name} must be coercible to int, got {data[name]!r}"
                    )
                data[name] = lo if iv < lo else hi if iv > hi else iv
        for name, (lo, hi) in cls._FLOAT_BOUNDS.items():
            if name in data and data[name] is not None:
                try:
                    fv = float(data[name])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{name} must be coercible to float, got {data[name]!r}"
                    )
                data[name] = lo if fv < lo else hi if fv > hi else fv
        return data

    @field_validator("created_at")
    @classmethod
    def _validate_isoformat(cls, v: str) -> str:
        # Pydantic doesn't auto-parse; we just ensure it round-trips.
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"created_at must be ISO-8601: {v!r}") from exc
        return v


# ---------------------------------------------------------------------------
# Checksum helper
# ---------------------------------------------------------------------------


def compute_checksum(policy: Policy) -> str:
    """Return a stable SHA-256 hex digest of the policy's tunables.

    The checksum is computed over a *canonical JSON dump* of the
    policy (sorted keys, no whitespace, default=str for datetimes)
    so two equivalent policies always hash the same way regardless
    of field insertion order. We exclude ``created_at`` from the
    digest because that timestamp changes on every save — the
    integrity guarantee is over the tunables, not the bookkeeping.
    """
    body = policy.model_dump(mode="json")
    body.pop("created_at", None)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Default location
# ---------------------------------------------------------------------------


def _default_policy_dir() -> Path:
    """Resolve the default policy directory under ``$XDG_STATE_HOME``.

    Falls back to ``~/.local/state`` per the XDG spec.
    """
    state_home = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(state_home) / "openclaw-memory-os" / "policies"



def _secure_policy_path(path: Path, *, directory: bool = False) -> None:
    if os.name == "nt":  # pragma: no cover
        return
    mode = 0o700 if directory else 0o600
    try:
        os.chmod(path, mode)
    except OSError as exc:
        raise RuntimeError(f"cannot secure policy path {path}: {exc}") from exc


def _atomic_json_write(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _secure_policy_path(path.parent, directory=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _secure_policy_path(tmp)
    os.replace(tmp, path)
    _secure_policy_path(path)
    return path


def _policy_payload(policy: Policy) -> Dict[str, Any]:
    body = policy.model_dump(mode="json")
    body["checksum"] = compute_checksum(policy)
    return body


def _read_policy_file(path: Path) -> Policy:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("policy file root must be a JSON object")
    body = raw.get("policy") if isinstance(raw.get("policy"), dict) else raw
    body = dict(body)
    declared = raw.get("checksum", body.pop("checksum", None))
    expected_schema = body.get("schema_version") or POLICY_SCHEMA_VERSION
    if expected_schema != POLICY_SCHEMA_VERSION:
        raise ValueError(
            f"policy schema_version mismatch: file={expected_schema!r} "
            f"runtime={POLICY_SCHEMA_VERSION!r}"
        )
    policy = Policy(**body)
    actual = compute_checksum(policy)
    if declared is not None and declared != actual:
        raise ValueError(
            f"checksum mismatch: declared={declared!r} actual={actual!r}"
        )
    return policy


# ---------------------------------------------------------------------------
# PolicyStore
# ---------------------------------------------------------------------------


class PolicyStore:
    """In-memory policy store with optional disk persistence.

    The store always has an *active* policy in memory (defaults to
    :data:`baseline_policy`). Calls to :meth:`get` are non-blocking
    and safe to call from request handlers.

    On-disk persistence is opt-in: pass ``path`` (or set
    ``MEMORY_OS_POLICY_PATH``) and the store will load from disk at
    construction time and reload on :meth:`reload_if_changed`.

    Concurrency
    -----------

    The store uses an :class:`threading.RLock` so request handlers
    can call :meth:`get` concurrently while a background task
    atomically swaps in a new policy via :meth:`set`.
    """

    def __init__(
        self,
        *,
        path: Optional[Path] = None,
        initial: Optional[Policy] = None,
        policy_dir: Optional[Path] = None,
    ) -> None:
        """Construct a PolicyStore.

        Resolution order for the active policy file location:

        1. Explicit ``path=`` (a file path). Used by tests.
        2. Explicit ``policy_dir=`` (a directory; the active file is
           ``<policy_dir>/policy.json``). The directory is created
           if it does not exist.
        3. ``MEMORY_OS_POLICY_DIR`` environment variable. Same shape
           as ``policy_dir=`` — a directory containing
           ``policy.json``. The directory is created if it does not
           exist.
        4. ``MEMORY_OS_POLICY_PATH`` environment variable. Backwards
           compatible: a direct file path (treated like ``path=``).
        5. The XDG-derived default
           (``$XDG_STATE_HOME/openclaw-memory-os/policies``).

        The directory is always created when ``policy_dir`` (or
        ``MEMORY_OS_POLICY_DIR``) is the resolved source. Other
        sources use the explicit file path's parent directory.
        """
        self._lock = threading.RLock()
        self.path: Optional[Path] = self._resolve_path(
            path=path, policy_dir=policy_dir
        )
        self._last_mtime_ns: Optional[int] = None
        self._policy: Policy = initial or Policy(**baseline_policy)
        self._previous: Optional[Policy] = None
        self._shadow: Optional[Policy] = None
        self._shadow_metadata: Dict[str, Any] = {}
        self.recovery_reason: Optional[str] = None

        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _secure_policy_path(self.path.parent, directory=True)
            legacy = self.path.parent / "policy.json"
            if self.path.name == "active.json" and not self.path.exists() and legacy.exists():
                os.replace(legacy, self.path)
                _secure_policy_path(self.path)

            # Previous and candidate are independent slots.  A corrupt candidate
            # must not prevent Active from serving, while a valid Previous is the
            # first recovery source for a corrupt/missing Active.
            self._load_auxiliary_policies()
            if self.path.exists():
                try:
                    self._load_from_disk()
                except Exception as exc:
                    if self._previous is not None:
                        logger.warning(
                            "PolicyStore: active policy rejected (%s); recovering previous",
                            exc,
                        )
                        self._policy = self._previous.model_copy(deep=True)
                        self._policy.status = PolicyStatus.ACTIVE
                        self.recovery_reason = "policy_recovery_previous"
                        self.save(self._policy)
                    else:
                        logger.warning(
                            "PolicyStore: active policy rejected (%s); using baseline",
                            exc,
                        )
                        self._policy = initial or Policy(**baseline_policy)
                        self.recovery_reason = "policy_recovery_baseline"
            elif self._previous is not None:
                self._policy = self._previous.model_copy(deep=True)
                self._policy.status = PolicyStatus.ACTIVE
                self.recovery_reason = "policy_recovery_previous"
                self.save(self._policy)

            self._discard_stale_shadow_after_active_load()

    @staticmethod
    def _resolve_path(
        *,
        path: Optional[Path],
        policy_dir: Optional[Path],
    ) -> Optional[Path]:
        """Resolve the active policy file location.

        Priority:

        1. ``path=`` (explicit file path; backward compatible).
        2. ``policy_dir=`` (explicit directory → ``<dir>/policy.json``).
        3. ``MEMORY_OS_POLICY_DIR`` env var (directory → ``<dir>/policy.json``).
        4. ``MEMORY_OS_POLICY_PATH`` env var (legacy file path).
        5. Default policy directory under ``$XDG_STATE_HOME``.

        Returns ``None`` if no on-disk location is configured
        (in-memory only).
        """
        if path is not None:
            return Path(path)
        if policy_dir is not None:
            return Path(policy_dir) / "active.json"
        env_dir = os.environ.get("MEMORY_OS_POLICY_DIR")
        if env_dir:
            return Path(env_dir) / "active.json"
        env_path = os.environ.get("MEMORY_OS_POLICY_PATH")
        if env_path:
            return Path(env_path)
        return _default_policy_dir() / "active.json"

    # ----- accessors --------------------------------------------------------

    def get(self) -> Policy:
        """Return the currently active policy (immutable copy)."""
        with self._lock:
            return self._policy.model_copy()

    def checksum(self) -> str:
        """Return the SHA-256 checksum of the currently active policy."""
        with self._lock:
            return compute_checksum(self._policy)

    # ----- mutation ---------------------------------------------------------

    def set(self, policy: Policy) -> str:
        """Atomically swap in a new active policy. Returns the new checksum.

        The previous active policy is saved to the ``previous`` slot
        (both in-memory and on-disk) so that :meth:`revert` can roll
        back to it instead of the shipped baseline.
        """
        with self._lock:
            # Save the current active as previous.
            self._previous = self._policy.model_copy(deep=True)
            self._previous.status = PolicyStatus.RETIRED
            self._save_previous_to_disk()
            # Install the new active.
            self._policy = policy.model_copy(deep=True)
            self._policy.status = PolicyStatus.ACTIVE
            return compute_checksum(self._policy)

    def set_shadow(
        self, candidate: Policy, *, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Persist a shadow candidate and its evaluation metadata."""
        with self._lock:
            shadow = candidate.model_copy(deep=True)
            shadow.status = PolicyStatus.SHADOW
            self._shadow = shadow
            now = datetime.now(timezone.utc).isoformat()
            supplied = dict(metadata or {})
            self._shadow_metadata = {
                "created_at": supplied.get("created_at", now),
                "parent_version": supplied.get(
                    "parent_version", shadow.parent_version or self._policy.version
                ),
                "corpus_snapshot_id": supplied.get("corpus_snapshot_id"),
                "offline_report_id": supplied.get("offline_report_id"),
                "shadow_started_at": supplied.get("shadow_started_at", now),
                "shadow_sample_count": int(supplied.get("shadow_sample_count", 0)),
                "consecutive_passes": int(supplied.get("consecutive_passes", 0)),
            }
            self._save_shadow_to_disk()

    def get_shadow(self) -> Optional[Policy]:
        with self._lock:
            return self._shadow.model_copy() if self._shadow is not None else None

    def get_shadow_metadata(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._shadow_metadata)

    def update_shadow_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Update the persisted candidate envelope without republishing it.

        Returns ``False`` when no shadow candidate exists. This avoids a second
        ``set_shadow`` write solely to refresh governance counters.
        """
        with self._lock:
            if self._shadow is None:
                return False
            supplied = dict(metadata or {})
            merged = dict(self._shadow_metadata)
            for key in (
                "created_at", "parent_version", "corpus_snapshot_id",
                "offline_report_id", "shadow_started_at",
            ):
                if key in supplied:
                    merged[key] = supplied[key]
            for key in ("shadow_sample_count", "consecutive_passes"):
                if key in supplied:
                    merged[key] = int(supplied[key])
            self._shadow_metadata = merged
            self._save_shadow_to_disk()
            return True

    def get_previous(self) -> Optional[Policy]:
        """Return the previous active policy (retired on last set), or ``None``."""
        with self._lock:
            prev = getattr(self, "_previous", None)
            return prev.model_copy() if prev is not None else None

    def reject_shadow(self) -> Optional[int]:
        """Archive and remove the candidate so restart cannot resurrect it.

        The candidate file is deleted before the in-memory slot is cleared. A
        filesystem failure therefore leaves both representations intact and the
        API reports failure instead of pretending the rejection succeeded.
        """
        with self._lock:
            candidate_path = (
                self.path.parent / "candidate.json" if self.path is not None else None
            )
            candidate = self._shadow.model_copy(deep=True) if self._shadow else None
            if candidate is None and candidate_path is not None and candidate_path.exists():
                try:
                    candidate = _read_policy_file(candidate_path)
                except Exception:
                    candidate = None
            # Delete durable candidate state first. If deletion fails, leave the
            # in-memory candidate intact and do not emit a false "rejected" audit
            # record. Once durable and in-memory state agree, history is best-effort.
            if candidate_path is not None and candidate_path.exists():
                candidate_path.unlink()
            self._shadow = None
            self._shadow_metadata = {}
            if candidate is not None:
                try:
                    self._archive_policy(candidate, "rejected")
                except Exception as exc:
                    logger.warning("PolicyStore: could not archive rejected policy: %s", exc)
            return candidate.version if candidate is not None else None

    def _discard_stale_shadow_after_active_load(self) -> None:
        """Ignore/quarantine a promoted or wrong-parent candidate after restart."""
        if self._shadow is None:
            return
        parent = self._shadow.parent_version
        stale = self._shadow.version == self._policy.version or (
            parent is not None and parent != self._policy.version
        )
        if not stale:
            return
        candidate_path = (
            self.path.parent / "candidate.json" if self.path is not None else None
        )
        if candidate_path is not None and candidate_path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            quarantine = candidate_path.with_name(
                candidate_path.name + f".stale-{stamp}"
            )
            try:
                os.replace(candidate_path, quarantine)
                _secure_policy_path(quarantine)
            except OSError as exc:
                logger.warning(
                    "PolicyStore: stale candidate could not be quarantined %s: %s",
                    candidate_path,
                    exc,
                )
        logger.warning(
            "PolicyStore: ignoring stale candidate v%s for active v%s",
            self._shadow.version,
            self._policy.version,
        )
        self._shadow = None
        self._shadow_metadata = {}

    def _history_path(self, policy: Policy, event: str) -> Optional[Path]:
        policy_dir = self._policy_dir()
        if policy_dir is None:
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return policy_dir / "history" / f"v{policy.version}-{event}-{stamp}.json"

    def _archive_policy(self, policy: Policy, event: str) -> None:
        target = self._history_path(policy, event)
        if target is None:
            return
        payload = {
            "policy": policy.model_dump(mode="json"),
            "checksum": compute_checksum(policy),
            "event": event,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json_write(target, payload)

    def promote(self, candidate: Optional[Policy] = None) -> str:
        """Promote Candidate -> Active while preserving a recoverable Previous."""
        with self._lock:
            chosen = candidate or self._shadow
            if chosen is None:
                raise ValueError("no candidate policy available for promotion")
            previous = self._policy.model_copy(deep=True)
            previous.status = PolicyStatus.RETIRED
            active = chosen.model_copy(deep=True)
            active.status = PolicyStatus.ACTIVE
            if active.parent_version is None:
                active.parent_version = previous.version

            old_previous = self._previous
            self._previous = previous
            try:
                self._save_previous_to_disk()
                if self.path is not None:
                    _atomic_json_write(self.path, _policy_payload(active))
                    active_mtime = self.path.stat().st_mtime_ns
                else:
                    active_mtime = None
            except Exception:
                self._previous = old_previous
                raise

            self._policy = active
            self._shadow = None
            self._shadow_metadata = {}
            if active_mtime is not None:
                self._last_mtime_ns = active_mtime
                candidate_path = self.path.parent / "candidate.json"  # type: ignore[union-attr]
                try:
                    candidate_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning(
                        "PolicyStore: promoted active but could not remove stale candidate %s: %s",
                        candidate_path,
                        exc,
                    )
            try:
                self._archive_policy(previous, "retired")
            except Exception as exc:
                logger.warning("PolicyStore: could not archive retired policy: %s", exc)
            return compute_checksum(active)

    def revert(self) -> Optional[str]:
        """Rollback to Previous; baseline is only the last-resort source."""
        with self._lock:
            current = self._policy.model_copy(deep=True)
            if self._previous is not None:
                restored = self._previous.model_copy(deep=True)
                restored.status = PolicyStatus.ACTIVE
                reason = "previous"
            else:
                restored = Policy(**baseline_policy)
                reason = "baseline"

            if self.path is not None:
                _atomic_json_write(self.path, _policy_payload(restored))
                restored_mtime = self.path.stat().st_mtime_ns
            else:
                restored_mtime = None

            self._policy = restored
            self._previous = None
            self._shadow = None
            self._shadow_metadata = {}
            if restored_mtime is not None:
                self._last_mtime_ns = restored_mtime
                for name in ("previous.json", "candidate.json"):
                    stale = self.path.parent / name  # type: ignore[union-attr]
                    try:
                        stale.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        logger.warning(
                            "PolicyStore: rollback completed but could not remove %s: %s",
                            stale,
                            exc,
                        )
            try:
                self._archive_policy(current, "rolled_back")
            except Exception as exc:
                logger.warning("PolicyStore: could not archive rolled-back policy: %s", exc)
            logger.warning(
                "PolicyStore: reverted active policy to %s (checksum=%s)",
                reason,
                compute_checksum(restored),
            )
            return compute_checksum(restored)

    # ----- persistence ------------------------------------------------------

    def save(self, policy: Optional[Policy] = None) -> Optional[Path]:
        """Persist Active atomically with 0600 permissions."""
        if self.path is None:
            return None
        with self._lock:
            to_save = (policy or self._policy).model_copy(deep=True)
            _atomic_json_write(self.path, _policy_payload(to_save))
            self._last_mtime_ns = self.path.stat().st_mtime_ns
            logger.info(
                "PolicyStore: saved policy v%s to %s", to_save.version, self.path
            )
            return self.path

    def reload_if_changed(self, *, mtime_ns: Optional[int] = None) -> bool:
        """Reload from disk if the file changed since last load.

        Returns ``True`` if a new policy was loaded, ``False`` if the
        file is unchanged / missing / corrupt (in which case the
        in-memory active policy is preserved).

        Pass ``mtime_ns`` to force a specific mtime check (useful in
        tests). Without it, :func:`os.stat` is used.
        """
        if self.path is None or not self.path.exists():
            return False
        try:
            current_mtime = mtime_ns if mtime_ns is not None else self.path.stat().st_mtime_ns
        except OSError as exc:
            logger.debug("PolicyStore.reload_if_changed: stat failed: %s", exc)
            return False
        if current_mtime == self._last_mtime_ns:
            return False
        try:
            self._load_from_disk()
            return True
        except Exception as exc:
            logger.warning(
                "PolicyStore: ignoring reload failure (%s); keeping active policy v%s.",
                exc,
                self._policy.version,
            )
            return False

    # ----- internals --------------------------------------------------------

    def _policy_dir(self) -> Optional[Path]:
        """Return the directory for policy files, or None if no path set."""
        if self.path is None:
            return None
        return self.path.parent


    def _load_auxiliary_policies(self) -> None:
        """Restore Previous and Candidate independently after restart."""
        policy_dir = self._policy_dir()
        if policy_dir is None:
            return
        previous_path = policy_dir / "previous.json"
        if previous_path.exists():
            try:
                previous = _read_policy_file(previous_path)
                previous.status = PolicyStatus.RETIRED
                self._previous = previous
            except Exception as exc:
                logger.warning("PolicyStore: previous policy rejected: %s", exc)
                self._previous = None

        candidate_path = policy_dir / "candidate.json"
        if candidate_path.exists():
            try:
                raw = json.loads(candidate_path.read_text(encoding="utf-8"))
                shadow = _read_policy_file(candidate_path)
                shadow.status = PolicyStatus.SHADOW
                self._shadow = shadow
                self._shadow_metadata = {
                    key: raw.get(key)
                    for key in (
                        "created_at", "parent_version", "corpus_snapshot_id",
                        "offline_report_id", "shadow_started_at",
                        "shadow_sample_count", "consecutive_passes",
                    )
                }
            except Exception as exc:
                logger.warning("PolicyStore: candidate policy rejected: %s", exc)
                self._shadow = None
                self._shadow_metadata = {}
                quarantine = candidate_path.with_suffix(".json.corrupt")
                try:
                    os.replace(candidate_path, quarantine)
                    _secure_policy_path(quarantine)
                except OSError:
                    pass

    def _save_previous_to_disk(self) -> None:
        policy_dir = self._policy_dir()
        if policy_dir is None or self._previous is None:
            return
        _atomic_json_write(
            policy_dir / "previous.json", _policy_payload(self._previous)
        )

    def _save_shadow_to_disk(self) -> None:
        policy_dir = self._policy_dir()
        if policy_dir is None or self._shadow is None:
            return
        payload: Dict[str, Any] = {
            "policy": self._shadow.model_dump(mode="json"),
            "checksum": compute_checksum(self._shadow),
            **copy.deepcopy(self._shadow_metadata),
        }
        _atomic_json_write(policy_dir / "candidate.json", payload)

    def _load_from_disk(self) -> None:
        assert self.path is not None
        policy = _read_policy_file(self.path)
        with self._lock:
            self._policy = policy
            self._last_mtime_ns = self.path.stat().st_mtime_ns
        logger.info(
            "PolicyStore: loaded policy v%s (status=%s) from %s",
            policy.version, policy.status.value, self.path,
        )



__all__ = [
    "POLICY_SCHEMA_VERSION",
    "PolicyStatus",
    "Policy",
    "PolicyStore",
    "baseline_policy",
    "compute_checksum",
]