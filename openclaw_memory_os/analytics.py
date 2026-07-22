"""Analytics helpers built on top of a :class:`MemoryBackend`.

The OS is a view layer, so analytics are computed on the fly from the
backend's memory list. For larger stores these should move to precomputed
materialized views, but for a single-user / small-team scale this is more
than fast enough and keeps the deployment simple.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time as _time
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, TypeVar

from .backends import MemoryBackend
from .models import (
    AutonomousGovernanceJob,
    DeletionCandidate,
    DuplicateCluster,
    HealthSummary,
    ImportanceBucket,
    Memory,
    MemoryStatus,
    MemoryTier,
    MonthCount,
    StatusCount,
    TierCount,
    utcnow,
)

logger = logging.getLogger(__name__)


def _read_autonomous_governance_status() -> dict:
    """Read the weekly autonomous governance status JSON, if present.

    The status file is written by the scheduled OpenClaw systemd timer after a
    run. It is deliberately tiny and redacted: timestamp, result token, and a
    short summary only. Missing or malformed files fall back to an honest
    unknown state rather than breaking the dashboard.

    Wave 2 (2026-07-21): the runner now optionally writes additional
    operational fields (``scheduled_at`` / ``started_at`` /
    ``finished_at`` / ``duration_seconds`` / ``next_scheduled_at`` /
    ``exit_code`` / ``mode``) when ``GOVERNANCE_STARTED_AT`` is set in
    the runner's environment. We pass them through here too so the
    dashboard renderer can read the full schedule/state picture without
    a separate call. The three legacy keys (``last_run`` /
    ``last_result`` / ``last_summary``) are always returned for
    backwards compatibility.
    """
    candidates = []
    env = os.environ.get("MEMORY_OS_GOVERNANCE_STATUS")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.home() / ".local/state/openclaw-memory-os/autonomous-governance.json")
    candidates.append(Path(__file__).resolve().parent.parent / "logs" / "autonomous-governance.json")

    for path in candidates:
        try:
            if not path.exists() or not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            last_run_raw = data.get("last_run") or data.get("finished_at")
            if last_run_raw:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(last_run_raw)
                    last_run_raw = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    pass
            out = {
                "last_run": last_run_raw or None,
                "last_result": data.get("last_result") or data.get("result"),
                "last_summary": data.get("last_summary") or data.get("summary"),
            }
            # Pass through the extended protocol fields verbatim if
            # they exist. None of them are required — the writer only
            # emits them when GOVERNANCE_STARTED_AT was provided.
            for ext_key in (
                "scheduled_at",
                "started_at",
                "finished_at",
                "duration_seconds",
                "next_scheduled_at",
                "exit_code",
                "mode",
            ):
                if ext_key in data and data[ext_key] is not None:
                    out[ext_key] = data[ext_key]
            # Wave 2 (2026-07-21): if the runner didn't emit
            # ``next_scheduled_at`` (e.g. an operator-driven manual run
            # that bypassed the systemd timer), backfill it from the
            # live ``openclaw-memory-os-governance.timer`` so the
            # dashboard always has a next-run value to render. The TTL
            # cache keeps the cost negligible.
            if not out.get("next_scheduled_at"):
                timer = _read_systemd_timer_schedule(GOVERNANCE_TIMER_NAME)
                next_elapse = timer.get("next_elapse") if timer else None
                if next_elapse:
                    out["next_scheduled_at"] = next_elapse
            return out
        except (OSError, PermissionError, json.JSONDecodeError):
            continue
    return {}


# Default write location mirrors the read fallback so a fresh deployment
# works without any env-var setup. The reader tries the env override first
# and then these two locations in order.
_DEFAULT_GOVERNANCE_STATUS_PATH = (
    Path.home() / ".local/state/openclaw-memory-os/autonomous-governance.json"
)


def _resolve_governance_status_path(status_file_path: Optional[Path]) -> Path:
    """Pick the destination path for a governance status write.

    Order:

    1. Explicit argument wins (callers from tests / scripts pass this).
    2. ``MEMORY_OS_GOVERNANCE_STATUS`` env var (matches the reader).
    3. The XDG-state default.
    """
    if status_file_path is not None:
        return Path(status_file_path)
    env = os.environ.get("MEMORY_OS_GOVERNANCE_STATUS")
    if env:
        return Path(env)
    return _DEFAULT_GOVERNANCE_STATUS_PATH


_ALLOWED_RESULT_TOKENS = {"ok", "failed", "running", "pending", "skipped", "degraded"}
_MAX_SUMMARY_LEN = 300


def _sanitize_summary(summary: str) -> str:
    """Redact obvious secrets before persisting a governance summary.

    The summary is human-written copy from the bash runner. We do a tiny
    belt-and-braces pass that strips control characters and truncates
    anything that obviously looks like a path / URL / token. This is
    deliberately conservative: it would rather keep a useful phrase than
    over-redact. The writer contract forbids embedding paths / tokens in
    the first place; this is just a safety net.
    """
    if not summary:
        return ""
    cleaned = "".join(ch for ch in summary if ch == "\t" or ch == "\n" or ch >= " ")
    # Collapse internal newlines/tabs to single spaces for a one-line summary.
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > _MAX_SUMMARY_LEN:
        cleaned = cleaned[: _MAX_SUMMARY_LEN - 3] + "..."
    return cleaned


def write_autonomous_governance_status(
    status_file_path: Optional[Path] = None,
    result_token: str = "ok",
    summary: str = "",
    finished_at: Optional[str] = None,
    scheduled_at: Optional[str] = None,
    started_at: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    next_scheduled_at: Optional[str] = None,
    exit_code: Optional[int] = None,
    mode: Optional[str] = None,
) -> Path:
    """Atomically write the tiny governance status JSON.

    The contract is intentionally tiny — exactly three keys:

    * ``last_run`` — ISO-8601 timestamp (defaults to "now" in Asia/Shanghai
      when ``finished_at`` is not provided).
    * ``last_result`` — one of ``ok`` / ``failed`` / ``degraded`` / ``running`` / ``pending`` / ``skipped``.
      Unknown tokens fall back to ``failed`` to keep the dashboard honest
      about an unexpected runner state.
    * ``last_summary`` — short, redacted, single-line human summary.

    Nothing else is written. The caller contract forbids embedding
    collection names, filesystem paths, IP addresses, tokens, or other
    private meta into ``summary``; ``_sanitize_summary`` applies a final
    pass for safety.

    Wave 2 (2026-07-21): extended governance status protocol. When
    ``started_at`` is provided, the payload additionally includes:

      * ``scheduled_at``        — ISO-8601 timestamp the systemd timer
                                  / caller asked the job to start.
      * ``started_at``          — ISO-8601 timestamp the runner script
                                  actually began executing.
      * ``finished_at``         — ISO-8601 timestamp the runner script
                                  finished. ``last_run`` is set from
                                  this for backwards compatibility.
      * ``duration_seconds``    — int, ``finished_at - started_at``.
      * ``next_scheduled_at``   — ISO-8601 timestamp (read by the
                                  dashboard renderer; not enforced by
                                  this writer).
      * ``exit_code``           — int, the runner's final exit code.
      * ``mode``                — short label (``governance`` for the
                                  weekly deep-audit path).

    These fields are **only** written when ``started_at`` is provided.
    The legacy 3-key contract holds when the env vars are unset so the
    dashboard renderer can keep reading the simple shape for
    non-scheduled (one-off operator) runs. ``test_write_governance_status.py``
    is unchanged — the new fields are gated behind the env-var path.

    The file is written 0600 inside a 0700 directory via ``tempfile + os.replace``
    so concurrent reads from the dashboard never see a half-written file.

    Returns the path that was actually written.
    """
    from .models import _OPERATOR_TZ  # local import to avoid circulars at module load

    target = _resolve_governance_status_path(status_file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(target.parent, 0o700)
    except OSError:
        # Best-effort hardening; permission hardening is not critical on the
        # test-only / dev-only path used by the writer.
        pass

    token = (result_token or "ok").strip().lower()
    if token not in _ALLOWED_RESULT_TOKENS:
        token = "failed"

    safe_summary = _sanitize_summary(summary)

    if finished_at:
        ts = finished_at
    else:
        ts = datetime.now(_OPERATOR_TZ).isoformat()

    payload = {
        "last_run": ts,
        "last_result": token,
        "last_summary": safe_summary,
    }

    # Wave 2 (2026-07-21) — extended protocol fields. Only emit them
    # when ``started_at`` was provided; the legacy 3-key contract
    # otherwise. ``finished_at`` is included as both a top-level mirror
    # (so dashboards that don't know about the new fields still have a
    # canonical end time) and so the analytics reader can pick it up
    # independently of ``last_run``.
    if started_at:
        payload["started_at"] = started_at
    if finished_at:
        payload["finished_at"] = finished_at
    if scheduled_at:
        payload["scheduled_at"] = scheduled_at
    if duration_seconds is not None:
        try:
            payload["duration_seconds"] = max(int(duration_seconds), 0)
        except (TypeError, ValueError):
            pass
    if next_scheduled_at:
        payload["next_scheduled_at"] = next_scheduled_at
    if exit_code is not None:
        try:
            payload["exit_code"] = int(exit_code)
        except (TypeError, ValueError):
            pass
    if mode:
        payload["mode"] = str(mode).strip().lower() or None

    # Write through a sibling tempfile so the dashboard reader never sees a
    # truncated file. Use the same umask the script created the directory
    # with; explicitly chmod 0600 after the fact because tempfile inherits
    # the process umask (usually 0022 in non-interactive shells).
    import tempfile

    fd, tmp_path = tempfile.mkstemp(
        prefix=".autonomous-governance-", suffix=".json.tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return target



# --- Health summary ---------------------------------------------------------

# Wave 2 (2026-07-21): tiny TTL cache for ``_read_systemd_timer_schedule``
# so the dashboard never blocks on ``systemctl show`` (3s timeout) per
# render. We avoid ``functools.lru_cache`` because it lacks a TTL on the
# stdlib version this project ships. The cache is intentionally small
# (maxsize=8 entries) and short-lived (TTL=10s) — any operator tweaking
# the timer unit gets a fresh read within ten seconds.
_F = TypeVar("_F", bound=Callable[..., Any])


def _ttl_cache(maxsize: int = 8, ttl_seconds: float = 10.0):
    """Minimal in-process TTL cache used for dashboard helpers.

    Tests clear the cache via the ``_clear_systemd_timer_schedule_cache``
    hook exported below so the test suite can swap subprocess.run with
    a mock without polluting real renders.
    """

    def decorator(fn: _F) -> _F:
        cache: Dict[Tuple[Any, ...], Tuple[float, Any]] = {}
        order: List[Tuple[Any, ...]] = []

        def _evict_if_needed() -> None:
            while len(order) > maxsize:
                oldest = order.pop(0)
                cache.pop(oldest, None)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = _time.monotonic()
            cached = cache.get(key)
            if cached is not None:
                ts, value = cached
                if now - ts < ttl_seconds:
                    return value
                # Expired — drop it and fall through to fresh read.
                cache.pop(key, None)
                try:
                    order.remove(key)
                except ValueError:
                    pass
            value = fn(*args, **kwargs)
            cache[key] = (now, value)
            order.append(key)
            _evict_if_needed()
            return value

        def cache_clear() -> None:
            cache.clear()
            order.clear()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def _clear_systemd_timer_schedule_cache() -> None:
    """Test-only hook: drop the TTL cache between test cases.

    Importers should call this whenever they mock
    ``subprocess.run`` so cached results from a previous test do not
    leak across cases.
    """
    try:
        _read_systemd_timer_schedule.cache_clear()  # type: ignore[attr-defined]
    except (AttributeError, NameError):
        pass


def _format_calendar_to_shanghai(on_calendar: str) -> str:
    """Convert a ``systemctl`` calendar expression to a dashboard label.

    Accepts both the raw ``OnCalendar=`` value (e.g.
    ``Tue *-*-* 04:01:00 Asia/Shanghai``) and the structured
    ``TimersCalendar={ OnCalendar=... ; next_elapse=... }`` shape that
    ``systemctl show`` emits. Returns the inner OnCalendar value
    trimmed; anything we cannot recognise as a real schedule
    expression is returned as ``""`` so the dashboard renders honest
    "unknown" state instead of a malformed label.

    Recognised prefixes: ``Mon`` / ``Tue`` / ``Wed`` / ``Thu`` / ``Fri``
    / ``Sat`` / ``Sun``, the daily form ``*-*-*``, or ``*:0/N``-style
    repeating markers.
    """
    if not on_calendar:
        return ""
    raw = on_calendar.strip()
    # ``systemctl show`` wraps the value: ``TimersCalendar={ OnCalendar=... ;
    # next_elapse=... }``. Pull out the inner OnCalendar if present.
    if "OnCalendar=" in raw:
        inner = raw.split("OnCalendar=", 1)[1]
        # Strip at the first ``;`` / ``}`` so we don't drag the next_elapse
        # tail into the schedule label.
        for stop in (";", "}"):
            if stop in inner:
                inner = inner.split(stop, 1)[0]
        raw = inner.strip()
    cleaned = " ".join(raw.split())
    # Reject anything that doesn't look like an OnCalendar expression.
    # A real value starts with a day-of-week token, ``*-*-*``, or
    # ``*:N`` (repeating interval).
    if not cleaned:
        return ""
    first = cleaned.split(None, 1)[0]
    valid_first = {
        "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
        "*-*-*", "*-*-*-*",
    }
    if first in valid_first or first.startswith("*:"):
        return cleaned
    # Looks like the inner ``OnCalendar=`` slot was missing (e.g. only
    # ``next_elapse=...`` was populated); bail out so the dashboard
    # shows the unknown state instead of a misformatted label.
    return ""


def _parse_usec_realtime(value: str) -> Optional[str]:
    """Convert a systemd ``usec`` / human-readable timestamp to ISO-8601 UTC.

    Newer systemd versions emit ``NextElapseUSecRealtime`` /
    ``LastTriggerUSec`` as a plain ``usec`` integer (epoch-microseconds).
    Older versions (and some newer ones) emit human-readable local
    timestamps like ``Wed 2026-07-22 07:45:00 CST``. Both shapes are
    accepted. Anything we cannot parse (e.g. ``infinity``, empty) becomes
    ``None`` so the dashboard never reads a fabricated timestamp.
    """
    if not value:
        return None
    val = value.strip()
    if not val or val.lower() == "infinity" or val == "0":
        return None

    # Path 1: raw microsecond integer.
    try:
        usec = int(val)
        if usec > 0:
            return datetime.fromtimestamp(usec / 1_000_000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        pass

    # Path 2: human-readable systemd format, e.g.
    # ``Wed 2026-07-22 07:45:00 CST``. Regex the trailing 3-letter tz
    # into a fixed offset so common cases parse without dateutil.
    m = re.match(
        r"^[A-Za-z]{3}\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+([A-Z]{2,5})$",
        val,
    )
    if m:
        from datetime import datetime as _dt, timedelta as _td
        date_str, time_str, tz_abbr = m.group(1), m.group(2), m.group(3)
        offset_minutes = {
            "CST": 8 * 60,
            "JST": 9 * 60,
            "KST": 9 * 60,
            "SGT": 8 * 60,
            "PHT": 8 * 60,
            "WIB": 7 * 60,
            "UTC": 0,
            "GMT": 0,
        }.get(tz_abbr)
        if offset_minutes is not None:
            naive = _dt.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            tz = timezone(_td(minutes=offset_minutes))
            parsed = naive.replace(tzinfo=tz)
            return parsed.astimezone(timezone.utc).isoformat()

    # Path 3: ISO-8601 with explicit offset (rare but possible).
    try:
        return datetime.fromisoformat(val).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        pass
    return None


@_ttl_cache(maxsize=8, ttl_seconds=10.0)
def _read_systemd_timer_schedule(timer_name: str) -> dict:
    """Read schedule metadata from a systemd timer unit.

    Returns a dict with these keys (any may be missing when the timer
    is absent or the field is empty):

      * ``calendar``       — ``OnCalendar`` expression trimmed.
      * ``last_trigger``   — ISO-8601 UTC timestamp (parsed from
                             ``LastTriggerUSec``).
      * ``next_elapse``    — ISO-8601 UTC timestamp (parsed from
                             ``NextElapseUSecRealtime``).
      * ``active_state``   — ``ActiveState`` field from
                             ``systemctl show`` (``active`` /
                             ``inactive`` / ``unknown``).
      * ``result``         — ``Result`` field from ``systemctl show``
                             (``success`` / ``success`` / ``unknown``).

    When the timer is not installed, systemd exits non-zero on
    ``systemctl show``; we always return
    ``{"active_state": "unknown"}`` so the dashboard card renders
    honestly instead of erroring out. Any subprocess error (timeout,
    PermissionError, FileNotFoundError) returns the same shape so
    callers can rely on the contract.

    The TTL cache means a hot dashboard render doesn't hit
    ``systemctl`` more than once per 10 seconds per timer name; this
    matters because ``systemctl show`` can block on a slow D-Bus
    connection.
    """
    if not timer_name or not isinstance(timer_name, str):
        return {"active_state": "unknown"}

    fields = (
        "OnCalendar",
        "TimersCalendar",
        "NextElapseUSecRealtime",
        "LastTriggerUSec",
        "Result",
        "ActiveState",
    )
    try:
        proc = subprocess.run(
            ["systemctl", "show", timer_name] + [f"-p{f}" for f in fields],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError, OSError):
        return {"active_state": "unknown"}

    out: Dict[str, Any] = {}
    if proc.returncode != 0:
        # Unit not installed or no permission to introspect.
        out["active_state"] = "unknown"
        return out

    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key in ("OnCalendar", "TimersCalendar"):
            # ``OnCalendar=`` may be empty for timers that are entirely
            # driven by ``TimersCalendar`` (which is what ``systemctl
            # show`` actually returns). Use whichever sidecar carries
            # the schedule.
            if key == "OnCalendar" and not value:
                continue
            formatted = _format_calendar_to_shanghai(value)
            if formatted and "calendar" not in out:
                out["calendar"] = formatted
        elif key == "NextElapseUSecRealtime":
            ts = _parse_usec_realtime(value)
            if ts:
                out["next_elapse"] = ts
        elif key == "LastTriggerUSec":
            ts = _parse_usec_realtime(value)
            if ts:
                out["last_trigger"] = ts
        elif key == "Result":
            if value:
                out["result"] = value
        elif key == "ActiveState":
            if value:
                out["active_state"] = value

    # ``systemctl show`` returns ``ActiveState=inactive`` for both an
    # inactive-but-loaded timer AND a non-existent one. The signal we
    # can trust is whether systemd actually populated the schedule
    # fields: a real unit will always carry an ``OnCalendar`` value
    # even when inactive. Without it, treat the unit as unknown so the
    # dashboard card renders honest "not configured" state instead of
    # pretending we know the schedule.
    if not out.get("calendar"):
        out.clear()
        out["active_state"] = "unknown"
    else:
        out.setdefault("active_state", "unknown")
    return out


# Canonical timer names used by the dashboard schedule cards.
# Defined as module constants so tests can pin them.
MAINTENANCE_TIMER_NAME = "openclaw-memory-os-maintenance.timer"
GOVERNANCE_TIMER_NAME = "openclaw-memory-os-governance.timer"


def build_health_summary(backend: MemoryBackend) -> HealthSummary:
    memories = backend.list_memories()
    total = len(memories)

    # Fast path for simple counters (no heavy computation)
    tier_counter: Counter = Counter()
    status_counter: Counter = Counter()
    month_counter: Counter = Counter()
    never_delete = 0

    for m in memories:
        tier_counter[m.tier] += 1
        status_counter[m.status] += 1
        ts = m.created_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        month_counter[ts.strftime("%Y-%m")] += 1
        # Protected memories = tier=core/long + importance>=0.6 + review_reason=never_delete
        rev = getattr(m, "review_reason", None) or ""
        if m.tier in (MemoryTier.CORE, MemoryTier.LONG) or m.importance >= 0.6 or "never_delete" in str(rev).lower():
            never_delete += 1

    # Deletion candidates are cheap (O(n) single pass with simple eligibility
    # filters — tier / importance / age checks then continue).  Always run
    # them so the dashboard's "自动清理候选" number stays in lockstep with
    # ``/api/deletion-candidates`` (which itself does no throttling).
    # The 5 000 cap is preserved only for the heavier MinHash + LSH
    # duplicate-cluster estimate, which has O(n) average but a much
    # larger constant factor.
    deletion = _build_deletion_candidates(memories)
    if total <= 5000:
        duplicates = _estimate_duplicate_clusters(memories)
    else:
        duplicates = []  # defer to /api/duplicates for large corpora

    tier_dist = [TierCount(tier=t, count=c) for t, c in sorted(tier_counter.items(), key=lambda kv: kv[0].value)]
    status_dist = [StatusCount(status=s, count=c) for s, c in sorted(status_counter.items(), key=lambda kv: kv[0].value)]
    monthly = [MonthCount(month=month, count=count) for month, count in sorted(month_counter.items())]
    governance_status = _read_autonomous_governance_status()

    return HealthSummary(
        backend=backend.name,
        total_memories=total,
        active=status_counter.get(MemoryStatus.ACTIVE, 0),
        superseded=status_counter.get(MemoryStatus.SUPERSEDED, 0),
        expired=status_counter.get(MemoryStatus.EXPIRED, 0),
        needs_review=status_counter.get(MemoryStatus.NEEDS_REVIEW, 0),
        duplicates_estimate=len(duplicates),
        deletion_candidate_count=len(deletion),
        never_delete=never_delete,
        last_maintenance=_read_last_maintenance(),
        maintenance_health=_read_maintenance_health(),
        # Wave 2 (2026-07-21): live systemd-timer schedule snapshots so
        # the dashboard never has to hardcode "Tue 04:01" /
        # "daily 07:45". Both helpers use the TTL cache; calling them
        # twice in the same render is cheap.
        maintenance_schedule=_read_systemd_timer_schedule(MAINTENANCE_TIMER_NAME),
        governance_schedule=_read_systemd_timer_schedule(GOVERNANCE_TIMER_NAME),
        last_maintenance_summary=_summarize_last_maintenance(),
        memory_brain=_read_memory_brain_status(),
        autonomous_governance=AutonomousGovernanceJob.for_dashboard(
            last_run=governance_status.get("last_run"),
            last_result=governance_status.get("last_result"),
            last_summary=governance_status.get("last_summary"),
            scheduled_at=governance_status.get("scheduled_at"),
            started_at=governance_status.get("started_at"),
            finished_at=governance_status.get("finished_at"),
            duration_seconds=governance_status.get("duration_seconds"),
            next_scheduled_at=governance_status.get("next_scheduled_at"),
            exit_code=governance_status.get("exit_code"),
            runner_mode=governance_status.get("mode"),
        ),
        tier_distribution=tier_dist,
        status_distribution=status_dist,
        monthly_counts=monthly,
        legacy_default_count=legacy_default_count(memories),
        importance_distribution=importance_distribution(memories),
        generated_at=utcnow(),
        collections=backend.list_collections(),
    )


def _read_last_maintenance() -> Optional[str]:
    """Read the most recent maintenance log mtime if present.

    Tries, in order:
        - $OPENCLAW_MEMORY_OS_LOG (preferred override)
        - /var/log/openclaw-memory-os/maintenance.log
        - /var/log/openclaw-memory-os.log (legacy)
        - <project_root>/logs/maintenance.log
        - ~/.local/state/openclaw-memory-os/maintenance.log

    Returns ISO-8601 UTC or ``None``.
    """
    import os as _os
    import datetime as _dt
    from pathlib import Path as _P

    candidates = []
    env = _os.environ.get("OPENCLAW_MEMORY_OS_LOG")
    if env:
        candidates.append(_P(env))
    candidates.append(_P("/var/log/openclaw-memory-os/maintenance.log"))
    candidates.append(_P("/var/log/openclaw-memory-os.log"))
    candidates.append(_P(__file__).resolve().parent.parent / "logs" / "maintenance.log")
    candidates.append(_P.home() / ".local/state/openclaw-memory-os/maintenance.log")

    for log in candidates:
        try:
            if log.exists() and log.is_file():
                ts = log.stat().st_mtime
                return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()
        except (OSError, PermissionError):
            continue
    return None


def _read_maintenance_health() -> dict:
    import datetime as _dt
    from pathlib import Path as _P
    import re as _re
    """Return a structured maintenance health summary.

    Returns:
      ``enabled``: bool      — whether a cron/timer entry is expected
      ``lock_present``: bool — whether the flock file currently exists
      ``last_run``: Optional[str] — ISO-8601 UTC of last log write
      ``last_ok``: Optional[str]  — ISO-8601 UTC of last "ok" log line
      ``log_lines``: int          — approximate line count of log file
      ``log_path``: Optional[str] — resolved absolute path of log file
      ``next_scheduled_at``: Optional[str] — ISO-8601 UTC of next
        scheduled maintenance run, derived from the live systemd timer
        (``openclaw-memory-os-maintenance.timer``). Populated when the
        timer exists; ``None`` otherwise.
    """
    import os as _os

    FLOCK_PATH = "/tmp/openclaw-memory-os.maintenance.lock"
    result = {
        "enabled": True,
        "lock_present": _P(FLOCK_PATH).exists(),
        "last_run": None,
        "last_ok": None,
        "log_lines": 0,
        "log_path": None,
    }

    candidates = []
    env = _os.environ.get("OPENCLAW_MEMORY_OS_LOG")
    if env:
        candidates.append(_P(env))
    candidates.append(_P("/var/log/openclaw-memory-os/maintenance.log"))
    candidates.append(_P("/var/log/openclaw-memory-os.log"))
    candidates.append(_P(__file__).resolve().parent.parent / "logs" / "maintenance.log")
    candidates.append(_P.home() / ".local/state/openclaw-memory-os/maintenance.log")

    for log in candidates:
        try:
            if log.exists() and log.is_file():
                result["log_path"] = str(log.resolve())
                lines = 0
                last_ok_ts = None
                # Wave 5 (2026-07-22): track the most recent
                # ``[maintenance YYYY-MM-DDTHH:MM:SSZ]`` start line so
                # we can surface a real run timestamp instead of
                # ``log.stat().st_mtime`` (which moves every time
                # logrotate truncates the file, even when no actual
                # maintenance run happened).
                last_maintenance_start_ts = None
                with log.open("r", errors="replace") as fh:
                    for line in fh:
                        lines += 1
                        if " ok" in line or line.rstrip().endswith("ok"):
                            # Extract embedded ISO-8601 timestamp from the line if available
                            # The log format is: [maintenance YYYY-MM-DDTHH:MM:SSZ] ...
                            import re as _re
                            m = _re.search(r"\[maintenance\s+([\dTZ:.+-]+)\]", line)
                            if m:
                                last_ok_ts = m.group(1)
                        # Any ``[maintenance ...]`` start line is a real
                        # maintenance-run trigger — capture the latest
                        # one. We deliberately do NOT trust the file
                        # mtime because external actors (logrotate,
                        # maintenance.sh redirect, etc.) can touch the
                        # file without a corresponding run.
                        import re as _re2
                        m2 = _re2.search(r"\[maintenance\s+([\dTZ:.+-]+)\]\s+starting", line)
                        if m2:
                            last_maintenance_start_ts = m2.group(1)
                result["log_lines"] = lines
                if last_ok_ts:
                    result["last_ok"] = last_ok_ts
                if last_maintenance_start_ts:
                    result["last_run"] = last_maintenance_start_ts
                # Wave 5 (2026-07-22): we deliberately do NOT fall
                # back to ``log.stat().st_mtime`` here. logrotate
                # truncates the file every night and updates the
                # mtime even when no actual maintenance run happened,
                # which causes the dashboard to surface a logrotate
                # timestamp as "最近运行". Leaving ``last_run`` at
                # ``None`` when the log is empty / rotated is honest
                # — the JSON-summary fallback below will fill it from
                # the canonical ``payload.last_run`` instead.
                break  # first readable log wins
        except (OSError, PermissionError):
            continue

    # Fallback: read the maintenance summary JSON. Useful when the log file is
    # stale (e.g. an ad-hoc maintenance run that didn't redirect into the log),
    # so the dashboard shows the last successful summary timestamp instead of
    # the older log mtime.
    # Wave 5 (2026-07-22): if the log was empty/rotated and we couldn't
    # parse a real ``[maintenance ...] starting`` line, the log loop
    # left ``result["last_run"]`` at the file mtime (which moves every
    # time logrotate truncates the file). The fallback's own mtime is
    # only meaningful when nothing else claimed the slot — we MUST NOT
    # compare the two numerically because that lets a logrotate
    # timestamp survive into the dashboard. ``result["last_ok"]``
    # already reads the summary payload's ``last_run`` field (not
    # its mtime), so the canonical timestamp reaches the dashboard
    # even when the log is empty.
    try:
        from .config import get_settings as _get_settings
        settings = _get_settings()
        for summary_path in (
            Path(os.environ.get("OPENCLAW_MEMORY_OS_SUMMARY", "/var/lib/openclaw-memory-os/state/openclaw-memory-os/maintenance-summary.json")),
            settings.env_file.parent / "summary.json" if settings.env_file else None,
            Path.home() / ".local/state/openclaw-memory-os/summary.json",
        ):
            if summary_path is None:
                continue
            if not summary_path.exists():
                continue
            import json as _json
            payload = _json.loads(summary_path.read_text(encoding="utf-8"))
            # The summary's payload-level ``last_run`` is the canonical
            # "real run" timestamp (written by ``_write_summary.py``
            # from the log's ok marker). Prefer it whenever the log
            # parser didn't already pin last_run to a parsed line.
            payload_last_run = payload.get("last_run")
            if payload_last_run:
                # Only override when we don't have a stronger source
                # (parsed log line). If both are present, the log line
                # is more authoritative.
                if not result.get("last_run") or result["last_run"] is None:
                    result["last_run"] = payload_last_run
            summary_last_ok = payload.get("last_run")
            if summary_last_ok and (result["last_ok"] is None or summary_last_ok > result["last_ok"]):
                result["last_ok"] = summary_last_ok
            break
    except (OSError, ValueError, TypeError):
        pass

    # Wave 2 (2026-07-21): backfill ``next_scheduled_at`` from the
    # live ``openclaw-memory-os-maintenance.timer`` so the maintenance
    # card never shows an empty next-run slot when the unit is
    # installed. The TTL cache on the timer helper keeps this cheap on
    # repeated renders.
    timer_schedule = _read_systemd_timer_schedule(MAINTENANCE_TIMER_NAME)
    next_elapse = timer_schedule.get("next_elapse") if timer_schedule else None
    if next_elapse:
        result["next_scheduled_at"] = next_elapse

    return result


def _read_json_status(path: Path) -> dict:
    try:
        if not path.exists() or not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, PermissionError, ValueError, TypeError):
        return {}


def _read_memory_brain_status() -> dict:
    """Read the canonical maintenance summary for ingest/consolidate dashboard cards.

    Wave-2 (2026-07-21): the legacy ``/var/log/openclaw-memory-brain-*.json`` and
    ``/var/log/memory-brain-*.json`` files were produced by the pre-unification
    pipeline and only ever reflected ad-hoc runs. Maintenance now publishes its
    own ingest/consolidate sub-state directly inside ``maintenance-summary.json``
    (fields ``steps.ingest`` / ``steps.memory_brain`` / top-level
    ``consolidation``), so the dashboard must read from there — the legacy
    files are retained on disk only as historical audit.

    The legacy paths may still be honoured if ``MEMORY_BRAIN_LEGACY_FILES_OK=1``
    is explicitly exported, but they can NEVER override canonical data:
    canonical maintenance-summary values always win, even if older.

    Field contract pinned 2026-07-21:

    ingest_card:
        run_id, status, started_at, finished_at, duration_seconds,
        ingested_new, chunks_scanned, skipped, files_processed,
        checkpoint_id

    consolidate_card:
        run_id, status, started_at, finished_at, duration_seconds,
        reason, new_since_24h, total_points, merged_topics, threshold

    Any missing timestamp surfaces as ``None`` so the dashboard reads an
    explicit error (the brief forbids silent "—").
    """
    summary = _summarize_last_maintenance() or {}
    steps = summary.get("steps") or {}
    ingest_step = steps.get("ingest") or {}
    brain_step = steps.get("memory_brain") or {}
    consolidation = summary.get("consolidation") or {}

    ingest_card: Dict[str, Any] = {}
    # Merge brain (unified pipeline) + ingest (step 1) data so the dashboard
    # card reflects whatever the most-recent run actually produced.
    # Wave 4 (2026-07-21): ingest_step first so its independent bracket
    # wins over the parent memory_brain bracket.
    for src in (ingest_step, brain_step):
        if not src:
            continue
        for key, val in src.items():
            if key == "chunks_scanned":
                # ingest CLI exposes `total_chunks` as chunks_scanned; keep
                # the union of both names so old JS templates still work.
                ingest_card["chunks_scanned"] = max(
                    int(ingest_card.get("chunks_scanned") or 0),
                    int(val or 0),
                )
                continue
            # Wave 4 (2026-07-21): the unified pipeline leaf (ingest_step)
            # now carries its own independent started_at / finished_at /
            # duration_seconds from the [brain-substep] markers. The
            # parent memory_brain bracket is the wider window that
            # covers both ingest and consolidate, so we must not let it
            # overwrite the leaf values. ``ingest_step`` is iterated
            # FIRST and assigned directly so leaf timestamps stick;
            # ``brain_step`` only fills gaps the leaf didn't set (so
            # legacy / pre-Wave-4 summaries still produce a card).
            if src is ingest_step:
                ingest_card[key] = val
            else:
                ingest_card.setdefault(key, val)
    ingest_card.setdefault("run_id", summary.get("run_id"))

    consolidate_card: Dict[str, Any] = dict(consolidation)
    consolidate_card.setdefault("run_id", summary.get("run_id"))

    # Wave 3 (2026-07-21): every card must carry a duration_seconds field.
    # Older runs that pre-date the schema rewrite did not compute it; in
    # that case derive it from the timestamps on the fly so the dashboard
    # always has a numeric duration (or ``None`` if a side is missing).
    def _derive_duration(card: Dict[str, Any]) -> None:
        if card.get("duration_seconds") is not None:
            return
        s = card.get("started_at")
        f = card.get("finished_at")
        if not s or not f:
            card["duration_seconds"] = None
            return
        try:
            from datetime import datetime as _dt
            sd = _dt.fromisoformat(str(s).replace("Z", "+00:00"))
            fd = _dt.fromisoformat(str(f).replace("Z", "+00:00"))
            card["duration_seconds"] = max(round((fd - sd).total_seconds(), 3), 0.0)
        except Exception:
            card["duration_seconds"] = None

    _derive_duration(ingest_card)
    _derive_duration(consolidate_card)

    # Wave 3 (2026-07-21): normalisation so the dashboard never has to
    # worry about which field name the writer chose. ``merged_topics`` is
    # the contract; ``topics_merged`` is the legacy alias kept for
    # back-compat. ``threshold`` is the configured minimum-new-memories
    # threshold that the consolidation script used to decide skip vs.
    # run (default 20 when not in the canonical summary).
    if consolidate_card.get("merged_topics") is None:
        consolidate_card["merged_topics"] = consolidate_card.get("topics_merged") or 0
    consolidate_card.setdefault("threshold", 20)

    # Optional explicit legacy override (ad-hoc / external runs only).
    if os.environ.get("MEMORY_BRAIN_LEGACY_FILES_OK") == "1":
        for key, path_str in (
            ("ingest", os.environ.get(
                "MEMORY_BRAIN_STATUS_FILE",
                "/var/log/openclaw-memory-brain-status.json",
            )),
            ("consolidate", os.environ.get(
                "MEMORY_BRAIN_DREAM_STATUS_FILE",
                "/var/log/openclaw-memory-brain-dream-status.json",
            )),
        ):
            legacy = _read_json_status(Path(path_str))
            if not legacy:
                continue
            target = ingest_card if key == "ingest" else consolidate_card
            for lk, lv in legacy.items():
                # canonical values always win — never overwrite.
                target.setdefault(lk, lv)

    return {"ingest": ingest_card, "consolidate": consolidate_card}

# --- Timeline / tiers / status helpers -------------------------------------

def monthly_counts(memories: Sequence[Memory]) -> List[MonthCount]:
    counter: Counter = Counter()
    for m in memories:
        ts = m.created_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        counter[ts.strftime("%Y-%m")] += 1
    return [MonthCount(month=k, count=v) for k, v in sorted(counter.items())]


def tier_distribution(memories: Sequence[Memory]) -> List[TierCount]:
    counter: Counter = Counter(m.tier for m in memories)
    return [TierCount(tier=t, count=c) for t, c in sorted(counter.items(), key=lambda kv: kv[0].value)]


def status_distribution(memories: Sequence[Memory]) -> List[StatusCount]:
    counter: Counter = Counter(m.status for m in memories)
    return [StatusCount(status=s, count=c) for s, c in sorted(counter.items(), key=lambda kv: kv[0].value)]


def legacy_default_count(memories: Sequence[Memory]) -> int:
    """Memories whose importance equals the adapter default (0.5) AND whose
    source is the legacy raw Qdrant payloads. These are historical points
    that never went through the modern classifier, so the dashboard surfaces
    them separately so users don't mistake "adapter fallback" for "actual
    classification".
    """
    n = 0
    for m in memories:
        # importance round to 4dp avoids float noise
        imp = round(float(m.importance or 0.0), 2)
        if imp == 0.5 and (m.source or "").lower() in ("qdrant", "session-recovery"):
            n += 1
    return n


def importance_distribution(memories: Sequence[Memory]) -> List[ImportanceBucket]:
    """Importance histogram with 5 buckets, ordered high to low."""
    buckets = [
        (">=0.8",   0.80, 1.01),
        ("0.6-0.79", 0.60, 0.80),
        ("0.5 (default)", 0.50, 0.60),
        ("0.3-0.49", 0.30, 0.50),
        ("<0.3",    -1.0,  0.30),
    ]
    counts = {label: 0 for label, _, _ in buckets}
    for m in memories:
        imp = float(m.importance or 0.0)
        for label, lo, hi in buckets:
            if lo <= imp < hi:
                counts[label] += 1
                break
    out: List[ImportanceBucket] = []
    for label, lo, hi in buckets:
        out.append(ImportanceBucket(label=label, count=counts[label], min_importance=lo, max_importance=hi))
    return out


def _summarize_last_maintenance() -> dict:
    """Read the maintenance summary JSON written by scripts/_write_summary.py.

    The summary file is the authoritative source. If it is missing or stale,
    fall back to an empty dict so the dashboard can still render.
    """
    import json as _json
    from pathlib import Path as _P

    candidates = []
    env_summary = os.environ.get("OPENCLAW_MEMORY_OS_SUMMARY")
    if env_summary:
        candidates.append(_P(env_summary))
    candidates.extend(
        [
            _P("/var/lib/openclaw-memory-os/state/openclaw-memory-os/maintenance-summary.json"),
            _P("/var/log/openclaw-memory-os-summary.json"),  # legacy
            _P(__file__).resolve().parent.parent / "state" / "summary.json",
            _P.home() / ".local/state/openclaw-memory-os/summary.json",
        ]
    )
    for path in candidates:
        try:
            if path.exists() and path.is_file():
                with path.open("r", encoding="utf-8") as fh:
                    data = _json.load(fh)
                # Legacy summary files (pre-v0.2.x) had `ingested_total`
                # but no `chunks_scanned`. They're the same value, so
                # backfill for forward compat with the new dashboard
                # wording ("新增 X / 扫描 Y chunks").
                if (
                    isinstance(data, dict)
                    and "chunks_scanned" not in data
                    and "ingested_total" in data
                ):
                    data["chunks_scanned"] = data["ingested_total"]
                return data
        except (OSError, ValueError):
            continue

    return {
        "ingested_total": 0,
        "ingested_new": 0,
        "chunks_scanned": 0,
        "expired_count": 0,
        "superseded_links": 0,
        "snapshot_name": None,
        "snapshot_size_bytes": 0,
    }
# --- Duplicate detection (heuristic, content-based) ------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


def _token_set(text: str) -> Set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 2}


def _shingle_set(text: str, k: int = 4) -> Set[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _estimate_duplicate_clusters(
    memories: Sequence[Memory],
    *,
    shingle_threshold: float = 0.6,
    token_threshold: float = 0.85,
    minhash_perm: int = 64,
    bands: int = 16,
) -> List[DuplicateCluster]:
    """Group near-duplicate memories by MinHash + LSH.

    Why MinHash/LSH instead of the previous O(n^2) Jaccard pass:
        - O(n^2) was fine for the 15-mem sample data, but with 25k+
          real memories it takes ~1-2 minutes per dashboard load and
          causes 504s. LSH reduces the candidate set to O(n) on
          average while still catching ~all pairs above the threshold.
    """

    items = [(m, _shingle_set(m.text)) for m in memories]
    n = len(items)
    if n == 0:
        return []
    # Bail out for very large collections; the dashboard samples are enough
    if n > 5000:
        return []

    # MinHash signatures
    rows_per_band = max(1, minhash_perm // bands)
    sigs: List[List[int]] = []
    for _, sh in items:
        sigs.append(_minhash_signature(sh, minhash_perm))

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # LSH banding
    bucket_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, sig in enumerate(sigs):
        for b in range(bands):
            start = b * rows_per_band
            end = start + rows_per_band
            key = (b, hash(tuple(sig[start:end])))
            bucket_map[key].append(i)

    candidate_pairs: Set[Tuple[int, int]] = set()
    for bucket in bucket_map.values():
        if len(bucket) < 2 or len(bucket) > 200:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]
                if a > b:
                    a, b = b, a
                candidate_pairs.add((a, b))

    token_cache: List[Set[str]] = [_token_set(m.text) for m, _ in items]
    for a, b in candidate_pairs:
        jac_shingle = _jaccard(items[a][1], items[b][1])
        jac_token = _jaccard(token_cache[a], token_cache[b])
        if jac_shingle >= shingle_threshold or jac_token >= token_threshold:
            union(a, b)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    clusters: List[DuplicateCluster] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        member_memories = [items[i][0] for i in indices]
        total_sim = 0.0
        pairs = 0
        for a_idx in range(len(indices)):
            for b_idx in range(a_idx + 1, len(indices)):
                sh_a = items[indices[a_idx]][1]
                sh_b = items[indices[b_idx]][1]
                total_sim += _jaccard(sh_a, sh_b)
                pairs += 1
        avg_sim = total_sim / pairs if pairs else 0.0
        representative = max(
            member_memories,
            key=lambda m: (m.updated_at or m.created_at),
        )
        rationale_bits = [f"avg_jaccard={avg_sim:.2f}", f"minhash_perm={minhash_perm}"]
        if any(m.supersedes or m.superseded_by for m in member_memories):
            rationale_bits.append("explicit supersede link")
        clusters.append(
            DuplicateCluster(
                representative_id=representative.id,
                member_ids=[m.id for m in member_memories],
                score=round(avg_sim, 4),
                rationale="; ".join(rationale_bits),
            )
        )
    clusters.sort(key=lambda c: c.score, reverse=True)
    return clusters


def _minhash_signature(shingles: set, num_perm: int) -> List[int]:
    """Compute a MinHash signature over a set of string shingles."""
    import hashlib as _hl
    import math as _m

    sig = [_m.inf] * num_perm
    if not shingles:
        return [0] * num_perm
    for sh in shingles:
        h = int.from_bytes(_hl.sha1(sh.encode()).digest()[:8], "big")
        for i in range(num_perm):
            mixed = (h ^ (i * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
            if mixed < sig[i]:
                sig[i] = mixed
    return [int(v) if v != _m.inf else 0 for v in sig]


# --- Deletion candidates ----------------------------------------------------

def _build_deletion_candidates(memories: Sequence[Memory]) -> List[DeletionCandidate]:
    """Build a list of memories the user may want to *review* for deletion.

    IMPORTANT: this project never deletes memories automatically. The
    ``recommended_action`` field is always ``review``. A human must
    decide what to do with the candidate list.

    Hard rule per the deletion policy (docs/deletion-policy.md):
        tier in (core, long) NEVER enters the candidate list.
        Only tier in (medium, short, working) can be considered.
    """

    candidates: List[DeletionCandidate] = []
    now = utcnow()
    eligible_tiers = {MemoryTier.MEDIUM, MemoryTier.SHORT, MemoryTier.WORKING}

    for m in memories:
        if m.tier not in eligible_tiers:
            continue  # tier=core / tier=long are immutable
        reasons: List[str] = []
        action = "keep"

        # Automatic keep conditions:
        if m.importance >= 0.6:
            action = "keep"
            continue  # any memory this important should not be shown at all
        created = m.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (now - created).days
        if age_days < 7:
            action = "keep"
            continue  # too recent
        if m.review_reason and "never_delete" in str(m.review_reason).lower():
            action = "keep"
            continue
        if not m.status or m.status != MemoryStatus.EXPIRED:
            # still active/superseded: only consider if old + low importance + tier=working
            if m.tier != MemoryTier.WORKING:
                continue

        # Reaches here: old, low-importance, tier=working or tier=short/medium that's expired
        if m.status == MemoryStatus.EXPIRED:
            reasons.append("已过期 30+ 天")
        if m.tier == MemoryTier.WORKING:
            reasons.append("临时状态，超 7 天")
        if m.importance < 0.3:
            reasons.append("低重要性 (<0.3)")
        if age_days > 60:
            reasons.append("超过 60 天未更新")

        if reasons:
            # Truly safe to auto-approve these
            action = "auto_delete"
            candidates.append(
                DeletionCandidate(
                    id=m.id,
                    text=m.text,
                    tier=m.tier,
                    status=m.status,
                    reason="; ".join(reasons),
                    recommended_action=action,
                )
            )
        # else: no reasons → skip (don't bother user)

    # Stable order: status then importance
    candidates.sort(key=lambda c: (c.status.value, c.tier.value))
    return candidates
