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
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
            return {
                "last_run": last_run_raw or None,
                "last_result": data.get("last_result") or data.get("result"),
                "last_summary": data.get("last_summary") or data.get("summary"),
            }
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

    # Only compute duplicates/deletion for smaller collections; skip for >5000
    if total <= 5000:
        duplicates = _estimate_duplicate_clusters(memories)
        deletion = _build_deletion_candidates(memories)
    else:
        duplicates = []
        deletion = []

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
        last_maintenance_summary=_summarize_last_maintenance(),
        memory_brain=_read_memory_brain_status(),
        autonomous_governance=AutonomousGovernanceJob.for_dashboard(
            last_run=governance_status.get("last_run"),
            last_result=governance_status.get("last_result"),
            last_summary=governance_status.get("last_summary"),
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
                result["log_lines"] = lines
                if last_ok_ts:
                    result["last_ok"] = last_ok_ts
                ts = log.stat().st_mtime
                result["last_run"] = (
                    _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()
                )
                break  # first readable log wins
        except (OSError, PermissionError):
            continue

    # Fallback: read the maintenance summary JSON. Useful when the log file is
    # stale (e.g. an ad-hoc maintenance run that didn't redirect into the log),
    # so the dashboard shows the last successful summary timestamp instead of
    # the older log mtime.
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
            ts = summary_path.stat().st_mtime
            summary_last_run = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()
            if result["last_run"] is None or summary_last_run > result["last_run"]:
                result["last_run"] = summary_last_run
            summary_last_ok = payload.get("last_run")
            if summary_last_ok and (result["last_ok"] is None or summary_last_ok > result["last_ok"]):
                result["last_ok"] = summary_last_ok
            break
    except (OSError, ValueError, TypeError):
        pass

    return result


def _read_json_status(path: Path) -> dict:
    try:
        if not path.exists() or not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, PermissionError, ValueError, TypeError):
        return {}


def _read_memory_brain_status() -> dict:
    """Read optional Memory Brain status JSON files for dashboard cards.

    Paths are configurable so public deployments can keep the feature generic
    while local operators choose their own log locations.
    """
    ingest_path = Path(os.environ.get("MEMORY_BRAIN_STATUS_FILE", "/var/log/openclaw-memory-brain-status.json"))
    consolidate_path = Path(os.environ.get("MEMORY_BRAIN_DREAM_STATUS_FILE", "/var/log/openclaw-memory-brain-dream-status.json"))

    # Backward-compatible local filenames from the standalone scripts. Public
    # code does not depend on them, but this lets existing deployments surface
    # their current status immediately after upgrade.
    if not ingest_path.exists():
        legacy = Path("/var/log/memory-brain-status.json")
        if legacy.exists():
            ingest_path = legacy
    if not consolidate_path.exists():
        legacy = Path("/var/log/memory-brain-dream-status.json")
        if legacy.exists():
            consolidate_path = legacy

    return {
        "ingest": _read_json_status(ingest_path),
        "consolidate": _read_json_status(consolidate_path),
    }

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
