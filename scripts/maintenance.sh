#!/usr/bin/env bash
# maintenance.sh — Daily maintenance entry point for OpenClaw Memory OS.
#
# The pipeline is best-effort within a run: later collections and cleanup
# stages still execute after an earlier error. The process exits non-zero at
# the end when any required stage failed, preventing cron/governance from
# reporting a partial run as green.

set -euo pipefail
if [ -n "${OPENCLAW_MEMORY_OS_MAINTENANCE_LOCK:-}" ]; then
  flock_path="$OPENCLAW_MEMORY_OS_MAINTENANCE_LOCK"
else
  default_lock_dir="${XDG_RUNTIME_DIR:-}"
  if [ -z "$default_lock_dir" ] && [ -n "${LOG_FILE:-}" ]; then
    default_lock_dir="$(dirname "$LOG_FILE")"
  fi
  if [ -z "$default_lock_dir" ]; then
    default_lock_dir="${XDG_STATE_HOME:-/tmp}"
  fi
  mkdir -p "$default_lock_dir" 2>/dev/null || default_lock_dir="/tmp"
  flock_path="$default_lock_dir/openclaw-memory-os.maintenance.lock"
fi
exec 9>"$flock_path"
if ! flock -n 9; then
  echo "maintenance.sh: another instance holds the lock; SKIPPED"
  exit 75
fi

# Wave 2 (2026-07-21): shared RUN_ID across maintenance.sh, the memory-brain
# pipeline, _write_summary, and _write_governance_status so a single run's
# sub-step states land in the canonical maintenance-summary.json under one
# run_id. Honour MAINTENANCE_RUN_ID when an upstream (governance.sh /
# manual) sets it so deep-audit runs share a single id with the daily path.
RUN_ID="${MAINTENANCE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(printf '%04x' $$)}"
export MAINTENANCE_RUN_ID="$RUN_ID"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$PROJECT_DIR/.venv/bin/python"
DEFAULT_COLLECTIONS="${DEFAULT_COLLECTIONS:-openclaw_memory_os}"
COLLECTIONS="${MAINTAIN_COLLECTIONS:-$DEFAULT_COLLECTIONS}"

DEFAULT_STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/openclaw-memory-os"
LOG_FILE="${LOG_FILE:-$DEFAULT_STATE_DIR/maintenance.log}"
SUMMARY_FILE="${SUMMARY_FILE:-$DEFAULT_STATE_DIR/maintenance-summary.json}"
mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$SUMMARY_FILE")"
: >> "$LOG_FILE"
chmod 600 "$LOG_FILE" 2>/dev/null || true

# Debug breadcrumb: write RUN_ID to the log right after flock acquisition
# so a forensics pass can correlate every sub-step line with the run that
# produced it. The legacy "[maintenance <ts>]" prefix is preserved so the
# existing regex-based parser in _write_summary.py keeps matching.
{
  printf '[maintenance %s] RUN_ID=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$RUN_ID"
} >> "$LOG_FILE"

LOG_PREFIX="[maintenance $(date -u +'%Y-%m-%dT%H:%M:%SZ')]"
log() {
  printf '%s %s\n' "$LOG_PREFIX" "$*"
  printf '%s %s\n' "$LOG_PREFIX" "$*" >> "$LOG_FILE"
}

failure_count=0
failed_step=""
last_error=""
mark_failure() {
  failure_count=$((failure_count + 1))
  if [ -z "$failed_step" ]; then
    failed_step="${1}"
  fi
  last_error="${1}"
  log "ERROR: $1"
}

if [ ! -x "$VENV_PY" ]; then
  log "ERROR: project venv is unavailable"
  exit 1
fi

export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

MEMORY_BRAIN="${MEMORY_BRAIN:-$SCRIPT_DIR/memory_brain.py}"
ENABLE_MEMORY_BRAIN="${ENABLE_MEMORY_BRAIN:-0}"

run_memory_brain() {
  if [ "$ENABLE_MEMORY_BRAIN" != "1" ]; then
    log "memory-brain: disabled"
    return 0
  fi
  if [ -f "$MEMORY_BRAIN" ]; then
    # Wave 2 (2026-07-21): bracket the unified pipeline run with
    # ``[brain-step] started=`` / ``[brain-step] finished=`` markers
    # so ``_write_summary.py`` can populate the ``started_at`` /
    # ``finished_at`` fields of ``steps.memory_brain`` without needing
    # to introspect the underlying ingest / consolidate subprocesses.
    log "memory-brain: unified pipeline (ingest + consolidate)"
    BRAIN_START="$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ || date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '[maintenance %s] [brain-step] run_id=%s started=%s\n' \
      "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$RUN_ID" "$BRAIN_START" >> "$LOG_FILE"
    set +e
    "$VENV_PY" "$MEMORY_BRAIN" >> "$LOG_FILE" 2>&1
    brain_rc=$?
    set -e
    BRAIN_END="$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ || date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '[maintenance %s] [brain-step] run_id=%s finished=%s exit=%s\n' \
      "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$RUN_ID" "$BRAIN_END" "$brain_rc" >> "$LOG_FILE"
    if [ "$brain_rc" -ne 0 ]; then
      mark_failure "memory-brain unified pipeline failed"
    fi
  else
    mark_failure "memory-brain component missing at $MEMORY_BRAIN"
  fi
}

log "starting maintenance"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ || date -u +%Y-%m-%dT%H:%M:%SZ)"
export MAINTENANCE_STARTED_AT="$STARTED_AT"
# Wave 2 (2026-07-21): tag the run with a mode token so the dashboard can
# distinguish daily cron runs (``daily``) from the weekly deep-audit
# invocation driven by autonomous_governance.sh (``governance``) and from
# operator-driven manual runs (``manual``). The default stays ``daily`` so
# existing callers and tests keep their semantics.
export MAINTENANCE_MODE="${MAINTENANCE_MODE:-daily}"
TOTAL_COLLECTIONS=$(echo "$COLLECTIONS" | wc -w | tr -d ' ')
STEPS_PER_COLLECTION=5
CUR=0
SKIP_INGEST_COLLECTIONS="${SKIP_INGEST_COLLECTIONS:-}"

for COLLECTION in $COLLECTIONS; do
  CUR=$((CUR + 1))
  export QDRANT_COLLECTION="$COLLECTION"
  # Bugfix 2026-07-21: run_memory_brain() was previously called once
  # before the collection loop, which meant QDRANT_COLLECTION was not
  # yet exported and memory_brain_consolidate.py fell back to the
  # hard-coded ``openclaw_memory_brain`` collection (404). The brain
  # now runs inside the loop against whatever collection is currently
  # being maintained.
  run_memory_brain
  log "--- [$CUR/$TOTAL_COLLECTIONS] collection maintenance ---"

  if echo " $SKIP_INGEST_COLLECTIONS " | grep -q " $COLLECTION "; then
    log "  step 1/$STEPS_PER_COLLECTION: ingest skipped (external ingest)"
  else
    log "  step 1/$STEPS_PER_COLLECTION: ingest"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-$PROJECT_DIR/..}" \
      "$VENV_PY" -m openclaw_memory_os.cli ingest --collection "$COLLECTION" >> "$LOG_FILE" 2>&1 \
      || mark_failure "ingest failed"
  fi

  log "  step 2/$STEPS_PER_COLLECTION: reclassify"
  "$VENV_PY" "$SCRIPT_DIR/tier_classifier.py" --collection "$COLLECTION" >> "$LOG_FILE" 2>&1 \
    || mark_failure "reclassification failed"

  log "  step 3/$STEPS_PER_COLLECTION: supersede detection"
  "$VENV_PY" "$SCRIPT_DIR/supersede_detect.py" --collection "$COLLECTION" --recency-gap-days 7 >> "$LOG_FILE" 2>&1 \
    || mark_failure "supersede detection failed"

  log "  step 4/$STEPS_PER_COLLECTION: expire old working-tier"
  "$VENV_PY" "$SCRIPT_DIR/expire_cron.py" --collection "$COLLECTION" >> "$LOG_FILE" 2>&1 \
    || mark_failure "expiry pass failed"

  log "  step 5/$STEPS_PER_COLLECTION: snapshot"
  "$SCRIPT_DIR/backup_snapshot.sh" "$COLLECTION" >> "$LOG_FILE" 2>&1 \
    || mark_failure "snapshot failed"
done

log "step 6/6: refresh lexical index"
"$VENV_PY" "$SCRIPT_DIR/refresh_lexical.py" >> "$LOG_FILE" 2>&1 \
  && log "  lexical refresh: ok" \
  || mark_failure "lexical refresh failed"
# Note: summary write happens at the end of the script so it can include
# the final exit code, finished_at, and failed_step.

WRITE_GOVERNANCE_STATUS="${WRITE_GOVERNANCE_STATUS:-0}"
if [ "$WRITE_GOVERNANCE_STATUS" = "1" ]; then
  if [ -z "${STATUS_FILE_PATH:-}" ] || [ -z "${RESULT_TOKEN:-}" ]; then
    mark_failure "governance status arguments missing"
  elif [ -x "$VENV_PY" ] && [ -f "$SCRIPT_DIR/_write_governance_status.py" ]; then
    "$VENV_PY" "$SCRIPT_DIR/_write_governance_status.py" \
      "$STATUS_FILE_PATH" "$RESULT_TOKEN" "${SUMMARY_STRING:-maintenance completed}" \
      || mark_failure "governance status write failed"
  else
    mark_failure "governance status writer unavailable"
  fi
fi

# Wave 6 (2026-07-22): when governance calls maintenance.sh, the
# maintenance summary must NOT be overwritten — governance has its own
# status file (autonomous-governance.json). Overwriting the daily
# maintenance summary with governance timestamps causes the dashboard
# to show governance start/finish times on the maintenance card,
# confusing operators.  When MAINTENANCE_MODE=governance, we skip the
# summary write entirely so the daily maintenance summary survives.
SKIP_SUMMARY_WRITE="${SKIP_SUMMARY_WRITE:-0}"
if [ "$MAINTENANCE_MODE" = "governance" ]; then
  SKIP_SUMMARY_WRITE=1
fi

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ || date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$failure_count" -gt 0 ]; then
  log "completed with failures=$failure_count failed_step=$failed_step"
  export MAINTENANCE_FINISHED_AT="$FINISHED_AT"
  export MAINTENANCE_FAILED_STEP="$failed_step"
  export MAINTENANCE_STATUS="failed"
  export MAINTENANCE_EXIT_CODE="1"
  if [ "$SKIP_SUMMARY_WRITE" != "1" ]; then
    "$VENV_PY" "$SCRIPT_DIR/_write_summary.py" "$LOG_FILE" "$SUMMARY_FILE" \
      || log "ERROR: summary write failed in failed-run path"
  else
    log "summary write skipped (governance mode)"
  fi
  exit 1
fi
log "ok"
export MAINTENANCE_FINISHED_AT="$FINISHED_AT"
export MAINTENANCE_FAILED_STEP=""
export MAINTENANCE_STATUS="success"
export MAINTENANCE_EXIT_CODE="0"
if [ "$SKIP_SUMMARY_WRITE" != "1" ]; then
  "$VENV_PY" "$SCRIPT_DIR/_write_summary.py" "$LOG_FILE" "$SUMMARY_FILE" \
    || log "ERROR: summary write failed in success-run path"
else
  log "summary write skipped (governance mode)"
fi
exit 0
