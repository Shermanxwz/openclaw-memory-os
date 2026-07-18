#!/usr/bin/env bash
# maintenance.sh — Daily maintenance entry point for OpenClaw Memory OS.
#
# The pipeline is best-effort within a run: later collections and cleanup
# stages still execute after an earlier error. The process exits non-zero at
# the end when any required stage failed, preventing cron/governance from
# reporting a partial run as green.

set -euo pipefail
flock_path="/tmp/openclaw-memory-os.maintenance.lock"
exec 9>"$flock_path"
if ! flock -n 9; then
  echo "maintenance.sh: another instance holds the lock; SKIPPED"
  exit 75
fi

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

LOG_PREFIX="[maintenance $(date -u +'%Y-%m-%dT%H:%M:%SZ')]"
log() {
  printf '%s %s\n' "$LOG_PREFIX" "$*"
  printf '%s %s\n' "$LOG_PREFIX" "$*" >> "$LOG_FILE"
}

failure_count=0
mark_failure() {
  failure_count=$((failure_count + 1))
  log "ERROR: $1"
}

if [ ! -x "$VENV_PY" ]; then
  log "ERROR: project venv is unavailable"
  exit 1
fi

export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

MEMORY_BRAIN_INGEST="${MEMORY_BRAIN_INGEST:-$SCRIPT_DIR/memory_brain_ingest.py}"
MEMORY_BRAIN_CONSOLIDATE="${MEMORY_BRAIN_CONSOLIDATE:-$SCRIPT_DIR/memory_brain_consolidate.py}"
ENABLE_MEMORY_BRAIN="${ENABLE_MEMORY_BRAIN:-0}"

run_memory_brain() {
  if [ "$ENABLE_MEMORY_BRAIN" != "1" ]; then
    log "memory-brain: disabled"
    return 0
  fi
  if [ -f "$MEMORY_BRAIN_INGEST" ]; then
    log "memory-brain: ingest"
    "$VENV_PY" "$MEMORY_BRAIN_INGEST" >> "$LOG_FILE" 2>&1 \
      || mark_failure "memory-brain ingest failed"
  else
    mark_failure "memory-brain ingest component missing"
  fi
  if [ -f "$MEMORY_BRAIN_CONSOLIDATE" ]; then
    log "memory-brain: consolidate"
    "$VENV_PY" "$MEMORY_BRAIN_CONSOLIDATE" >> "$LOG_FILE" 2>&1 \
      || mark_failure "memory-brain consolidate failed"
  else
    mark_failure "memory-brain consolidate component missing"
  fi
}

log "starting maintenance"
run_memory_brain
TOTAL_COLLECTIONS=$(echo "$COLLECTIONS" | wc -w | tr -d ' ')
STEPS_PER_COLLECTION=5
CUR=0
SKIP_INGEST_COLLECTIONS="${SKIP_INGEST_COLLECTIONS:-}"

for COLLECTION in $COLLECTIONS; do
  CUR=$((CUR + 1))
  export QDRANT_COLLECTION="$COLLECTION"
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

log "step 6/6: refresh lexical index and write summary"
"$VENV_PY" "$SCRIPT_DIR/refresh_lexical.py" >> "$LOG_FILE" 2>&1 \
  && log "  lexical refresh: ok" \
  || mark_failure "lexical refresh failed"
"$VENV_PY" "$SCRIPT_DIR/_write_summary.py" "$LOG_FILE" "$SUMMARY_FILE" \
  || mark_failure "summary write failed"

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

if [ "$failure_count" -gt 0 ]; then
  log "completed with failures=$failure_count"
  exit 1
fi
log "ok"
