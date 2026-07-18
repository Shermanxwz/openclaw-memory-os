#!/usr/bin/env bash
# autonomous_governance.sh — Weekly deep governance runner.
#
# Hard contracts:
#   * Never physically delete memories.
#   * Never write collection names / paths / IPs / tokens to the status file.
#   * Continue far enough to persist an honest final status after failures.
#   * Exit zero only when every required stage succeeds.

set -euo pipefail

flock_path="/tmp/openclaw-memory-os.governance.lock"
exec 9>"$flock_path"
if ! flock -n 9; then
  echo "autonomous_governance.sh: another instance holds the lock; skipping" >&2
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$PROJECT_DIR/.venv/bin/python"
MAINTENANCE_SH="$SCRIPT_DIR/maintenance.sh"
WRITE_STATUS_PY="$SCRIPT_DIR/_write_governance_status.py"
EVOLUTION_PY="$SCRIPT_DIR/run_evolution_cycle.py"
REPLAY_PY="$SCRIPT_DIR/replay_feedback.py"

DEFAULT_STATUS_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/openclaw-memory-os/autonomous-governance.json"
STATUS_FILE_PATH="${STATUS_FILE_PATH:-$DEFAULT_STATUS_FILE}"

FORCE_CONTENT_SUPERSEDE="${FORCE_CONTENT_SUPERSEDE:-1}"
ENABLE_AUTO_SUPERSEDE="${ENABLE_AUTO_SUPERSEDE:-1}"
SUPERSEDE_MAX_APPLY="${SUPERSEDE_MAX_APPLY:-50}"
GOVERNANCE_COLLECTIONS_DEFAULT="${GOVERNANCE_COLLECTIONS:-openclaw_memory_os}"
MAINTAIN_COLLECTIONS="${MAINTAIN_COLLECTIONS:-$GOVERNANCE_COLLECTIONS_DEFAULT}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$PROJECT_DIR/..}"

if [ -n "${LOG_FILE:-}" ]; then
  LOG_FILE="$LOG_FILE"
else
  preferred_log="/var/log/openclaw-memory-os/governance.log"
  if mkdir -p "$(dirname "$preferred_log")" 2>/dev/null && : >> "$preferred_log" 2>/dev/null; then
    LOG_FILE="$preferred_log"
  else
    LOG_FILE="$(dirname "$STATUS_FILE_PATH")/governance.log"
  fi
fi
mkdir -p "$(dirname "$LOG_FILE")"
: >> "$LOG_FILE"
chmod 600 "$LOG_FILE" 2>/dev/null || true

LOG_PREFIX="[governance $(date -u +'%Y-%m-%dT%H:%M:%SZ')]"
log() { printf '%s %s\n' "$LOG_PREFIX" "$*"; }

write_early_failure() {
  local summary="$1"
  local fallback_py
  fallback_py="$(command -v /usr/bin/python3 || command -v python3 || true)"
  if [ -n "$fallback_py" ] && [ -f "$WRITE_STATUS_PY" ]; then
    "$fallback_py" "$WRITE_STATUS_PY" "$STATUS_FILE_PATH" "failed" "$summary" \
      || log "ERROR: status write failed during preflight"
  fi
}

if [ ! -x "$VENV_PY" ]; then
  log "ERROR: project venv is unavailable"
  write_early_failure "project venv unavailable"
  exit 1
fi
if [ ! -x "$MAINTENANCE_SH" ]; then
  log "ERROR: maintenance runner is unavailable"
  "$VENV_PY" "$WRITE_STATUS_PY" "$STATUS_FILE_PATH" "failed" "maintenance runner unavailable" || true
  exit 1
fi
if [ ! -f "$WRITE_STATUS_PY" ] || [ ! -f "$EVOLUTION_PY" ] || [ ! -f "$REPLAY_PY" ]; then
  log "ERROR: governance component is unavailable"
  if [ -f "$WRITE_STATUS_PY" ]; then
    "$VENV_PY" "$WRITE_STATUS_PY" "$STATUS_FILE_PATH" "failed" "governance component unavailable" || true
  fi
  exit 1
fi

log "starting deep governance"

# Required stage 1: durable feedback must be replayable before evaluation.
set +e
"$VENV_PY" "$REPLAY_PY" >> "$LOG_FILE" 2>&1
feedback_rc=$?
set -e
if [ "$feedback_rc" -eq 0 ]; then
  log "feedback replay: ok"
else
  log "feedback replay: failed (exit=$feedback_rc)"
fi

# Required stage 2: run the full maintenance pipeline. It continues through
# individual failures and returns non-zero at the end when any required step
# failed, so this runner can persist one honest aggregate result.
set +e
FORCE_CONTENT_SUPERSEDE="$FORCE_CONTENT_SUPERSEDE" \
ENABLE_AUTO_SUPERSEDE="$ENABLE_AUTO_SUPERSEDE" \
SUPERSEDE_MAX_APPLY="$SUPERSEDE_MAX_APPLY" \
MAINTAIN_COLLECTIONS="$MAINTAIN_COLLECTIONS" \
WORKSPACE_ROOT="$WORKSPACE_ROOT" \
  "$MAINTENANCE_SH" >> "$LOG_FILE" 2>&1
audit_rc=$?
set -e
log "deep audit exit=$audit_rc"

# Required stage 3: evolve only after a complete maintenance pass. Running
# evolution against a partially updated corpus would make the candidate gate
# misleading. Lock-held evolution remains a successful no-op because the
# Python runner returns status=skipped with exit zero and leaves Active intact.
evolution_rc=0
EVOLUTION_STATE="not-run"
if [ "$audit_rc" -eq 0 ]; then
  set +e
  EVOLUTION_STATE="$("$VENV_PY" "$EVOLUTION_PY" 2>&1)"
  evolution_rc=$?
  set -e
  if [ "$evolution_rc" -eq 0 ]; then
    log "evolution: $EVOLUTION_STATE"
  else
    log "evolution: failed (exit=$evolution_rc)"
  fi
elif [ "$audit_rc" -eq 75 ]; then
  log "evolution: skipped because maintenance lock is held"
else
  log "evolution: skipped because maintenance did not complete"
fi

RESULT_TOKEN="ok"
SUMMARY_TEXT="governance completed"
final_rc=0

if [ "$audit_rc" -eq 75 ]; then
  RESULT_TOKEN="skipped"
  SUMMARY_TEXT="deep audit skipped; maintenance lock held"
  final_rc=75
elif [ "$audit_rc" -ne 0 ]; then
  RESULT_TOKEN="failed"
  SUMMARY_TEXT="maintenance failed; exit=$audit_rc"
  final_rc="$audit_rc"
elif [ "$evolution_rc" -ne 0 ]; then
  RESULT_TOKEN="failed"
  SUMMARY_TEXT="evolution failed; exit=$evolution_rc"
  final_rc="$evolution_rc"
elif [ "$feedback_rc" -ne 0 ]; then
  RESULT_TOKEN="degraded"
  SUMMARY_TEXT="feedback replay failed; maintenance and evolution completed"
  final_rc="$feedback_rc"
fi

# Persist the final result only after all attempted stages have completed.
# A writer failure is itself an alertable failure because the dashboard would
# otherwise retain stale green state.
set +e
"$VENV_PY" "$WRITE_STATUS_PY" "$STATUS_FILE_PATH" "$RESULT_TOKEN" "$SUMMARY_TEXT"
status_rc=$?
set -e
if [ "$status_rc" -ne 0 ]; then
  log "ERROR: final status write failed (exit=$status_rc)"
  [ "$final_rc" -ne 0 ] || final_rc="$status_rc"
else
  log "status written: result=$RESULT_TOKEN"
fi

if [ "$final_rc" -eq 0 ]; then
  log "ok"
else
  log "completed with non-zero result=$RESULT_TOKEN exit=$final_rc"
fi
exit "$final_rc"
