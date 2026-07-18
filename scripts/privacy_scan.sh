#!/usr/bin/env bash
# Privacy scanner wrapper. Exits non-zero if the in-repo scanner finds
# anything not suppressed by either:
#   - the per-line `privacy-allow: <RULE_ID>` marker, or
#   - the JSON baseline pinned at scripts/privacy_baseline.json
#
# Usage:
#   ./scripts/privacy_scan.sh                # scan the whole repo
#   ./scripts/privacy_scan.sh openclaw_memory_os   # scan a subtree
#
# Pass --update-baseline to refresh scripts/privacy_baseline.json instead of
# failing.

set -euo pipefail

cd "$(dirname "$0")/.."

# Prefer the project's venv so we get the same deps as the rest of the
# repo. Fall back to a system python only if the venv is missing.
if [ -x .venv/bin/python ]; then
  PY_BIN=.venv/bin/python
elif command -v python >/dev/null 2>&1; then
  PY_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN=python3
else
  echo "no python interpreter found" >&2
  exit 2
fi

BASELINE="${BASELINE:-scripts/privacy_baseline.json}"

ARGS=()
if [ -f "$BASELINE" ]; then
  ARGS+=(--baseline "$BASELINE")
fi
if [ "${UPDATE_BASELINE:-0}" = "1" ]; then
  ARGS+=(--update-baseline)
fi

exec "$PY_BIN" -m openclaw_memory_os.privacy "${ARGS[@]}" "$@"
