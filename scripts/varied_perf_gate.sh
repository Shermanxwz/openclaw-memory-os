#!/usr/bin/env bash
# Host-only varied-query performance gate. Does not run on GitHub-hosted CI.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
OUT="${VARIED_PERF_REPORT:-$PROJECT_DIR/varied-perf-report.json}"
exec "$PYTHON" "$SCRIPT_DIR/varied_perf_gate.py" --out "$OUT" "$@"
