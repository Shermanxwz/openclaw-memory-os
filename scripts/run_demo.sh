#!/usr/bin/env bash
# Run a quick demo against the in-repo sample backend.
#
# Usage:
#   ./scripts/run_demo.sh
#
# What it does:
#   1. Optionally create a virtualenv.
#   2. Install the package in editable mode with the demo extras.
#   3. Print the health summary via the CLI (offline-friendly).
#   4. Run a few sample recall queries.
#   5. Optionally boot the HTTP server (set DEMO_SERVE=1).
#
# All credentials in this script are placeholders. No real secrets.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"
if [ ! -d ".venv" ]; then
  echo "[demo] creating virtualenv under .venv ..."
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --quiet -e ".[dev]"

echo
echo "[demo] health summary (via CLI, hits the sample backend):"
echo "------------------------------------------------------------"
openclaw-memory-os health | head -40
echo

echo "[demo] recall --query 'recall test':"
echo "------------------------------------------------------------"
openclaw-memory-os recall --query "recall test" --mode hybrid --limit 5
echo

echo "[demo] recall --query 'deletion policy':"
echo "------------------------------------------------------------"
openclaw-memory-os recall --query "deletion policy" --mode hybrid --limit 5
echo

echo "[demo] privacy-scan (should be clean):"
echo "------------------------------------------------------------"
openclaw-memory-os privacy-scan .
echo

if [ "${DEMO_SERVE:-0}" = "1" ]; then
  echo "[demo] booting HTTP server at http://127.0.0.1:7788 ... (Ctrl-C to stop)"
  echo "[demo] auth: MEMORY_OS_TOKEN=demo-token ; visit /login"
  MEMORY_OS_TOKEN=demo-token openclaw-memory-os serve --host 127.0.0.1 --port 7788
fi
