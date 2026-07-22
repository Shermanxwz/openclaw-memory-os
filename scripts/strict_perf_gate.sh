#!/usr/bin/env bash
# Strict performance gate runner for the recall pipeline.
#
# Boots the live uvicorn (assumed already running on 127.0.0.1:7788)
# and runs scripts/strict_perf_gate.py with:
#   - 10 warmup, 200 measured iterations
#   - concurrency 1 and 5
# Then asserts the JSON report shows overall_pass=true for both
# HTTP (≤ 300 ms p95) and server-side (≤ 150 ms p95) gates.
#
# Exit codes:
#   0  — both gates green at both concurrency levels
#   1  — gate failed
#   2  — script not runnable (missing python / not on PATH)
#   3  — service unreachable
#
# This script is invoked from .github/workflows/ci.yml under
# `jobs.strict-perf`. Run it manually with:
#   bash scripts/strict_perf_gate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not on PATH; strict perf gate cannot run" >&2
    exit 2
fi

# Sanity-check the service is up before spending 200 × 2 = 400 requests
# on a dead port.
if ! curl -sf -m 3 http://127.0.0.1:7788/health >/dev/null; then
    echo "service not reachable on 127.0.0.1:7788; strict perf gate deferred" >&2
    exit 3
fi

REPORT_DIR="${REPO_ROOT}/.bench-cache/strict-perf"
mkdir -p "${REPORT_DIR}"
REPORT_PATH="${REPORT_DIR}/$(date -u +%Y%m%dT%H%M%SZ).json"

set +e
python3 scripts/strict_perf_gate.py \
    --warmup 10 \
    --measured 200 \
    --concurrency 1,5 \
    --out "${REPORT_PATH}"
exit_code=$?
set -e

# Symlink "latest" so CI / docs can reference a stable name.
ln -sfn "${REPORT_PATH}" "${REPORT_DIR}/latest.json" 2>/dev/null || true

if [[ ${exit_code} -eq 0 ]]; then
    echo "STRICT PERF GATE: PASS"
else
    echo "STRICT PERF GATE: FAIL (exit=${exit_code}); report at ${REPORT_PATH}"
fi
exit ${exit_code}
