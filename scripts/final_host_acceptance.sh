#!/usr/bin/env bash
# Final, real-host graduation gate for OpenClaw Memory OS v0.3.0.
#
# This script is intentionally host-only. It exercises the installed service,
# real Qdrant/Ollama dependencies, authentication persistence, varied-query
# performance, the complete test suite, and two real governance invocations.
# It never physically deletes memories.
#
# Required acknowledgement:
#   sudo FINAL_ACCEPTANCE_ACK=YES \
#     ACCEPTANCE_COLLECTIONS="collection_a collection_b" \
#     scripts/final_host_acceptance.sh

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
SERVICE_NAME="${SERVICE_NAME:-openclaw-memory-os.service}"
MAINTENANCE_TIMER="${MAINTENANCE_TIMER:-openclaw-memory-os-maintenance.timer}"
GOVERNANCE_SERVICE="${GOVERNANCE_SERVICE:-openclaw-memory-os-governance.service}"
GOVERNANCE_TIMER="${GOVERNANCE_TIMER:-openclaw-memory-os-governance.timer}"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
ACCEPTANCE_COLLECTIONS="${ACCEPTANCE_COLLECTIONS:-}"
ACCEPTANCE_MIN_POINTS="${ACCEPTANCE_MIN_POINTS:-20000}"
ACCEPTANCE_QUERY="${ACCEPTANCE_QUERY:-}"
ACCEPTANCE_MIN_HITS="${ACCEPTANCE_MIN_HITS:-1}"
ACCEPTANCE_GOVERNANCE_GAP_SECONDS="${ACCEPTANCE_GOVERNANCE_GAP_SECONDS:-610}"
RESTORE_PROOF_FILE="${RESTORE_PROOF_FILE:-}"
EVOLUTION_PROOF_FILE="${EVOLUTION_PROOF_FILE:-}"
STAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
EVIDENCE_DIR="${ACCEPTANCE_EVIDENCE_DIR:-/var/lib/openclaw-memory-os/acceptance/$STAMP}"
STATUS_FILE="/var/lib/openclaw-memory-os/state/openclaw-memory-os/autonomous-governance.json"

pass() { printf 'PASS  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*" >&2; exit 1; }
run_logged() {
    local name="$1"; shift
    printf '\n==== %s ====\n' "$name"
    "$@" 2>&1 | tee "$EVIDENCE_DIR/${name}.log"
}

[[ "${FINAL_ACCEPTANCE_ACK:-}" == "YES" ]] \
    || fail "set FINAL_ACCEPTANCE_ACK=YES after confirming this is the real acceptance host"
[[ "$(id -u)" -eq 0 ]] || fail "run as root; auth persistence and systemd gates restart services"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "python3.12 is required"
command -v gitleaks >/dev/null 2>&1 || fail "gitleaks is required for complete-history acceptance"
[[ -d "$PROJECT_ROOT/.git" ]] || fail "a full Git checkout is required for history scanning"
[[ "$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.12" ]] \
    || fail "acceptance requires CPython 3.12"
[[ -r "$ENV_FILE" ]] || fail ".env is not readable"
[[ -n "$ACCEPTANCE_COLLECTIONS" ]] || fail "set ACCEPTANCE_COLLECTIONS to the real Qdrant collections"
[[ "$ACCEPTANCE_MIN_POINTS" =~ ^[0-9]+$ ]] || fail "ACCEPTANCE_MIN_POINTS must be an integer"
[[ -n "$ACCEPTANCE_QUERY" ]] || fail "set ACCEPTANCE_QUERY to a known phrase that must return real hits"
[[ "$ACCEPTANCE_MIN_HITS" =~ ^[0-9]+$ ]] || fail "ACCEPTANCE_MIN_HITS must be an integer"
[[ "$ACCEPTANCE_GOVERNANCE_GAP_SECONDS" =~ ^[0-9]+$ ]] || fail "ACCEPTANCE_GOVERNANCE_GAP_SECONDS must be an integer"
[[ "$ACCEPTANCE_GOVERNANCE_GAP_SECONDS" -ge 600 ]] || fail "governance windows must be at least 600 seconds apart"
[[ -r "$RESTORE_PROOF_FILE" ]] || fail "RESTORE_PROOF_FILE must point to OpenClaw disposable-restore evidence"
[[ -r "$EVOLUTION_PROOF_FILE" ]] || fail "EVOLUTION_PROOF_FILE must point to OpenClaw controlled evolution evidence"
install -d -m 0700 "$EVIDENCE_DIR"

printf '%s\n' "$STAMP" > "$EVIDENCE_DIR/started-at.txt"
if git -C "$PROJECT_ROOT" rev-parse HEAD > "$EVIDENCE_DIR/git-sha.txt" 2>/dev/null; then
    git -C "$PROJECT_ROOT" status --porcelain > "$EVIDENCE_DIR/git-status.txt"
    [[ ! -s "$EVIDENCE_DIR/git-status.txt" ]] || fail "working tree is not clean"
fi
sha256sum "$PROJECT_ROOT/requirements/runtime-py312.lock" > "$EVIDENCE_DIR/runtime-lock.sha256"

# 1. Service identity, isolation and timers.
run_logged systemd-status systemctl status --no-pager "$SERVICE_NAME" "$MAINTENANCE_TIMER" "$GOVERNANCE_TIMER"
[[ "$(systemctl show -p User --value "$SERVICE_NAME")" == "openclaw-memory-os" ]] \
    || fail "web service is not running as openclaw-memory-os"
systemctl is-active --quiet "$SERVICE_NAME" || fail "web service inactive"
systemctl is-active --quiet "$MAINTENANCE_TIMER" || fail "maintenance timer inactive"
systemctl is-active --quiet "$GOVERNANCE_TIMER" || fail "governance timer inactive"
pass "service identity and persistent timers"

# 2. Recreate the audited test environment separately from production.
ACCEPTANCE_VENV="$EVIDENCE_DIR/test-venv"
"$PYTHON_BIN" -m venv "$ACCEPTANCE_VENV"
run_logged install-test-environment "$ACCEPTANCE_VENV/bin/python" -m pip install \
    --disable-pip-version-check -r "$PROJECT_ROOT/requirements/dev-py312.lock"
run_logged install-product "$ACCEPTANCE_VENV/bin/python" -m pip install \
    --disable-pip-version-check --no-deps "$PROJECT_ROOT"
run_logged pip-check "$ACCEPTANCE_VENV/bin/python" -m pip check
run_logged compile "$ACCEPTANCE_VENV/bin/python" -W error::SyntaxWarning -m compileall -q \
    "$PROJECT_ROOT/openclaw_memory_os" "$PROJECT_ROOT/scripts"
run_logged pytest env PYTHONPATH="$PROJECT_ROOT" "$ACCEPTANCE_VENV/bin/python" -m pytest -q "$PROJECT_ROOT/tests"
run_logged privacy-scan "$PROJECT_ROOT/scripts/privacy_scan.sh"
run_logged test-pip-freeze "$ACCEPTANCE_VENV/bin/python" -m pip freeze
install -d -m 0700 "$EVIDENCE_DIR/wheelhouse" "$EVIDENCE_DIR/dist"
run_logged download-runtime-wheelhouse "$ACCEPTANCE_VENV/bin/python" -m pip download \
    --disable-pip-version-check --only-binary=:all: \
    --dest "$EVIDENCE_DIR/wheelhouse" -r "$PROJECT_ROOT/requirements/runtime-py312.lock"
sha256sum "$EVIDENCE_DIR"/wheelhouse/* > "$EVIDENCE_DIR/wheelhouse.sha256"
install -m 0600 "$PROJECT_ROOT/requirements/runtime-py312.lock" \
    "$PROJECT_ROOT/requirements/dev-py312.lock" "$EVIDENCE_DIR/"
run_logged build-wheel "$ACCEPTANCE_VENV/bin/python" -m pip wheel \
    --disable-pip-version-check --no-deps --wheel-dir "$EVIDENCE_DIR/dist" "$PROJECT_ROOT"
WHEEL_PATH="$(find "$EVIDENCE_DIR/dist" -maxdepth 1 -type f -name 'openclaw_memory_os-*.whl' -print -quit)"
[[ -n "$WHEEL_PATH" ]] || fail "wheel build did not produce an artifact"
sha256sum "$WHEEL_PATH" > "$EVIDENCE_DIR/wheel.sha256"
WHEEL_VENV="$EVIDENCE_DIR/wheel-smoke-venv"
"$PYTHON_BIN" -m venv "$WHEEL_VENV"
run_logged install-wheel-runtime "$WHEEL_VENV/bin/python" -m pip install \
    --disable-pip-version-check -r "$PROJECT_ROOT/requirements/runtime-py312.lock"
run_logged install-built-wheel "$WHEEL_VENV/bin/python" -m pip install \
    --disable-pip-version-check --no-deps "$WHEEL_PATH"
run_logged wheel-pip-check "$WHEEL_VENV/bin/python" -m pip check
run_logged wheel-smoke "$WHEEL_VENV/bin/python" -c \
    "import openclaw_memory_os; from openclaw_memory_os.app import create_app; assert create_app().title"
rm -rf "$WHEEL_VENV" "$ACCEPTANCE_VENV"
pass "clean source-test and wheel-smoke environments removed"

# 3. Real Qdrant corpus, required local models and live recall modes.
ENV_FILE="$ENV_FILE" ACCEPTANCE_COLLECTIONS="$ACCEPTANCE_COLLECTIONS" \
ACCEPTANCE_MIN_POINTS="$ACCEPTANCE_MIN_POINTS" ACCEPTANCE_QUERY="$ACCEPTANCE_QUERY" \
ACCEPTANCE_MIN_HITS="$ACCEPTANCE_MIN_HITS" PROJECT_ROOT="$PROJECT_ROOT" \
"$PROJECT_ROOT/.venv/bin/python" - <<'PY' | tee "$EVIDENCE_DIR/live-dependencies.json"
from __future__ import annotations
import json, os, re, urllib.request
from pathlib import Path


def dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_json(url: str, *, headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"GET failed: HTTP {response.status}")
        return json.load(response)


def post_json(url: str, body: dict, token: str) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        if response.status != 200:
            raise RuntimeError(f"recall failed: HTTP {response.status}")
        return json.load(response)

cfg = {**dotenv(Path(os.environ["ENV_FILE"])), **os.environ}
qdrant = cfg.get("QDRANT_URL", "").rstrip("/")
ollama = cfg.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
token = cfg.get("MEMORY_OS_TOKEN", "")
qdrant_headers = {"api-key": cfg["QDRANT_API_KEY"]} if cfg.get("QDRANT_API_KEY") else {}
if not qdrant or not token:
    raise SystemExit("QDRANT_URL and MEMORY_OS_TOKEN are required")
collections = os.environ["ACCEPTANCE_COLLECTIONS"].split()
minimum = int(os.environ["ACCEPTANCE_MIN_POINTS"])
counts: dict[str, int] = {}
for collection in collections:
    payload = get_json(f"{qdrant}/collections/{collection}", headers=qdrant_headers)
    result = payload.get("result") or {}
    count = int(result.get("points_count") or result.get("vectors_count") or 0)
    if count <= 0:
        raise SystemExit(f"collection has no points: {collection}")
    counts[collection] = count
if sum(counts.values()) < minimum:
    raise SystemExit(f"corpus too small: {sum(counts.values())} < {minimum}")
models = get_json(f"{ollama}/api/tags").get("models") or []
names = {str(m.get("name") or m.get("model") or "") for m in models}
for required in ("nomic-embed-text", "qwen2.5:1.5b"):
    if not any(name == required or name.startswith(required + ":") for name in names):
        raise SystemExit(f"required local model unavailable: {required}")
recalls = {}
for mode in ("keyword", "dense", "hybrid"):
    payload = post_json(
        "http://127.0.0.1:7788/api/recall-test",
        {"query": os.environ["ACCEPTANCE_QUERY"], "mode": mode, "limit": 10},
        token,
    )
    hits = payload.get("hits") or []
    if len(hits) < int(os.environ["ACCEPTANCE_MIN_HITS"]):
        raise SystemExit(f"{mode} returned too few hits: {len(hits)}")
    if not payload.get("query_id") or not payload.get("policy_version"):
        raise SystemExit(f"{mode} missing query/policy identity")
    diagnostics = payload.get("diagnostics") or {}
    if not diagnostics:
        raise SystemExit(f"{mode} missing diagnostics")
    for hit in hits:
        key = str(hit.get("candidate_key") or "")
        if key and ":" not in key:
            raise SystemExit(f"{mode} returned unqualified candidate key")
    recalls[mode] = {
        "hits": len(hits),
        "status": diagnostics.get("status"),
        "collections_searched": diagnostics.get("collections_searched") or [],
    }
print(json.dumps({"collections": counts, "total_points": sum(counts.values()), "recall": recalls}, indent=2, sort_keys=True))
PY
pass "real Qdrant, fixed Ollama models and three recall modes"

# 4. Authentication/session persistence and varied-query performance.
run_logged auth-smoke env ENV_FILE="$ENV_FILE" SERVICE_NAME="${SERVICE_NAME%.service}" \
    HOME=/var/lib/openclaw-memory-os XDG_STATE_HOME=/var/lib/openclaw-memory-os/state \
    "$PROJECT_ROOT/scripts/auth_smoke.sh"
run_logged varied-perf env ENV_FILE="$ENV_FILE" \
    VARIED_PERF_REPORT="$EVIDENCE_DIR/varied-perf.json" \
    "$PROJECT_ROOT/scripts/varied_perf_gate.sh"

# 5. Two real governance invocations separated by the production minimum
# window gap. Controlled promotion/rollback/circuit-breaker evidence is
# validated separately below so a normal corpus is never forced to promote a
# worse candidate merely to satisfy acceptance.
run_logged governance-window-1 systemctl start --wait "$GOVERNANCE_SERVICE"
printf "waiting %s seconds for a distinct governance window\n" "$ACCEPTANCE_GOVERNANCE_GAP_SECONDS" \
    | tee "$EVIDENCE_DIR/governance-gap.log"
sleep "$ACCEPTANCE_GOVERNANCE_GAP_SECONDS"
run_logged governance-window-2 systemctl start --wait "$GOVERNANCE_SERVICE"
STATUS_FILE="$STATUS_FILE" "$PROJECT_ROOT/.venv/bin/python" - <<'PY' | tee "$EVIDENCE_DIR/governance-status.json"
import json, os
from pathlib import Path
p = Path(os.environ["STATUS_FILE"])
payload = json.loads(p.read_text(encoding="utf-8"))
if payload.get("last_result") != "ok":
    raise SystemExit(f"governance is not green: {payload!r}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
pass "two real governance windows completed with honest green status"

# 6. Controlled host proofs supplied by OpenClaw. The restore drill must use a
# disposable Qdrant clone and must not mutate the production collections. The
# evolution drill must restore the original production policy/state afterward.
RESTORE_PROOF_FILE="$RESTORE_PROOF_FILE" EVOLUTION_PROOF_FILE="$EVOLUTION_PROOF_FILE" \
ACCEPTANCE_MIN_POINTS="$ACCEPTANCE_MIN_POINTS" \
"$PROJECT_ROOT/.venv/bin/python" - <<'PY' | tee "$EVIDENCE_DIR/controlled-host-proofs.json"
import json, os, re
from pathlib import Path

def load(name: str) -> dict:
    return json.loads(Path(os.environ[name]).read_text(encoding="utf-8"))

restore = load("RESTORE_PROOF_FILE")
evolution = load("EVOLUTION_PROOF_FILE")
if restore.get("status") != "passed":
    raise SystemExit("restore proof is not passed")
if restore.get("environment") != "disposable":
    raise SystemExit("restore proof must come from a disposable environment")
if restore.get("production_mutated") is not False:
    raise SystemExit("restore proof must state production_mutated=false")
if int(restore.get("restored_points") or 0) < int(os.environ["ACCEPTANCE_MIN_POINTS"]):
    raise SystemExit("restore proof does not cover the accepted corpus scale")
if not re.fullmatch(r"[0-9a-fA-F]{64}", str(restore.get("source_snapshot_sha256") or "")):
    raise SystemExit("restore proof is missing a SHA-256 snapshot digest")
for proof_name, proof in (("restore", restore), ("evolution", evolution)):
    if not str(proof.get("tested_at") or "").strip():
        raise SystemExit(f"{proof_name} proof is missing tested_at")
if evolution.get("status") != "passed":
    raise SystemExit("evolution proof is not passed")
for key in ("same_candidate_two_windows", "rollback_to_previous", "circuit_breaker", "production_policy_restored"):
    if evolution.get(key) is not True:
        raise SystemExit(f"evolution proof failed contract: {key}")
print(json.dumps({"restore": restore, "evolution": evolution}, indent=2, sort_keys=True))
PY
pass "disposable restore and controlled evolution proofs"

# 7. Complete-history secret scan is mandatory for final acceptance.
run_logged gitleaks-history gitleaks git "$PROJECT_ROOT" --redact --no-banner

COMMIT_SHA="$(cat "$EVIDENCE_DIR/git-sha.txt")"
SOURCE_ARCHIVE="$EVIDENCE_DIR/openclaw-memory-os-${COMMIT_SHA}.tar.gz"
git -C "$PROJECT_ROOT" archive --format=tar.gz --output="$SOURCE_ARCHIVE" HEAD
sha256sum "$SOURCE_ARCHIVE" > "$EVIDENCE_DIR/source-archive.sha256"
sha256sum "$RESTORE_PROOF_FILE" "$EVOLUTION_PROOF_FILE" > "$EVIDENCE_DIR/host-proofs.sha256"

cat > "$EVIDENCE_DIR/result.json" <<JSON
{"status":"passed","completed_at":"$(date -u +'%Y-%m-%dT%H:%M:%SZ')","evidence_dir":"$EVIDENCE_DIR"}
JSON
chmod 0600 "$EVIDENCE_DIR/result.json"
printf '\nHOST_ACCEPTANCE_PASSED evidence=%s\n' "$EVIDENCE_DIR"
