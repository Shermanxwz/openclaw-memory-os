#!/usr/bin/env bash
# Full host-side authentication graduation smoke.
#
# Requires a live FastAPI service, its real SessionStore SQLite DB, curl,
# and the project Python environment. This script intentionally is NOT
# run by GitHub-hosted CI because systemd/restart persistence must be exercised
# on the operator host.
#
# IMPORTANT: This script NEVER uses /usr/bin/sqlite3 to query the production
# sessions DB. The standalone sqlite3 binary opens databases in read-write mode
# by default, which modifies WAL/SHM files even for SELECT queries and can
# corrupt the WAL if the process is interrupted. Instead, all DB queries go
# through scripts/session_readonly_helper.py which uses Python's sqlite3 module
# with ?mode=ro URI and PRAGMA query_only=ON.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:7788}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
SERVICE_NAME="${SERVICE_NAME:-openclaw-memory-os}"
SESSIONS_DB="${SESSIONS_DB:-}"
REQUIRE_SYSTEMD="${REQUIRE_SYSTEMD:-1}"
AUTH_SMOKE_SERVICE_WAIT_SECONDS="${AUTH_SMOKE_SERVICE_WAIT_SECONDS:-300}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

pass() { printf 'PASS  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*" >&2; exit 1; }
banner() { printf '\n==== %s ====\n' "$*"; }

need_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command missing: $1"
}

# Read a dotenv value without sourcing arbitrary shell code.
env_value() {
    local key="$1"
    [[ -r "$ENV_FILE" ]] || return 0
    awk -v k="$key" '
      /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
      {
        line=$0
        sub(/^[[:space:]]*export[[:space:]]+/, "", line)
        pos=index(line, "=")
        if (!pos) next
        lhs=substr(line,1,pos-1)
        gsub(/[[:space:]]/, "", lhs)
        if (lhs != k) next
        rhs=substr(line,pos+1)
        sub(/^[[:space:]]+/, "", rhs)
        sub(/[[:space:]]+#.*$/, "", rhs)
        if ((substr(rhs,1,1)=="\"" && substr(rhs,length(rhs),1)=="\"") ||
            (substr(rhs,1,1)=="\047" && substr(rhs,length(rhs),1)=="\047")) {
          rhs=substr(rhs,2,length(rhs)-2)
        }
        print rhs
        exit
      }
    ' "$ENV_FILE"
}

http_code() {
    curl -k -sS --max-time 20 -o /dev/null -w '%{http_code}' "$@" || printf '000'
}

expect_code() {
    local label="$1" expected="$2"
    shift 2
    local got
    got="$(http_code "$@")"
    [[ "$got" == "$expected" ]] || fail "$label expected=$expected got=$got"
    pass "$label -> $got"
}

read_cookie() {
    local jar="$1" name="$2"
    [[ -r "$jar" ]] || return 0
    awk -v n="$name" '$6 == n { print $7 }' "$jar" | tail -n1
}

get_login_csrf() {
    local jar="$1"
    rm -f "$jar"
    curl -k -sS --max-time 20 -o /dev/null -c "$jar" "$BASE_URL/login"
    local csrf
    csrf="$(read_cookie "$jar" csrf_token)"
    [[ -n "$csrf" ]] || fail "GET /login did not issue csrf_token"
    printf '%s' "$csrf"
}

current_totp() {
    TOTP_SECRET_VALUE="$TOTP_SECRET" "$PYTHON" - <<'PY'
import os
from openclaw_memory_os.auth import totp_now
print(totp_now(os.environ["TOTP_SECRET_VALUE"]))
PY
}

invalid_totp() {
    TOTP_SECRET_VALUE="$TOTP_SECRET" "$PYTHON" - <<'PY'
import os
from openclaw_memory_os.auth import verify_totp
secret = os.environ["TOTP_SECRET_VALUE"]
for value in range(1000000):
    code = f"{value:06d}"
    if not verify_totp(code, secret):
        print(code)
        break
PY
}

random_token() {
    TOKEN_LENGTH="$1" "$PYTHON" - <<'PY'
import os, secrets
length = int(os.environ["TOKEN_LENGTH"])
value = ""
while len(value) < length:
    value += secrets.token_urlsafe(length)
print(value[:length])
PY
}

cookie_hash() {
    COOKIE_VALUE="$1" "$PYTHON" - <<'PY'
import hashlib, os
print(hashlib.sha256(os.environ["COOKIE_VALUE"].encode()).hexdigest())
PY
}

resolve_sessions_db() {
    if [[ -n "$SESSIONS_DB" ]]; then printf '%s\n' "$SESSIONS_DB"; return; fi
    local configured
    configured="$(env_value MEMORY_OS_SESSIONS_DB || true)"
    if [[ -n "$configured" ]]; then printf '%s\n' "$configured"; return; fi
    local state_home="${XDG_STATE_HOME:-$HOME/.local/state}"
    printf '%s/openclaw-memory-os/sessions.db\n' "$state_home"
}

wait_for_service() {
    local timeout="${AUTH_SMOKE_SERVICE_WAIT_SECONDS}"
    case "$timeout" in
        ''|*[!0-9]*) fail "AUTH_SMOKE_SERVICE_WAIT_SECONDS must be a positive integer (got: '$timeout')" ;;
    esac
    if (( timeout < 30 || timeout > 900 )); then
        fail "AUTH_SMOKE_SERVICE_WAIT_SECONDS must be between 30 and 900 (got: $timeout)"
    fi
    local started now elapsed
    started="$(date +%s)"
    while true; do
        if [[ "$(http_code "$BASE_URL/health")" == "200" ]]; then
            now="$(date +%s)"
            elapsed=$(( now - started ))
            pass "service returned after restart in ${elapsed}s (ceiling=${timeout}s)"
            return 0
        fi
        now="$(date +%s)"
        elapsed=$(( now - started ))
        if (( elapsed >= timeout )); then
            printf 'TIMEOUT  waited %ss for %s to return 200 after restart\n' "$elapsed" "$SERVICE_NAME" >&2
            systemctl status "$SERVICE_NAME" --no-pager 2>&1 | head -40 >&2 || true
            journalctl -u "$SERVICE_NAME" --since "@$started" --no-pager -n 100 2>&1 | tail -60 >&2 || true
            fail "service failed to return after restart within ${timeout}s"
        fi
        sleep 1
    done
}

restart_service() {
    if ! command -v systemctl >/dev/null 2>&1 || ! systemctl is-active --quiet "$SERVICE_NAME"; then
        if [[ "$REQUIRE_SYSTEMD" == "1" ]]; then
            fail "systemd service $SERVICE_NAME is required for restart persistence tests"
        fi
        printf 'SKIP  restart persistence (systemd unavailable)\n'
        return 1
    fi
    systemctl restart "$SERVICE_NAME"
    wait_for_service || fail "service failed to return after restart"
    pass "service restart"
    return 0
}

login_with_totp() {
    local jar="$1"
    local csrf code status
    csrf="$(get_login_csrf "$jar")"
    code="$(current_totp)"
    status="$(http_code -X POST "$BASE_URL/login" -b "$jar" -c "$jar" \
        --data-urlencode "password=$PASSWORD" \
        --data-urlencode "totp_code=$code" \
        --data-urlencode "csrf_token=$csrf")"
    [[ "$status" == "303" ]] || fail "valid password+TOTP login expected=303 got=$status"
    local session
    session="$(read_cookie "$jar" memory_os_session)"
    [[ -n "$session" ]] || fail "valid login did not issue memory_os_session"
    printf '%s' "$session"
}

banner "preflight"
need_command curl
[[ -f "$SCRIPT_DIR/session_readonly_helper.py" ]] || fail "session_readonly_helper.py not found in $SCRIPT_DIR"
[[ -r "$ENV_FILE" ]] || fail "environment file not readable: $ENV_FILE"
[[ -x "$PYTHON" ]] || fail "python executable unavailable"
expect_code "public health" 200 "$BASE_URL/health"

TOKEN="${MEMORY_OS_TOKEN:-$(env_value MEMORY_OS_TOKEN || true)}"
PASSWORD="${MEMORY_OS_PASSWORD:-$(env_value MEMORY_OS_PASSWORD || true)}"
TOTP_SECRET="${MEMORY_OS_TOTP_SECRET:-$(env_value MEMORY_OS_TOTP_SECRET || true)}"
RECOVERY_CODE="${AUTH_SMOKE_RECOVERY_CODE:-$(env_value AUTH_SMOKE_RECOVERY_CODE || true)}"
[[ -n "$TOKEN" ]] || fail "MEMORY_OS_TOKEN is required"
[[ -n "$PASSWORD" ]] || fail "MEMORY_OS_PASSWORD is required"
[[ -n "$TOTP_SECRET" ]] || fail "MEMORY_OS_TOTP_SECRET is required"
SESSIONS_DB_PATH="$(resolve_sessions_db)"
[[ -r "$SESSIONS_DB_PATH" ]] || fail "sessions DB not readable: $SESSIONS_DB_PATH"
TMP_DIR="$(mktemp -d -t memory-os-auth-smoke.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

banner "bearer authentication"
expect_code "anonymous API" 401 "$BASE_URL/api/health"
expect_code "exact MEMORY_OS_TOKEN" 200 -H "Authorization: Bearer $TOKEN" "$BASE_URL/api/health"
expect_code "password cannot be bearer" 401 -H "Authorization: Bearer $PASSWORD" "$BASE_URL/api/health"
for length in 32 64 128; do
    value="$(random_token "$length")"
    expect_code "random ${length}-character bearer" 401 -H "Authorization: Bearer $value" "$BASE_URL/api/health"
done
expect_code "empty bearer" 401 -H "Authorization: Bearer " "$BASE_URL/api/health"

banner "login CSRF"
expect_code "login without CSRF cookie/form" 403 -X POST "$BASE_URL/login" \
    --data-urlencode "password=$PASSWORD"
JAR_CSRF="$TMP_DIR/login-csrf.jar"
csrf="$(get_login_csrf "$JAR_CSRF")"
expect_code "login with cookie but missing form CSRF" 403 -X POST "$BASE_URL/login" -b "$JAR_CSRF" \
    --data-urlencode "password=$PASSWORD"
expect_code "login with mismatched CSRF" 403 -X POST "$BASE_URL/login" -b "$JAR_CSRF" \
    --data-urlencode "password=$PASSWORD" --data-urlencode "csrf_token=$(random_token 64)"

banner "TOTP and password negative cases"
valid_code="$(current_totp)"
invalid_code="$(invalid_totp)"
JAR_MISSING="$TMP_DIR/missing-totp.jar"; csrf="$(get_login_csrf "$JAR_MISSING")"
expect_code "correct password missing TOTP" 401 -X POST "$BASE_URL/login" -b "$JAR_MISSING" \
    --data-urlencode "password=$PASSWORD" --data-urlencode "csrf_token=$csrf"
JAR_BAD_TOTP="$TMP_DIR/bad-totp.jar"; csrf="$(get_login_csrf "$JAR_BAD_TOTP")"
expect_code "correct password wrong TOTP" 401 -X POST "$BASE_URL/login" -b "$JAR_BAD_TOTP" \
    --data-urlencode "password=$PASSWORD" --data-urlencode "totp_code=$invalid_code" \
    --data-urlencode "csrf_token=$csrf"
JAR_BAD_PASSWORD="$TMP_DIR/bad-password.jar"; csrf="$(get_login_csrf "$JAR_BAD_PASSWORD")"
expect_code "wrong password correct TOTP" 401 -X POST "$BASE_URL/login" -b "$JAR_BAD_PASSWORD" \
    --data-urlencode "password=$(random_token 48)" --data-urlencode "totp_code=$valid_code" \
    --data-urlencode "csrf_token=$csrf"

banner "valid session and disk secrecy"
JAR_SESSION="$TMP_DIR/session.jar"
session_cookie="$(login_with_totp "$JAR_SESSION")"
[[ "$session_cookie" != "$TOKEN" ]] || fail "session cookie equals MEMORY_OS_TOKEN"
[[ "$session_cookie" != "$PASSWORD" ]] || fail "session cookie equals MEMORY_OS_PASSWORD"
expect_code "fresh session cookie" 200 -b "$JAR_SESSION" "$BASE_URL/api/health"
session_hash="$(cookie_hash "$session_cookie")"
DB_PATH="$SESSIONS_DB_PATH" RAW_SESSION="$session_cookie" RAW_TOKEN="$TOKEN" RAW_PASSWORD="$PASSWORD" SESSION_HASH="$session_hash" "$PYTHON" - <<'PY'
import os, sqlite3
from pathlib import Path
path = Path(os.environ["DB_PATH"])
# Open in strict read-only mode via URI to avoid modifying WAL/SHM.
conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
conn.execute("PRAGMA query_only=ON;")
columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
assert "token_hash" in columns, columns
assert "token" not in columns, columns
assert conn.execute("SELECT COUNT(*) FROM sessions WHERE token_hash=?", (os.environ["RAW_SESSION"],)).fetchone()[0] == 0
assert conn.execute("SELECT COUNT(*) FROM sessions WHERE token_hash=? AND revoked=0", (os.environ["SESSION_HASH"],)).fetchone()[0] == 1
conn.close()
for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
    if not candidate.exists():
        continue
    data = candidate.read_bytes()
    for env_name in ("RAW_SESSION", "RAW_TOKEN", "RAW_PASSWORD"):
        value = os.environ[env_name].encode()
        assert not value or value not in data, f"raw secret {env_name} found in {candidate}"
PY
pass "sessions DB stores only token hash and contains no raw credentials"

banner "logout CSRF and persistent revocation"
expect_code "logout missing CSRF" 403 -X POST "$BASE_URL/logout" -b "$JAR_SESSION"
expect_code "session remains valid after missing logout CSRF" 200 -b "$JAR_SESSION" "$BASE_URL/api/health"
expect_code "logout wrong CSRF" 403 -X POST "$BASE_URL/logout" -b "$JAR_SESSION" \
    --data-urlencode "csrf_token=$(random_token 64)"
expect_code "session remains valid after wrong logout CSRF" 200 -b "$JAR_SESSION" "$BASE_URL/api/health"
logout_csrf="$(read_cookie "$JAR_SESSION" csrf_token)"
[[ -n "$logout_csrf" ]] || fail "session jar missing csrf_token"
expect_code "logout correct CSRF" 303 -X POST "$BASE_URL/logout" -b "$JAR_SESSION" -c "$JAR_SESSION" \
    --data-urlencode "csrf_token=$logout_csrf"
expect_code "revoked cookie immediately rejected" 401 \
    -H "Cookie: memory_os_session=$session_cookie" "$BASE_URL/api/health"
revoked="$(python3 "$SCRIPT_DIR/session_readonly_helper.py" "$SESSIONS_DB_PATH" \
    "SELECT revoked FROM sessions WHERE token_hash='$session_hash';")"
# The helper prints a Python list like [(1,)] or [(0,)]; extract the value.
revoked="$(echo "$revoked" | grep -oP '\d+' | head -1)"
[[ "$revoked" == "1" ]] || fail "logout did not persist revoked=1 (helper output: $revoked)"
if restart_service; then
    expect_code "revoked cookie rejected after restart" 401 \
        -H "Cookie: memory_os_session=$session_cookie" "$BASE_URL/api/health"
fi

banner "active session survives restart"
JAR_PERSIST="$TMP_DIR/persist.jar"
persist_cookie="$(login_with_totp "$JAR_PERSIST")"
expect_code "active session before restart" 200 -b "$JAR_PERSIST" "$BASE_URL/api/health"
if restart_service; then
    expect_code "active session after restart" 200 \
        -H "Cookie: memory_os_session=$persist_cookie" "$BASE_URL/api/health"
fi

banner "expired session rejection"
expired_cookie="$(random_token 64)"
DB_PATH="$SESSIONS_DB_PATH" EXPIRED_COOKIE="$expired_cookie" "$PYTHON" - <<'PY'
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from openclaw_memory_os.sessions import SessionStore
store = SessionStore(Path(os.environ["DB_PATH"]))
store.create(
    os.environ["EXPIRED_COOKIE"],
    1,
    issued_at=datetime.now(timezone.utc) - timedelta(minutes=5),
)
store.close()
PY
expect_code "expired SessionStore cookie" 401 \
    -H "Cookie: memory_os_session=$expired_cookie" "$BASE_URL/api/health"

if [[ -n "$RECOVERY_CODE" ]]; then
    banner "one-time recovery code"
    JAR_RECOVERY="$TMP_DIR/recovery.jar"; csrf="$(get_login_csrf "$JAR_RECOVERY")"
    expect_code "unused recovery code" 303 -X POST "$BASE_URL/login" -b "$JAR_RECOVERY" -c "$JAR_RECOVERY" \
        --data-urlencode "password=$PASSWORD" --data-urlencode "recovery_code=$RECOVERY_CODE" \
        --data-urlencode "csrf_token=$csrf"
    JAR_RECOVERY_REUSE="$TMP_DIR/recovery-reuse.jar"; csrf="$(get_login_csrf "$JAR_RECOVERY_REUSE")"
    expect_code "reused recovery code" 401 -X POST "$BASE_URL/login" -b "$JAR_RECOVERY_REUSE" \
        --data-urlencode "password=$PASSWORD" --data-urlencode "recovery_code=$RECOVERY_CODE" \
        --data-urlencode "csrf_token=$csrf"
else
    printf 'SKIP  recovery-code one-time test (set AUTH_SMOKE_RECOVERY_CODE explicitly)\n'
fi

banner "ALL AUTH GRADUATION SMOKE PASSED"
printf 'Service: %s\n' "$BASE_URL"
printf 'Sessions DB: %s\n' "$SESSIONS_DB_PATH"
