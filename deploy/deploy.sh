#!/usr/bin/env bash
# deploy.sh — idempotent production installer for OpenClaw Memory OS v0.3.0.
#
# The installer creates an unprivileged service account, installs the audited
# CPython 3.12 dependency lock, installs the application without dependency
# re-resolution, and deploys systemd/nginx configuration.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
UNIT_SRC="${PROJECT_ROOT}/deploy/systemd/openclaw-memory-os.service"
UNIT_DST="/etc/systemd/system/openclaw-memory-os.service"
UNIT_NAME="openclaw-memory-os.service"
SYSTEMD_DIR="${PROJECT_ROOT}/deploy/systemd"
MAINTENANCE_TIMER="openclaw-memory-os-maintenance.timer"
GOVERNANCE_TIMER="openclaw-memory-os-governance.timer"
NGINX_SRC="${PROJECT_ROOT}/deploy/nginx/memory-os.example.com.conf"
NGINX_DST="/etc/nginx/conf.d/openclaw-memory-os.conf"
ACME_SRC="${PROJECT_ROOT}/deploy/acme-issue.sh"
LOGROTATE_SRC="${PROJECT_ROOT}/deploy/logrotate/openclaw-memory-os"
LOGROTATE_DST="/etc/logrotate.d/openclaw-memory-os"
ENV_FILE="${PROJECT_ROOT}/.env"
ENV_EXAMPLE="${PROJECT_ROOT}/.env.example"
RUNTIME_LOCK="${PROJECT_ROOT}/requirements/runtime-py312.lock"
SERVICE_USER="openclaw-memory-os"
SERVICE_GROUP="openclaw-memory-os"
STATE_DIR="/var/lib/openclaw-memory-os"
CACHE_DIR="/var/cache/openclaw-memory-os"
LOG_DIR="/var/log/openclaw-memory-os"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
DOMAIN="${MEMORY_OS_DOMAIN:-${ACME_DOMAIN:-}}"
VENV_PY="${PROJECT_ROOT}/.venv/bin/python"

ok()   { printf '[ok]   %s\n' "$*"; }
skip() { printf '[skip] %s\n' "$*"; }
fail() { printf '[fail] %s\n' "$*" >&2; exit 1; }
note() { printf '       %s\n' "$*"; }

[[ "$(id -u)" -eq 0 ]] || fail "must run as root"
[[ -d "$PROJECT_ROOT" ]] || fail "project root not found"
case "$PROJECT_ROOT" in
    /root/*|/home/*) fail "deploy from a system path such as /opt/openclaw-memory-os; ProtectHome=yes blocks home directories" ;;
esac
[[ -f "$RUNTIME_LOCK" ]] || fail "runtime lock missing"
for required_command in "$PYTHON_BIN" curl tar sha256sum systemctl getent groupadd useradd usermod install sed awk nginx logrotate; do
    command -v "$required_command" >/dev/null 2>&1 || fail "required command not found: ${required_command}"
done
PY_MINOR="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PY_MINOR" == "3.12" ]] || fail "runtime lock requires CPython 3.12; found ${PY_MINOR}"

# ---------- 1. service identity and private state ----------
echo "[1/6] Ensuring service identity and private state"
if getent group "$SERVICE_GROUP" >/dev/null 2>&1; then
    skip "group already exists"
else
    groupadd --system "$SERVICE_GROUP"
    ok "created system group"
fi
if getent passwd "$SERVICE_USER" >/dev/null 2>&1; then
    usermod --gid "$SERVICE_GROUP" --home "$STATE_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
    skip "service user already exists; identity normalized"
else
    useradd --system --gid "$SERVICE_GROUP" --home-dir "$STATE_DIR" \
        --create-home --shell /usr/sbin/nologin "$SERVICE_USER"
    ok "created unprivileged service user"
fi
install -d -m 0700 -o "$SERVICE_USER" -g "$SERVICE_GROUP" \
    "$STATE_DIR" "$STATE_DIR/state" "$STATE_DIR/audit" "$CACHE_DIR" "$LOG_DIR"
ok "private state/cache/log directories ready"

# ---------- 2. environment ----------
echo "[2/6] Ensuring environment file"
if [[ -f "$ENV_FILE" ]]; then
    skip ".env already exists; preserving values"
else
    [[ -f "$ENV_EXAMPLE" ]] || fail "missing .env.example"
    cp -- "$ENV_EXAMPLE" "$ENV_FILE"
    ok "seeded .env from example"
fi
TOKEN_VALUE=$(awk -F= '/^MEMORY_OS_TOKEN=/{sub(/^[^=]*=/, ""); print; exit}' "$ENV_FILE" 2>/dev/null || true)
TOKEN_VALUE="${TOKEN_VALUE#\"}"
TOKEN_VALUE="${TOKEN_VALUE%\"}"
if [[ -z "$TOKEN_VALUE" || "$TOKEN_VALUE" == "<"*">"* || ${#TOKEN_VALUE} -lt 32 ]]; then
    fail "MEMORY_OS_TOKEN is missing, placeholder, or shorter than 32 characters"
fi
if [[ -z "$DOMAIN" ]]; then
    DOMAIN=$(awk -F= '/^MEMORY_OS_DOMAIN=/{sub(/^[^=]*=/, ""); print; exit}' "$ENV_FILE" 2>/dev/null || true)
fi
DOMAIN="${DOMAIN#\"}"
DOMAIN="${DOMAIN%\"}"
DOMAIN="${DOMAIN#'}"
DOMAIN="${DOMAIN%'}"
[[ -n "$DOMAIN" && "$DOMAIN" != "memory-os.example.com" ]] \
    || fail "set MEMORY_OS_DOMAIN to the real public hostname in .env"
[[ "$DOMAIN" == *.* && "$DOMAIN" =~ ^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$ ]] \
    || fail "MEMORY_OS_DOMAIN is not a valid DNS hostname: ${DOMAIN}"
chown root:"$SERVICE_GROUP" "$ENV_FILE"
chmod 0640 "$ENV_FILE"
ok ".env is root-owned and readable only by the service group"

# The service must be able to read an immutable checkout even when deploy.sh
# was invoked from a restrictive administrator umask. Keep the dedicated group
# readable, remove access for other users, and preserve executable bits in the
# virtual environment and operational scripts.
chown -R root:"$SERVICE_GROUP" "$PROJECT_ROOT"
find "$PROJECT_ROOT" -path "$PROJECT_ROOT/.venv" -prune -o -type d -exec chmod 0750 {} +
find "$PROJECT_ROOT" -path "$PROJECT_ROOT/.venv" -prune -o -type f -exec chmod 0640 {} +
find "$PROJECT_ROOT/scripts" "$PROJECT_ROOT/deploy" -type f -exec chmod 0750 {} +
chmod 0640 "$ENV_FILE"

# ---------- 3. reproducible Python runtime ----------
echo "[3/6] Installing audited Python runtime"
if [[ ! -x "$VENV_PY" ]]; then
    "$PYTHON_BIN" -m venv "${PROJECT_ROOT}/.venv"
    ok "created CPython 3.12 virtual environment"
else
    INSTALLED_MINOR="$($VENV_PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "$INSTALLED_MINOR" == "3.12" ]] || fail "existing .venv uses Python ${INSTALLED_MINOR}; recreate it with 3.12"
fi
"$VENV_PY" -m pip install --disable-pip-version-check -r "$RUNTIME_LOCK"
"$VENV_PY" -m pip install --disable-pip-version-check --no-deps "$PROJECT_ROOT"
"$VENV_PY" -m pip check
"$VENV_PY" -m compileall -q "${PROJECT_ROOT}/openclaw_memory_os"
ok "locked dependencies and application installed"

# Code and venv stay root-owned/read-only to the service; only state paths and
# the group-readable .env are mutable/readable as required.
chown -R root:"$SERVICE_GROUP" "${PROJECT_ROOT}/.venv"
chmod -R g+rX,o-rwx "${PROJECT_ROOT}/.venv"

# ---------- 4. systemd ----------
echo "[4/6] Installing systemd service and durable timers"
for unit in \
    openclaw-memory-os.service \
    openclaw-memory-os-maintenance.service \
    openclaw-memory-os-maintenance.timer \
    openclaw-memory-os-governance.service \
    openclaw-memory-os-governance.timer; do
    src="${SYSTEMD_DIR}/${unit}"
    [[ -f "$src" ]] || fail "missing systemd unit: ${unit}"
    tmp_unit="$(mktemp)"
    sed "s|/opt/openclaw-memory-os|${PROJECT_ROOT}|g" "$src" > "$tmp_unit"
    install -m 0644 -- "$tmp_unit" "/etc/systemd/system/${unit}"
    rm -f "$tmp_unit"
done
systemctl daemon-reload
systemctl enable "$UNIT_NAME" "$MAINTENANCE_TIMER" "$GOVERNANCE_TIMER" >/dev/null
if systemctl is-active --quiet "$UNIT_NAME"; then
    systemctl restart "$UNIT_NAME"
else
    systemctl start "$UNIT_NAME"
fi
systemctl start "$MAINTENANCE_TIMER" "$GOVERNANCE_TIMER"
sleep 1
systemctl is-active --quiet "$UNIT_NAME" \
    || fail "service failed to start; inspect journalctl -u ${UNIT_NAME}"
systemctl is-active --quiet "$MAINTENANCE_TIMER" \
    || fail "maintenance timer failed to start"
systemctl is-active --quiet "$GOVERNANCE_TIMER" \
    || fail "governance timer failed to start"
ok "service and persistent timers active under ${SERVICE_USER}"

# ---------- 5. nginx and TLS ----------
echo "[5/6] Installing nginx and TLS configuration"
[[ -f "$NGINX_SRC" ]] || fail "missing nginx vhost"
[[ -f "$ACME_SRC" ]] || fail "missing ACME script"
# Issue the certificate before installing the TLS vhost; nginx -t would fail on
# a first deployment if the referenced certificate did not exist yet.
ACME_DOMAIN="$DOMAIN" bash "$ACME_SRC"
tmp_nginx="$(mktemp)"
nginx_backup="$(mktemp)"
had_nginx_config=0
sed "s|memory-os.example.com|${DOMAIN}|g" "$NGINX_SRC" > "$tmp_nginx"
if [[ -f "$NGINX_DST" ]]; then
    cp -- "$NGINX_DST" "$nginx_backup"
    had_nginx_config=1
fi
install -m 0644 -- "$tmp_nginx" "$NGINX_DST"
rm -f "$tmp_nginx"
if ! nginx -t >/dev/null 2>&1; then
    if [[ "$had_nginx_config" -eq 1 ]]; then
        install -m 0644 -- "$nginx_backup" "$NGINX_DST"
    else
        rm -f -- "$NGINX_DST"
    fi
    rm -f "$nginx_backup"
    fail "nginx configuration validation failed; previous configuration restored"
fi
rm -f "$nginx_backup"
if systemctl is-active --quiet nginx; then
    systemctl reload nginx
else
    systemctl enable --now nginx
fi
ok "nginx and TLS active for ${DOMAIN}"

[[ -f "$LOGROTATE_SRC" ]] || fail "missing logrotate policy"
logrotate_backup="$(mktemp)"
had_logrotate_config=0
if [[ -f "$LOGROTATE_DST" ]]; then
    cp -- "$LOGROTATE_DST" "$logrotate_backup"
    had_logrotate_config=1
fi
install -m 0644 -- "$LOGROTATE_SRC" "$LOGROTATE_DST"
if ! logrotate -d "$LOGROTATE_DST" >/dev/null 2>&1; then
    if [[ "$had_logrotate_config" -eq 1 ]]; then
        install -m 0644 -- "$logrotate_backup" "$LOGROTATE_DST"
    else
        rm -f -- "$LOGROTATE_DST"
    fi
    rm -f "$logrotate_backup"
    fail "logrotate policy validation failed; previous policy restored"
fi
rm -f "$logrotate_backup"
ok "owner-private maintenance/governance log rotation installed"

# ---------- 6. local health ----------
echo "[6/6] Verifying local service health"
"$VENV_PY" - <<'PY'
import json
import urllib.request
with urllib.request.urlopen("http://127.0.0.1:7788/health", timeout=10) as response:
    if response.status != 200:
        raise SystemExit(f"health returned HTTP {response.status}")
    payload = json.load(response)
    if payload.get("status") not in {"ok", "healthy"}:
        raise SystemExit(f"unexpected health payload: {payload!r}")
PY
ok "local health endpoint passed"

cat <<EOF

[done] deployment finished
       Domain:       ${DOMAIN}
       Service user: ${SERVICE_USER}
       Project:      ${PROJECT_ROOT}
       State:        ${STATE_DIR}
       Logs:         ${LOG_DIR}
       Unit:         ${UNIT_NAME}
       Maintenance:  systemd timer ${MAINTENANCE_TIMER}
       Governance:   systemd timer ${GOVERNANCE_TIMER}
       Next:         run scripts/final_host_acceptance.sh on the real host
EOF
