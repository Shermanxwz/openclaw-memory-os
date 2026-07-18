#!/usr/bin/env bash
# acme-issue.sh — issue a Let's Encrypt certificate for ACME_DOMAIN.
#
# Strategy: acme.sh standalone http-01 challenge on port 80.
#   1. Stop nginx (it owns :80 and would block the challenge server).
#   2. Run `acme.sh --issue --standalone ...` against :80.
#   3. Install the cert to /etc/letsencrypt/live/<domain>/ so nginx can
#      load the rendered nginx vhost installed by deploy/deploy.sh.
#   4. Restart nginx.
#
# The script is idempotent: re-running while a cert is already valid just
# refreshes it. It does NOT print or persist any private key material.
#
# Required env / assumptions:
#   * You are root (needed for /etc/letsencrypt and systemctl).
#   * ACME_DOMAIN is set to the real hostname and its DNS A/AAAA record already
#     points at this host (the challenge only works after DNS propagates).
#   * acme.sh is installed and on PATH (or set ACMESH_BIN below).

set -Eeuo pipefail
IFS=$'\n\t'

# ---------- config (override via env if you really need to) ----------
DOMAIN="${ACME_DOMAIN:-}"
CERT_DIR="/etc/letsencrypt/live/${DOMAIN}"
NGINX_SERVICE="${ACME_NGINX_SERVICE:-nginx}"
# Allow pointing at acme.sh in a non-default location.
ACME_BIN="${ACME_BIN:-acme.sh}"

log()  { printf '[acme] %s\n' "$*"; }
fail() { printf '[acme][fail] %s\n' "$*" >&2; exit 1; }

# ---------- preflight ----------
[[ -n "$DOMAIN" && "$DOMAIN" != "memory-os.example.com" ]] \
    || fail "set ACME_DOMAIN to the real public hostname"
command -v "$ACME_BIN" >/dev/null 2>&1 \
    || fail "acme.sh not found on PATH (\"$ACME_BIN\"). Install it first:
               curl -fsSL https://get.acme.sh | sh
               export PATH=\"\$HOME/.acme.sh:\$PATH\"
             Aborting."

[[ "$(id -u)" -eq 0 ]] || fail "must run as root (needs to write to /etc/letsencrypt and stop nginx)."

# Nginx must be present (we stop/start it). Fail fast if not.
command -v systemctl >/dev/null 2>&1 || fail "systemctl not found; this script targets systemd hosts."
systemctl is-active --quiet "$NGINX_SERVICE" && NGINX_WAS_ACTIVE=1 || NGINX_WAS_ACTIVE=0
# If nginx is up, note it — we'll restore it at the end regardless.
log "domain           = ${DOMAIN}"
log "cert install dir = ${CERT_DIR}"
log "nginx was active = ${NGINX_WAS_ACTIVE}"

restart_nginx() {
    if [[ "$NGINX_WAS_ACTIVE" -eq 1 ]]; then
        log "starting ${NGINX_SERVICE} (was active before run)"
        systemctl start "$NGINX_SERVICE"
    else
        log "leaving ${NGINX_SERVICE} stopped (was not active before run)"
    fi
}
ensure_nginx_restored() {
    if [[ "$NGINX_WAS_ACTIVE" -eq 1 ]] && ! systemctl is-active --quiet "$NGINX_SERVICE"; then
        systemctl start "$NGINX_SERVICE" || true
    fi
}
trap ensure_nginx_restored EXIT

# ---------- issue / renew ----------
issue() {
    # --standalone : spin up acme.sh's own http-01 server on :80.
    # --httpport 80 : be explicit (default).
    # --server letsencrypt : default; spelled out for clarity.
    # --issue      : (re-)request a cert. For renewals acme.sh picks up
    #                an existing account and re-uses the order.
    # --install-cert: write the issued material to a fixed location so the
    #                nginx vhost in deploy/nginx/...conf can load it.
    log "running acme.sh --issue --standalone (http-01 on :80) for ${DOMAIN}"
    "$ACME_BIN" --issue -d "$DOMAIN" \
        --standalone \
        --httpport 80 \
        --server letsencrypt

    log "installing cert to ${CERT_DIR}"
    install -d -m 0750 "$CERT_DIR"
    "$ACME_BIN" --install-cert -d "$DOMAIN" \
        --key-file       "${CERT_DIR}/privkey.pem" \
        --fullchain-file "${CERT_DIR}/fullchain.pem" \
        --reloadcmd      "systemctl try-reload-or-restart ${NGINX_SERVICE}"
}

main() {
    # We only need to stop nginx for the brief window acme.sh binds :80.
    if [[ "$NGINX_WAS_ACTIVE" -eq 1 ]]; then
        log "stopping ${NGINX_SERVICE} to free port 80"
        systemctl stop "$NGINX_SERVICE"
    fi

    # Run issuance. On any error, ensure nginx is restored.
    if ! issue; then
        restart_nginx || true
        fail "acme.sh failed to issue/renew the cert for ${DOMAIN}"
    fi

    restart_nginx

    log "[ok] certificate material at:"
    log "      ${CERT_DIR}/fullchain.pem"
    log "      ${CERT_DIR}/privkey.pem"
}

main "$@"
