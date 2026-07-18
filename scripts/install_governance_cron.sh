#!/usr/bin/env bash
# Compatibility entry point retained for operators of older installations.
# v0.3.0 final freeze uses a persistent systemd timer rather than root cron so
# the runner receives the same EnvironmentFile, HOME, XDG state, permissions,
# and unprivileged identity as the web service.

set -euo pipefail

TIMER_NAME="${TIMER_NAME:-openclaw-memory-os-governance.timer}"
UNIT_PATH="/etc/systemd/system/$TIMER_NAME"

if [ "$(id -u)" -ne 0 ]; then
  echo "install_governance_cron: run as root to enable $TIMER_NAME" >&2
  exit 1
fi
if [ ! -f "$UNIT_PATH" ]; then
  echo "install_governance_cron: $UNIT_PATH is missing" >&2
  echo "  run deploy/deploy.sh first; root cron installation is no longer supported" >&2
  exit 1
fi

systemctl daemon-reload
systemctl enable --now "$TIMER_NAME" >/dev/null
systemctl is-active --quiet "$TIMER_NAME" || {
  echo "install_governance_cron: timer failed to activate" >&2
  exit 1
}
echo "install_governance_cron: $TIMER_NAME is enabled and active"
systemctl list-timers "$TIMER_NAME" --no-pager
