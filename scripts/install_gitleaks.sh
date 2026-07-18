#!/usr/bin/env bash
# scripts/install_gitleaks.sh
# v0.3.0 G10.1 — install gitleaks binary (no apt).
# Wave 5 deliverable: ships a stable binary to /usr/local/bin so the
# Runbook G10 verification gate can run as part of CI / governance.
#
# The script is idempotent: it skips the download when /usr/local/bin/gitleaks
# is already present and matches the requested version, then prints the
# installed version so the caller can verify the gate.
#
# Usage:
#   scripts/install_gitleaks.sh                 # install v8.18.4
#   GITLEAKS_VERSION=v8.18.4 scripts/install_gitleaks.sh
#   GITLEAKS_DEST=/opt/bin scripts/install_gitleaks.sh
#
# Manual fallback (documented):
#   If the network is unreachable, download
#       https://github.com/gitleaks/gitleaks/releases/download/${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION#v}_linux_x64.tar.gz
#   manually, extract the `gitleaks` binary, and place it at $GITLEAKS_DEST/gitleaks.

set -euo pipefail

GITLEAKS_VERSION="${GITLEAKS_VERSION:-v8.18.4}"
GITLEAKS_DEST="${GITLEAKS_DEST:-/usr/local/bin}"
GITLEAKS_TMP="${TMPDIR:-/tmp}/gitleaks-install.$$"

version_stripped="${GITLEAKS_VERSION#v}"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) TARGET_ARCH="x64" ;;
  aarch64|arm64) TARGET_ARCH="arm64" ;;
  *)
    echo "unsupported architecture: $ARCH" >&2
    exit 1
    ;;
esac

TARBALL_URL="https://github.com/gitleaks/gitleaks/releases/download/${GITLEAKS_VERSION}/gitleaks_${version_stripped}_linux_${TARGET_ARCH}.tar.gz"
TARBALL_PATH="${GITLEAKS_TMP}/gitleaks.tar.gz"
EXTRACT_DIR="${GITLEAKS_TMP}/extract"

echo "[install_gitleaks] target=${GITLEAKS_DEST}/gitleaks version=${GITLEAKS_VERSION} arch=${TARGET_ARCH}"

mkdir -p "$GITLEAKS_TMP" "$EXTRACT_DIR"

# Idempotency check
if command -v gitleaks >/dev/null 2>&1; then
  installed="$(gitleaks version 2>/dev/null || true)"
  if [[ "$installed" == *"$version_stripped"* ]]; then
    echo "[install_gitleaks] already present: $installed"
    exit 0
  fi
fi

# Prefer curl, fall back to wget
download_ok=0
if command -v curl >/dev/null 2>&1; then
  if curl -fsSL --retry 3 --connect-timeout 15 "$TARBALL_URL" -o "$TARBALL_PATH"; then
    download_ok=1
  fi
elif command -v wget >/dev/null 2>&1; then
  if wget -q --tries=3 --timeout=15 "$TARBALL_URL" -O "$TARBALL_PATH"; then
    download_ok=1
  fi
else
  echo "[install_gitleaks] neither curl nor wget is available" >&2
  exit 2
fi

if [[ "$download_ok" -ne 1 ]]; then
  echo "[install_gitleaks] download failed: $TARBALL_URL" >&2
  echo "[install_gitleaks] manual fallback: download that URL and run 'sudo mv gitleaks ${GITLEAKS_DEST}/gitleaks'" >&2
  rm -rf "$GITLEAKS_TMP"
  exit 3
fi

tar -xzf "$TARBALL_PATH" -C "$EXTRACT_DIR"

if [[ ! -x "$EXTRACT_DIR/gitleaks" ]]; then
  echo "[install_gitleaks] gitleaks binary not found after extraction" >&2
  rm -rf "$GITLEAKS_TMP"
  exit 4
fi

# Move into place (sudo when target is not writable by current user)
if [[ -w "$GITLEAKS_DEST" ]] || [[ -w "$(dirname "$GITLEAKS_DEST")" ]]; then
  install -m 0755 "$EXTRACT_DIR/gitleaks" "$GITLEAKS_DEST/gitleaks"
else
  echo "[install_gitleaks] ${GITLEAKS_DEST} is not writable; falling back to sudo"
  sudo install -m 0755 "$EXTRACT_DIR/gitleaks" "$GITLEAKS_DEST/gitleaks"
fi

rm -rf "$GITLEAKS_TMP"

echo "[install_gitleaks] installed:"
"${GITLEAKS_DEST}/gitleaks" version