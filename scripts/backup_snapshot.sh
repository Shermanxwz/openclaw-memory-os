#!/usr/bin/env bash
# backup_snapshot.sh — Snapshot the Qdrant collection used by OpenClaw
# Memory OS to a local snapshots dir. Designed for off-host transfer
# (e.g. rclone, WebDAV, S3, or another remote sink).
#
# Usage:
#   ./backup_snapshot.sh [COLLECTION]
#
# What it does:
#   1. Calls Qdrant snapshot API for the collection.
#   2. Waits for completion.
#   3. Tars the snapshot into a timestamped archive under
#      $BACKUP_DIR (default $HOME/snapshots).
#   4. Removes any temporary download artifacts created in step 2/3.
#   5. Keeps the latest 5 archives on disk; older ones are removed.
#   6. After successful archiving, prunes old Qdrant-internal snapshots
#      for the same collection, keeping the latest BACKUP_KEEP by
#      creation_time (and name as tiebreaker).
#
# Environment:
#   QDRANT_URL       (default http://127.0.0.1:6333)
#   QDRANT_API_KEY   (optional; sent as the Qdrant api-key header)
#   BACKUP_DIR       (default $HOME/snapshots)
#   BACKUP_KEEP      (default 5)
#   QDRANT_BACKUP_CACHE_DIR (default /opt/qdrant/backup; set empty to skip)
#   QDRANT_CACHE_KEEP       (default 0; cache is redundant after archive)

set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"
COLLECTION="${1:-${QDRANT_COLLECTION:-openclaw_memory_os}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="${VENV_PY:-$SCRIPT_DIR/../.venv/bin/python}"
BACKUP_DIR="${BACKUP_DIR:-${HOME:-/var/lib/openclaw-memory-os}/snapshots}"
BACKUP_KEEP="${BACKUP_KEEP:-5}"
QDRANT_BACKUP_CACHE_DIR="${QDRANT_BACKUP_CACHE_DIR:-/opt/qdrant/backup}"
QDRANT_CACHE_KEEP="${QDRANT_CACHE_KEEP:-0}"

[ -x "$VENV_PY" ] || { echo "[snapshot] project venv unavailable" >&2; exit 1; }
mkdir -p "$BACKUP_DIR"
stamp="$(date -u +'%Y%m%dT%H%M%SZ')"
if command -v zstd >/dev/null 2>&1; then
  archive="$BACKUP_DIR/${COLLECTION}-${stamp}.tar.zst"
else
  archive="$BACKUP_DIR/${COLLECTION}-${stamp}.tar.gz"
fi
curl_auth=()
if [ -n "$QDRANT_API_KEY" ]; then
  curl_auth=(-H "api-key: $QDRANT_API_KEY")
fi
tmp=""
is_tmp=0

log() { printf '[snapshot] %s\n' "$*"; }

# Best-effort cleanup: remove the .tmp download we created on success,
# and prune any stale .tmp-* files left behind by previous runs.
cleanup_tmp() {
  if [ "$is_tmp" = "1" ] && [ -n "$tmp" ] && [ -f "$tmp" ]; then
    rm -f -- "$tmp" && log "removed temp download: $tmp"
  fi
  # Sweep any leftover .tmp-* files in the backup dir (e.g. from a prior
  # crashed run). Logged so operators can spot recurring issues.
  if [ -d "$BACKUP_DIR" ]; then
    shopt -s nullglob
    for stale in "$BACKUP_DIR"/.tmp-*; do
      [ -f "$stale" ] || continue
      rm -f -- "$stale" && log "removed stale temp file: $stale"
    done
    shopt -u nullglob
  fi
}
trap cleanup_tmp EXIT

log "triggering snapshot for $COLLECTION"
response="$(curl -fsS "${curl_auth[@]}" -X POST "$QDRANT_URL/collections/$COLLECTION/snapshots")"
echo "$response" | head -c 400; echo
name="$(echo "$response" | "$VENV_PY" -c 'import json,sys; print(json.load(sys.stdin)["result"]["name"])')"
log "snapshot name: $name"

# Poll until the snapshot file appears on disk (Qdrant default storage).
snapshot_path=""
for _ in $(seq 1 30); do
  sleep 1
  snapshot_path="$(find /var/lib/qdrant/snapshots/"$COLLECTION" -name "$name*" 2>/dev/null | head -1 || true)"
  if [ -n "$snapshot_path" ]; then break; fi
done

if [ -z "$snapshot_path" ]; then
  log "snapshot file not found locally; downloading via API"
  tmp="$BACKUP_DIR/.tmp-$name"
  curl -fsS "${curl_auth[@]}" -o "$tmp" "$QDRANT_URL/collections/$COLLECTION/snapshots/$name"
  snapshot_path="$tmp"
  is_tmp=1
fi

log "archiving snapshot to $archive"
if command -v zstd >/dev/null 2>&1; then
  tar --use-compress-program=zstd -cf "$archive" -C "$(dirname "$snapshot_path")" "$(basename "$snapshot_path")"
else
  tar -czf "$archive" -C "$(dirname "$snapshot_path")" "$(basename "$snapshot_path")"
fi

# Drop the temp download now that the archive is on disk.
cleanup_tmp
is_tmp=0  # already removed; let EXIT trap be a no-op

# Prune old local archives (keep latest $BACKUP_KEEP, by mtime).
shopt -s nullglob
archives=("$BACKUP_DIR/${COLLECTION}-"*.tar.*)
shopt -u nullglob
if [ "${#archives[@]}" -gt 0 ] && [ "${#archives[@]}" -gt "$BACKUP_KEEP" ]; then
  # Sort by modification time, newest first; tail skips the kept ones.
  # Using ls -1t on the expanded array is safe: filenames are quoted above.
  # shellcheck disable=SC2012  # we just globbed, no surprises
  while IFS= read -r old; do
    [ -n "$old" ] || continue
    rm -f -- "$old" && log "pruned old archive: $old"
  done < <(ls -1t "${archives[@]}" 2>/dev/null | tail -n +$((BACKUP_KEEP + 1)))
fi
log "kept latest $BACKUP_KEEP archives"

# Prune Qdrant's local snapshot/download cache. This is distinct from the
# Memory OS archive directory and from Qdrant's snapshot API listing. Keep the
# rule narrow: only files named ${COLLECTION}-*.snapshot under the configured
# cache directory are candidates, and only after the archive succeeded. The
# actual delete logic lives in scripts/_prune_helpers.py so it's unit-tested
# without needing a real Qdrant.
if [ -n "$QDRANT_BACKUP_CACHE_DIR" ] && [ -d "$QDRANT_BACKUP_CACHE_DIR" ]; then
  log "pruning Qdrant backup cache for $COLLECTION in $QDRANT_BACKUP_CACHE_DIR (keep latest $QDRANT_CACHE_KEEP)"
  if "$VENV_PY" "$SCRIPT_DIR/_prune_helpers.py" cache-prune "$QDRANT_BACKUP_CACHE_DIR" "$COLLECTION" "$QDRANT_CACHE_KEEP" 2>&1; then
    :
  else
    log "  cache prune failed; continuing"
  fi
fi

# Prune old Qdrant snapshots for this collection. Done only after a
# successful archive so we never delete a snapshot that wasn't archived.
# Use Python (already a hard dep for the snapshot-name parse above) so
# we don't depend on jq being installed.
log "pruning old Qdrant snapshots for $COLLECTION (keep latest $BACKUP_KEEP)"
prune_output="$(QDRANT_URL="$QDRANT_URL" QDRANT_API_KEY="$QDRANT_API_KEY" COLLECTION="$COLLECTION" BACKUP_KEEP="$BACKUP_KEEP" "$VENV_PY" - <<'PY' || true
import json, os, sys, urllib.request, urllib.error

base = os.environ["QDRANT_URL"].rstrip("/")
api_key = os.environ.get("QDRANT_API_KEY", "")
headers = {"api-key": api_key} if api_key else {}
collection = os.environ["COLLECTION"]
keep = int(os.environ["BACKUP_KEEP"])

list_url = f"{base}/collections/{collection}/snapshots"
try:
    req = urllib.request.Request(list_url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
except Exception as exc:
    print(f"[snapshot] list failed: {exc}", file=sys.stderr)
    sys.exit(0)  # never fail the backup because prune couldn't list

snaps = data.get("result") or []
# Sort newest-first by (creation_time, name). Missing creation_time
# falls back to empty string so older entries without it sink.
snaps.sort(key=lambda s: (s.get("creation_time") or "", s.get("name") or ""), reverse=True)

if len(snaps) <= keep:
    print(f"[snapshot] {len(snaps)} qdrant snapshot(s); nothing to prune")
    sys.exit(0)

deleted = 0
for s in snaps[keep:]:
    name = s.get("name")
    if not name:
        continue
    del_url = f"{base}/collections/{collection}/snapshots/{name}"
    try:
        req = urllib.request.Request(del_url, method="DELETE", headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
        ct = s.get("creation_time") or "?"
        sz = s.get("size") or 0
        print(f"[snapshot] deleted qdrant snapshot {name} (created={ct}, size={sz})")
        deleted += 1
    except urllib.error.HTTPError as exc:
        body = exc.read()[:160].decode("utf-8", "replace")
        print(f"[snapshot] delete failed for {name}: HTTP {exc.code} {body}", file=sys.stderr)
    except Exception as exc:
        print(f"[snapshot] delete error for {name}: {exc}", file=sys.stderr)

print(f"[snapshot] pruned {deleted} qdrant snapshot(s); kept {min(len(snaps), keep)}")
PY
)"
if [ -n "$prune_output" ]; then
  printf '%s\n' "$prune_output"
fi

log "ok: $archive"
