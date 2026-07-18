# Deployment

This document covers running the OS in production-ish conditions.

## Requirements

* CPython 3.12 for the frozen production deployment
* `gitleaks` for the mandatory final complete-history scan
* A reverse proxy that terminates TLS (nginx, Caddy, traefik, ...).
  The OS does not handle certificates directly.

## Minimal install

Set `MEMORY_OS_DOMAIN` in `.env` to the real public DNS hostname before using
the production deployer. The placeholder `memory-os.example.com` is rejected.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/runtime-py312.lock
.venv/bin/python -m pip install --no-deps .
```

## Running

```bash
# Foreground (development)
uvicorn openclaw_memory_os.app:app --host 127.0.0.1 --port 7788

# Foreground (binding to all interfaces, behind a TLS-terminating proxy)
uvicorn openclaw_memory_os.app:app --host 0.0.0.0 --port 7788

# CLI shortcut
openclaw-memory-os serve --host 127.0.0.1 --port 7788
```

Proxies should forward to `127.0.0.1:7788` and refuse anything that didn't
authenticate, ideally via an additional network rule.

## Authentication

Set a long, random bearer token:

```bash
export MEMORY_OS_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

Clients authenticate either via header:

```
Authorization: Bearer <the-token>
```

For browser access, open `/login` and submit the token through the form.
Query-string tokens are rejected on every route so bearer credentials never
appear in browser history, Referer headers, redirects, or access logs. Use the
HttpOnly dashboard session cookie or `Authorization: Bearer ...` for API calls.

`/health` is intentionally public so monitoring can scrape liveness
without a token.

## Multi-user auth

Out of scope. The model is "single shared token, small team, trusted
VPS". Multi-user auth would be a significantly larger change (sessions,
per-user tokens, an auth service) and would push the project out of "safe
to open-source at this scope". Document it explicitly: if you have multiple
unrelated tenants, run multiple instances.

## Storage backends

* **Sample** (default): a JSON file, suitable for dev / CI / small demos.
* **Qdrant** (optional): set `QDRANT_URL` and optionally `QDRANT_API_KEY`.
  The adapter is read-only.

## Logging

The web process logs to journald. Maintenance and governance logs are
owner-private under `/var/log/openclaw-memory-os/` and rotate through the
shipped `/etc/logrotate.d/openclaw-memory-os` policy. Query strings are omitted
from the nginx access format.

## Persistent state and backups

Qdrant remains the memory store and must be protected with collection
snapshots. Memory OS also persists session revocations, recall feedback,
evaluation reports, policy Active/Previous/Candidate/history, lexical caches,
and governance state below `/var/lib/openclaw-memory-os`. Preserve that private
state directory together with `.env` and the exact source/wheel artifact.

Final release acceptance requires a restore drill against a disposable Qdrant
instance or clone. It must not mutate or delete production memories. See
[`final-host-acceptance.md`](final-host-acceptance.md).

## Health probes

* `GET /health` — liveness; always public, no auth.
* `GET /api/health` — memory-store health; auth-gated when
  `MEMORY_OS_TOKEN` is set.

## Reverse proxy (nginx)

Use [`deploy/nginx/memory-os.example.com.conf`](../deploy/nginx/memory-os.example.com.conf)
as the production baseline. `deploy/deploy.sh` renders the real hostname from
`MEMORY_OS_DOMAIN`, obtains the certificate before installing the TLS vhost,
and rejects `?token=` before redirect/proxying and
uses an access-log format that never includes query strings.

## Automatic maintenance and governance (systemd timers)

`sudo deploy/deploy.sh` installs two persistent timers under the dedicated
`openclaw-memory-os` service account:

* `openclaw-memory-os-maintenance.timer` — daily at 07:45 Asia/Shanghai.
* `openclaw-memory-os-governance.timer` — Tuesday at 04:01 Asia/Shanghai.

`Persistent=true` means a missed run is executed after the host returns. Both
scripts retain their own `flock` locks, continue far enough to finish cleanup
and status writing, and return non-zero whenever any required stage failed.
A snapshot, ingest, lexical refresh, feedback replay, evolution, or status-write
failure can therefore no longer leave a stale green dashboard state.

Inspect the schedule and latest results with:

```bash
systemctl list-timers 'openclaw-memory-os-*'
systemctl status openclaw-memory-os-maintenance.service
systemctl status openclaw-memory-os-governance.service
journalctl -u openclaw-memory-os-maintenance.service
journalctl -u openclaw-memory-os-governance.service
```

The mutable state is owner-private under `/var/lib/openclaw-memory-os`; logs are
under `/var/log/openclaw-memory-os`. The web service and both timer services run
as the unprivileged `openclaw-memory-os` account.

### Status JSON contract

The governance runner writes exactly `last_run`, `last_result`, and
`last_summary`. Allowed result values are `ok`, `failed`, `degraded`, `running`,
`pending`, and `skipped`. Unknown values normalize to `failed`. A run is green
only after feedback replay, maintenance, evolution, and final status writing all
succeed.

### Manual trigger

```bash
sudo systemctl start --wait openclaw-memory-os-maintenance.service
sudo systemctl start --wait openclaw-memory-os-governance.service
```

The weekly runner never physically deletes memories. Active-first /
Superseded-fallback remains the retrieval contract, and `tier=core` plus
`tier=long` are never automatically superseded.

## Backups and snapshots

[`scripts/backup_snapshot.sh`](../scripts/backup_snapshot.sh) is the
canonical backup flow. It:

1. triggers a Qdrant snapshot for the requested collection;
2. waits for the snapshot to materialise (or downloads it via the
   API if Qdrant hasn't written it locally yet);
3. archives the snapshot with `zstd` into a timestamped `.tar.zst`
   (or a correctly named `.tar.gz` fallback when `zstd` is unavailable)
   under `BACKUP_DIR` (default `${HOME}/snapshots`, which resolves to
   `/var/lib/openclaw-memory-os/snapshots` for the service account);
4. prunes local archives so only the latest `BACKUP_KEEP` (default
   `5`) survive;
5. sweeps the Qdrant backup cache at `QDRANT_BACKUP_CACHE_DIR`,
   keeping at most `QDRANT_CACHE_KEEP` recent files (default `0` so
   the redundant cache copy is dropped after archiving);
6. prunes old Qdrant-internal snapshots for the same collection
   (keeping the latest `BACKUP_KEEP` by creation_time, then name).

When `QDRANT_API_KEY` is configured, every snapshot create, download, list,
and prune request carries the Qdrant `api-key` header. Re-running the script is idempotent. Each step prints `[snapshot]`
log lines so operators can spot recurring issues (e.g. stale `.tmp-*`
files from a prior crashed run are swept on entry).
