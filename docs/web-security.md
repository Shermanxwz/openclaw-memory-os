# Web Security

OpenClaw Memory OS dashboard security overview.

The dashboard is intentionally designed to **fail closed**: state-changing
endpoints require either a valid session cookie + CSRF token or a
correct `Authorization: Bearer <token>` header. This document is the
operator-facing reference; the authoritative code lives in
`openclaw_memory_os/auth.py` and `openclaw_memory_os/app.py` (the
`require_auth` / `require_csrf_for_cookie_session` dependencies).

## Authentication Layers

The auth flow has three complementary mechanisms that may be enabled in
any combination:

### 1. Shared Bearer Token (Legacy)

- `MEMORY_OS_TOKEN` in `.env`
- Accepted via `Authorization: Bearer <token>` header
- API clients and CLI scripts use this path
- Considered "always-on" because it never depends on cookies; if set,
  `auth_enabled=True` even when no password is configured

### 2. Password + TOTP (Recommended)

- `MEMORY_OS_PASSWORD` + `MEMORY_OS_TOTP_SECRET` in `.env`
- Login form: password + 6-digit TOTP
- Password storage comparison supports both the legacy plaintext
  equality path (`hmac.compare_digest`) and the modern argon2id hash
  path (`$argon2id$v=19$...`); `verify_password()` selects
  automatically.
- TOTP: RFC 6238, HMAC-SHA1, 30-second step, 6 digits, ±1 step window
  (90-second total)
- Stdlib-only implementation (no external deps except `argon2-cffi`
  when an argon2id password hash is configured)
- Backwards-compatible: if no password is set, falls back to the
  bearer-token path (mode 1)

### 3. Session Cookie

- `memory_os_session` — HttpOnly, Secure, SameSite=Lax
- Cookie value is a hex secret (`secrets.token_hex(32)` for
  password+TOTP logins, or the bearer token itself when the bearer
  path is used as a session identifier)
- Configurable lifetime: `MEMORY_OS_SESSION_MAX_AGE`. The shipped
  default is 43200s = 12h, but the deployment `.env` pins it to
  2592000s (30 days) for this environment. Operators should set
  `MEMORY_OS_SESSION_MAX_AGE` explicitly to document the intended
  duration rather than relying on the fallback.
- In-process revocation set on `/logout`. The revocation set is
  in-memory and is cleared on process restart.
- Session cookie receives `Secure` whenever the request reached the
  app via HTTPS. In local development (`http://127.0.0.1`) the
  `Secure` flag is dropped so the cookie still reaches the browser.

### Feature flag — `PASSWORD_TOTP_AUTH`

The `PASSWORD_TOTP_AUTH` feature flag (see `docs/feature-flags.md`)
controls **only** the upgrade path:

- `PASSWORD_TOTP_AUTH=on` (default) — when a password is configured
  the password+TOTP path is preferred; otherwise the bearer-token
  path is used.
- `PASSWORD_TOTP_AUTH=off` — forces the legacy bearer-token login flow
  even when `MEMORY_OS_PASSWORD` and `MEMORY_OS_TOTP_SECRET` are
  configured. Useful when an operator wants to temporarily disable
  password+TOTP without removing the env vars.

The flag is **additive**: bearer-token-only auth (no password
configured) remains enabled in either case.

## CSRF Protection

- `csrf_token` cookie (non-HttpOnly, Secure, SameSite=Lax) issued
  via `issue_csrf_cookie()` on successful login
- `X-CSRF-Token` header (or hidden form field) verified by
  `verify_csrf()` on every state-changing request
- Login + logout enforce CSRF; likewise for all `POST /api/*` state
  changes (`/api/recall-test`, `/api/feedback`,
  `/api/evolution/pause`, `/api/evolution/resume`,
  `/api/evolution/candidate/reject`, `/api/evolution/rollback`, etc.)
- Dashboard state-changing operations always require CSRF
- Bearer-token clients (CLI / scripts) bypass the CSRF cookie check by
  design — they authenticate via the `Authorization: Bearer <token>`
  header instead

## Query String Tokens

- `?token=**` is intentionally NOT accepted. Tokens in query strings
  leak to: nginx access logs, browser history, Referer headers,
  monitoring tools that ingest GET URLs.
- Only the cookie and `Authorization` header are accepted for
  authentication; the only query-string paths are public
  redirects (`/`, `/dashboard`).

## Cookie Flags

| Cookie              | HttpOnly | Secure | SameSite | Purpose                                      |
| ------------------- | :------: | :----: | :------: | -------------------------------------------- |
| `memory_os_session` |   Yes    |  Yes   |   Lax    | Auth session.                                |
| `csrf_token`        |    No    |  Yes   |   Lax    | CSRF token (must be JS-readable).            |

Both cookies are issued with `Secure` in production (HTTPS); in local
HTTP development the `Secure` flag is dropped so the cookie still
reaches the browser.

## Session Management

- `/logout` clears both cookies and adds the session token to the
  in-process revocation set
- Revocation set is **in-memory**; it resets on process restart. This
  is a deliberate trade-off: a restart equals a clean slate for
  revoked sessions, eliminating any stale-token risk.
- There is no persistent session store. Session tokens are derived
  from the bearer token, the password+TOTP login, or a random
  hex-secret. Persistent storage would require either encryption at
  rest or a backing service, neither of which is part of this
  project.

## Token Verification Constants

- `hmac.compare_digest` is used for every secret comparison (TOTP
  digest, session cookie, bearer-token match). Plain `==` is forbidden
  in this path because `compare_digest` is constant-time and resists
  timing-attack side channels.
- Argon2id (when configured) compares via `argon2-cffi`'s
  `PasswordHasher.verify()`. Plaintext legacy passwords are matched
  via `hmac.compare_digest`.

## Evolution Endpoints

All `/api/evolution/*` endpoints require the same auth + CSRF contract
as any other state-changing API. They are also gated by the
`EVOLUTION_ENABLED` feature flag (see `docs/feature-flags.md`):

- `POST /api/evolution/pause` — flip `paused=True` on
  `evolution-state.json`. Returns
  `{"status": "disabled", "reason": "evolution_enabled=off"}` when the
  flag is off.
- `POST /api/evolution/resume` — flip `paused=False` (same behaviour
  when the flag is off).
- `POST /api/evolution/candidate/reject` — clear the shadow-candidate
  list; required CSRF + auth; same disabled envelope when off.
- `POST /api/evolution/rollback` — revert policy to the baseline
  (`Policy(**baseline_policy)`), stamp the manual-rollback timestamp
  in state, increment `consecutive_rollbacks`. Same disabled envelope
  when off.

These endpoints never receive the bearer token from a query string;
they accept it via the standard `Authorization` header or the
session cookie path. See `openclaw_memory_os/app.py` for the exact
dependency wiring.

## Hard Safety Boundary

The dashboard inherits the OS-wide hard contract (see
`openclaw_memory_os/contracts.py:HARD_CONTRACTS`):

- Dashboard never physically deletes memories. The Governance and
  Memories pages are explicitly read-only.
- Deletion candidates are review-only. The OS returns a list of
  candidates (cluster report, status filter, etc.) and humans decide
  what to do with them.
- Governance scope = `memory-content` only.
- No access to: repo, system, config, secrets, personal taxonomy,
  external services.
- Evolution endpoints touch only the policy file and the
  `evolution-state.json` file under
  `~/.local/state/openclaw-memory-os/`. They never reach into the
  Qdrant backend.

## Privacy Scanner

The in-repo privacy scanner is documented separately in
`docs/privacy-scanner.md`. It is not a substitute for `gitleaks` in CI;
both should be run as part of the regular release flow.

## Threat Model Summary

| Threat                                  | Mitigation                                                                    |
| --------------------------------------- | ----------------------------------------------------------------------------- |
| Session hijack via XSS                   | `HttpOnly` session cookie; CSP-friendly dashboard with no inline scripts except via `<script src=...>` |
| CSRF                                    | `csrf_token` cookie + `X-CSRF-Token` header check on all state changes         |
| Replay of revoked session                | In-process revocation set on `/logout`; restart = clean slate                 |
| Token leak via URL                       | `?token=` is **never** accepted                                              |
| Timing attacks on token comparison       | `hmac.compare_digest` everywhere; argon2id constant-time for password hashes   |
| Network capture                          | `Secure` cookie flag in production HTTPS                                      |
| Open redirect on `/login`                | Login posts to a fixed page; only the `next` redirect target is resolved server-side and only to known internal routes |
| Privilege escalation via evolution       | Consecutive-rollback circuit breaker; manual `/api/evolution/rollback` requires CSRF + auth; audit-logged |
| Memory deletion from dashboard           | `NO_PHYSICAL_DELETION` hard contract; no UI affordance for delete             |
