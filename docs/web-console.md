# Web Console Guide

OpenClaw Memory OS Dashboard — single-page server-rendered console at `/dashboard`.

The dashboard is intentionally **read-only by default**. State-changing
endpoints (recall-test, feedback, evolution pause/resume/rollback) are
guarded by the same CSRF + session-cookie rules that protect the rest
of the UI. The console itself never physically deletes memories.

## Sections

### 总览 (Overview) — `/dashboard/overview`

- Memory health: total, active, superseded, expired, needs_review
- Maintenance health: status, last run, snapshots, lock state
- Memory Brain: ingest + consolidate status
- Autonomous governance status card
- Retrieval strategy card (policy_version, last_promotion_at,
  promotion_count_30d, consecutive_rollbacks, shadow_enabled)
- Charts: tier distribution, status distribution, importance histogram

### 层级分类 (Tiers) — `/dashboard/tiers`

- 5-tier lifecycle mapping: core → long → medium → short → working
- Tier → OpenClaw 4-tier recall priority mapping
- Status distribution table

### 去重审核 (Duplicates) — `/dashboard/duplicates`

- Jaccard similarity duplicate clusters
- Consolidation analysis (no physical delete)

### 召回测试 (Recall) — `/dashboard/recall`

- Test retrieval with any query
- Mode: `hybrid` / `keyword` / `dense`
- Results with scores, badges, explanations
- Diagnostics envelope: `status`, `degraded_reason`, `dense_available`,
  `lexical_available`, `collections_searched`, `candidate_count`,
  `embedding_ms`, `lexical_ms`, `ranking_ms` (and the per-hit
  `query_id` / `policy_version` from the structured-feedback path).
- Feedback buttons (useful/not-useful) routed through the structured
  `record_feedback_v030()` writer when `STRUCTURED_FEEDBACK=on`
  (default). See `docs/feature-flags.md`.

### 自主治理 (Governance) — `/dashboard/governance`

- Autonomous governance status: last run, next run, result
- Cleanup candidates (auto-filtered: no core/long/high-importance/7d-recent)
- One-click candidate confirmation (review-only, no physical delete)

### 检索策略 (Strategy) — `/dashboard/strategy`

- Policy version, 30d promotions, rollbacks, shadow comparisons
- Evolution cycle: every Tuesday 04:01 Asia/Shanghai
- guarded_auto mode: max 2 promotions/30d,
  2 consecutive rollbacks → circuit breaker
- Evolution controls: pause / resume / reject candidate /
  rollback to baseline. All four endpoints are POSTs and require a
  valid CSRF token. When `EVOLUTION_ENABLED=off` the endpoints return
  `{"status": "disabled"}` and the buttons are rendered as disabled.
  See `docs/self-evolution.md`.
- Live policy state (read by `strategy.js` from `/api/strategy`):
  version, checksum, shadow comparison count, `shadow_enabled` flag.

### 评估 (Evaluation) — `/dashboard/evaluation`

- Feedback summary cards: 24h / 7d / 30d positive ratios + total
  events. When `total_events` is zero, every ratio is rendered as
  `—` (honest-null contract).
- Offline evaluation metrics: tiles for Recall@1/5/10, MRR@10,
  nDCG@10, useful@1/5, explicit-negative@5, no-result rate,
  p50/p95 latency, degraded/fallback rate. v0.3.0.x graded fields
  (`judged_ndcg_at_10`, `useful_superseded_fallback_rate`) render
  as `—` with an "unavailable" badge until the offline pipeline has
  produced graded judgments. See `docs/retrieval-diagnostics.md`.
- Replay history: short table of the last few offline runs.
- The "Note" footer reminds operators that this page is read-only and
  is updated by `scripts/run_evolution_cycle.py` /
  `scripts/evaluate_retrieval.py`.

### 记忆浏览 (Memories) — `/dashboard/memories`

- Read-only browser over the active memory corpus
- Filter by status / tier / collection
- "No physical delete" badge is rendered prominently

### 系统健康 (Health) — `/dashboard/health`

- Qdrant status + point count
- Ollama model availability
- Lexical index status (`dense_available`, `lexical_available`)
- Policy DB status (active policy file path, checksum, version)
- SQLite feedback table status (`recall_runs`, `recall_results`,
  `feedback_events` row counts, retention window)
- Feature-flag readout (every v0.3.0.x flag surfaced live so
  operators can confirm the on-path / off-path at a glance)

### 安全设置 (Security) — `/dashboard/security`

- Auth status (Password + TOTP or shared token)
- CSRF status
- Session lifetime + cookie flags
- Logout / session revocation

## Key API endpoints the console consumes

| Endpoint                                  | Purpose                                              | CSRF |
| ----------------------------------------- | ---------------------------------------------------- | :--: |
| `GET  /api/health`                         | Backend health + feature-flag snapshot              | no   |
| `GET  /api/strategy` (alias `/api/dashboard/strategy`) | Live policy + evolution state             | no   |
| `GET  /api/dashboard/evaluation`           | Read-only offline-evaluation envelope               | no   |
| `POST /api/recall-test`                    | Run recall (writes to `recall_runs` / `recall_results`) | yes  |
| `POST /api/feedback`                       | Record a useful / not-useful verdict                | yes  |
| `POST /api/evolution/pause`                | Pause the next evolution cycle                      | yes  |
| `POST /api/evolution/resume`               | Resume after a pause                                | yes  |
| `POST /api/evolution/candidate/reject`     | Drop the candidate currently in shadow              | yes  |
| `POST /api/evolution/rollback`            | Revert policy to the baseline                       | yes  |
| `POST /logout`                             | Clear the session cookie                            | yes  |

Endpoint CSRF rules are enforced by `require_csrf_for_cookie_session`
in `openclaw_memory_os/app.py`. Bearer-token clients (CLI / scripts)
are exempt because they authenticate via the
`Authorization: Bearer <token>` header instead of a browser session.

## Architecture

- Server: FastAPI + Jinja2 templates
- Auth: HttpOnly session cookie + CSRF cookie + X-CSRF-Token header
- CSS: `/static/css/dashboard.css`
- JS bundle: `/static/js/dashboard.js` (plus per-page modules
  `evaluation.js`, `strategy.js`, `recall.js`, `governance.js`,
  `health.js`, `memories.js`, `overview.js`, `security.js`,
  `tiers_duplicates.js`)
- Charts: Chart.js (local `/static/chart.umd.min.js`)
- All POST endpoints enforce the same CSRF token check; non-browser
  bearer-token clients bypass the CSRF check by design.

## Feature-flag notes

The Strategy and Health pages render different states based on
`EVOLUTION_ENABLED` and `SHADOW_ENABLED` (see `docs/feature-flags.md`).
Operators who set `EVOLUTION_ENABLED=off` see the pause/resume/rollback
buttons rendered as disabled, with a "Disabled by feature flag" badge.
Operators who set `SHADOW_ENABLED=off` see a small "shadow=off" badge
next to the shadow-comparison counter.

The Recall page always returns a `diagnostics.retrieval_engine` field
(`"v2"` or `"legacy"`) so the dashboard can render which path served
the request.

## Hard safety boundary

This dashboard inherits the OS-wide hard contract:

- Dashboard never physically deletes memories
- Deletion candidates are review-only (the Governance and Memories
  pages are explicitly read-only)
- Governance scope = `memory-content` only
- No access to: repo, system, config, secrets, personal taxonomy,
  external services
- Evolution endpoints touch **only** the policy file and the
  evolution state JSON; both live under
  `~/.local/state/openclaw-memory-os/`.
