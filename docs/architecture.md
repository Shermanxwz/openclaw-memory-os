# Architecture

> **Independent community project. Not affiliated with, endorsed by, or
> supported by the OpenClaw project.** Complements the OpenClaw built-in
> memory engine; does not replace it.

OpenClaw Memory OS is a small Python service that sits in front of any
Qdrant collection (or the JSON development backend) and exposes a
dashboard, a recall-testing API, and a governance runner. It is one of
many possible control-plane layers; users are expected to use it together
with their own memory store and (optionally) the OpenClaw built-in
memory engine for typical recall.

## High-level layout

```
┌──────────────────────────────────────────────────────────────────────┐
│                          OpenClaw Memory OS                          │
│                                                                      │
│   FastAPI app  ─── Jinja2 templates (dashboard.html, login.html)      │
│       │                                                              │
│       ├── /api/health, /api/timeline, /api/tiers                     │
│       ├── /api/duplicates, /api/deletion-candidates                  │
│       └── /api/recall-test   <-- RetrievalEngine.retrieve()          │
│                       │                                              │
│                       ▼                                              │
│              RetrievalEngine                                         │
│           ┌──────────┬──────────────┐                                │
│           ▼          ▼              ▼                                │
│        keyword     dense         hybrid                              │
│        (BM25)    (Qdrant vec)   (RRF + feature rerank)               │
│                       │                                              │
│                       ▼                                              │
│      SampleBackend / QdrantBackend  (memory corpus)                  │
│      SQLite + JSON state files       (operational state)             │
└──────────────────────────────────────────────────────────────────────┘
```

The memory corpus lives in Qdrant or the development JSON backend.
Memory OS keeps its own operational state in local SQLite and JSON files,
including sessions, audit events, structured recall feedback, evaluation
state, and policy-evolution state.

## Storage split

OpenClaw Memory OS deliberately keeps two different classes of state
in two different places:

* **Memory corpus.** Stored in Qdrant (production) or the JSON
  development backend (offline / smoke). The corpus contains the
  Memory records the recall pipeline reads from.
* **Operational state.** Stored as local SQLite databases and JSON
  files under `$MEMORY_OS_RECALL_STATE_DIR` (default
  `$XDG_STATE_HOME/openclaw-memory-os/`, fallback
  `~/.local/state/openclaw-memory-os/`). This includes:
  * sessions,
  * audit events,
  * structured recall feedback (`recall_runs`, `recall_results`,
    `feedback_events`),
  * evaluation state,
  * policy store and policy-evolution state,
  * maintenance and backup/restore artefacts.

The split exists so an uninstall does not destroy operator-owned
signals and so the corpus can be snapshotted independently from the
operational state.

## Why a control plane?

A memory store tends to grow organically: rules, scratchpads, expired notes,
superseded entries, and near-duplicates accumulate faster than anyone
reviews them. This OS provides:

* **Visibility** — a dashboard that summarises counts, tiers, statuses,
  duplicates, and deletion candidates.
* **Recall-testing** — a real ranking pipeline over the corpus so a
  user can sanity-check that "the right memory comes back" for a
  canonical query, with explanations and policy metadata attached.
* **Evaluation** — an offline harness with time-split metrics and
  per-policy comparison.
* **Policy evolution** — a deterministic search plus guarded
  auto-promotion with rollback, so the recall pipeline can be tuned
  without manual replay of every recall run.
* **Review-only deletion candidates** — a list of memories the human
  might want to clean up. The OS never deletes.

## Modules

| Module | Role |
| --- | --- |
| `openclaw_memory_os/config.py` | Env-driven settings, cached. |
| `openclaw_memory_os/models.py` | Pydantic models (Memory, RecallHit, ...). |
| `openclaw_memory_os/auth.py`   | Bearer-token + password/TOTP gate. |
| `openclaw_memory_os/sessions.py` | SQLite-backed dashboard sessions. |
| `openclaw_memory_os/audit.py` | Audit-event recording. |
| `openclaw_memory_os/backends/__init__.py` | `SampleBackend` + `QdrantBackend`. |
| `openclaw_memory_os/lexical.py` | BM25 lexical index with on-disk cache. |
| `openclaw_memory_os/retrieval_engine.py` | `RetrievalEngine.retrieve()` — the v0.3.0 production retrieval path. |
| `openclaw_memory_os/ranking.py` | Legacy compatibility scoring. Not the v0.3.0 production path. |
| `openclaw_memory_os/candidate_pool.py` | Shared candidate pool for evaluation and live recall. |
| `openclaw_memory_os/recall_feedback.py` | Structured SQLite recall feedback. |
| `openclaw_memory_os/evaluation.py` | Time-split offline metrics. |
| `openclaw_memory_os/evaluation_reports.py` | Report rendering for the dashboard. |
| `openclaw_memory_os/evolution.py` | Policy search and guarded auto-promotion. |
| `openclaw_memory_os/policy_store.py` | Persistent policy store. |
| `openclaw_memory_os/ingestion.py` | Ingestion paths (CLI + programmatic). |
| `openclaw_memory_os/ingestion_validation.py` | Pydantic schema for LLM ingestion output. |
| `openclaw_memory_os/feedback.py` | Legacy audit-log feedback path. |
| `openclaw_memory_os/consolidation.py` | Maintenance helpers. |
| `openclaw_memory_os/migration.py` | Migration helpers. |
| `openclaw_memory_os/personal_taxonomy.py` | Tier/topic keyword loader. |
| `openclaw_memory_os/privacy.py` | Privacy scanner. |
| `openclaw_memory_os/analytics.py` | Health summary, dup clustering, deletion candidates. |
| `openclaw_memory_os/app.py` | FastAPI app + routes. |
| `openclaw_memory_os/cli.py` | CLI entry point (`openclaw-memory-os ...`). |
| `openclaw_memory_os/templates/*.html` | Server-rendered UI. |
| `openclaw_memory_os/static/*` | Dashboard assets (CSS, JS, Chart.js). |

## Configuration

All runtime config is environment-driven. See [`.env.example`](../.env.example) for
the canonical list. The notable variables:

* `MEMORY_OS_TOKEN` — if set, all non-health routes require
  `Authorization: Bearer <token>` or the dashboard session cookie.
* `MEMORY_OS_ADMIN_PASSWORD` / `MEMORY_OS_TOTP_SECRET` — credentials for
  the password-plus-TOTP login flow on the dashboard.
* `QDRANT_URL` — if set, the Qdrant backend is used.
* `QDRANT_COLLECTION` — primary collection name (default
  `openclaw_memories`).
* `QDRANT_SECONDARY_COLLECTIONS` — optional comma-separated list of
  additional collections the dashboard should report on.
* `RECALL_FALLBACK_SUPERSEDED` / `RECALL_FALLBACK_SUPERSEDED_MIN_RESULTS`
  — control the recall fallback strategy (active-first, expand to
  superseded below the threshold).
* `MEMORY_OS_SAMPLE_PATH` — path to the JSON sample backend.
* `MEMORY_OS_RECALL_STATE_DIR` — root for the operational SQLite and
  JSON state files.

## What the OS does NOT do

* It does not physically delete memories.
* It does not push or pull from the outside world at runtime.
* It does not store credentials. `QDRANT_API_KEY` flows straight through to
  the Qdrant client.
* It does not auto-supersede by default. The bundled supersede detector
  is conservative by default — it tags near-duplicates for review rather
  than mutating status — and requires `ENABLE_AUTO_SUPERSEDE=1` for
  lifecycle-changing runs.

## Threading model

FastAPI runs handlers in an asyncio loop. The current code does all backend
calls synchronously inside the handlers. For stores measured in the tens of
thousands of memories that is fine. Larger stores should swap the backend for
an async one — the `MemoryBackend` ABC keeps that localised.

## OpenClaw integration (optional)

The bundled adapter under `plugins/openclaw-memory-os-adapter/` exposes
Memory OS as a native OpenClaw exclusive memory plugin. It is loaded
only when an operator explicitly installs it and sets
`plugins.slots.memory = "openclaw-memory-os"`. When the adapter is not
installed or not enabled, OpenClaw recall is unaffected — the built-in
memory engine continues to handle recall on its own.

See `docs/openclaw-memory-plugin-adapter.md` for the runtime contract,
config example, and test entry point.
