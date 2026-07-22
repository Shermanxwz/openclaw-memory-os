# Architecture

OpenClaw Memory OS is a small Python service that sits next to a memory store
(either a JSON-backed sample store or a Qdrant collection) and exposes a
read-only dashboard and a recall-testing API on top of it.

## High-level layout

```
┌──────────────────────────────────────────────────────────────────────┐
│                          OpenClaw Memory OS                          │
│                                                                      │
│   FastAPI app  ─── Jinja2 templates (dashboard.html, login.html)      │
│       │                                                              │
│       ├── /api/health, /api/timeline, /api/tiers                     │
│       ├── /api/duplicates, /api/deletion-candidates                  │
│       └── /api/recall-test   <-- rank_memories(...)                  │
│                       │                                              │
│                       ▼                                              │
│                MemoryBackend interface                                │
│            ┌──────────┴──────────┐                                    │
│            ▼                     ▼                                    │
│       SampleBackend          QdrantBackend                            │
│       (JSON file)         (lazy, optional)                            │
└──────────────────────────────────────────────────────────────────────┘
```

The OS is intentionally one process. The web app, the analytics helpers, and
the ranking code share the same Python module so the surface area is small
and auditable. There is no database of its own — the OS borrows data shape
from whatever it points at.

## Why a thin layer?

A memory store tends to grow organically: rules, scratchpads, expired notes,
superseded entries, and near-duplicates accumulate faster than anyone
reviews them. This OS provides:

* **Visibility** — a dashboard that summarises counts, tiers, statuses,
  duplicates, and deletion candidates.
* **Recall-testing** — a deterministic-ish ranking function over the store
  so a user can sanity-check that "the right memory comes back" for a
  canonical query.
* **Review-only deletion candidates** — a list of memories the human
  might want to clean up. The OS never deletes.

## Modules

| Module | Role |
| --- | --- |
| `openclaw_memory_os/config.py` | Env-driven settings, cached. |
| `openclaw_memory_os/models.py` | Pydantic models (Memory, RecallHit, ...). |
| `openclaw_memory_os/auth.py`   | Bearer-token gate. |
| `openclaw_memory_os/backends/__init__.py` | `SampleBackend` + `QdrantBackend`. |
| `openclaw_memory_os/ranking.py` | Pure, testable scoring. |
| `openclaw_memory_os/analytics.py` | Health summary, dup clustering, deletion candidates. |
| `openclaw_memory_os/app.py` | FastAPI app + routes. |
| `openclaw_memory_os/cli.py` | CLI entry point (`openclaw-memory-os ...`). |
| `openclaw_memory_os/privacy.py` | Privacy scanner. |
| `openclaw_memory_os/templates/*.html` | Server-rendered UI. |

## Configuration

All runtime config is environment-driven. See [`.env.example`](../.env.example) for
the canonical list. The notable variables:

* `MEMORY_OS_TOKEN` — if set, all non-health routes require
  `Authorization: Bearer <token>` or the dashboard session cookie.
* `QDRANT_URL` — if set, the Qdrant backend is used.
* `QDRANT_COLLECTION` — primary collection name (default
  `openclaw_memories`).
* `QDRANT_SECONDARY_COLLECTIONS` — optional comma-separated list of
  additional collections the dashboard should report on.
* `RECALL_FALLBACK_SUPERSEDED` / `RECALL_FALLBACK_SUPERSEDED_MIN_RESULTS`
  — control the recall fallback strategy (active-only first, expand to
  superseded below the threshold).
* `MEMORY_OS_SAMPLE_PATH` — path to the JSON sample backend.

## What the OS does NOT do

* It does not physically delete memories.
* It does not push or pull from the outside world at runtime.
* It does not embed by default. Dense recall uses importance + recency as a
  testable placeholder (see `docs/recall-ranking.md`).
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
