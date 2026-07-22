# OpenClaw Memory OS

A privacy-clean, self-evolving FastAPI dashboard and recall engine for
Qdrant-backed agent memory. Sits between an agent's memory store and a
human operator. **Never deletes memories** — produces review-only
deletion candidate lists. v0.3.0 adds real hybrid retrieval (Dense +
BM25), offline evaluation, deterministic policy evolution with
auto-promotion and rollback, Pydantic-validated LLM ingestion, and
structured feedback.

[![CI](https://img.shields.io/badge/ci-pytest%20%2B%20privacy%20scan-blue)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Project status

This repository is a **privacy-clean public reference distribution** of a
single-operator Memory OS control plane. It is an independent community
project that complements OpenClaw-style Qdrant memory deployments; it is
not affiliated with, endorsed by, or supported by the OpenClaw project.

The public release contains source code, templates, tests, and sanitized
documentation only. It does **not** include operator memories, audit logs,
session databases, credentials, production `.env` files, private taxonomy
overrides, or deployment-specific host identifiers. See [PRIVACY.md](PRIVACY.md).

The implementation is designed and tested for a single trusted operator on
one host (`uvicorn --workers 1`). Treat it as a reference distribution, not
a managed SaaS or production-certified multi-tenant memory platform.

## What it does

OpenClaw Memory OS is an operator-facing control plane for any agent
that keeps memories in a Qdrant collection (or a JSON file for
development). It surfaces a dashboard, runs recall with a real hybrid
(Dense + BM25 + RRF fusion) engine, evaluates recall quality from
structured feedback, and evolves retrieval policy automatically — all
while enforcing hard safety boundaries: no physical deletion, no
touching secrets, no unvalidated LLM output.

It is **not** an LLM agent. It is **not** the memory store itself. It is
the operator's control plane in front of one.

## TL;DR

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/runtime-py312.lock
.venv/bin/python -m pip install --no-deps .
.venv/bin/openclaw-memory-os serve --host 127.0.0.1 --port 7788
# open http://127.0.0.1:7788/login and authenticate there
```

## Architecture (v0.3.0)

```
                         ┌─────────────────────────────────────┐
                         │         Operator / Dashboard        │
                         │  (HTML + Chart.js, TOTP login)      │
                         └──────────────┬──────────────────────┘
                                        │ HTTP / JSON API
                         ┌──────────────▼──────────────────────┐
                         │           FastAPI App                │
                         │  app.py · auth.py · cli.py          │
                         └──────────────┬──────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
         ┌──────────▼──────┐ ┌──────────▼──────┐ ┌─────────▼────────┐
         │ RetrievalEngine  │ │  Feedback Loop  │ │  Maintenance     │
         │ (hybrid RRF)     │ │  (structured    │ │  (ingest →       │
         │                  │ │   SQLite)       │ │   reclassify →   │
         │ Dense + BM25 +   │ │                 │ │   dedupe →       │
         │ feature rerank   │ │  Evaluation     │ │   expire →       │
         │ + active-first   │ │  (time-split,   │ │   snapshot)      │
         │ + superseded     │ │   Recall@k,     │ │                  │
         │   fallback       │ │   MRR, nDCG)    │ │  Evolution       │
         └────────┬─────────┘ └────────┬────────┘ │  (policy search │
                  │                    │          │   + shadow +     │
         ┌────────▼────────────────────▼──────────▼─┐ auto-promote)  │
         │              Policy Store                 │                │
         │  (JSON, hot-reload, checksum)             │                │
         └────────────────────┬──────────────────────┘                │
                              │                         ┌────────────┘
                   ┌──────────▼──────────┐              │
                   │   Memory Backend    │              │
                   │  (Qdrant / JSON)    │◄─────────────┘
                   │  + BM25 Lexical     │  (ingestion_validation,
                   │    Index            │   tier_classifier,
                   └─────────────────────┘   supersede_detect)
```

## Features

### Retrieval & Recall

- **Unified RetrievalEngine** (`retrieval_engine.py`): single entry
  point for all recall. Three modes: `keyword` (BM25 only), `dense`
  (vector only, graceful degradation to BM25 on embedding failure),
  `hybrid` (Weighted RRF + feature rerank). Active-first pass with
  automatic superseded-hybrid fallback when active hits fall below
  `fallback_min_results`. Expired memories are never auto-included.
- **BM25 lexical index** (`lexical.py`): multi-field tokenisation
  (English, Chinese 2-3 grams, snake_case, kebab-case, env vars,
  IP:port, version strings, file paths, model names). Exact-identifier
  boost at search time. On-disk cache with checksum verification.
- **Policy Store** (`policy_store.py`): JSON-based retrieval weights
  with SHA-256 tamper-evident checksum, hot-reload from disk, and a
  known-good baseline policy. The engine re-resolves the active policy
  on every request.
- **Recall fallback strategy**: active memories ranked first;
  superseded memories included only when active hits are below
  threshold, clamped below the lowest active score.

### Self-Evolution

- **Offline evaluation** (`evaluation.py`): time-split (60/20/20),
  judged metrics (Recall@1/5/10, MRR@10, nDCG@10, useful@1/5,
  explicit-negative@5, no-result rate, latency percentiles).
  CandidatePool design so each query is run once and multiple policies
  re-rank the same pool.
- **Policy evolution** (`evolution.py`): deterministic coordinate-style
  search + bounded random candidate generation. Cold-start gates
  (30/100 queries). Guarded auto-promotion (2-consecutive-window,
  7-day cooldown, max 2/30d). Immediate rollback (corrupt file,
  error >5%, latency 2x) and statistical rollback (useful_at_1
  down >8%, MRR down >5%).
- **Governance integration**: `autonomous_governance.sh` runs the full
  evolution cycle (acquire lock → rollback check → cold-start gate →
  candidate generation → evaluation → shadow → promotion). Independent
  lock file prevents concurrent runs.

### Ingestion & Validation

- **Pydantic-validated LLM ingestion** (`ingestion_validation.py`):
  `ClassificationSchema` rejects unknown fields, unknown type/topic,
  out-of-range importance, enforces 80-char summary and 8-item keyword/
  entity/trigger limits. Single retry with corrective prompt;
  deterministic fallback on double failure. Records
  `classification_status` and `prompt_version` in the Qdrant payload.
- **Robust ingestion** (`ingestion.py`): checkpoint/resume, skip-existing,
  progress state, longer Ollama timeouts, SIGINT/SIGTERM checkpointing.
- **Hard contracts** (`contracts.py`): canonical memory identity
  (`collection:memory_id`), cross-module data shapes (`MemoryRef`,
  `MemoryRecord`, `ScoredMemoryCandidate`, `RecallHit`), and runtime
  invariants documented as constants.

### Feedback & Audit

- **Structured feedback** (`recall_feedback.py`): three SQLite tables
  (recall_runs, recall_results, feedback_events) with per-hit scores,
  policy version, query_id, and 180-day retention. Legacy audit-log
  feedback can be migrated idempotently.
- **SQLite audit log** (`audit.py`): feedback, ingestion, and
  consolidation events.
- **Duplicate consolidation** (`consolidation.py`): merge / keep-newest /
  keep-best strategies while preserving `tier=core` memories.

### Dashboard & Auth

- **FastAPI dashboard** (server-rendered HTML + Chart.js): overview,
  timeline, tiers, duplicates, recall, deletion review, audit log,
  maintenance status, Memory Brain status, autonomous governance
  status card, and strategy/policy visualisation. CSS and JS are
  modular (`static/css/`, `static/js/`).
- **JSON API**: `/api/health`, `/api/timeline`, `/api/tiers`,
  `/api/duplicates`, `/api/deletion-candidates`, `POST /api/recall-test`,
  `POST /api/feedback`, `GET /api/audit-log`,
  `POST /api/consolidate-duplicates`, `POST /api/maintenance/reclassify`,
  `GET /api/strategy`, `GET /api/minimax`.
  v0.3.0.x dashboard endpoints:
  `/api/dashboard/strategy`, `/api/dashboard/evaluation`,
  `/api/dashboard/memories`, `/api/security/sessions`,
  `POST /api/security/sessions/revoke-all`,
  `POST /api/evolution/pause|resume|candidate/reject|rollback`.
- **Layered auth**: shared bearer token (`MEMORY_OS_TOKEN`) for
  development; optional Password + TOTP two-step login for production.
  HttpOnly `memory_os_session` cookie for browsers; `Authorization:
  Bearer` for non-browser clients. API query-string tokens are
  intentionally rejected.
- **OpenClaw plugin adapter** (`plugins/openclaw-memory-os-adapter/`):
  Node.js plugin that queries Memory OS before falling back to built-in
  memory search.

### Maintenance & Governance

- **Multi-collection maintenance**: `scripts/maintenance.sh` iterates
  over a configurable list of Qdrant collections and runs ingest →
  reclassify → supersede detection → expiry → snapshot → summary for
  each. Takes its own `flock` lock; do **not** wrap it in an outer one.
  Optional Memory Brain steps enabled with `ENABLE_MEMORY_BRAIN=1`.
- **Conservative-by-default supersede modes**: tag-only by default —
  writes `review_reason: near_duplicate` and never touches `status`.
  Lifecycle-changing auto-supersede gated behind
  `ENABLE_AUTO_SUPERSEDE=1` and capped by `SUPERSEDE_MAX_APPLY`.
  `tier=core` and `tier=long` memories are never auto-superseded.
- **Backup retention**: `scripts/backup_snapshot.sh` triggers a Qdrant
  snapshot, archives locally with zstd, prunes old archives by
  `BACKUP_KEEP`, then prunes Qdrant-internal snapshots and sweeps the
  backup cache directory.
- **Autonomous governance**: weekly run surfaces as a compact dashboard
  card (last run / next run / result). Production deployment installs a
  persistent systemd timer under the dedicated service account.

### Privacy & Safety

- **Personal taxonomy**: operators keep their own keywords in a
  gitignored `config/personal_taxonomy.json` so the public repo never
  embeds operator-specific data. The bundled example uses generic
  placeholders.
- **Privacy scanner** (`privacy.py`): flags private filesystem paths,
  internal hostnames, provider IDs, API keys, JWTs, IPs, account
  patterns. Per-line `privacy-allow:` markers and a JSON baseline
  cover legitimate mentions in docs.
- **CI workflows**: pytest matrix, smoke import, in-repo privacy
  scanner, and `gitleaks` for deeper coverage.

## Quickstart

```bash
git clone https://github.com/<your-org>/openclaw-memory-os.git
cd openclaw-memory-os
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/dev-py312.lock
.venv/bin/python -m pip install --no-deps -e .

# Optional: try the dashboard with sample data
DEMO_SERVE=1 ./scripts/run_demo.sh
# visit http://127.0.0.1:7788/login
```

For production:

```bash
cp .env.example .env
$EDITOR .env                    # set MEMORY_OS_TOKEN, MEMORY_OS_DOMAIN, QDRANT_URL, etc.
cp config/personal_taxonomy.example.json config/personal_taxonomy.json
$EDITOR config/personal_taxonomy.json    # add your tier / topic keywords
.venv/bin/python -m pip install -r requirements/runtime-py312.lock
.venv/bin/python -m pip install --no-deps .
.venv/bin/openclaw-memory-os serve --host 127.0.0.1 --port 7788
```

Deploy with systemd + nginx (see `deploy/`):

```bash
sudo bash deploy/deploy.sh
```

## Repository layout

```
openclaw-memory-os/
├── openclaw_memory_os/            # FastAPI app + core engine
│   ├── app.py                     # FastAPI app, route definitions
│   ├── auth.py                    # bearer-token + TOTP gate
│   ├── cli.py                     # openclaw-memory-os CLI
│   ├── config.py                  # env-driven settings
│   ├── contracts.py               # hard contracts, identity model, data shapes
│   ├── retrieval_engine.py        # unified hybrid retrieval (Dense + BM25 + RRF)
│   ├── lexical.py                 # BM25 lexical index, multi-field tokeniser
│   ├── ranking.py                 # legacy ranking (retained for compat)
│   ├── policy_store.py            # JSON policy with hot-reload + checksum
│   ├── evolution.py               # deterministic policy search + shadow + auto-promote
│   ├── evaluation.py              # offline evaluation (time-split, IR metrics)
│   ├── recall_feedback.py         # structured SQLite feedback (3 tables)
│   ├── feedback.py                # legacy feedback recording
│   ├── ingestion.py               # checkpoint/resume ingestion
│   ├── ingestion_validation.py    # Pydantic-validated LLM ingestion contract
│   ├── consolidation.py           # duplicate consolidation analysis
│   ├── analytics.py               # deletion-candidate analytics
│   ├── audit.py                   # SQLite audit log
│   ├── personal_taxonomy.py       # operator-specific keyword loader
│   ├── privacy.py                 # in-house privacy scanner
│   ├── backends/__init__.py       # SampleBackend + QdrantBackend
│   ├── models.py                  # Pydantic models
│   ├── static/
│   │   ├── css/dashboard.css      # modular dashboard styles
│   │   └── js/                    # modular section scripts (common, recall, strategy, etc.)
│   └── templates/
│       ├── dashboard.html         # server-rendered dashboard
│       └── login.html             # TOTP login form
├── plugins/
│   └── openclaw-memory-os-adapter/  # OpenClaw Node.js plugin
│       ├── src/adapter.js
│       └── src/index.js
├── config/
│   └── personal_taxonomy.example.json  # public-safe template
├── data/                          # sample memories for JSON backend
├── deploy/                        # systemd unit, nginx vhost, logrotate, deploy.sh
├── docs/                          # architecture, retrieval, security, deployment, evolution
│   ├── architecture.md
│   ├── auth-totp.md
│   ├── database-migrations.md     # recall feedback schema + no-physical-delete invariant
│   ├── deletion-policy.md
│   ├── deployment.md
│   ├── feature-flags.md           # v0.3.0.x flag contract + safe-disable playbook
│   ├── feedback-loop-roadmap.md
│   ├── gap-analysis.md
│   ├── openclaw-integration.md
│   ├── openclaw-memory-plugin-adapter.md
│   ├── privacy-scanner.md
│   ├── recall-ranking.md
│   ├── retrieval-diagnostics.md   # engine envelope + honest-null contract
│   ├── self-evolution.md          # gated policy evolution cycle
│   ├── web-console.md             # dashboard sections + API map
│   └── web-security.md            # auth + CSRF + hard safety boundary
├── examples/                      # CLI/API examples
├── scripts/
│   ├── maintenance.sh             # daily timer task; honest aggregate exit
│   ├── autonomous_governance.sh   # weekly governance + evolution cycle
│   ├── final_host_acceptance.sh   # one-command real-host graduation gate
│   ├── backup_snapshot.sh         # Qdrant snapshot + local archive + cache prune
│   ├── refresh_lexical.py         # BM25 index rebuild after snapshot
│   ├── run_evolution_cycle.py     # policy evolution runner
│   ├── evaluate_retrieval.py      # offline evaluation JSON reporter
│   ├── replay_feedback.py         # feedback replay for evaluation
│   ├── supersede_detect.py        # conservative/full-auto supersede modes
│   ├── expire_cron.py             # mark stale working-tier memories expired
│   ├── tier_classifier.py         # tier + importance reclassify pass
│   ├── ingest_memory.py           # one-shot ingestion helper
│   ├── dedup_cron.py              # near-duplicate detection
│   ├── memory_brain_ingest.py     # optional structured-memory ingestion
│   ├── memory_brain_consolidate.py # optional topic-summary compaction
│   ├── _write_summary.py          # log parser → atomic summary JSON
│   ├── _write_governance_status.py # governance status JSON writer
│   ├── _qdrant_helpers.py         # Qdrant helper (integer-ID coercion, batched writes)
│   ├── _prune_helpers.py          # backup-cache prune helpers
│   ├── run_demo.sh
│   └── privacy_scan.sh
├── schemas/                       # JSON schemas (memory, recall)
├── tests/                         # 593+ pytest tests
├── .github/workflows/
│   ├── ci.yml                     # pytest + privacy scan
│   └── secret-scan.yml            # gitleaks
├── requirements/                  # audited CPython 3.12 runtime/dev locks
├── pyproject.toml
├── README.md
├── CHANGELOG.md
└── LICENSE
```

## Authentication model

### Bearer token (development / simple setups)

When `MEMORY_OS_TOKEN` is set, the app uses a single shared token:

1. Browser login: open `/login` and submit the token, or Password + TOTP when configured.
2. The server sets an HttpOnly, SameSite=Strict `memory_os_session`
   cookie.
3. Dashboard JavaScript uses that cookie for JSON API calls.
4. Non-browser clients can use `Authorization: Bearer <TOKEN>`.
5. API query-string tokens are intentionally rejected.

### Password + TOTP (production)

When `MEMORY_OS_PASSWORD` is set, the login form upgrades to a two-step
challenge (password + 6-digit TOTP). Set both `MEMORY_OS_PASSWORD` and
`MEMORY_OS_TOTP_SECRET` to enable. Session lifetime is configurable via
`MEMORY_OS_SESSION_MAX_AGE` (default 12 hours). See
[docs/auth-totp.md](docs/auth-totp.md) for setup instructions.

## Configuration (env vars)

| Variable | Default | Notes |
| --- | --- | --- |
| `MEMORY_OS_TOKEN` | unset | Single bearer token. Off → auth disabled. |
| `MEMORY_OS_PASSWORD` | unset | Set to enable Password + TOTP login. |
| `MEMORY_OS_TOTP_SECRET` | unset | Base32 TOTP secret (required when PASSWORD is set). |
| `MEMORY_OS_SESSION_MAX_AGE` | `43200` | Session cookie lifetime in seconds (12h default). |
| `QDRANT_URL` | unset | Set to enable the Qdrant backend. |
| `QDRANT_COLLECTION` | `openclaw_memories` | Primary Qdrant collection. |
| `QDRANT_SECONDARY_COLLECTIONS` | unset | Comma-separated additional collections. |
| `QDRANT_API_KEY` | unset | Forwarded to the Qdrant client. |
| `MEMORY_OS_SAMPLE_PATH` | `data/sample_memories.json` | Sample backend file. |
| `MEMORY_OS_MAX_RECALL` | `25` | Hard cap on recall result size. |
| `MEMORY_OS_RECENCY_HALFLIFE_DAYS` | `30` | Recency half-life for the recency boost term. |
| `MEMORY_OS_SUPERSEDED_PENALTY` | `0.25` | Multiplier on superseded memories. |
| `MEMORY_OS_EXPIRED_PENALTY` | `0.10` | Multiplier on expired memories. |
| `MEMORY_OS_IMPORTANCE_BOOST` | `0.6` | Scale on the importance term. |
| `RECALL_FALLBACK_SUPERSEDED` | `on` | `on` / `off`. Enables the recall fallback strategy. |
| `RECALL_FALLBACK_SUPERSEDED_MIN_RESULTS` | `5` | Active-hit threshold below which fallback engages. |
| `MEMORY_OS_TAXONOMY_PATH` | `config/personal_taxonomy.json` | Path to gitignored personal taxonomy. |
| `MEMORY_OS_GOVERNANCE_STATUS` | `~/.local/state/openclaw-memory-os/autonomous-governance.json` | Governance status JSON. |
| `WORKSPACE_ROOT` | repo parent | Workspace containing `MEMORY.md` and `memory/*.md`. |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint for ingestion. |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for ingestion. |
| `BACKUP_DIR` | `/var/lib/openclaw-memory-os/snapshots` | Local archive directory. |
| `BACKUP_KEEP` | `5` | Local archive retention. |
| `QDRANT_BACKUP_CACHE_DIR` | `/opt/qdrant/backup` | Qdrant download/cache directory to prune. |
| `QDRANT_CACHE_KEEP` | `0` | Cache retention. `0` drops redundant copies after archive. |
| `SUPERSEDE_MAX_APPLY` | `200` | Cap on per-collection auto-supersede writes per run. |
| `ENABLE_AUTO_SUPERSEDE` | unset (off) | Opt-in for lifecycle-changing auto-supersede. |
| `ENABLE_TOPIC_SUPERSEDE` | unset (off) | Opt-in for keyword-topic supersede pass. |
| `ENABLE_MEMORY_BRAIN` | `0` | Enable optional Memory Brain ingest/consolidate steps. |
| `MEMORY_BRAIN_COLLECTION` | `QDRANT_COLLECTION` or `openclaw_memory_brain` | Target collection for Memory Brain scripts. |
| `MEMORY_BRAIN_MAX_FILES` | `20` | Max changed memory files per ingest run (`0` = all). |
| `MEMORY_OS_SESSIONS_DB` | `<XDG_STATE_HOME>/openclaw-memory-os/sessions.db` | Path to the persistent session SQLite DB. |
| `DREAM_MIN_HOURS` | `24` | Minimum hours between Memory Brain consolidation runs. |
| `RETRIEVAL_ENGINE_V2` | `on` | v0.3.0.x feature flag: `on` uses the unified `RetrievalEngine`; `off` falls back to the legacy scorer. See [docs/feature-flags.md](docs/feature-flags.md). |
| `STRUCTURED_FEEDBACK` | `on` | v0.3.0.x feature flag: `on` writes recall feedback to the structured SQLite tables; `off` falls back to the legacy audit-log path. |
| `EVOLUTION_ENABLED` | `on` | v0.3.0.x feature flag: `off` makes every `/api/evolution/*` endpoint and the weekly governance cycle a safe no-op. See [docs/self-evolution.md](docs/self-evolution.md). |
| `SHADOW_ENABLED` | `on` | v0.3.0.x feature flag: `off` skips the shadow-comparison stage of the evolution cycle. |
| `PASSWORD_TOTP_AUTH` | `on` | v0.3.0.x feature flag: `off` forces the legacy bearer-token login flow even when password + TOTP are configured. |
| `MEMORY_OS_RECALL_STATE_DIR` | `$XDG_STATE_HOME/openclaw-memory-os` | Override the state directory used for the recall-feedback DB + evolution state JSON. |

## Documentation

- [docs/architecture.md](docs/architecture.md) — module / data-flow overview
- [docs/retrieval-diagnostics.md](docs/retrieval-diagnostics.md) — `RetrievalDiagnostics` envelope, `EvalResult` honest-null contract, `scripts/evaluate_retrieval.py`
- [docs/self-evolution.md](docs/self-evolution.md) — policy-evolution cycle, gates, rollback, timer
- [docs/database-migrations.md](docs/database-migrations.md) — recall-feedback schema + no-physical-delete invariant
- [docs/feature-flags.md](docs/feature-flags.md) — v0.3.0.x flag contract + safe-disable playbook
- [docs/web-console.md](docs/web-console.md) — dashboard sections + API map
- [docs/web-security.md](docs/web-security.md) — auth + CSRF + hard safety boundary
- [docs/recall-ranking.md](docs/recall-ranking.md) — ranking formulas + fallback strategy
- [docs/auth-totp.md](docs/auth-totp.md) — Password + TOTP setup
- [docs/feedback-loop-roadmap.md](docs/feedback-loop-roadmap.md) — feedback loop background
- [docs/deployment.md](docs/deployment.md) — systemd, nginx, logrotate
- [docs/final-host-acceptance.md](docs/final-host-acceptance.md) — real Qdrant/Ollama and 20k+ final gate
- [docs/deletion-policy.md](docs/deletion-policy.md) — no-physical-delete contract details
- [docs/privacy-scanner.md](docs/privacy-scanner.md) — in-repo privacy scanner rules

## Graduation status

As of v0.3.0, OpenClaw Memory OS is feature-complete for the
single-operator Memory OS control-plane scope:

- The full audited suite passes across Python 3.10, 3.11, and 3.12, covering
  auth, ranking, recall fallback, API routes, privacy, audit, feedback,
  consolidation, ingestion, backups, Qdrant adapters, evaluation, policy
  lifecycle, evolution, deployment, and governance.
- The hybrid retrieval engine (Dense + BM25 + RRF) is the single
  production recall path; `ranking.py` is retained for backward
  compatibility only.
- Policy evolution runs automatically inside the weekly governance
  cycle with cold-start gates, guarded auto-promotion, and immediate
  + statistical rollback.
- Destructive operations remain outside the dashboard: the project
  never physically deletes memories; lifecycle changes are performed
  by explicit maintenance scripts with conservative defaults and caps.
- Public repo hygiene is enforced through pytest, compile checks, the
  in-repo privacy scanner, and GitHub secret-scan workflow.

## Tests

```bash
pytest -v
```

Coverage spans the following core contracts (the suite grows without pinning a stale test count):

| Test file | Scope |
| --- | --- |
| `test_auth.py` | Token gate, TOTP, header/query extraction, health-route exemption |
| `test_ranking.py` | Pure ranking: filters, mode switches, recency, importance, tier, since_days |
| `test_recall_fallback.py` | Active-first default, fallback threshold, no duplication, superseded clamping |
| `test_real_hybrid.py` | End-to-end hybrid retrieval with RRF fusion |
| `test_active_first_hybrid.py` | Active-first + superseded-hybrid fallback integration |
| `test_dense_search_v030.py` | Dense-only mode with graceful BM25 degradation |
| `test_lexical_bm25.py` | BM25 tokeniser, CJK, exact-identifier boost, cache |
| `test_evaluation_metrics.py` | Time-split, Recall@k, MRR, nDCG, useful-at-k |
| `test_candidate_pool.py` | CandidatePool, judged nDCG, useful superseded fallback metrics |
| `test_candidate_search.py` | Deterministic policy search, cold-start gates |
| `test_policy_store.py` | JSON policy load/save, checksum, hot-reload |
| `test_api.py` | Every route, auth-on/off matrix, feedback, audit, consolidation |
| `test_dashboard_ui_compliance.py` | Modular dashboard sections, JS modules, no-CDN contract |
| `test_evaluate_retrieval_script.py` | Offline evaluation script smoke and honest-null output |
| `test_feature_flags.py` | v0.3.0.x feature-flag defaults and safe-disable behavior |
| `test_privacy.py` | Scanner rules, per-line marker, baseline, real-world fixtures |
| `test_audit.py` | SQLite audit store |
| `test_feedback.py` | Legacy feedback recording |
| `test_feedback_schema.py` | Structured feedback schema validation |
| `test_ingestion.py` | Checkpoint/resume, governance payload fields |
| `test_consolidation.py` | Duplicate consolidation strategies |
| `test_supersede_detect.py` | Conservative vs. full-auto, tier guards, cap |
| `test_superseded_hybrid_fallback.py` | Superseded-hybrid fallback edge cases |
| `test_personal_taxonomy.py` | Taxonomy loader, env override, no-leak guard |
| `test_backup_snapshot_cache.py` | Backup-cache prune helpers |
| `test_qdrant_helpers.py` | Point-ID coercion matrix |
| `test_qdrant_backend_payload.py` | Backend payload cast for integer-ID collections |
| `test_qdrant_backend_search.py` | Qdrant backend search integration |
| `test_write_summary.py` | Multi-collection maintenance summary parser |
| `test_write_governance_status.py` | Governance status JSON writer |
| `test_autonomous_governance_dashboard.py` | Governance dashboard card |
| `test_autonomous_governance_dashboard_e2e.py` | End-to-end governance flow |
| `test_autonomous_governance_runner.py` | Governance runner with evolution |
| `test_analytics.py` | Deletion-candidate edge cases |
| `test_maintenance_uses_venv.py` | Maintenance script venv activation |
| `test_memory_brain_delete_optin.py` | Memory Brain delete opt-in guard |
| `test_memory_brain_importance_normalize.py` | Importance normalisation |
| `test_qwen_validation.py` | Pydantic ingestion validation |
| `test_replay_feedback.py` | Feedback replay for evaluation |
| `test_version.py` | Version string consistency |
| `test_contracts.py` | Hard contract invariants |

## License

MIT. See [`LICENSE`](LICENSE).
