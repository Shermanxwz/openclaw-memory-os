## Project status (added at public release)

This is the **permanently archived** final source release. No further
development or maintenance is planned.

OpenClaw Memory OS is an **independent community project** and is
**not affiliated with, endorsed by, or supported by the OpenClaw
project**. It is a personal/single-operator reference distribution. The
author publishes the source for inspection, archival, and educational
use.

It **complements** the OpenClaw built-in memory engine; it does not
replace it. For typical keyword, vector, or hybrid search, the
OpenClaw built-in memory is the recommended path. Memory OS adds an
optional layer for multi-collection governance, lifecycle metadata,
evaluation, audit, feedback, policy evolution, and backup/restore
*on top of* an existing memory store (such as a Qdrant collection that
an OpenClaw-compatible agent may write to).

Recommended deployment profile: personal/single-user, recommended
interactive concurrency 1, on a single small host. Sustained
five-request concurrent performance is **not** in the supported
profile and is **not** certified by any release of this project.

This `CHANGELOG.md` is preserved for historical reference only. See
`README.md` for the current public positioning.

---

## v0.3.0 — Real hybrid retrieval + self-evolution + structured feedback

Released: 2026-07-14

v0.3.0 is the largest single upgrade since the project's inception. It replaces
the approximate "dense" and "hybrid" ranking formulas with a real dual-pipeline
(nomic-embed-text Dense + BM25 Lexical), introduces a unified RetrievalEngine
with RRF fusion, Active-first / Superseded-hybrid fallback, Pydantic-validated
LLM ingestion, structured SQLite feedback for offline evaluation, deterministic
policy search, guarded auto-promotion with automatic rollback, and the evolution
lock integrated into the weekly governance runner.

### Added

- **BM25 lexical index** (`openclaw_memory_os/lexical.py`). Tokenizer supports
  English, Chinese 2-3 grams, snake_case, kebab-case, environment variables,
  IP:port, version strings, file paths, and model names. Exact-identifier
  boost at search time. On-disk cache with checksum verification.
- **Unified RetrievalEngine** (`openclaw_memory_os/retrieval_engine.py`). Single
  entry point for all three modes: `keyword` (lexical only), `dense` (vector
  only with graceful degradation to lexical on embedding failure), and
  `hybrid` (Weighted RRF + feature rerank). Mode-specific semantics are
  enforced inside the engine and shared by API and CLI.
- **Active-first / Superseded-hybrid fallback**. The first pass always searches
  Active only. When fewer than `fallback_min_results` active hits are found,
  the engine re-runs the same hybrid engine on Superseded memories and clamps
  their display scores below the lowest active score. Expired memories are
  never auto-included.
- **Pydantic-validated LLM ingestion** (`openclaw_memory_os/ingestion_validation.py`).
  `ClassificationSchema` rejects unknown fields, unknown type/topic, out-of-range
  importance, and enforces 80-char summary, 8-item keyword/entity/trigger limits.
  Single retry with corrective prompt; deterministic fallback on double failure.
  Records `classification_status` and `prompt_version` in the Qdrant payload.
- **Structured feedback** (`openclaw_memory_os/recall_feedback.py`). Three
  SQLite tables (recall_runs, recall_results, feedback_events) with per-hit
  scores, policy version, query_id, and 180-day retention. Legacy audit-log
  feedback can be migrated idempotently.
- **Offline evaluation** (`openclaw_memory_os/evaluation.py`). Time-split
  (60/20/20), judged metrics (Recall@1/5/10, MRR@10, nDCG@10, useful@1/5,
  explicit-negative@5, no-result rate, latency percentiles). CandidatePool
  design so each query is run once and multiple policies re-rank the same pool.
- **Policy evolution** (`openclaw_memory_os/evolution.py`). Deterministic
  coordinate-style search + bounded random candidate generation. Cold-start
  gate (30/100 queries), guarded auto-promotion (2-consecutive-window, 7-day
  cooldown, max 2/30d, max 2 consecutive rollbacks). Shadow comparison,
  immediate rollback (corrupt file, error >5%, latency 2x) and statistical
  rollback (useful_at_1 down >8%, MRR down >5%).
- **Governance integration** (`scripts/refresh_lexical.py`, `scripts/run_evolution_cycle.py`).
  maintenance.sh now runs lexical index refresh after snapshots. autonomous_governance.sh
  runs the full evolution cycle (acquire lock → rollback check → cold-start gate →
  candidate generation → evaluation → shadow → promotion). Independent
  `/tmp/openclaw-memory-os.evolution.lock`.

### Changed

- `ranking.py` is retained for backward compatibility but is **not** the
  v0.3.0 production retrieval path. All production recall paths route
  through `RetrievalEngine.retrieve()`.
- `MemoryBackend` gains an abstract `lexical_search()` method.
- `MemoryRecord` in `contracts.py` extended with `keywords`, `recall_triggers`,
  `entities` fields.
- `FeedbackEntry` model accepts new `query_id` + `candidate_key` fields; the
  old `memory_id` + `query` pair is still accepted.
- `RecallResponse` gains `query_id`, `policy_version`, `diagnostics` fields.
- maintenance.sh runs lexical refresh after Qdrant snapshots.
- autonomous_governance.sh runs evolution after maintenance.

### Fixed

- The previous recommendation to set `embeddings_failure → zero vector` has been
  replaced with a hard contract: embedding failure raises `EmbeddingUnavailable`
  and the engine either silently degrades to lexical or sets `degraded_reason =
  'embedding_unavailable'`. Zero vectors are never sent to Qdrant.
- LLM output is now validated through a Pydantic schema before being written to
  the payload, eliminating the risk of corrupted type/topic values.

### Removed

- `?token=` has been removed from all dashboard pages (token auth still works via
  Bearer header for CLI/scripts).

### Safety

The hard safety boundary statement on the dashboard remains unchanged:
Scope = `memory-content`. Allowed actions = `supersede`, `expire`, `archive`,
`dedupe`, `promote`. Hard boundary = never physically delete; never touch
repo / system / config / secrets / personal taxonomy / external services.

### Verification

- Automated test suite passes.
- `bash scripts/privacy_scan.sh` — 0 findings.
- `bash -n` clean on all shell scripts.
- Manual run of the full autonomous_governance.sh on the operator host
  confirms the evolution lock, feedback replay, deep audit, and status
  write paths.

### Upgrade notes

No manual migration is required. The new SQLite feedback tables are created
lazily on first use. The legacy audit-log feedback can be migrated by running:

```bash
python -c "from openclaw_memory_os.recall_feedback import migrate_legacy_feedback; print(migrate_legacy_feedback())"
```
