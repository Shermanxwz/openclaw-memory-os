# Database migrations

The v0.3.0 series introduces a structured recall-feedback SQLite store
that runs **side-by-side** with the legacy audit-log feedback path. This
document explains how the schema is created, how new columns are added
to pre-existing databases, how legacy feedback is migrated, and which
invariants the database layer is **forbidden** to break — including the
no-physical-deletion hard contract that the rest of the OS depends on.

The authoritative code lives in `openclaw_memory_os/recall_feedback.py`.
The migration tests live in `tests/test_feedback_schema.py` and
`tests/test_evaluate_retrieval_script.py` (for the migration plumbing of
the offline CLI).

## Storage layout

The structured feedback database lives under
`$MEMORY_OS_RECALL_STATE_DIR` (default
`$XDG_STATE_HOME/openclaw-memory-os/`, fallback
`~/.local/state/openclaw-memory-os/`). It is a single SQLite file:

```text
recall_feedback.db
├── recall_runs        (one row per recall-test query)
├── recall_results     (one row per ranked hit)
└── feedback_events    (one row per thumbs-up / thumbs-down)
```

A second file `evolution-state.json` is stored in the same directory and
governs the weekly evolution cycle (see `docs/self-evolution.md`).
**Never** move these files inside the Qdrant or repository path — the
state directory is intentionally `~/.local/state` so an uninstall does
not destroy operator-owned signals.

## Schema and the migration contract

`_ensure_schema(conn)` in `recall_feedback.py` is the single entry point
for the schema. It is:

* **Idempotent.** Running it twice is a no-op. Each new column is
  declared in `_SCHEMA_COLUMNS` and is checked against
  `PRAGMA table_info(table)` before being added.
* **Backward-compatible.** Fresh databases receive the new columns
  inline in `CREATE TABLE IF NOT EXISTS`. Pre-existing databases
  receive them via `ALTER TABLE ... ADD COLUMN ...` guarded by the
  `table_info` check.
* **Failure-tolerant.** A SQLite race where two callers race on the
  same migration is logged and treated as success (`duplicate column
  name` is the only known race).
* **Auditable.** Each migration is reproducible: the `_SCHEMA_COLUMNS`
  tuple is the ordered, named source of truth. Adding a new column
  means appending a triple `(table, column, sql_type)` to that tuple;
  removing one means leaving it in place but adding a guard to skip.

The current `_SCHEMA_COLUMNS` tuple lives at the top of
`recall_feedback.py`. New fields are added in v0.3.0.x releases only
after the offline evaluator and the dashboard have learned to ignore
them (a missing column means "no judgement yet", never an error).

## Retention policy

The retention cleanup in `_retention_cleanup()` runs in two explicit
steps against the structured feedback database. It does not use SQLite
ON DELETE CASCADE; the schema has no `ON DELETE CASCADE` clauses, and
`PRAGMA foreign_keys=ON` only enables the declared foreign-key
constraint checks, not auto-cascading deletes.

* **`recall_runs`** — 180-day retention. Rows older than that are
  pruned by `_retention_cleanup()`.
* **`recall_results`** — retained for the same 180-day window as the
  parent `recall_runs` row. The retention cleanup **explicitly
  deletes the corresponding `recall_results` rows first**, then
  deletes the parent `recall_runs` rows. The intent is that no orphan
  `recall_results` rows are ever produced: every `recall_results` row
  is either deleted alongside its parent or retained alongside it.
* **`feedback_events`** — never auto-pruned. These rows are the only
  source of truth for what the user actually wanted, and the offline
  evaluation pipeline would be silently biased if they were removed.

The actual sequence is:

```sql
-- Step 1: delete recall_results whose parent recall_run is older than
-- the retention window.
DELETE FROM recall_results
WHERE query_id IN (
    SELECT query_id FROM recall_runs
    WHERE created_at < datetime('now', ?)
);

-- Step 2: delete the parent recall_runs.
DELETE FROM recall_runs
WHERE created_at < datetime('now', ?);
```

Operators who need stricter retention must edit
`_RECALL_RUNS_RETENTION_DAYS` and ship a custom build; there is **no**
runtime env var that bypasses the 180-day default because feedback
retention is the contract that the offline evaluator depends on.

## Migration runner — recall feedback

The legacy v0.2.x feedback path stored thumbs-up / thumbs-down as
`action="feedback"` rows in the audit log. v0.3.0 introduced the
structured SQLite tables. `migrate_legacy_feedback()` is the
authoritative migration entry point:

```text
openclaw_memory_os.recall_feedback.migrate_legacy_feedback() -> int
```

It reads every `action="feedback"` audit row, parses the legacy
`detail` string for `(query, memory_id, useful)`, and upserts the row
into `feedback_events` with `feedback_source="migrated"` and
`migration_status="migrated:audit"`. The legacy audit rows are left
**read-only**: the function never deletes or edits them.

Properties:

* **Idempotent.** Each legacy row is keyed by `(query, memory_id,
  created_at)`; re-running the function is a no-op on already-migrated
  rows.
* **Bounded.** The function never opens Qdrant, never calls the
  embedder, never blocks on the network.
* **Auditable.** Each migrated row carries `migration_status =
  "migrated:audit"`, so operators can filter for provenance in
  ad-hoc queries.

The function is also exposed via the CLI:

```bash
python -c "from openclaw_memory_os.recall_feedback import migrate_legacy_feedback; print(migrate_legacy_feedback())"
```

## Adding a new column

To add a new column to one of the three structured-feedback tables,
follow the recipe below. **Do not** rewrite the existing tuples — append
to them.

1. Identify which table the column belongs to (`recall_runs`,
   `recall_results`, `feedback_events`).
2. Choose a SQL type (`TEXT` / `INTEGER` / `REAL`).
3. Append `(table, column, sql_type)` to `_SCHEMA_COLUMNS` in
   `recall_feedback.py`. The order matters: the schema migration runs
   in tuple order on pre-existing databases.
4. Add an optional default to the `record_recall_*` writer that
   populates the field. Make the default `None` / `""` so existing
   callers don't have to be updated.
5. Update `EvalResult.to_dict()` and the `/api/dashboard/evaluation`
   shape to surface the new field.
6. Update this document with the column's purpose and its default.

A complete example lives in the v0.3.0.x history:

* `recall_runs.query_hash`, `corpus_snapshot_id`, `dense_available`,
  `lexical_available`, `collections_succeeded_json`,
  `collections_failed_json`.
* `recall_results.vector_score_raw`, `vector_score_calibrated`,
  `lexical_score_raw`, `lexical_score_calibrated`, `importance_score`,
  `recency_score`, `feedback_score`, `display_score`.
* `feedback_events.migration_status`.

## Migration runner — adding tables

There is currently **one** structured-feedback database (the SQLite file
above). The legacy audit store is separate and survives every
migration. If you ever introduce a second table or a second database:

1. Place it under `$MEMORY_OS_RECALL_STATE_DIR/openclaw-memory-os/`.
2. Reuse `_ensure_schema()` with its own `_SCHEMA_COLUMNS` tuple. The
   existing pattern is intentionally copyable rather than abstracted
   — the second table just gets its own `_SCHEMA_COLUMNS` and its own
   `_ensure_schema()`.
3. Document the new file in this document under a new section so
   operators can find it on a fresh install.

## Hard contract: no physical delete

The OS is forbidden from physically deleting memory records. The
contract lives in `openclaw_memory_os/contracts.py` as
`NO_PHYSICAL_DELETION` (and is re-asserted in the `HARD_CONTRACTS`
tuple).

This database-migrations document does **not** soften that rule. The
allowed operations on the **memory record** are:

* `supersede` — a new record replaces the old via the same
  candidate-key; the old one is flagged `status="superseded"`.
* `expire` — the record's `status` becomes `expired`.
* `archive` — the record is moved to an "archive" collection or
  annotation.
* `dedupe` — two near-duplicate records are merged into one; the loser
  is **superseded**, never deleted.

The retention cleanup in `_retention_cleanup()` is **not** a violation
of this rule because:

* It only deletes `recall_runs` and `recall_results` rows (audit-style
  trace data).
* It never touches memory records or `feedback_events` rows.
* `feedback_events` rows are preserved indefinitely so the offline
  evaluation pipeline keeps its source of truth even when the
  corresponding recall trace is aged out.

If you ever modify `_retention_cleanup()` to also prune `feedback_events`,
the contract reviewer must:

1. Verify the offline evaluation pipeline still has enough signal.
2. Verify no dashboard surfaces a counter that depends on the deleted
   rows.
3. Document the change here **and** in `CHANGELOG.md`.

## Cross-version compatibility

v0.3.0 → v0.3.0.x is forward-compatible. A v0.3.0 process that opens a
database written by v0.3.0.x will simply not see the new columns
(`SELECT *` returns them as `NULL` on SQLite versions that support
default-`NULL` columns). The reverse direction — v0.3.0.x opening a
v0.3.0 database — runs the `_ensure_schema()` migration silently the
first time the writer is called.

The legacy audit-log feedback path is **still readable** in both
directions. The `_parse_old_feedback()` helper preserves
backwards-compatible parsing so an audit row from v0.2.x migrates
cleanly into `feedback_events`.

## Forensic notes for operators

* The schema is `PRAGMA journal_mode=WAL` and `PRAGMA
  foreign_keys=ON`. Operators running multi-process writers should
  keep WAL mode; do not edit it.
* Connections have `timeout=5` (5 seconds) to absorb transient lock
  contention. A contended lock raises `sqlite3.OperationalError`; the
  caller wraps it in a `with _lock:` so the write retried on the next
  request rather than blocking the event loop.
* `feedback_events` rows are preserved across all upgrades. Operators
  who need to **export** the signals (e.g. for offline research) can
  run a `sqlite3 -header -csv recall_feedback.db "SELECT * FROM
  feedback_events"` against the `~/.local/state/openclaw-memory-os/`
  directory; the export is non-destructive.

## Related files

| File | Purpose |
| ---- | ------- |
| `openclaw_memory_os/recall_feedback.py` | Schema, writers, migration helpers. |
| `openclaw_memory_os/evaluation.py` | Read-only consumer; never modifies the DB. |
| `scripts/evaluate_retrieval.py` | Offline CLI wrapper around the read path. |
| `tests/test_feedback_schema.py` | Pins the migration contract. |
| `tests/test_evaluate_retrieval_script.py` | Pins the offline CLI / DB compatibility contract. |
