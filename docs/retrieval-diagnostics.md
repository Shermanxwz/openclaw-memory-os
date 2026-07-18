# Retrieval diagnostics

Every response from the recall API carries a **diagnostics envelope**
that lets the dashboard and the offline evaluator reason about what
actually happened during the request. This document is the contract
reference for that envelope and the related metrics, and explains the
"honest null" contract used by the offline evaluator.

The authoritative types live in `openclaw_memory_os/contracts.py`
(`RetrievalDiagnostics`, `ScoredMemoryCandidate`, `MemoryRecord`,
`MemoryRef`) and `openclaw_memory_os/evaluation.py` (`EvalResult`,
`CandidatePool`). The JSON shape is pinned by `schemas/recall.schema.json`.

## Recall path overview

The unified `RetrievalEngine` (`openclaw_memory_os/retrieval_engine.py`)
is the single entry point for all recall modes:

1. **First pass.** Defaults to `status=["active"]`. The engine runs the
   requested `mode` (`keyword` | `dense` | `hybrid`) on the active
   memory corpus and produces a list of `ScoredMemoryCandidate`s.
2. **Recency / importance / RRF fusion.** For `hybrid`, the engine
   performs Reciprocal Rank Fusion between the dense and lexical
   channels, then calibrates per-signal scores into `[0, 1]` and
   re-ranks using the active `Policy` weights.
3. **Superseded fallback.** If the active pass yielded fewer hits
   than `Policy.fallback_min_results`, the engine re-runs the same
   hybrid engine on `status=["superseded"]` and merges the results;
   the merged superseded hits are clamped below the lowest active
   score so they never outrank live memories.
4. **Diagnostics + response shaping.** Each request returns a
   `RetrievalDiagnostics` envelope that the API layer turns into the
   `diagnostics` JSON field of `RecallResponse`.

The legacy `ranking.build_recall_response` scorer still exists for
backwards compatibility; it is selected by setting
`RETRIEVAL_ENGINE_V2=off`. The Response shape stays compatible — the
legacy path emits the same fields it did in v0.2.x.

## Mode semantics

| Mode       | First channel          | Fallback on channel failure                                | Notes                                                                                  |
| ---------- | ---------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `keyword`  | BM25 lexical only      | n/a (lexical is in-process; cannot be "unavailable")       | `lexical_available=False` is set when the index could not be loaded; the response may be empty. |
| `dense`    | Embedding + vector DB  | Lexical-only on `EmbeddingUnavailable`                     | Empty index returns empty hits, never a zero-vector fake success.                       |
| `hybrid`   | Dense + Lexical, RRF   | Lexical-only if embedding fails; full no-result if both fail| Weighted RRF + feature rerank per active `Policy` weights.                              |

A `keyword`-mode response that hits nothing **does not** mean the
dense channel is broken — the engine doesn't even touch the dense
channel in that mode. The dashboard surfaces this through the
`diagnostics.lexical_available` and `diagnostics.dense_available`
fields.

## `RetrievalDiagnostics` envelope

The envelope is defined as a pydantic model
(`openclaw_memory_os/contracts.py`). The fields below are stable; new
fields may be added in v0.3.0.x without bumping a major version.

| Field                  | Type                          | Meaning                                                                                |
| ---------------------- | ----------------------------- | -------------------------------------------------------------------------------------- |
| `status`               | `"ok" \| "degraded" \| "failed"` | Top-level signal the dashboard / caller switches on. `"ok"` means every channel succeeded; `"degraded"` means at least one channel was unavailable but the pipeline still produced a result; `"failed"` means no channel could serve. |
| `degraded_reason`      | `string \| null`              | When `status != "ok"`, a human-readable reason. Common values: `"embedding_unavailable"`, `"empty_query"`, `"no_collections_configured"`. |
| `dense_available`      | `bool`                        | True iff the dense (vector) channel was consulted this request. False when the embedder is unavailable. |
| `lexical_available`    | `bool`                        | True iff the BM25 index was loaded. False when the index could not be constructed.     |
| `collections_searched` | `list[str]`                   | Qdrant collections (and `"sample"` for the JSON backend) actually queried this request. |
| `candidate_count`      | `int` (>= 0)                  | Total candidate records considered before ranking (active + superseded).              |
| `embedding_ms`         | `float` (>= 0)                | Wall-clock time spent on embedding + dense search.                                     |
| `lexical_ms`           | `float` (>= 0)                | Wall-clock time spent on BM25 lexical search.                                          |
| `ranking_ms`           | `float` (>= 0)                | Wall-clock time spent on fusion + re-ranking.                                          |

The envelope is `extra="ignore"`, so additional fields returned by
newer engine versions are silently dropped by older clients. Operators
who want to extend the envelope should follow the recipe in
`docs/database-migrations.md` (the same migration discipline applies).

## `EvalResult` envelope

The offline evaluator (`openclaw_memory_os/evaluation.py`) returns
`EvalResult`. The fields below split into **legacy** fields that default
to `0.0` and **v0.3.0.x graded** fields that default to `None`.

| Field                              | Type        | Default | Meaning                                                                       |
| ---------------------------------- | ----------- | ------- | ----------------------------------------------------------------------------- |
| `recall_at_1` / `_5` / `_10`       | `float`     | `0.0`   | Legacy. Fraction of relevant docs in top-k.                                   |
| `mrr_at_10`                        | `float`     | `0.0`   | Legacy. Mean reciprocal rank, first relevant at top-10.                        |
| `ndcg_at_10`                       | `float`     | `0.0`   | Legacy. nDCG@10 with binary relevance.                                         |
| `useful_at_1` / `_5`               | `float`     | `0.0`   | Legacy. Fraction of top-k that are useful.                                    |
| `explicit_negative_at_5`           | `float`     | `0.0`   | Legacy. Fraction of top-5 that are explicitly not-useful.                     |
| `no_result_rate`                   | `float`     | `0.0`   | Legacy. Fraction of cases where the rank_fn returned an empty list.            |
| `p50_latency`, `p95_latency`       | `float`     | `0.0`   | Legacy. Latency percentiles (ms).                                              |
| `degraded_rate`                    | `float`     | `0.0`   | Legacy. Fraction of cases where the engine reported `status="degraded"`.       |
| `fallback_rate`                    | `float`     | `0.0`   | Legacy. Fraction of cases where the superseded fallback was used.             |
| `num_cases`                        | `int`       | `0`     | Total judged cases included in the run.                                       |
| `judged_ndcg_at_10`                | `float \| None` | `None` | v0.3.0.x. Graded nDCG@10; `None` when no graded judgements are available.      |
| `useful_superseded_fallback_rate`  | `float \| None` | `None` | v0.3.0.x. Of the fallback expansions, how many surfaced a judged-useful hit.   |
| `num_judged_cases`                 | `int`       | `0`     | Subset of `num_cases` with at least one graded positive label.                |
| `corpus_snapshot_id`               | `string \| None` | `None` | Snapshot id the metrics were computed against; `None` until the offline pipeline records one. |
| `judged_ndcg_status`               | `string`    | `"unavailable"` | `"ok"` once graded judgments exist; `"unavailable"` otherwise.                  |
| `fallback_rate_status`             | `string`    | `"unavailable"` | Same as above, but for `useful_superseded_fallback_rate`.                      |

`EvalResult.to_dict()` emits `None` for the graded fields exactly as
stored; the dashboard / API consumers must not coerce `None` to `0.0`
before rendering.

## Honest-null contract

The "honest-null" rule is pinned by `tests/test_evaluation_metrics.py`
and surfaced as a stable contract by
`scripts/evaluate_retrieval.py` and `/api/dashboard/evaluation`:

* When the offline pipeline has produced **no judgement yet** (fresh
  install, empty DB), graded fields return `None` with an explicit
  `status="unavailable"` marker. They never return `0.0`. The
  dashboard renders `None` as "not scored yet" instead of "scored
  zero".
* When the offline pipeline has produced judgements, graded fields
  return the real number and `status="ok"`.
* Legacy fields continue to default to `0.0` because older dashboards
  render zeros as "no useful hits" without crashing.
* Warnings are non-fatal: a transient DB read failure surfaces as
  `"warnings": [...]` in the envelope, never as a fake metric.

The contract is exercised by `scripts/evaluate_retrieval.py` (see
below). The same rule applies to `/api/dashboard/evaluation`, with
the additional guarantee that the endpoint is cheap: it never calls
Qdrant, never loads the embedder, and never blocks on I/O.

## Offline evaluation: `scripts/evaluate_retrieval.py`

`scripts/evaluate_retrieval.py` is the **side-effect-free** CLI
companion to `/api/dashboard/evaluation`. It reads the
`recall_feedback.db` file, runs the offline evaluation helpers
(`time_split`, `evaluate`), and emits a JSON envelope on stdout
(or to a file via `--out`).

```text
scripts/evaluate_retrieval.py                # JSON on stdout
scripts/evaluate_retrieval.py --pretty       # JSON, indented
scripts/evaluate_retrieval.py --out FILE     # write JSON to FILE
scripts/evaluate_retrieval.py --limit 50     # cap judged cases
```

Envelope shape:

```text
{
  "status": "ok" | "unavailable" | "error",
  "generated_at": "...ISO-8601 UTC...",
  "corpus_snapshot_id": "snap-..." | null,
  "metrics": { ... EvalResult.to_dict() ... },
  "feedback": { ... get_feedback_summary() ... },
  "history": [ ... recent offline runs ... ],
  "notes": [ "..." ],
  "warnings": [ "..." ]
}
```

Guarantees:

* **No Qdrant access.** The script never imports the Qdrant backend
  and never queries a vector store.
* **No embedder access.** The script never calls an embedding model.
* **No network.** Everything is read from the local SQLite file.
* **No fabricated metrics.** When judged cases are absent, every
  graded field is `None` and `status="unavailable"`. The script
  never returns `0.0` as a substitute.
* **Bounded.** Capped by `--limit` (default 500) so the run is cheap
  on a 50k-point corpus.

The script is exercised by
`tests/test_evaluate_retrieval_script.py`. Tests pin:

* `--help` exits zero;
* import is clean (no Qdrant / no embedder);
* the empty-DB path returns `status="unavailable"` (not an error);
* the metrics envelope contains the documented `EvalResult` fields;
* the script is callable as a module from any working directory.

## Diagnostics in the dashboard

The `/api/dashboard/evaluation` endpoint surfaces the same envelope as
`scripts/evaluate_retrieval.py`. The Evaluation page on the dashboard
renders:

* **Feedback summary cards.** Total events, 24h / 7d / 30d positive
  ratios (with "no data yet" rendering via `null`).
* **Metric tiles.** Each metric row with a value-or-`—` placeholder.
* **History table.** Recent offline runs (compact form).
* **Note line.** Human-readable explanation that offline evaluation is
  read-only and is produced by `scripts/run_evolution_cycle.py` /
  `scripts/evaluate_retrieval.py`.

The dashboard never fakes a metric. If the offline pipeline has not
produced a result, every graded field renders as `—` and the row
carries an explicit "unavailable" badge.

## Operations

* **Force a recall-side refresh.** A recall run already records its
  trace via `recall_feedback.record_recall_run()` /
  `record_recall_result()`. No manual action is needed.
* **Force an evaluation refresh.** Run `scripts/evaluate_retrieval.py
  --out /var/tmp/memory-os-evaluation.json` or call
  `/api/dashboard/evaluation` and read the JSON response.
* **Inspect a single recall.** `GET /api/health` returns
  `dense_available` / `lexical_available` for the **next** request, not
  for a past one. For diagnostics on a past request, look up the
  `query_id` in `recall_runs`.

## Related files

| File | Purpose |
| ---- | ------- |
| `openclaw_memory_os/retrieval_engine.py` | Unified engine; produces `RetrievalDiagnostics`. |
| `openclaw_memory_os/contracts.py` | Type definitions: `RetrievalDiagnostics`, `MemoryRecord`, `ScoredMemoryCandidate`, `MemoryRef`. |
| `openclaw_memory_os/evaluation.py` | Offline evaluator; produces `EvalResult`, `CandidatePool`. |
| `scripts/evaluate_retrieval.py` | Side-effect-free CLI companion. |
| `schemas/recall.schema.json` | Public JSON shape for `/api/recall-test`. |
| `docs/recall-ranking.md` | Legacy ranking path (kept for `RETRIEVAL_ENGINE_V2=off`). |
| `docs/feedback-loop-roadmap.md` | Background on the feedback loop. |
| `tests/test_evaluation_metrics.py` | Pins the honest-null contract. |
| `tests/test_evaluate_retrieval_script.py` | Pins the CLI contract. |
