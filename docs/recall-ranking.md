# Recall ranking

This document describes the v0.3.0 production recall pipeline. The
production entry point is `RetrievalEngine.retrieve()` in
`openclaw_memory_os/retrieval_engine.py`. `openclaw_memory_os/ranking.py`
is retained for legacy compatibility. It is **not** the final v0.3.0
production retrieval implementation.

## Inputs

A `RecallRequest` carries:

* `query` — text to match.
* `mode` — `hybrid` (default), `keyword`, or `dense`.
* `since_days` — optional recency window.
* `include_superseded` / `include_expired` — opt-ins for filtered statuses.
* `tier_filter` — optional list of allowed tiers.
* `limit` — cap on returned hits (clamped to `settings.max_recall_results`).

The `RetrievalEngine` consumes the configured memory backend (Qdrant or
the JSON development backend) and returns a `RecallResponse` with hits,
diagnostics, policy metadata, and optional superseded-fallback information.

## Production flow

```
RecallRequest
   ↓
RetrievalEngine.retrieve()
   ↓
active-only candidate pass
   ↓
keyword mode: BM25 lexical search
dense mode:   Qdrant vector similarity
hybrid mode:  weighted reciprocal-rank fusion
   ↓
feature reranking
   ↓
optional superseded fallback (active-first)
   ↓
diagnostics, explanations, and policy metadata
```

### Active-only candidate pass

The first pass always searches `status == active`. Superseded and expired
memories are filtered out unless the caller opts in via
`include_superseded` / `include_expired`.

### Mode-specific scoring

* **`keyword`** runs a BM25 lexical search over the candidate pool.
  The BM25 index lives in `openclaw_memory_os/lexical.py` and supports
  English, Chinese 2-3 grams, snake_case, kebab-case, environment
  variables, IP:port, version strings, file paths, and model names.
  An exact-identifier boost is applied at search time, and the on-disk
  cache is checksum-verified on load.
* **`dense`** runs Qdrant vector similarity using the configured
  embedding backend (`nomic-embed-text` is the v0.3.0 reference). When
  the embedding backend is unavailable, the engine records
  `degraded_reason = "embedding_unavailable"` in the response and falls
  back to lexical results for the dense call. Zero vectors are never
  sent to Qdrant.
* **`hybrid`** combines the keyword and dense candidate lists with
  weighted reciprocal-rank fusion (RRF) and then applies feature
  reranking (recency, importance, feedback signals) on top of the fused
  list.

### Feature reranking

After the mode-specific ranking, the engine applies a feature rerank that
combines:

* `recency` — exponential decay over `updated_at` (or `created_at`).
* `importance` — the memory's recorded importance score.
* `feedback` — recent structured `feedback_events` signals when present.

Every hit carries a `components` breakdown and a human-readable
`explanation`. The breakdown is what makes the scoring debuggable end-to-end
without firing up a notebook.

### Active-first / Superseded fallback

The active pass uses the same filters as the request and applies the
`superseded` / `expired` opt-ins exactly as written.

* **Default active-only pass.** The active pass always runs first.
* **Fallback trigger.** If the active pass returns fewer than
  `settings.recall_fallback_superseded_min_results` hits
  (default `5`, configurable via
  `RECALL_FALLBACK_SUPERSEDED_MIN_RESULTS`), and the request did **not**
  opt in to superseded memories directly
  (`include_superseded=False`), and there is at least one superseded
  memory in the corpus, the API re-runs the same hybrid engine with
  `include_superseded=True` and merges the additional hits.
* **No duplication.** Every hit returned by the active pass wins its
  slot; superseded hits are filtered out by id before they can be
  appended.
* **Superseded stays below active.** Superseded hits added by the
  fallback are clamped to a score strictly below the lowest active hit,
  so they never outrank live memory. The active-pass score distribution
  sets the floor; if no active hit matched, the floor is `0`. Each
  clamped hit carries a `fallback_floor` component for diagnostics.
* **Result cap.** The merged result is still capped at
  `min(request.limit, settings.max_recall_results)`, so a fallback that
  would explode the result is trimmed to the same envelope.
* **`total_considered` stability.** The response's `total_considered`
  continues to reflect the active-pass count so dashboards don't
  suddenly report a different corpus size when the fallback engages.
* **Diagnostics.** `RecallResponse.fallback` reports
  `enabled` / `min_results` / `used` / `added` so the dashboard / API
  consumer can surface when a superseded memory was appended.
* **Opt-out.** Set `RECALL_FALLBACK_SUPERSEDED=off` (or
  `settings.recall_fallback_superseded=False`) to disable the strategy
  entirely. An explicit `include_superseded=True` in the request always
  bypasses the fallback logic — the caller already asked for
  superseded in the same ranking.

## Pre-filters

Pre-filters reject a memory *before* scoring:

* `status == superseded` rejected unless `include_superseded=True`.
* `status == expired` rejected unless `include_expired=True`.
* `updated_at/created_at` older than `since_days` rejected.
* Tiers outside `tier_filter` rejected.

Rejections are recorded in the per-hit `explanation`, never silently dropped.

## Why this shape?

* **Decoupled components** so swapping any single piece (lexical
  backend, dense backend, rerank weights) does not change the API
  surface.
* **Explanations** so the dashboard can render "why was this hit returned".
* **Determinism** so CI tests are stable.
* **Active-first safety net** so a recall query never silently returns
  "nothing useful" when the answer is hiding in superseded memories.

## Compatibility code

`openclaw_memory_os/ranking.py` is retained for legacy callers and for
tests that exercise the older scoring formula directly. It is **not** the
final v0.3.0 production retrieval implementation and should not be wired
into new code paths. All production recall (dashboard, CLI, API, adapter)
goes through `RetrievalEngine.retrieve()`.

## Related files

| File | Purpose |
| ---- | ------- |
| `openclaw_memory_os/retrieval_engine.py` | `RetrievalEngine.retrieve()` — the v0.3.0 production path. |
| `openclaw_memory_os/lexical.py` | BM25 lexical index and tokenizer. |
| `openclaw_memory_os/ranking.py` | Legacy compatibility scoring. |
| `openclaw_memory_os/candidate_pool.py` | Shared candidate pool for evaluation and live recall. |
| `tests/test_recall_fallback.py` | Pins the active-first / superseded-fallback contract. |
| `tests/test_retrieval_engine.py` | Pins the mode-specific and hybrid semantics. |
