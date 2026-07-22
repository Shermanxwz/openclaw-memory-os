# Recall ranking

This document describes the current ranking logic. The function lives in
`openclaw_memory_os/ranking.py` and is intentionally pure: same inputs →
same outputs, easy to test, no I/O.

## Inputs

A `RecallRequest` carries:

* `query` — text to match.
* `mode` — `hybrid` (default), `keyword`, or `dense`.
* `since_days` — optional recency window.
* `include_superseded` / `include_expired` — opt-ins for filtered statuses.
* `tier_filter` — optional list of allowed tiers.
* `limit` — cap on returned hits (clamped to `settings.max_recall_results`).

A list of `Memory` objects feeds the scorer.

## Scoring (current)

For each memory that survives the pre-filters, we compute:

* `base` — from the status: `active=1.0`, `superseded=settings.superseded_penalty`,
  `expired=settings.expired_penalty`, `needs_review=0.4`.
* `recency` — exponential decay over `updated_at` (or `created_at`):
  `exp(-ln 2 * age_days / recency_half_life_days)`.
* `importance_boost` — `importance_boost_scale * memory.importance`.
* `keyword` — Jaccard-style overlap over tokens extracted from the
  memory's text, summary, and tags. Coverage (0..1) is weighted 0.7,
  density 0.3.

The composite score depends on mode:

* `keyword`: `composite = keyword`
* `dense`: `composite = 0.5 * base + recency + importance_boost`
  (no embeddings in the sample backend; this is an intentional testable
  placeholder).
* `hybrid` (default): `composite = base * (0.4 + 0.6 * recency)
  + importance_boost + 0.5 * keyword`.

Every hit carries a `components` breakdown (rounded) and a human-readable
`explanation`. The breakdown is what makes the scoring debuggable end-to-end
without firing up a notebook.

## Filters

Pre-filters reject a memory *before* scoring:

* `status == superseded` rejected unless `include_superseded=True`.
* `status == expired` rejected unless `include_expired=True`.
* `updated_at/created_at` older than `since_days` rejected.
* Tiers outside `tier_filter` rejected.

Rejections are recorded in the per-hit `explanation`, never silently dropped.

## Why this shape?

* **Decoupled components** so swapping the placeholder dense score for an
  actual embedding similarity later doesn't change the API surface.
* **Explanations** so the dashboard can render "why was this hit returned".
* **Determinism** so CI tests are stable.

## Future work

* Replace the placeholder `dense` mode with real embedding similarity.
* Add a cross-encoder rerank as an opt-in step after scoring.
* For very large stores, precompute shingle-based MinHash signatures and
  use LSH for the candidate set (mirrors what the duplicate detector
  could do at scale).

## Fallback strategy (default: active-only with safety net)

`build_recall_response()` ranks active memories first and surfaces a
safety net so a recall query never silently returns "nothing useful" when
the answer is hiding in superseded memories.

* **Default active-only pass.** The active pass uses the same filters as
  the request and applies the `superseded` / `expired` opt-ins exactly
  as written.
* **Fallback trigger.** If the active pass returns fewer than
  `settings.recall_fallback_superseded_min_results` hits
  (default `5`, configurable via
  `RECALL_FALLBACK_SUPERSEDED_MIN_RESULTS`), and the request did **not**
  opt in to superseded memories directly
  (`include_superseded=False`), and there is at least one superseded
  memory in the corpus, the API re-ranks with
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

See [`tests/test_recall_fallback.py`](../tests/test_recall_fallback.py)
for the behavioral contract.

## Personal taxonomy (tier / topic keywords)

The tier classifier (`scripts/tier_classifier.py`) and the ingestion
module (`openclaw_memory_os/ingestion.py`, `scripts/ingest_memory.py`)
share a small set of base keyword lists that govern tier classification
and topic inference (`core`, `long`, `amazon`, `personal`, …).

The base lists are intentionally generic — they contain only
public-safe keywords (Chinese phrases, English equivalents, generic
category words). Operator-specific keyword lists (e.g. real Amazon
brand names used by the topic classifier) live in a separate JSON
config file so they never get committed to the public repo.

* **Loader**: `openclaw_memory_os.personal_taxonomy.load_personal_taxonomy()`
* **Default path**: `config/personal_taxonomy.json`
* **Override**: set `MEMORY_OS_TAXONOMY_PATH=/absolute/path/to/your.json`
* **Public template**: `config/personal_taxonomy.example.json` (checked in,
  contains placeholder `brand_a` / `brand_b` / `brand_c` entries)
* **Real config**: gitignored; copy the example, replace the
  placeholders with your real keywords, and the loader picks them up
  on next ingest / classify run.

The loader is fault-tolerant: a missing, unreadable, or malformed
config file degrades gracefully to an empty taxonomy (with a single
INFO log line), so classification still works — it just doesn't have
the operator-specific keyword enrichments for that run.

See [`tests/test_personal_taxonomy.py`](../tests/test_personal_taxonomy.py)
for the loader contract (missing-file fall-through, env override,
malformed-JSON warning, no-real-brand-leak guard).
