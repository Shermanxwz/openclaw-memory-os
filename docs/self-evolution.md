# Self-evolution

The policy-evolution loop in OpenClaw Memory OS tunes the retrieval
scorer **automatically** against real user feedback, with explicit
gates and rollback hooks so it can never degrade the live ranking.
This document explains how it works, how it stays safe, and how to
operate it.

The authoritative code lives in `openclaw_memory_os/evolution.py`,
with state persisted to
`~/.local/state/openclaw-memory-os/evolution-state.json`. The
governance runner that calls the cycle is
`scripts/autonomous_governance.sh`; the entry-point script that
invokes one cycle on demand is `scripts/run_evolution_cycle.py`. The
offline evaluation pipeline is documented in
`docs/retrieval-diagnostics.md`.

## Why a self-evolution loop

The retrieval scorer has roughly a dozen parameters: `rrf_dense_weight`,
`rrf_lexical_weight`, `final_*_weight`, `importance_weight`,
`recency_weight`, `feedback_weight`, `dense_k`, `lexical_k`, `rrf_k`,
`fallback_min_results`, `exact_match_boost`. Hand-tuning them is slow,
and the optimum drifts as the corpus grows. The v0.3.0 series adds a
**deterministic**, **gated** cycle so the policy improves on its own
when there is enough signal, and stays put when there isn't.

The loop is **not** a machine-learning model and it never calls an LLM
during the online path. It perturbs weights, runs the scorer through
the offline evaluator (`openclaw_memory_os.evaluation.evaluate`), and
promotes the candidate only if it improves nDCG@10 by ≥1% while leaving
Recall@5, MRR@10, useful@1, negative@5 and no-result rate unchanged.

## Tunable parameters and where they live

The live tunables are stored in `openclaw_memory_os/policy_store.py`
as a `Policy` pydantic model. The store reads and writes a
`policy.json` file under the project root (or the configured policy
dir). The hot path (`/api/recall-test` and the CLI `recall` command)
reads `store.get()` on **every** request, so a successful promotion
takes effect immediately — no restart needed.

The default `Policy` is shipped in `policy_store.py`; operators can
pin a custom file by setting the directory in the operator's policy
config. `scripts/run_evolution_cycle.py` boots its own `PolicyStore`
so the timer path doesn't depend on the running web service.

## Candidate generation

`evolution.generate_candidates()` builds up to N candidate policies
(default 20) by:

1. **Coordinate-style perturbation.** Each tunable is perturbed by
   `+max_delta` and `-max_delta` independently. After perturbation
   the (rrf / vector / lexical / importance / recency / feedback)
   weights are renormalised to sum to 1.0; if their unnormalised sum
   was below 0.75, they are first scaled so the post-normalisation
   weights are not absurdly large.
2. **Bounded random combinations.** If coordinate-only is not enough,
   fill the remaining slots with random uniform combinations in
   the same range. Bounded so no parameter exits its safety range.
3. **De-duplication.** A candidate whose score vector is identical to
   an earlier candidate is dropped (`score` delta ≤ 1e-9).

`max_delta` defaults to 0.05; during the cold-start window it is
clamped to 0.03 so weights cannot drift by more than 3% per cycle
while the population of judged queries is thin. The list of
perturbable parameters is in `evolution.py:_Param`; adding a new
parameter requires updating `_fix_weights()` so the post-perturbation
normalisation stays consistent.

## Cold-start gate

The evolution cycle refuses to operate until enough judged queries
exist. The thresholds are:

| Window            | Judged queries required | Behaviour                                                                  |
| ----------------- | ----------------------- | -------------------------------------------------------------------------- |
| Cold start        | < 30                    | Cycle status `"skipped"`, reason `"cold_start: N/30"`. No candidates.    |
| Mid-confidence    | 30 .. 99                | Cycle runs, but `max_delta` is clamped to 0.03. All hard metrics must not degrade. |
| Full range        | ≥ 100                   | Full ±0.05 perturbation range, full hard-metric contract.                  |

The gate is checked once per cycle in `run_evolution_cycle()` (right
before candidate generation). The threshold counts live in
`evolution.py`:

```python
_COLD_START_MIN_QUERIES = 30
_QUERIES_FOR_FULL_RANGE = 100
```

Both are module-level constants on purpose: changing them changes the
gating contract and requires updating both `self-evolution.md` and
`tests/test_feature_flags.py` / `tests/test_active_first_hybrid.py`
together.

## Shadow comparison and promotion rules

When the candidate is generated, `run_evolution_cycle()` evaluates
each candidate against the same hold-out split used by the offline
evaluator. The candidate with the highest `nDCG@10` (and at least
+0.005 over the baseline) is selected. From there:

1. **Shadow is set.** The candidate is published as `PolicyStore.shadow`
   so the live API can include shadow comparisons in
   `/api/strategy` without affecting live scoring.
2. **Promotion gates are checked.** `_can_promote()` re-validates
   cold-start, 7-day cooldown (since the last successful promotion),
   `promotion_count_30d <= 2`, and a circuit breaker on consecutive
   rollbacks.
3. **Promotion happens via `PolicyStore.set(best_cand)`.**
   `consecutive_rollbacks` resets to 0; `last_promotion_at` is
   stamped; `promotion_count_30d` increments.
4. **`SHADOW_ENABLED=off`.** If the operator disabled the shadow
   stage, the cycle short-circuits to a deterministic verdict without
   touching the live policy. This is the safe default for CI and
   evaluation runs.

The thresholds are:

```python
_PROMOTION_WINDOWS_REQUIRED = 2   # consecutive passing eval windows
_PROMOTION_COOLDOWN_DAYS = 7
_MAX_PROMOTIONS_PER_30D = 2
_MAX_CONSECUTIVE_ROLLBACKS = 2     # circuit breaker
```

## Rollback

Rollback is **immediate** (file corruption, error rate > 5%) or
**statistical** (latency 2x baseline, no-result +15pp, useful@1 down
> 8pp, MRR down > 5%). `_check_rollback()` is the entry point:

* **File corrupt / checksum mismatch.** `_force_rollback()` reverts to
  the previous active policy.
* **Error rate > 5%.** `_check_rollback()` measures `degraded_rate`
  from the offline evaluator and forces a rollback.
* **Latency 2x.** Implicit — `_check_rollback()` records the
  `p95_latency` of the last 50 cases and forces a rollback if it
  doubles.

`/api/evolution/rollback` is the manual escape hatch. It accepts a
reason string (which is appended to the audit log) and reverts the
active policy to the last good state.

**Circuit breaker.** When `_MAX_CONSECUTIVE_ROLLBACKS` is reached the
cycle refuses to promote again until an operator intervenes via
`/api/evolution/resume` (which resets the breaker after manual
review). This is the only way the cycle can become a long-term
no-op without operator action.

## Feature-flag interaction

Evolution respects two feature flags (see `docs/feature-flags.md`):

* `EVOLUTION_ENABLED=off` — every evolution endpoint becomes a safe
  no-op (`status="disabled"`). The weekly systemd-timer cycle is a no-op.
  `consecutive_rollbacks` and other counters stay readable but are
  not mutated. Policy is effectively static.
* `SHADOW_ENABLED=off` — the shadow-comparison stage is skipped; the
  cycle goes straight from "best candidate" to "deterministic verdict
  for documentation". This is the safe default for reproducible
  CI runs.

Both are checked at the top of `run_evolution_cycle()` and at every
`/api/evolution/*` endpoint. The defaults are `on`.

## Weekly runner

The production schedule is installed as
`openclaw-memory-os-governance.timer` and runs Tuesday at 04:01
Asia/Shanghai. The timer invokes `openclaw-memory-os-governance.service` under
the dedicated `openclaw-memory-os` account with the same `.env`, XDG state,
and cache paths as the web service.

Within `scripts/autonomous_governance.sh`, feedback replay runs first, followed
by maintenance and then evolution. Evolution is not run against a partial
maintenance result. The final status JSON is written only after all attempted
stages finish, and the process exits zero only for a complete green run.
Maintenance lock contention is recorded as `skipped`; feedback replay failure
with otherwise successful stages is `degraded`; maintenance or evolution
failure is `failed`.

Manual execution uses the same unit and environment:

```bash
sudo systemctl start --wait openclaw-memory-os-governance.service
journalctl -u openclaw-memory-os-governance.service -n 200 --no-pager
```

The inner evolution runner retains `/tmp/openclaw-memory-os.evolution.lock`, so
two evolution cycles cannot run concurrently even if the outer governance lock
is bypassed.

## State files

The evolution cycle owns one persistent file and one lock:

| Path                                                            | Purpose                                                                |
| --------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `~/.local/state/openclaw-memory-os/evolution-state.json`        | Promotion counters, cooldown timestamps, shadow-comparison log.        |
| `/tmp/openclaw-memory-os.evolution.lock`                        | `fcntl.lockf` so only one `run_evolution_cycle()` runs at a time.     |
| `/tmp/openclaw-memory-os.governance.lock`                       | Outer governance lock held by `autonomous_governance.sh`.              |

The state file is written with mode `0600` to keep the JSON readable
only by the operator. Operators who back it up should encrypt the
backup.

## Manual operation

* **Force a cycle now.** Run `scripts/run_evolution_cycle.py` (it
  acquires the same lock as the timer unit). The script prints a JSON
  envelope with `status` ∈ {`"ok"`, `"shadow"`, `"promoted"`,
  `"rolled_back"`, `"skipped"`, `"disabled"`, `"error"`}.
* **Pause evolution.** `POST /api/evolution/pause` flips
  `evolution-state.json` so the next cycle returns
  `status="disabled"` without touching policy. Reversible via
  `/api/evolution/resume`.
* **Force rollback.** `POST /api/evolution/rollback {reason:
  "manual"}` reverts to the last good state and increments
  `consecutive_rollbacks`.
* **Inspect history.** `/api/strategy` returns
  `state.last_promotion_at`, `state.promotion_count_30d`,
  `state.consecutive_rollbacks`, and the count of recent shadow
  comparisons.
* **Snapshot current policy.** `cp <policy_dir>/policy.json
  <backup>/policy-$(date -u +%FT%TZ).json`. Snapshots are
  human-readable JSON, suitable for diff and grep.

## Safety summary

* `EVOLUTION_ENABLED=off` makes the cycle a no-op. Use this as the
  first-line incident-response lever.
* `SHADOW_ENABLED=off` makes the cycle deterministic. Use this for
  reproducible CI evaluations and post-incident replay.
* Rollback is automatic on file corruption, error-rate spikes, and
  latency regressions.
* The circuit breaker on consecutive rollbacks prevents thrash.
* The cycle never calls an LLM (see `QWEN_NOT_IN_ONLINE_PATH` in
  `openclaw_memory_os/contracts.py`).

## Related files

| File | Purpose |
| ---- | ------- |
| `openclaw_memory_os/evolution.py` | Candidate generation, shadow, promote / rollback. |
| `openclaw_memory_os/evaluation.py` | Offline evaluation runner. |
| `openclaw_memory_os/policy_store.py` | `Policy` model + on-disk store. |
| `scripts/run_evolution_cycle.py` | Cron entry-point wrapper. |
| `scripts/autonomous_governance.sh` | Weekly governance runner. |
| `tests/test_candidate_search.py` | Candidate generation contract. |
| `tests/test_candidate_pool.py` | Offline-eval candidate-pool contract. |
| `tests/test_evaluation_metrics.py` | Metrics + honest-null contract. |
