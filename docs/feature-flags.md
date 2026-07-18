# Feature flags

Feature flags let operators roll v0.3.0 capabilities **on** or **off** at
runtime through environment variables, without editing code or
restarting dependent processes. Every flag ships with a **safe default**
that preserves the previous behaviour, so a missing env var is never a
silent semantic change.

This document is the single source of truth for the v0.3.0 flag contract.
The authoritative parser lives in `openclaw_memory_os/config.py`
(`_env_flag`), and the runtime defaults are surfaced via `/api/health`,
`/api/strategy` and `/api/dashboard/evaluation`.

## Acceptable values

All flags accept the canonical boolean-ish spellings:

```text
1 | 0 | true | false | yes | no | on | off
```

Case-insensitive. Empty / unset values fall back to the **on-path**
default documented below. Matching is performed by
`openclaw_memory_os.config._env_flag`.

## Flags

The columns below list each flag, its **default** (what happens when the
env var is unset or empty), and the **safe-disable** behaviour (what
operators get when they explicitly set the flag to `off`).

| Flag env var             | Default | Disable behaviour (`off`)                                                                                                                                  |
| ------------------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RETRIEVAL_ENGINE_V2`    | `on`    | Use the legacy `ranking.build_recall_response` scorer. The `/api/recall-test` and CLI paths route around the unified `RetrievalEngine`. Diagnostics envelope (`diagnostics.dense_available`, `lexical_available`, …) degrades to legacy shape. |
| `STRUCTURED_FEEDBACK`    | `on`    | Persist feedback through the legacy audit-log path (`record_feedback`). The v0.3.0 SQLite tables (`recall_runs` / `recall_results` / `feedback_events`) stop receiving new rows. The offline evaluation pipeline degrades to "no fresh cases"; migrations can still be replayed via `migrate_legacy_feedback()`. |
| `EVOLUTION_ENABLED`      | `on`    | Every evolution endpoint (`/api/evolution/*`) becomes a safe no-op (`status="disabled"`). The weekly `autonomous_governance.sh` cycle is a no-op. Policy remains statically equal to the last good active file — the rollback/circuit-breaker state continues to be readable but is not mutated. |
| `SHADOW_ENABLED`         | `on`    | Skip the shadow-comparison stage of the evolution cycle. `run_evolution_cycle()` short-circuits to a deterministic candidate verdict. Useful for reproducible CI / evaluation runs that should not block on background traffic. |
| `PASSWORD_TOTP_AUTH`     | `on`    | Force the legacy bearer-token login flow even when `MEMORY_OS_PASSWORD` / `MEMORY_OS_TOTP_SECRET` are configured. The login form upgrades to a single-step bearer login. If no token is set the OS stays usable in local dev (`auth_enabled=False`). |
| `RECALL_FALLBACK_SUPERSEDED` | `on` | Disable the Active-first / Superseded-fallback contract. The recall engine returns only the active-pass hits; an explicit `include_superseded=True` on the request **always wins** and bypasses the flag. Default `on`; tunable threshold via `RECALL_FALLBACK_SUPERSEDED_MIN_RESULTS`. |

## How disable behaviour is implemented

* The flag values are loaded by `openclaw_memory_os.config.get_settings()`
  into a frozen `Settings` dataclass. The cache is process-local
  (`functools.lru_cache(maxsize=1)`) and can be cleared by
  `openclaw_memory_os.config.reset_settings_cache()` (tests and the CLI
  rely on this).
* Each flag is read **once per process** by default. Changes to the env
  vars are picked up on the next process start, or by calling
  `reset_settings_cache()` followed by a `Settings` rebuild.
* `RETRIEVAL_ENGINE_V2=off` switches `/api/recall-test` to a thin wrapper
  around the legacy scorer. The response shape is preserved (the legacy
  scorer already emits `query_id`, `policy_version`, `components`).
* `STRUCTURED_FEEDBACK=off` short-circuits `record_feedback_v030()` and
  `record_recall_run()` to a no-op; legacy `record_feedback()` still
  receives the writes.
* `EVOLUTION_ENABLED=off` is checked at the top of every evolution
  endpoint (`/api/evolution/pause`, `/api/evolution/resume`,
  `/api/evolution/rollback` and the `run_evolution_cycle()` background
  function). All of them return
  `{"status": "disabled", "reason": "evolution_enabled=off"}` and skip
  state mutation.
* `SHADOW_ENABLED=off` is consulted by `evolution.py:run_evolution_cycle`
  before entering the shadow stage. The cycle then performs a
  deterministic candidate comparison and logs `shadow=skipped`.
* `PASSWORD_TOTP_AUTH=off` is applied by `Settings.auth_enabled` together
  with the password/token env vars: when `False`, only the bearer-token
  path can enable auth, regardless of whether a password is configured.
* `RECALL_FALLBACK_SUPERSEDED=off` is checked inside the engine at the
  same point where the second pass is triggered; a request-level
  `include_superseded=True` always wins.

## Why every flag is a safe-default-on

Every flag in the table above starts in the **safe** position when the
env var is missing:

* `RETRIEVAL_ENGINE_V2=off` would force the legacy scorer — that is
  backwards compatible, not a regression.
* `STRUCTURED_FEEDBACK=off` would re-enable the audit-log path — also
  backwards compatible.
* `EVOLUTION_ENABLED=off` makes policy static and predictable.
* `SHADOW_ENABLED=off` skips a non-essential comparison step.
* `PASSWORD_TOTP_AUTH=off` keeps the bearer-only path available.
* `RECALL_FALLBACK_SUPERSEDED=off` keeps recall predictable.

The only **unsafe** flag operators sometimes add during incident
response is `EVOLUTION_ENABLED=off`, which we surface as the first row
of the strategy card on the dashboard.

## Safe-disable playbook

In an incident, you typically want to:

1. **Pause further policy churn.** Set `EVOLUTION_ENABLED=off` in the
   service `.env` and reload the service. The shadow / rollback stage
   will not run; previously promoted policies stay in force.
2. **Freeze evaluation while you investigate.** Set `SHADOW_ENABLED=off`
   to make `run_evolution_cycle()` deterministic and cheap.
3. **Roll back to a known-good scorer.** Set `RETRIEVAL_ENGINE_V2=off`
   if the new `RetrievalEngine` is suspected of returning bad hits. The
   legacy scorer is preserved in `openclaw_memory_os/ranking.py` and is
   tested by `tests/test_ranking.py` and `tests/test_recall_fallback.py`.
4. **Re-enable audit-log feedback.** Set `STRUCTURED_FEEDBACK=off` if
   you suspect a SQLite write is wedging the request path; the legacy
   `feedback.record_feedback()` continues to work.
5. **Disable password + TOTP.** Set `PASSWORD_TOTP_AUTH=off` if the
   TOTP path is causing lock-outs; the bearer token still works as a
   secondary path.

Each of those disable values is safe to flip **independently**, and each
is reversible by re-exporting the env var or removing the line from
`.env`. A process restart (or `reset_settings_cache()` in the CLI) is
required for the change to take effect.

## Migration notes

The flags are a v0.3.0 addition. Operators on v0.2.x do not have to
unset anything — the v0.2.x process simply ignores the new env vars, and
all features on the v0.2.x path remain available because the flag
defaults preserve the v0.2.x behaviour at the **runtime** layer.

If you operate a `.env` that already pins the old ranking tunables
(`MEMORY_OS_MAX_RECALL`, `MEMORY_OS_RECENCY_HALF_LIFE`,
`MEMORY_OS_SUPERSEDED_PENALTY`, `MEMORY_OS_EXPIRED_PENALTY`,
`MEMORY_OS_IMPORTANCE_BOOST`), leave them in place — they continue to
work, and the unified engine reads them on every request.

## Where the flags are surfaced

* `/api/health` includes `settings.retrieval_engine_v2`,
  `settings.structured_feedback`, `settings.evolution_enabled`,
  `settings.shadow_enabled`, and `settings.password_totp_auth` so
  operators can confirm the live values.
* `/api/strategy` (alias `/api/dashboard/strategy`) embeds the current
  `state.shadow_enabled` so the dashboard can render "shadow on / off".
* `/api/recall-test` and the CLI `recall` command emit a
  `diagnostics.retrieval_engine` field whose value is `"v2"` or
  `"legacy"` depending on the resolved flag.

## Testing the flag contract

The contract is pinned by `tests/test_feature_flags.py`, which:

* forces each flag to `on` and `off` via `monkeypatch.setenv` plus
  `reset_settings_cache()`;
* builds a `TestClient` and asserts the expected behaviour for each
  flag pair;
* uses `_clean_env` autouse fixture from `tests/conftest.py` to make
  sure no env var leaks across tests.

Do not edit the parser without updating the table above and the tests
together — the table is documentation, but the tests are the only
executable source of truth for the contract.
