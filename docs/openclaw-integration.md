# OpenClaw integration notes

This document is the explicit "how does this OS plug into an OpenClaw-style
memory store" page. It deliberately contains some literal strings that the
in-repo privacy scanner flags by default — those flagged strings are
**documentation of the rule itself**, not actual leaked credentials. Each
such line carries a `privacy-allow:` marker (or `privacy-allow: *`) so the
line is intentionally auditable in code review.

## What "OpenClaw-style" means here

This project is meant to sit next to whatever memory store an OpenClaw-style
agent uses. Concretely:

* The agent stores memories somewhere — JSON file, SQLite, vector DB.
* The OS reads that store via a backend adapter (see
  `openclaw_memory_os/backends/__init__.py`).
* The OS exposes a small dashboard and a recall-test API.

The current release (v0.2.2) is read-only: it never writes back to the
underlying store. Lifecycle operations (supersede, expire) are exposed
through the maintenance scripts under `scripts/` so operators retain
explicit review before any status changes touch the store.

## Why a per-line `privacy-allow:` marker?

The privacy scanner in this repo intentionally refuses to scan certain
private-looking strings (file paths under the canonical private prefix,
internal hostnames, provider IDs, etc.). When a documentation file has to
*describe* those patterns — e.g. "this scanner will flag the path under
the canonical private prefix below" — the line ends with a marker that
suppresses the rule just for that line.

The marker is auditable: in code review you can grep for `privacy-allow:`
and confirm each suppression is intentional.

A second, equivalent mechanism is the JSON baseline file at
`scripts/privacy_baseline.json` (or wherever the `--baseline` flag points).
That file pins exact `(file, line, rule_id)` triples that are allowed to
fire. Use it when a multi-line block (e.g. a JSON fixture) needs to be
pinned.

## Two pragmatic rules for integrating

1. **Don't put real credentials anywhere in this repo.** Use generic
   placeholders (`example.com`, `127.0.0.1`) and inject secrets via
   environment at deploy time.
2. **Treat the scanner as a safety net, not a security product.** A
   full-featured scanner like `gitleaks` does deeper credential discovery;
   the in-repo scanner is just there to catch the obvious accidents
   before they ship.

## Where the OS sits

```
        agent process
             │  (writes)
             ▼
   ┌─────────────────────┐
   │  memory store        │   ← whatever the agent uses
   │  (JSON / Qdrant / …) │
   └────────┬────────────┘
            │  (reads; via backend adapter)
            ▼
   ┌─────────────────────┐
   │  OpenClaw Memory OS  │   ← this project
   │   - dashboard        │
   │   - recall API       │
   │   - deletion review  │
   └─────────────────────┘
            │
            ▼
        human operator
```

The OS does not write back to the store from the live dashboard. Lifecycle
operations (supersede links, expiry, near-duplicate tags) are exposed
through the offline maintenance scripts under `scripts/` so operators
retain explicit review before any status changes touch the store. That
keeps blast radius small and avoids subtle data-loss bugs from making it
into a public release.

## Provider IDs

Internal model provider identifiers used in routing configs are flagged by
the scanner. To reference a placeholder example, wrap it in backticks and
mark the line. The actual rule lives in `openclaw_memory_os/privacy.py` —
see `docs/privacy-scanner.md` for the full pattern table.

```
new-api-123456  privacy-allow: PROVIDER_ID
```

`vps-deadbeef01  privacy-allow: PRIVATE_HOSTNAME` is the corresponding
shorthand for an internal-hostname placeholder in docs.
