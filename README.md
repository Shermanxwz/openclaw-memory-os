# OpenClaw Memory OS

Optional governance, evaluation, audit, and observability layer for
Qdrant-backed agent memory. **Complements** the OpenClaw built-in memory
engine; does not replace it.

> **Independent community project. Not affiliated with, endorsed by, or
> supported by the OpenClaw project. Use the OpenClaw built-in memory for
> typical keyword, vector, and hybrid search.**

This is the **permanently archived** final source release.

## Project status

- Final release: `v0.3.0-archived`
- Intended profile: personal or single operator
- Recommended interactive concurrency: 1
- Status: archived and unmaintained
- No further feature, compatibility, maintenance, or security updates are planned

## What it provides

For users who already store agent memory in a Qdrant collection (or the
JSON development backend) and want **a control plane in front of it**:

- A FastAPI dashboard for browsing, tier, duplicate, and deletion-candidate views
- A recall-testing API with keyword (BM25), dense (Qdrant vector), and
  hybrid (weighted reciprocal-rank fusion + feature rerank) ranking
- An offline evaluation harness with time-split metrics
- Structured SQLite feedback collection for policy tuning
- A policy store plus a policy-evolution runner with guarded auto-promotion
  and rollback
- Maintenance and backup/restore tooling for the Qdrant corpus and the
  SQLite operational state
- A TOTP-authenticated, rate-limited, never-deleting safety model
- An optional OpenClaw adapter that loads Memory OS as a supplemental
  memory runtime when explicitly installed and enabled

It does **not**:

- Replace the OpenClaw built-in memory engine (use that for typical
  keyword, vector, and hybrid search)
- Ship as a hosted service
- Promise any specific throughput, latency, or concurrency on a
  particular host

## Quick start (single-user)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/runtime-py312.lock
.venv/bin/python -m pip install --no-deps .
.venv/bin/openclaw-memory-os serve --host 127.0.0.1 --port 7788
# open http://127.0.0.1:7788/login and authenticate there
```

See `docs/deployment.md` for systemd unit templates, nginx example
config, and acme.sh hooks. Adjust paths for your own installation.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          OpenClaw Memory OS                          │
│  Optional governance layer sitting in front of any Qdrant-backed    │
│  agent memory (or the JSON development backend).                     │
│                                                                      │
│   FastAPI app  ─── Jinja2 templates (dashboard.html, login.html)      │
│       │                                                              │
│       ├── /api/health, /api/timeline, /api/tiers                     │
│       ├── /api/duplicates, /api/deletion-candidates                  │
│       └── /api/recall-test   <-- RetrievalEngine.retrieve()          │
│                       │                                              │
│                       ▼                                              │
│              RetrievalEngine                                         │
│           ┌──────────┬──────────────┐                                │
│           ▼          ▼              ▼                                │
│        keyword     dense         hybrid                              │
│        (BM25)    (Qdrant vec)   (RRF + feature rerank)               │
│                       │                                              │
│                       ▼                                              │
│      SampleBackend / QdrantBackend  (memory corpus)                  │
│      SQLite + JSON state files       (operational state)             │
└──────────────────────────────────────────────────────────────────────┘
```

The memory corpus lives in Qdrant or the development JSON backend.
Memory OS keeps its own operational state in local SQLite and JSON files,
including sessions, audit events, structured recall feedback, evaluation
state, and policy-evolution state.

## What it is *not*

- **Not the memory store itself.** It sits in front of one.
- **Not an LLM agent.** It does not generate answers.
- **Not production-certified.** This is a personal/single-operator
  reference source distribution. Operate accordingly.
- **Not a competitor to the OpenClaw built-in memory engine.** For
  basic keyword, vector, or hybrid search, the OpenClaw built-in memory
  is the recommended path. Memory OS adds multi-collection governance,
  lifecycle metadata, evaluation, audit, feedback, policy evolution,
  and backup/restore *on top of* an existing memory store.

## What you get if you run it

For the personal/single-user recommended profile:

- Single-machine deployment
- One or more configured Qdrant collections, or the JSON development backend
- One operator using the configured bearer-token flow or password plus TOTP
- Recommended interactive concurrency: 1
- No external authentication, no scaling, no clustering

If you need higher throughput, you should test and tune on your own
host. The included self-tests run on a 2-vCPU, 4-GiB reference host
under single-thread interactive load; sustained multi-request
concurrency is not in the supported profile.

## Notes on this repository

The public root commit contains the frozen runtime source, sanitized
documentation and metadata, plus a test-harness portability correction.
It does not contain any production data, private memories, credentials,
host evidence, or operator-specific configuration.

`openclaw_memory_os/ranking.py` is compatibility code retained for
legacy callers. It is not the v0.3.0 production retrieval path; the
production path is `openclaw_memory_os.retrieval_engine.RetrievalEngine.retrieve()`.

## Contributing

This project is archived. No new contributions are accepted.

## License

MIT — see `LICENSE`.
