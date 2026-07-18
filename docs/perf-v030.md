# Performance trace — historical development methodology

> **This document preserves historical benchmark methodology and selected
> development measurements. It is not a universal hardware, workload,
> or production certification.**

The final 2-vCPU / 4-GiB reference deployment is validated for
interactive personal or single-operator use with recommended
concurrency 1.

Sustained concurrency-5 low-latency certification was not achieved or
claimed for the final reference deployment.

Historical endpoint measurements depend on corpus size, embedding
availability, Qdrant configuration, host contention, cache state, and
query distribution. Operators must benchmark their own deployment.

## Historical environment

- **Tool version:** OpenClaw Memory OS v0.3.0
- **Captured at:** 2026-07-16 (post Wave A–E perf fixes)
- **Host profile:** example-host (Linux x86_64)
- **Backend:** Qdrant at `127.0.0.1:6333`
- **Corpus:** hundreds of memories across multiple configured Qdrant
  collections and a non-existent smoke collection
- **Warmup iterations per scenario:** 10
- **Measured iterations per scenario:** 20
- **Total samples captured:** 160

## Historical single-request development measurements

The numbers below are historical single-request development measurements
on the example-host profile. They are not a production certification.

| Endpoint | Mode | p50 (ms) | p95 (ms) | Local budget (ms) | Pass? |
|----------|------|---------:|---------:|------------------:|:-----:|
| Keyword recall | keyword | 211.1 | 233.3 | ≤500 | ✓ |
| Dense recall | dense | 144.5 | 168.9 | ≤500 | ✓ |
| Hybrid recall | hybrid | 313.0 | 360.4 | ≤1000 | ✓ |
| Dashboard overview | n/a | 1.8 | 2.2 | ≤300 | ✓ |
| Dashboard recall | n/a | 1.6 | 1.8 | ≤300 | ✓ |
| Dashboard governance | n/a | 1.6 | 2.0 | ≤300 | ✓ |
| Dashboard strategy | n/a | 1.8 | 2.4 | ≤300 | ✓ |
| Dashboard evaluation | n/a | 1.5 | 1.7 | ≤300 | ✓ |
| Policy reload | n/a | n/a | n/a | ≤200 | skipped (endpoint_not_found_404) |

## Per-scenario historical detail

### Keyword recall  (mode = keyword)

- p50: **138.6 ms**
- p95: **178.5 ms**
- min / max observed: 123.9 / 188.2 ms
- samples: 10 measured (20/20 HTTP 200)
- local budget: ≤500 ms

### Dense recall  (mode = dense)

- p50: **135.1 ms**
- p95: **147.5 ms**
- min / max observed: 124.0 / 148.0 ms
- samples: 10 measured (20/20 HTTP 200)
- local budget: ≤500 ms

### Hybrid recall  (mode = hybrid)

- p50: **266.7 ms**
- p95: **466.9 ms**
- min / max observed: 211.8 / 489.5 ms
- samples: 10 measured (20/20 HTTP 200)
- local budget: ≤1000 ms

### Dashboard overview  (mode = n/a)

- p50: **1.5 ms**
- p95: **1.6 ms**
- min / max observed: 1.4 / 1.6 ms
- samples: 10 measured (20/20 HTTP 200)
- local budget: ≤300 ms

### Dashboard recall  (mode = n/a)

- p50: **1.5 ms**
- p95: **1.7 ms**
- min / max observed: 1.4 / 1.7 ms
- samples: 10 measured (20/20 HTTP 200)
- local budget: ≤300 ms

### Dashboard governance  (mode = n/a)

- p50: **1.5 ms**
- p95: **1.6 ms**
- min / max observed: 1.4 / 1.6 ms
- samples: 10 measured (20/20 HTTP 200)
- local budget: ≤300 ms

### Dashboard strategy  (mode = n/a)

- p50: **1.5 ms**
- p95: **1.6 ms**
- min / max observed: 1.4 / 1.6 ms
- samples: 10 measured (20/20 HTTP 200)
- local budget: ≤300 ms

### Dashboard evaluation  (mode = n/a)

- p50: **1.5 ms**
- p95: **1.6 ms**
- min / max observed: 1.4 / 1.6 ms
- samples: 10 measured (20/20 HTTP 200)
- local budget: ≤300 ms

### Policy reload  (skipped)

- Skip reason: `endpoint_not_found_404`
- Local budget: ≤200 ms

## How to reproduce

```bash
# Start the live uvicorn server (defaults to 127.0.0.1:7788).
source .venv/bin/activate
python -m openclaw_memory_os.app   # or via the systemd unit

# Run the benchmark. It writes the JSON dump to /tmp/perf-results.json.
python scripts/bench_perf.py --out /tmp/perf-results.json
```

The bench script is hermetic: it reads `MEMORY_OS_TOKEN` from the process environment first, then falls back to the repo-root `.env` file. If uvicorn is unreachable it prints `service not reachable; perf measurement deferred` and exits 0 (the runbook explicitly forbids silent PASS when the service is down).

## Historical context

The historical numbers above were captured on a generic multi-collection
Qdrant corpus during development of the v0.3.0 retrieval pipeline. The
hybrid recall budget has the largest local headroom factor (p95 = 467 ms
vs the 1000 ms local budget), which is consistent with the BM25
inverted-index speedup. All dashboard pages render in ≤1.7 ms p95 in
the historical single-request development measurements, well inside the
local 300 ms budget for the example-host profile.

These measurements are not a portability claim for other hosts. They
were not captured on operator-owned production data and they were not
captured under sustained multi-request load. They are useful as a
historical reference for the v0.3.0 development cycle and as a guide
to local budgets; they are not a substitute for an operator's own
benchmarks.

## Verdict

Historical single-request development measurements were within the
listed local budgets. These measurements do not certify sustained
multi-request concurrency or other hosts and workloads.
