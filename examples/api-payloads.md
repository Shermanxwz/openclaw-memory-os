# Request / response shapes

Concrete examples of the JSON the API serves. Mirrors
`schemas/memory.schema.json` and `schemas/recall.schema.json`.

## `GET /api/health`

```json
{
  "backend": "sample",
  "total_memories": 15,
  "active": 11,
  "superseded": 2,
  "expired": 1,
  "needs_review": 1,
  "duplicates_estimate": 1,
  "deletion_candidate_count": 4,
  "tier_distribution": [
    {"tier": "core", "count": 4},
    {"tier": "long",  "count": 4},
    {"tier": "medium","count": 4},
    {"tier": "short", "count": 2},
    {"tier": "working","count": 1}
  ],
  "status_distribution": [
    {"status": "active",        "count": 11},
    {"status": "superseded",    "count": 2},
    {"status": "expired",       "count": 1},
    {"status": "needs_review",  "count": 1}
  ],
  "monthly_counts": [
    {"month": "2025-07", "count": 1},
    {"month": "2025-08", "count": 1},
    {"month": "2025-09", "count": 1},
    {"month": "2025-10", "count": 1},
    {"month": "2025-11", "count": 2},
    {"month": "2025-12", "count": 2},
    {"month": "2026-01", "count": 5},
    {"month": "2026-02", "count": 1},
    {"month": "2026-03", "count": 1}
  ],
  "generated_at": "2026-03-01T12:34:56.789012+00:00"
}
```

## `POST /api/recall-test`

Request:

```json
{
  "query": "recall test",
  "mode": "hybrid",
  "since_days": null,
  "include_superseded": false,
  "include_expired": false,
  "tier_filter": null,
  "limit": 5
}
```

Response (`schemas/recall.schema.json`):

```json
{
  "query": "recall test",
  "mode": "hybrid",
  "took_ms": 1.42,
  "backend": "sample",
  "total_considered": 15,
  "hits": [
    {
      "id": "mem-0002",
      "text": "Recall tests must be runnable in CI without external services. Use the bundled sample backend by default.",
      "summary": "Recall tests use the bundled sample backend in CI.",
      "tier": "long",
      "status": "active",
      "importance": 0.78,
      "score": 1.21,
      "components": {
        "base": 1.0,
        "recency": 0.95,
        "importance": 0.47,
        "keyword": 0.4,
        "composite": 1.21
      },
      "explanation": "status:active; recent(0.95); importance=0.78; matched[recall,test]"
    }
  ]
}
```

## `GET /api/duplicates`

```json
{
  "backend": "sample",
  "count": 1,
  "clusters": [
    {
      "representative_id": "mem-0008",
      "member_ids": ["mem-0007", "mem-0008"],
      "score": 1.0,
      "rationale": "avg_jaccard=1.00"
    }
  ]
}
```

## `GET /api/deletion-candidates`

```json
{
  "backend": "sample",
  "count": 4,
  "policy": "review-only; no physical deletion is performed by this OS.",
  "candidates": [
    {
      "id": "mem-0004",
      "text": "Temporary scratchpad for the 2026-03 deployment dry run. Will be pruned after the run.",
      "tier": "working",
      "status": "needs_review",
      "reason": "tier=working (session-scoped)",
      "recommended_action": "review"
    },
    {
      "id": "mem-0006",
      "text": "Expired TTL note from the 2025-09 promo campaign. No longer relevant; flagged for review.",
      "tier": "short",
      "status": "expired",
      "reason": "status=expired; expires_at in the past; low importance (0.05)",
      "recommended_action": "review"
    }
  ]
}
```
