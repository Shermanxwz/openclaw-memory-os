# Examples

Runnable invocations against the bundled sample backend. None of these use
real credentials; tokens are placeholders.

## 1. Run the dashboard locally

```bash
cd openclaw-memory-os
pip install -e .
export MEMORY_OS_TOKEN="dev-only-token-please-rotate"
openclaw-memory-os serve --host 127.0.0.1 --port 7788
# open http://127.0.0.1:7788/login and submit the development token
```

## 2. Run a recall test from the CLI

```bash
openclaw-memory-os recall --query "worker model rule" --mode hybrid --limit 5
```

Sample output (truncated):

```json
{
  "query": "worker model rule",
  "mode": "hybrid",
  "took_ms": 1.42,
  "backend": "sample",
  "total_considered": 15,
  "hits": [
    {
      "id": "mem-0001",
      "text": "Project COFFEE uses a worker model for routine summarization ...",
      "score": 1.62,
      "components": {"base": 1.0, "recency": 0.93, "importance": 0.55, "keyword": 0.6, "composite": 1.62},
      "explanation": "status:active; recent(0.93); importance=0.92; matched[model,rule,worker]; no-keyword-match=false"
    }
  ]
}
```

## 3. Hit the recall-test API with `curl`

```bash
TOKEN="dev-only-token-please-rotate"
curl -sS -X POST http://127.0.0.1:7788/api/recall-test \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"deletion policy","mode":"hybrid","limit":3}' | jq .
```

## 4. Generate the deletion candidate list

```bash
curl -sS -H "Authorization: Bearer $TOKEN" http://127.0.0.1:7788/api/deletion-candidates | jq .
```

Returns a JSON object with `policy: "review-only; no physical deletion is performed by this OS."`
and a `candidates: [...]` array, every entry's `recommended_action` set to
the literal string `review`.

## 5. Run the privacy scanner against the in-repo docs

```bash
./scripts/privacy_scan.sh
```

Exit code is non-zero if any rule fires without a marker. The
`docs/openclaw-integration.md` page has a few literal strings
the scanner would otherwise flag — each line that mentions them ends with
the `privacy-allow: <RULE_ID>` marker.
