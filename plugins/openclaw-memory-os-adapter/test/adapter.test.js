import test from "node:test";
import assert from "node:assert/strict";
import { buildRecallPayload, callMemoryOsSearch, mapMemoryOsHit, MemoryOsSearchManager, normalizeConfig } from "../src/adapter.js";

test("normalizeConfig uses safe defaults", () => {
  const cfg = normalizeConfig({});
  assert.equal(cfg.baseUrl, "http://127.0.0.1:7788");
  assert.equal(cfg.timeoutMs, 2500);
  assert.equal(cfg.fallback, true);
});

test("buildRecallPayload defaults to current active recall", () => {
  assert.deepEqual(buildRecallPayload("worker", { maxResults: 3 }), {
    query: "worker",
    mode: "hybrid",
    include_superseded: false,
    include_expired: false,
    limit: 3
  });
});

test("mapMemoryOsHit converts recall hit to OpenClaw memory result shape", () => {
  const hit = mapMemoryOsHit({ id: "m1", text: "hello", score: 0.9, source: "MEMORY.md", line_start: 2, line_end: 4, tier: "core", status: "active" });
  assert.equal(hit.path, "MEMORY.md");
  assert.equal(hit.startLine, 2);
  assert.equal(hit.endLine, 4);
  assert.equal(hit.source, "memory");
  assert.match(hit.snippet, /hello/);
});

test("callMemoryOsSearch posts recall request", async () => {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url, init });
    return new Response(JSON.stringify({ hits: [{ id: "1", text: "ok", score: 0.8 }] }), { status: 200, headers: { "content-type": "application/json" } });
  };
  const hits = await callMemoryOsSearch("q", { maxResults: 1 }, { baseUrl: "http://x", token: "t" }, fetchImpl);
  assert.equal(hits.length, 1);
  assert.equal(calls[0].url, "http://x/api/recall-test");
  assert.equal(calls[0].init.headers.authorization, "Bearer t");
});

test("manager falls back when Memory OS fails", async () => {
  const fallbackManager = { search: async () => [{ path: "fallback", startLine: 1, endLine: 1, score: 0.1, snippet: "fb", source: "memory" }], status: () => ({ backend: "builtin", provider: "local" }) };
  const manager = new MemoryOsSearchManager({ config: { baseUrl: "http://bad", timeoutMs: 100, fallback: true }, fallbackManager });
  const oldFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response("no", { status: 503 });
  try {
    const hits = await manager.search("q", { maxResults: 1 });
    assert.equal(hits[0].path, "fallback");
    assert.match(manager.lastError, /HTTP 503/);
  } finally {
    globalThis.fetch = oldFetch;
  }
});
