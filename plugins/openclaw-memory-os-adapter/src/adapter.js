const DEFAULT_BASE_URL = "http://127.0.0.1:7788";
const DEFAULT_TIMEOUT_MS = 2500;

export function normalizeConfig(raw = {}) {
  const cfg = raw && typeof raw === "object" ? raw : {};
  const baseUrl = String(cfg.baseUrl || process.env.MEMORY_OS_URL || DEFAULT_BASE_URL).replace(/\/+$/, "");
  const token = cfg.token || process.env.MEMORY_OS_TOKEN || "";
  const timeoutMs = Number.isFinite(Number(cfg.timeoutMs)) ? Math.max(100, Number(cfg.timeoutMs)) : DEFAULT_TIMEOUT_MS;
  const fallback = cfg.fallback !== false;
  const preferMemoryOsWhenEmpty = cfg.preferMemoryOsWhenEmpty === true;
  return { baseUrl, token, timeoutMs, fallback, preferMemoryOsWhenEmpty };
}

export function mapCorpus(corpus) {
  if (corpus === "sessions") return ["sessions"];
  return ["memory"];
}

export function buildRecallPayload(query, opts = {}) {
  const limit = Math.max(1, Math.min(100, Number(opts.maxResults || 10)));
  const payload = {
    query,
    mode: "hybrid",
    include_superseded: false,
    include_expired: false,
    limit
  };
  return payload;
}

export function mapMemoryOsHit(hit, index = 0) {
  const id = String(hit.id || `memory-os-${index}`);
  const lineStart = Number.isFinite(Number(hit.line_start)) ? Number(hit.line_start) : 1;
  const lineEnd = Number.isFinite(Number(hit.line_end)) ? Number(hit.line_end) : lineStart;
  const source = hit.source || hit.path || "memory-os";
  const topic = hit.topic ? ` [${hit.topic}]` : "";
  const status = hit.status ? ` status=${hit.status}` : "";
  const tier = hit.tier ? ` tier=${hit.tier}` : "";
  const explanation = hit.explanation ? `\nReason: ${hit.explanation}` : "";
  const snippet = `${hit.summary || hit.text || ""}${topic}${tier}${status}${explanation}`.trim();
  return {
    path: String(source),
    startLine: lineStart,
    endLine: lineEnd,
    score: Number(hit.score || 0),
    vectorScore: Number(hit.components?.base || hit.score || 0),
    textScore: Number(hit.components?.keyword || 0),
    snippet,
    source: "memory",
    citation: `${source}#${lineStart}`,
    _memoryOsId: id
  };
}

export async function callMemoryOsSearch(query, opts = {}, config = {}, fetchImpl = globalThis.fetch) {
  const cfg = normalizeConfig(config);
  if (typeof fetchImpl !== "function") {
    throw new Error("fetch is not available in this Node.js runtime");
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(new Error("Memory OS request timed out")), cfg.timeoutMs);
  if (opts.signal) {
    if (opts.signal.aborted) controller.abort(opts.signal.reason);
    else opts.signal.addEventListener("abort", () => controller.abort(opts.signal.reason), { once: true });
  }
  try {
    const headers = { "content-type": "application/json" };
    if (cfg.token) headers.authorization = `Bearer ${cfg.token}`;
    const res = await fetchImpl(`${cfg.baseUrl}/api/recall-test`, {
      method: "POST",
      headers,
      body: JSON.stringify(buildRecallPayload(query, opts)),
      signal: controller.signal
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Memory OS recall failed: HTTP ${res.status}${text ? ` ${text.slice(0, 160)}` : ""}`);
    }
    const data = await res.json();
    const hits = Array.isArray(data?.hits) ? data.hits : [];
    return hits.map(mapMemoryOsHit);
  } finally {
    clearTimeout(timeout);
  }
}

export class MemoryOsSearchManager {
  constructor({ config = {}, fallbackManager = null, logger = null } = {}) {
    this.config = normalizeConfig(config);
    this.fallbackManager = fallbackManager;
    this.logger = logger;
    this.lastError = null;
  }

  async search(query, opts = {}) {
    const sources = mapCorpus(opts.corpus);
    if (opts.sources && !opts.sources.some((s) => sources.includes(s))) {
      return this.fallbackManager ? this.fallbackManager.search(query, opts) : [];
    }
    try {
      const results = await callMemoryOsSearch(query, opts, this.config);
      this.lastError = null;
      if (results.length > 0 || !this.config.fallback || this.config.preferMemoryOsWhenEmpty) {
        return results;
      }
    } catch (error) {
      this.lastError = error instanceof Error ? error.message : String(error);
      this.logger?.warn?.(`Memory OS recall unavailable, falling back: ${this.lastError}`);
      if (!this.config.fallback) throw error;
    }
    if (this.fallbackManager) {
      return this.fallbackManager.search(query, opts);
    }
    return [];
  }

  async readFile(params) {
    if (this.fallbackManager?.readFile) return this.fallbackManager.readFile(params);
    return { text: "", path: params.relPath, truncated: false, from: params.from, lines: params.lines };
  }

  status() {
    const fallbackStatus = this.fallbackManager?.status?.();
    return {
      backend: "builtin",
      provider: "openclaw-memory-os",
      custom: {
        adapter: "openclaw-memory-os",
        baseUrl: this.config.baseUrl,
        fallback: this.config.fallback,
        fallbackStatus,
        lastError: this.lastError
      }
    };
  }

  async sync(params) {
    return this.fallbackManager?.sync?.(params);
  }

  getCachedEmbeddingAvailability() {
    return { ok: true, checked: true, cached: true };
  }

  async probeEmbeddingAvailability() {
    try {
      await callMemoryOsSearch("health", { maxResults: 1 }, this.config);
      return { ok: true, checked: true };
    } catch (error) {
      return { ok: false, checked: true, error: error instanceof Error ? error.message : String(error) };
    }
  }

  async probeVectorStoreAvailability() {
    return true;
  }

  async probeVectorAvailability() {
    return true;
  }

  async close() {
    await this.fallbackManager?.close?.();
  }
}
