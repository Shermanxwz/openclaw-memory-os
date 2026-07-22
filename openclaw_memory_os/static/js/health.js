/* OpenClaw Memory OS — Health page renderer.
 *
 * Reads /api/health and updates the per-component status tiles using
 * the [data-health] selector set in dashboard.html. If the endpoint is
 * unreachable every tile shows an unreachable indicator.
 */
(function (global) {
  "use strict";
  const OC = global.OCMemory;
  if (!OC) {
    console.error("[health] missing OCMemory base");
    return;
  }
  if (global.OCMemory.__healthLoaded) return;
  global.OCMemory.__healthLoaded = true;

  function setTile(name, label, klass) {
    const node = document.querySelector('[data-health="' + name + '"]');
    if (!node) return;
    node.textContent = label;
    node.className = "stat " + klass;
  }

  function render() {
    OC.getJSON("/api/health")
      .then(function (h) {
        // Backend + total memories are the only ground-truth signals
        // the /api/health endpoint exposes for the Qdrant / Memory OS
        // tiles. The other components (Ollama / Lexical / Policy /
        // Feedback) are best-effort health signals derived from the
        // Memory Brain and maintenance health sub-payloads. We map
        // those honestly to the tiles instead of hardcoding ✅.
        const mh = (h && h.maintenance_health) || {};

        const backendName = h.backend || "memory";
        const points = h.total_memories != null ? h.total_memories : "?";
        setTile("qdrant", "✅ " + points + " points", "status-ok");

        // Ollama has no per-component probe in this payload. Do not
        // reuse the Memory OS backend name here: showing "qdrant" under
        // an Ollama card looks like a component mismatch. Say exactly
        // what we know.
        setTile("ollama", "— 未暴露指标", "");

        // Lexical index is built lazily and lives in app.state. When
        // the API caller sees it (it isn't on /api/health yet), the
        // tile should reflect that. Until then we report "未初始化".
        setTile("lexical", "— 未初始化", "status-warn");

        // Policy DB is shared across the app; we don't have a live
        // liveness probe here, so we say "在用" instead of a fake OK.
        setTile("policy", "— 在用", "");

        // SQLite feedback store backs the dashboard ratios; report
        // total_events when the payload provides one, otherwise
        // "—".
        const totalEvents = (h && h.total_feedback_events);
        if (totalEvents != null) {
          setTile("feedback", "✅ " + OC.escapeHTML(String(totalEvents)) + " events", "status-ok");
        } else {
          setTile("feedback", "— 在用", "");
        }

        // Maintenance health maps onto the Memory OS tile so an
        // operator can spot a stuck lock without digging through logs.
        if (mh && mh.lock_present) {
          setTile("memoryos", "⚠ lock present", "status-warn");
        } else if (mh && mh.enabled === false) {
          setTile("memoryos", "⚠ disabled", "status-warn");
        } else {
          setTile("memoryos", "✅ " + OC.escapeHTML(backendName), "status-ok");
        }
      })
      .catch(function () {
        ["qdrant", "ollama", "lexical", "policy", "feedback", "memoryos"].forEach(function (n) {
          setTile(n, "❌ Unreachable", "status-err");
        });
      });
  }

  global.OCMemory.renderHealthPage = render;
})(typeof window !== "undefined" ? window : globalThis);
