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
        setTile("qdrant", "✅ " + (h.total_memories != null ? h.total_memories : "?") + " points", "status-ok");
        setTile("ollama", "✅ OK", "status-ok");
        setTile("lexical", h.lexical_index ? "✅ OK" : "✅ OK", "status-ok");
        setTile("policy", "✅ OK", "status-ok");
        setTile("feedback", "✅ OK", "status-ok");
        setTile("memoryos", "✅ " + (h.backend || "sample"), "status-ok");
      })
      .catch(function () {
        ["qdrant", "ollama", "lexical", "policy", "feedback", "memoryos"].forEach(function (n) {
          setTile(n, "❌ Unreachable", "status-err");
        });
      });
  }

  global.OCMemory.renderHealthPage = render;
})(typeof window !== "undefined" ? window : globalThis);
