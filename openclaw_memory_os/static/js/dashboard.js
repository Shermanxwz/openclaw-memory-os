/* OpenClaw Memory OS — Dashboard bootstrap (v0.3.0)
 *
 * This file is loaded LAST (after chart.umd.min.js, common.js, and
 * every per-section module). Its only responsibility is to dispatch
 * to the right renderer based on the current section, and to attach
 * the page-wide reclassify buttons on the overview page.
 *
 * The bulk of the rendering logic now lives in the per-section
 * modules under /static/js/. They all run inside the
 * ``window.OCMemory`` namespace that ``common.js`` provides.
 */
(function (global) {
  "use strict";
  const OC = global.OCMemory;
  if (!OC) {
    console.error("[dashboard] OCMemory base missing — common.js not loaded?");
    return;
  }
  if (global.__OCMemoryDashboardBooted) return;
  global.__OCMemoryDashboardBooted = true;

  function section() {
    return (global.__SECTION__ || "");
  }

  function boot() {
    const s = section();
    const healthPromise = OC.getJSON("/api/health")
      .then(function (h) {
        if (typeof OC.renderHealthPill === "function") OC.renderHealthPill(h);
        return h;
      })
      .catch(function (e) { console.error("[health-pill]", e); return null; });
    try {
      switch (s) {
        case "overview":
          healthPromise
            .then(function (h) { if (h) OC.renderOverview(h); })
            .catch(function (e) { console.error("[overview]", e); });
          // Reclassify buttons are page-wide (sit under the charts row).
          attachReclassifyButtons();
          break;
        case "tiers":
          OC.getJSON("/api/tiers").then(OC.renderTiers).catch(function (e) { console.error("[tiers]", e); });
          break;
        case "duplicates":
          OC.getJSON("/api/duplicates").then(OC.renderDuplicates).catch(function (e) { console.error("[duplicates]", e); });
          break;
        case "recall":
          OC.attachRecallForm();
          break;
        case "governance":
          OC.getJSON("/api/deletion-candidates").then(OC.renderGovernanceCandidates).catch(function (e) { console.error("[governance]", e); });
          break;
        case "strategy":
          if (typeof OC.renderStrategyPage === "function") OC.renderStrategyPage();
          break;
        case "evaluation":
          if (typeof OC.renderEvaluationPage === "function") OC.renderEvaluationPage();
          break;
        case "memories":
          if (typeof OC.renderMemoriesPage === "function") OC.renderMemoriesPage();
          break;
        case "security":
          if (typeof OC.renderSecurityPage === "function") OC.renderSecurityPage();
          break;
        case "health":
          if (typeof OC.renderHealthPage === "function") OC.renderHealthPage();
          break;
        default:
          // Unknown / root: nothing dynamic.
          break;
      }
    } catch (e) {
      console.error("[dashboard] boot error", e);
    }
  }

  function attachReclassifyButtons() {
    const live = OC.el("reclassifyBtn");
    const dry = OC.el("reclassifyDryBtn");
    const status = OC.el("reclassifyStatus");
    if (live) {
      live.addEventListener("click", function () {
        OC.setStatus(status, "运行中...", "info");
        OC.postJSON("/api/maintenance/reclassify", { dry_run: false })
          .then(function (r) { return r.json(); })
          .then(function (j) { OC.setStatus(status, "✅ 完成 exit=" + OC.escapeHTML(String(j.exit_code)), "ok"); })
          .catch(function (e) { OC.setStatus(status, "❌ 错误: " + e.message, "err"); });
      });
    }
    if (dry) {
      dry.addEventListener("click", function () {
        OC.setStatus(status, "Dry-run 计算中...", "info");
        OC.postJSON("/api/maintenance/reclassify", { dry_run: true })
          .then(function (r) { return r.json(); })
          .then(function (j) { OC.setStatus(status, "✅ 完成 exit=" + OC.escapeHTML(String(j.exit_code)), "ok"); })
          .catch(function (e) { OC.setStatus(status, "❌ 错误: " + e.message, "err"); });
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})(typeof window !== "undefined" ? window : globalThis);
