/* OpenClaw Memory OS — Recall page renderer.
 *
 * Posts to /api/recall-test, renders candidate hits with diagnostics
 * and wires 👍/👎 feedback buttons. The hit element exposes
 * ``data-query-id`` / ``data-candidate-key`` so the feedback buttons
 * can be wired even after re-renders.
 */
(function (global) {
  "use strict";
  const OC = global.OCMemory;
  if (!OC) {
    console.error("[recall] missing OCMemory base");
    return;
  }
  if (global.OCMemory.__recallLoaded) return;
  global.OCMemory.__recallLoaded = true;

  const ZH = {
    status: { active: "当前生效", superseded: "已被取代", expired: "已过期", needs_review: "待审核" },
  };
  const ZH_LABEL = {
    no_results: "无结果",
  };

  function badge(status) {
    return '<span class="badge badge-' + OC.escapeHTML(status) + '">' + OC.escapeHTML(ZH.status[status] || status) + '</span>';
  }

  function buildHit(hit, queryId) {
    const ck = hit.candidate_key || ((hit.collection || "") + ":" + (hit.id || ""));
    const comps = hit.components || {};
    const scoreBits = ["rrf", "vector", "lexical", "importance"].filter(function (k) {
      return comps[k] !== undefined;
    }).map(function (k) { return k + "=" + comps[k]; }).join(" · ");
    return '' +
      '<div class="hit" data-query-id="' + OC.escapeHTML(queryId || "") + '" data-candidate-key="' + OC.escapeHTML(ck) + '">' +
        '<div class="head"><span>' + OC.escapeHTML(hit.id) + ' ' + badge(hit.status) + '</span><span class="score">' + OC.escapeHTML(String(hit.score)) + '</span></div>' +
        '<div class="text">' + OC.escapeHTML(hit.text) + '</div>' +
        '<div class="why">' + OC.escapeHTML(hit.explanation) + '</div>' +
        '<div class="why"><code>' + OC.escapeHTML(ck) + '</code> · ' + OC.escapeHTML(scoreBits) + '</div>' +
        '<div class="controls">' +
          '<button class="fb-btn" data-useful="true" data-id="' + OC.escapeHTML(hit.id) + '" data-candidate-key="' + OC.escapeHTML(ck) + '">👍 有用</button>' +
          '<button class="fb-btn" data-useful="false" data-id="' + OC.escapeHTML(hit.id) + '" data-candidate-key="' + OC.escapeHTML(ck) + '">👎 没用</button>' +
        '</div>' +
      '</div>';
  }

  function attachFeedbackHandlers(data) {
    document.querySelectorAll(".fb-btn").forEach(function (btn) {
      btn.addEventListener("click", async function () {
        const useful = btn.getAttribute("data-useful") === "true";
        const candidateKey = btn.getAttribute("data-candidate-key") || "";
        const memoryId = btn.getAttribute("data-id") || "";
        btn.disabled = true;
        try {
          const resp = await OC.postJSON("/api/feedback", {
            query_id: data.query_id,
            candidate_key: candidateKey,
            memory_id: memoryId,
            query: data.query,
            useful: useful,
          });
          btn.textContent = resp.ok ? "✅ 已记录" : "错误 " + resp.status;
        } catch (e) {
          btn.textContent = "网络错误";
        }
      });
    });
  }

  async function runRecall() {
    const qEl = OC.el("q");
    const query = qEl ? qEl.value.trim() : "";
    if (!query) return;
    const body = {
      query: query,
      mode: (OC.el("mode") || {}).value || "hybrid",
      limit: parseInt((OC.el("limit") || {}).value || "10", 10),
      include_superseded: !!(OC.el("incSup") || {}).checked,
      include_expired: !!(OC.el("incExp") || {}).checked,
    };
    const meta = OC.el("recallMeta");
    const hits = OC.el("recallHits");
    if (meta) meta.textContent = "运行中...";
    try {
      const resp = await OC.postJSON("/api/recall-test", body);
      if (!resp.ok) {
        if (meta) meta.textContent = "错误 " + resp.status;
        return;
      }
      const data = await resp.json();
      const diag = data.diagnostics || {};
      if (meta) {
        meta.innerHTML =
          "模式=" + OC.escapeHTML(data.mode) +
          " 后端=" + OC.escapeHTML(data.backend) +
          " 候选=" + data.total_considered +
          " 耗时=" + data.took_ms + "ms" +
          " · query_id=<code>" + OC.escapeHTML(data.query_id || "") + "</code>" +
          " · policy=<code>" + OC.escapeHTML(data.policy_version || "") + "</code>" +
          (diag.degraded_reason ? " · degraded=<code>" + OC.escapeHTML(diag.degraded_reason) + "</code>" : "");
      }
      if (!data.hits || !data.hits.length) {
        if (hits) hits.innerHTML = '<div class="empty">' + ZH_LABEL.no_results + '</div>';
        return;
      }
      if (hits) hits.innerHTML = data.hits.map(function (h) { return buildHit(h, data.query_id); }).join("");
      attachFeedbackHandlers(data);
    } catch (e) {
      if (meta) meta.textContent = "错误: " + e.message;
    }
  }

  function attachRecallForm() {
    const btn = OC.el("runBtn");
    const q = OC.el("q");
    if (btn) btn.addEventListener("click", runRecall);
    if (q) q.addEventListener("keydown", function (e) { if (e.key === "Enter") runRecall(); });
  }

  global.OCMemory.attachRecallForm = attachRecallForm;
})(typeof window !== "undefined" ? window : globalThis);
