/* OpenClaw Memory OS — Strategy page renderer.
 *
 * Fetches /api/dashboard/strategy (alias of /api/strategy) and renders
 * the policy version, checksum, paused/active state, promotion count,
 * shadow comparisons and the safe action buttons (pause / resume /
 * reject candidate / rollback). These buttons talk to the
 * /api/evolution/* endpoints via postJSON so CSRF is included.
 */
(function (global) {
  "use strict";
  const OC = global.OCMemory;
  if (!OC) {
    console.error("[strategy] missing OCMemory base");
    return;
  }
  if (global.OCMemory.__strategyLoaded) return;
  global.OCMemory.__strategyLoaded = true;

  function setText(node, text) {
    if (node) node.textContent = text;
  }

  function setStatus(node, text, kind) {
    if (!node) return;
    node.textContent = text;
    node.dataset.status = kind || "info";
  }

  function renderStrategyCard(s) {
    if (!s) return;
    // Overview card (when present) — same data attributes the
    // overview page uses.
    setText(document.querySelector('[data-strategy="policy-version"]'), s.policy_version || "v1");
    const noteNode = document.querySelector('[data-strategy="note"]');
    if (noteNode) {
      noteNode.textContent = "策略由 autonomous_governance.sh 自动优化；guarded_auto 模式。pause/resume 用于紧急人工介入。";
    }
  }

  async function callEvolution(path, statusNode, successLabel) {
    setStatus(statusNode, "请求 " + path + " ...", "info");
    try {
      const r = await OC.postJSON(path, {});
      if (!r.ok) {
        setStatus(statusNode, "❌ HTTP " + r.status, "err");
        return;
      }
      const body = await r.json();
      setStatus(statusNode, "✅ " + successLabel + " (state=" + JSON.stringify(body.state || {}).slice(0, 80) + ")", "ok");
      return body;
    } catch (e) {
      setStatus(statusNode, "❌ " + e.message, "err");
    }
  }

  function render() {
    // Wave 2 (2026-07-21): also fetch /api/health so the strategy page
    // can surface the live ``governance_schedule`` (read from the
    // systemd timer unit). Without this, the schedule card would have
    // to hardcode ``Tue 04:01`` which the dashboard contract forbids.
    Promise.all([
      OC.getJSON("/api/dashboard/strategy"),
      OC.getJSON("/api/health").catch(function () { return null; }),
    ])
      .then(function (results) {
        const s = results[0] || {};
        const h = results[1] || {};
        const policy = (s && s.policy) || {};
        const state = (s && s.state) || {};
        const checksum = (s && s.checksum) || "";

        // Schedule calendar (live systemd timer). Empty when the timer
        // is not reachable — the dashboard never hardcodes ``Tue 04:01``
        // as a string literal in source.
        const gsched = (h && h.governance_schedule) || {};
        const scheduleNode = document.querySelector('[data-strategy="schedule-calendar"]');
        if (scheduleNode) {
          const cal = gsched.calendar || "";
          scheduleNode.textContent = cal || "—";
        }

        // Server-rendered header is still the source of truth for the
        // displayed policy_version; we update the data-attribute nodes.
        setText(document.querySelector('[data-strategy-card="status"] [data-strategy="policy-version"]'),
                policy.version ? "v" + policy.version : "v1");

        // Dynamic state block (paused / shadow comparisons)
        const stateNode = OC.el("strategy-state");
        if (stateNode) {
          stateNode.innerHTML = '' +
            '<table><tbody>' +
              '<tr><th>当前状态</th><td>' + (state.paused ? '<span class="badge badge-superseded">paused</span>' : '<span class="badge badge-active">active</span>') + '</td></tr>' +
              '<tr><th>Policy version</th><td><code>' + OC.escapeHTML("v" + (policy.version || "?")) + '</code></td></tr>' +
              '<tr><th>Checksum</th><td><code>' + OC.escapeHTML(checksum || "—") + '</code></td></tr>' +
              '<tr><th>30d 自动晋升</th><td>' + OC.escapeHTML(String(state.promotion_count_30d != null ? state.promotion_count_30d : 0)) + '</td></tr>' +
              '<tr><th title="自上次策略晋升以来的累计回滚数（无晋升时仅递增）">累计回滚</th><td>' + OC.escapeHTML(String(state.consecutive_rollbacks || 0)) + '</td></tr>' +
              '<tr><th>Shadow 对照</th><td>' + OC.escapeHTML(String(typeof state.shadow_comparisons === "number" ? state.shadow_comparisons : ((state.shadow_comparisons || []).length))) + '</td></tr>' +
              '<tr><th>上次回滚</th><td>' + OC.escapeHTML(state.last_manual_rollback_at || "—") + '</td></tr>' +
              '<tr><th>上次拒绝候选</th><td>' + OC.escapeHTML(state.candidate_rejected_at || "—") + '</td></tr>' +
            '</tbody></table>';
        }

        // Safe action buttons (guarded by CSRF on the server).
        const actions = OC.el("strategy-actions");
        if (actions && !actions.dataset.bound) {
          actions.dataset.bound = "1";
          const statusNode = OC.el("strategy-status");
          actions.innerHTML = '' +
            '<button id="strategy-pause-btn" style="padding:.4rem .9rem;border:1px solid #d29922;background:#d2992222;color:#ffdf5d;border-radius:6px;cursor:pointer;">⏸ 暂停自动优化</button> ' +
            '<button id="strategy-resume-btn" style="padding:.4rem .9rem;border:1px solid #2ea043;background:#2ea04322;color:#7ee787;border-radius:6px;cursor:pointer;">▶ 恢复</button> ' +
            '<button id="strategy-reject-btn" style="padding:.4rem .9rem;border:1px solid #1f6feb;background:#1f6feb22;color:#79c0ff;border-radius:6px;cursor:pointer;">✕ 拒绝当前候选</button> ' +
            '<button id="strategy-rollback-btn" style="padding:.4rem .9rem;border:1px solid #f85149;background:#f8514922;color:#ff7b72;border-radius:6px;cursor:pointer;" title="将策略回滚到 baseline v1。">⟲ 回滚到 baseline</button>' +
            '<div id="strategy-status" class="muted" style="margin-top:.5rem;font-size:.85rem;"></div>';
          OC.el("strategy-pause-btn").addEventListener("click", function () {
            callEvolution("/api/evolution/pause", statusNode, "已暂停");
          });
          OC.el("strategy-resume-btn").addEventListener("click", function () {
            callEvolution("/api/evolution/resume", statusNode, "已恢复");
          });
          OC.el("strategy-reject-btn").addEventListener("click", function () {
            callEvolution("/api/evolution/candidate/reject", statusNode, "已拒绝候选");
          });
          OC.el("strategy-rollback-btn").addEventListener("click", function () {
            callEvolution("/api/evolution/rollback", statusNode, "已回滚到 baseline");
          });
        }

        // Keep the SSR card rows in sync.
        renderStrategyCard(s);
      })
      .catch(function (e) {
        setStatus(OC.el("strategy-status"), "❌ 加载策略失败: " + e.message, "err");
      });
  }

  global.OCMemory.renderStrategyPage = render;
})(typeof window !== "undefined" ? window : globalThis);
