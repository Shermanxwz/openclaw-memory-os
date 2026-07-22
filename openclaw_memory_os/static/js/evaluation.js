/* OpenClaw Memory OS — Evaluation page renderer.
 *
 * Reads /api/dashboard/evaluation, which returns:
 *   {
 *     "feedback": { ratio_24h, ratio_7d, ratio_30d, total_events, ... },
 *     "metrics":  {...},
 *     "history":  [...],
 *     "note":     "offline evaluation is produced by scripts/run_evolution_cycle.py"
 *   }
 *
 * Renders the feedback summary cards (positive ratios per window),
 * metric tiles and any replay history rows.
 */
(function (global) {
  "use strict";
  const OC = global.OCMemory;
  if (!OC) {
    console.error("[evaluation] missing OCMemory base");
    return;
  }
  if (global.OCMemory.__evaluationLoaded) return;
  global.OCMemory.__evaluationLoaded = true;

  function fmtPct(value) {
    if (value == null) return "—";
    const num = Number(value);
    if (isNaN(num)) return "—";
    return (num * 100).toFixed(1) + "%";
  }

  function fmtMetric(value) {
    // Honest-null contract: ``None`` / ``null`` / ``undefined`` from
    // the API means "not scored", not "scored zero". Surface the
    // em-dash placeholder so the dashboard can distinguish the two.
    if (value == null) return "—";
    if (typeof value === "number") {
      if (Number.isNaN(value)) return "—";
      // Round tiny floats so the dashboard doesn't show 0.000000001.
      if (value === 0) return "0";
      if (Math.abs(value) < 0.001) return value.toExponential(2);
      return String(Number(value.toFixed(4)));
    }
    if (typeof value === "string") return value;
    return OC.escapeHTML(JSON.stringify(value));
  }

  function ratioColor(value) {
    if (value == null) return "#9da7b3";
    if (value >= 0.8) return "#7ee787";
    if (value >= 0.5) return "#ffa657";
    return "#ff7b72";
  }

  function renderFeedbackSummary(node, feedback) {
    if (!node) return;
    const f = feedback || {};
    const card = function (label, value) {
      return '<div class="card"><h2>' + label + '</h2><div class="stat" style="color:' + ratioColor(value) + ';">' + fmtPct(value) + '</div></div>';
    };
    const totalCard = '<div class="card"><h2>总反馈事件</h2><div class="stat">' + OC.escapeHTML(String(f.total_events || 0)) + '</div></div>';
    node.innerHTML = card("有用率 (24h)", f.ratio_24h) + card("有用率 (7d)", f.ratio_7d) + card("有用率 (30d)", f.ratio_30d) + totalCard;
  }

  function renderMetrics(node, metrics) {
    if (!node) return;
    if (!metrics || !Object.keys(metrics).length) {
      node.innerHTML = '<div class="empty">离线评估指标尚未生成。运行 scripts/run_evolution_cycle.py 后会出现。</div>';
      return;
    }
    const rows = Object.keys(metrics).map(function (key) {
      return '<tr><th>' + OC.escapeHTML(key) + '</th><td>' + fmtMetric(metrics[key]) + '</td></tr>';
    });
    node.innerHTML = '<table><tbody>' + rows.join("") + '</tbody></table>';
  }

  function renderHistory(node, history) {
    if (!node) return;
    // /api/dashboard/evaluation returns one row per persisted offline
    // report. The canonical fields are:
    //   report_id, generated_at, status, corpus_snapshot_id,
    //   policy.{version, ...}, decision.{...}
    // We surface report_id / generated_at / status / policy version /
    // decision verdict. Older fields (run_id / mode / useful_total /
    // feedback_total / finished_at) are only kept as defensive
    // fallbacks for back-compat.
    const rows = (history || []).map(function (h) {
      const policy = (h && h.policy) || {};
      const decision = (h && h.decision) || {};
      const policyVer = policy.version != null ? ("v" + policy.version) : (h.policy_version || "—");
      const status = h.status || (decision && decision.status) || "—";
      const id = h.report_id || h.run_id || h.query_id || h.id || "—";
      const ts = h.generated_at || h.finished_at || h.evaluated_at || "—";
      return '<tr>' +
        '<td><code>' + OC.escapeHTML(id) + '</code></td>' +
        '<td><span class="badge">' + OC.escapeHTML(status) + '</span></td>' +
        '<td>' + OC.escapeHTML(policyVer) + '</td>' +
        '<td>' + OC.escapeHTML(String(h.corpus_snapshot_id || "—")) + '</td>' +
        '<td>' + OC.escapeHTML(OC.formatTimestamp(ts)) + '</td>' +
      '</tr>';
    });
    if (!rows.length) {
      node.innerHTML = '<div class="empty">暂无历史 run。执行 scripts/run_evolution_cycle.py 后会陆续出现。</div>';
      return;
    }
    node.innerHTML = '<table><thead><tr>' +
      '<th>Report ID</th><th>Status</th><th>Policy</th><th>Corpus</th><th>时间</th>' +
      '</tr></thead><tbody>' + rows.join("") + '</tbody></table>';
  }

  function render() {
    const summaryNode = OC.el("evaluation-feedback");
    const metricsNode = OC.el("evaluation-metrics");
    const historyNode = OC.el("evaluation-history");
    const noteNode = OC.el("evaluation-note");

    OC.getJSON("/api/dashboard/evaluation")
      .then(function (e) {
        renderFeedbackSummary(summaryNode, e.feedback);
        renderMetrics(metricsNode, e.metrics);
        renderHistory(historyNode, e.history);
        if (noteNode) noteNode.textContent = e.note || "";
      })
      .catch(function (err) {
        if (summaryNode) summaryNode.innerHTML = '<div class="empty">加载评估失败: ' + OC.escapeHTML(err.message) + '</div>';
      });
  }

  global.OCMemory.renderEvaluationPage = render;
})(typeof window !== "undefined" ? window : globalThis);
