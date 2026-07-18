/* OpenClaw Memory OS — Overview / health page renderer.
 *
 * Renders the top-of-dashboard cards (maintenance health, totals,
 * doughnut tier/status charts, importance distribution) and the
 * global "记忆健康度" header pill. Reads from /api/health.
 */
(function (global) {
  "use strict";
  const OC = global.OCMemory;
  if (!OC) {
    console.error("[overview] missing OCMemory base — common.js must be loaded first");
    return;
  }
  if (global.OCMemory.__overviewLoaded) return;
  global.OCMemory.__overviewLoaded = true;

  const TIER_PLAN = {
    core: { n: 1, role: "当前事实", recall: "默认优先", color: "#7ee787" },
    long: { n: 1, role: "长期事实", recall: "默认优先", color: "#7ee787" },
    medium: { n: 2, role: "中期记录", recall: "常规召回", color: "#79c0ff" },
    short: { n: 3, role: "短期记录", recall: "常规召回", color: "#d2a8ff" },
    working: { n: 4, role: "临时状态", recall: "有 expires_at 才召回", color: "#ffa657" },
  };

  const ZH = {
    tier: { core: "核心事实", long: "长期偏好", medium: "中期记录", short: "短期记录", working: "临时状态" },
    status: { active: "当前生效", superseded: "已被取代", expired: "已过期", needs_review: "待审核" },
  };

  function chartColors() {
    return ["#58a6ff", "#7ee787", "#d2a8ff", "#ffa657", "#ff7b72", "#79c0ff", "#56d4dd"];
  }

  const DOUGHNUT_OPTS = {
    responsive: true,
    maintainAspectRatio: false,
    layout: { padding: 12 },
    plugins: {
      legend: { position: "bottom", fullWidth: true, labels: { color: "#c9d1d9", boxWidth: 12, padding: 14, font: { size: 12 } } },
      tooltip: { enabled: true },
    },
    cutout: "55%",
  };

  function translateTier(name) { return ZH.tier[name] || name; }
  function translateStatus(name) { return ZH.status[name] || name; }

  function renderMaintenanceCards(h) {
    const mh = h.maintenance_health || {};
    const ms = h.last_maintenance_summary || {};
    const healthEl = OC.el("maintenance-health");
    if (!healthEl) return;

    const lastRunDate = mh.last_run ? new Date(mh.last_run) : null;
    const lastOkDate = mh.last_ok ? new Date(mh.last_ok) : null;
    const now = Date.now();
    const HOUR = 3_600_000;
    let runStatus = "未知", runColor = "#9da7b3";
    const lockExplain = mh.lock_present
      ? '🔒 <span style="font-weight:400;color:#9da7b3;">互斥锁持有中</span>'
      : '🔓 <span style="font-weight:400;color:#9da7b3;">互斥锁空闲</span>';
    if (lastRunDate) {
      const ago = now - lastRunDate.getTime();
      const agoH = (ago / HOUR).toFixed(1);
      if (ago < 25 * HOUR)        { runStatus = "✅ 正常 · " + agoH + "h 前"; runColor = "#7ee787"; }
      else if (ago < 50 * HOUR)   { runStatus = "⚠️ 接近 · " + agoH + "h 前"; runColor = "#ffa657"; }
      else                        { runStatus = "🔴 过久 · " + agoH + "h 前"; runColor = "#ff7b72"; }
    }

    const snapSize = ms.snapshot_size_bytes || 0;
    const snapFmt = snapSize > 0
      ? (snapSize >= 1_048_576 ? (snapSize / 1_048_576).toFixed(1) + " MB" : (snapSize / 1024).toFixed(1) + " KB")
      : "—";
    const snapName = (ms.snapshot_name && String(ms.snapshot_name)) || "—";
    const mb = h.memory_brain || {};
    const mbi = mb.ingest || {};
    const mbc = mb.consolidate || {};
    const fmtTime = function (ts) { return ts ? new Date(ts).toLocaleString() : "—"; };
    const ingestValue = mbi.last_run
      ? fmtTime(mbi.last_run) + ' <span style="color:#7ee787;font-size:.9rem;">+' + Number(mbi.total_ingested || 0).toLocaleString() + ' new</span>'
      : "—";
    const consolidateStatus = mbc.status || (mbc.last_run ? "completed" : "");
    const consolidateReason = mbc.reason ? ": " + OC.escapeHTML(String(mbc.reason)) : "";
    const consolidateValue = mbc.last_run
      ? fmtTime(mbc.last_run) + ' <span style="color:' + (consolidateStatus === "skipped" ? "#ffa657" : "#7ee787") + ';font-size:.9rem;">' + OC.escapeHTML(consolidateStatus || "ok") + consolidateReason + '</span>'
      : "—";

    const cards = [
      ["维护状态", '<span style="color:' + runColor + ';font-weight:700;">' + runStatus + '</span><div style="font-size:.7rem;margin-top:.3rem;">' + lockExplain + '</div>'],
      ["最近运行", lastRunDate ? lastRunDate.toLocaleString() : "从未运行"],
      ["最后成功", lastOkDate ? lastOkDate.toLocaleString() : (mh.last_run ? "日志中未找到 ok 标记" : "—")],
      ["文件摄入 chunks", (ms.chunks_scanned || ms.ingested_total) ? '<strong style="font-size:1.3rem;">新增 ' + (ms.ingested_new || 0) + '</strong> / 文件扫描 ' + (ms.chunks_scanned || ms.ingested_total) : "—"],
      ["治理扫描 points", (ms.totals && ms.totals.points_scanned) ? '<strong style="font-size:1.3rem;">' + ms.totals.points_scanned.toLocaleString() + '</strong>' : "—"],
      ["过期待处理", (ms.expired_count || 0) + " 条"],
      ["取代链接", (ms.superseded_links || 0) + " 条"],
      ["最近快照", '<span title="' + OC.escapeHTML(snapName) + '">' + OC.escapeHTML(snapName.substring(0, 40)) + '…</span>'],
      ["记忆摄取", ingestValue, mbi.last_run ? '文件 ' + (mbi.files_processed || 0) + ' · 跳过 ' + (mbi.total_skipped || 0) : ""],
      ["记忆整合", consolidateValue, mbc.last_run ? '合并 ' + (mbc.topics_merged || 0) + ' 话题' : ""],
    ];
    healthEl.innerHTML = cards.map(function (pair) {
      const k = pair[0], v = pair[1], n = pair[2];
      return '<div class="card"><h2>' + k + '</h2><div class="stat">' + v + (n ? ' <small>' + n + '</small>' : '') + '</div></div>';
    }).join("");
  }

  function renderOverview(h) {
    renderMaintenanceCards(h);
    const totalCards = [
      ["总记忆数", (h.total_memories || 0).toLocaleString() + " 条"],
      ["当前生效", h.active || 0],
      ["已被取代", h.superseded || 0],
      ["已过期", h.expired || 0],
      ["待审核", h.needs_review || 0],
      ["重复簇数", h.duplicates_estimate || 0],
      ["自动清理候选", h.deletion_candidate_count || 0],
      ["受保护记忆", h.never_delete || 0],
    ];
    const stats = OC.el("stats");
    if (stats) {
      stats.innerHTML = totalCards.map(function (pair) {
        return '<div class="card"><h2>' + pair[0] + '</h2><div class="stat">' + pair[1] + '</div></div>';
      }).join("");
    }

    if (h.tier_distribution && h.status_distribution) {
      const tierCanvas = OC.el("tierChart");
      const statusCanvas = OC.el("statusChart");
      if (tierCanvas && global.Chart) {
        new Chart(tierCanvas, {
          type: "doughnut",
          data: {
            labels: h.tier_distribution.map(function (t) { return translateTier(t.tier); }),
            datasets: [{ data: h.tier_distribution.map(function (t) { return t.count; }), backgroundColor: chartColors() }],
          },
          options: DOUGHNUT_OPTS,
        });
      }
      if (statusCanvas && global.Chart) {
        new Chart(statusCanvas, {
          type: "doughnut",
          data: {
            labels: h.status_distribution.map(function (s) { return translateStatus(s.status); }),
            datasets: [{ data: h.status_distribution.map(function (s) { return s.count; }), backgroundColor: chartColors() }],
          },
          options: DOUGHNUT_OPTS,
        });
      }
    }

    const impData = h.importance_distribution || [];
    const impCanvas = OC.el("impChart");
    if (impData.length && impCanvas && global.Chart) {
      new Chart(impCanvas, {
        type: "bar",
        data: {
          labels: impData.map(function (b) { return b.label; }),
          datasets: [{ data: impData.map(function (b) { return b.count; }), backgroundColor: ["#7ee787", "#79c0ff", "#d2a8ff", "#ffa657", "#9da7b3"], barThickness: 18 }],
        },
        options: {
          indexAxis: "y", responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { beginAtZero: true, ticks: { color: "#9da7b3", font: { size: 11 } }, grid: { color: "#21262d" } },
            y: { ticks: { color: "#c9d1d9", font: { size: 12 } }, grid: { color: "#21262d", drawTicks: false } },
          },
          layout: { padding: { top: 4, right: 12, bottom: 4, left: 4 } },
        },
      });
    }

    const pill = document.querySelector(".health-pill");
    if (pill && h.total_memories) {
      const pct = Math.round(((h.active || 0) / h.total_memories) * 100);
      pill.textContent = "记忆健康度 " + pct + "%";
      pill.style.borderColor = pct >= 80 ? "#7ee787" : pct >= 50 ? "#ffa657" : "#ff7b72";
    }
  }

  global.OCMemory.renderOverview = renderOverview;
  global.OCMemory.TIER_PLAN = TIER_PLAN;
})(typeof window !== "undefined" ? window : globalThis);
