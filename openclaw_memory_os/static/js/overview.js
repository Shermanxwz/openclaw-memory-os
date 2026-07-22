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

  function fmtUtcToShanghai(value) {
    if (!value) return "";
    const d = new Date(value);
    if (isNaN(d.getTime())) return String(value);
    return d.toLocaleString();
  }

  function statusColor(token) {
    switch ((token || "").toLowerCase()) {
      case "ok":
      case "success":
        return "#7ee787";
      case "noop":
        return "#9da7b3";
      case "skipped":
        return "#ffa657";
      case "failed":
      case "failure":
        return "#ff7b72";
      case "degraded":
        return "#d29922";
      default:
        return "#9da7b3";
    }
  }

  function statusLabel(token) {
    switch ((token || "").toLowerCase()) {
      case "ok":
      case "success":
        return "ok";
      case "noop":
        return "noop";
      case "skipped":
        return "skipped";
      case "failed":
      case "failure":
        return "failed";
      case "degraded":
        return "degraded";
      default:
        return token || "unknown";
    }
  }

  function renderMaintenanceCards(h) {
    const mh = h.maintenance_health || {};
    const ms = h.last_maintenance_summary || {};
    const healthEl = OC.el("maintenance-health");
    if (!healthEl) return;

    // last_run: ISO UTC. last_ok: ISO UTC from "ok" line. last_success_at: NEW field — only
    // updated when a run actually succeeded. Treat them separately so a failed run
    // doesn't roll back the "last success" display.
    const lastRunDate = mh.last_run ? new Date(mh.last_run) : null;
    const lastOkDate = mh.last_ok ? new Date(mh.last_ok) : null;
    const lastSuccessAt = ms.last_success_at ? new Date(ms.last_success_at) : null;
    const summaryStatus = ms.status || null;
    const summaryExit = (ms.exit_code !== undefined && ms.exit_code !== null) ? ms.exit_code : null;
    const summaryFailedStep = ms.failed_step || null;
    const now = Date.now();
    const HOUR = 3_600_000;
    let runStatus = "未知", runColor = "#9da7b3";
    const lockExplain = mh.lock_present
      ? '🔒 <span style="font-weight:400;color:#9da7b3;">互斥锁持有中</span>'
      : '🔓 <span style="font-weight:400;color:#9da7b3;">互斥锁空闲</span>';
    // Wave 6 (2026-07-22): use lastSuccessAt for the relative-time
    // display so it matches the "最后成功" absolute time on the next
    // card. Fall back to lastRunDate when no summary exists yet.
    const statusRefDate = lastSuccessAt || lastRunDate;
    if (statusRefDate) {
      const ago = now - statusRefDate.getTime();
      const agoH = (ago / HOUR).toFixed(1);
      // Override status text if the summary says we failed
      if (summaryStatus === "failed") {
        runStatus = "🔴 失败 · " + agoH + "h 前";
        runColor = "#ff7b72";
      } else if (ago < 25 * HOUR)  { runStatus = "✅ 正常 · " + agoH + "h 前"; runColor = "#7ee787"; }
      else if (ago < 50 * HOUR)     { runStatus = "⚠️ 接近 · " + agoH + "h 前"; runColor = "#ffa657"; }
      else                          { runStatus = "🔴 过久 · " + agoH + "h 前"; runColor = "#ff7b72"; }
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
    // Wave 3 (2026-07-21): compact Asia/Shanghai formatter used by the
    // ingest / consolidation time blocks. Missing / malformed timestamps
    // surface as an explicit error span so the operator never sees a
    // silent em-dash on a critical path.
    const fmtCstCompact = function (ts) {
      if (!ts) return '<span style="color:#ff7b72;">状态数据缺少时间</span>';
      const d = new Date(ts);
      if (isNaN(d.getTime())) return '<span style="color:#ff7b72;">时间格式错误</span>';
      const pad = function (n) { return n < 10 ? "0" + n : "" + n; };
      const cst = new Date(d.getTime() + 8 * 3600 * 1000);
      return cst.getUTCFullYear() + "/" + (cst.getUTCMonth() + 1) + "/" +
             cst.getUTCDate() + " " + pad(cst.getUTCHours()) + ":" +
             pad(cst.getUTCMinutes()) + ":" + pad(cst.getUTCSeconds());
    };
    const fmtDuration = function (v) {
      if (v === null || v === undefined) {
        return '<span style="color:#ff7b72;">缺少耗时</span>';
      }
      const n = Number(v);
      if (!isFinite(n) || n < 0) return '<span style="color:#ff7b72;">缺少耗时</span>';
      if (n >= 1) return Math.round(n) + "s";
      return n.toFixed(2).replace(/0+$/, "").replace(/\.$/, "") + "s";
    };
    // Wave 2 (2026-07-21): ingest card now distinguishes ok / noop /
    // failed / unknown. ``mbi.status`` is sourced from
    // ``steps.ingest.status`` in the canonical maintenance-summary JSON;
    // a totally missing card (no run_id, no last_run, no status) still
    // renders as ``—`` so a clean dashboard doesn't fabricate ticks.
    const ingestHasAnySignal = !!(mbi.last_run || mbi.run_id || mbi.status || mbi.started_at);
    let ingestValue;
    let ingestSubtitle;
    if (!ingestHasAnySignal) {
      ingestValue = '<span style="color:#9da7b3;">—</span>';
      ingestSubtitle = '<span style="color:#9da7b3;">尚无子步骤记录</span>';
    } else {
      const ingestStatus = (mbi.status || "").toLowerCase();
      const newCount = Number(mbi.ingested_new || mbi.total_ingested || 0);
      const statusTok = statusLabel(ingestStatus || (ingestStatus === "" ? "ok" : ingestStatus));
      const statusColour = statusColor(ingestStatus || "ok");
      const chip = '<span style="color:' + statusColour + ';font-size:.9rem;font-weight:600;">' + OC.escapeHTML(statusTok) + '</span>';
      const startStr = fmtCstCompact(mbi.started_at);
      const finishStr = fmtCstCompact(mbi.finished_at);
      const durStr = fmtDuration(mbi.duration_seconds);
      ingestValue =
        '<span style="color:#c9d1d9;font-weight:700;">+' + newCount.toLocaleString() + ' new</span> · ' + chip +
        '<div class="mb-substep-time">' +
          '<div>开始：' + startStr + '</div>' +
          '<div>完成：' + finishStr + '</div>' +
          '<div>耗时：' + durStr + '</div>' +
        '</div>';
      if (mbi.files_processed !== undefined || mbi.skipped !== undefined || mbi.total_skipped !== undefined) {
        const files = Number(mbi.files_processed || 0);
        const skip = Number(mbi.skipped || mbi.total_skipped || 0);
        ingestSubtitle = '文件 ' + files + ' · 跳过 ' + skip;
      } else {
        ingestSubtitle = "";
      }
    }
    // Wave 2 (2026-07-21): consolidation card surfaces skipped / ok /
    // failed explicitly so a no-op run doesn't render as ``ok`` and a
    // failure shows up red. ``mbc.run_id`` always reflects the parent
    // run (canonical maintenance-summary.json ``run_id``); if it is
    // missing we still render the card so the user sees ``unknown``.
    // Wave 3 (2026-07-21): skipped runs are NOT equivalent to
    // "never ran" — the bracket is always rendered so the operator sees
    // when the trigger was evaluated and how long the decision took.
    const consolidateHasAnySignal = !!(mbc.last_run || mbc.run_id || mbc.status || mbc.started_at);
    let consolidateValue;
    let consolidateSubtitle;
    if (!consolidateHasAnySignal) {
      consolidateValue = '<span style="color:#9da7b3;">—</span>';
      consolidateSubtitle = '<span style="color:#9da7b3;">尚无子步骤记录</span>';
    } else {
      const consolidateStatus = (mbc.status || "").toLowerCase();
      const statusTok = statusLabel(consolidateStatus || "ok");
      const statusColour = statusColor(consolidateStatus || "ok");
      const merged = Number(mbc.merged_topics || mbc.topics_merged || 0);
      const startStr = fmtCstCompact(mbc.started_at);
      const finishStr = fmtCstCompact(mbc.finished_at);
      const durStr = fmtDuration(mbc.duration_seconds);
      const reasonSuffix = mbc.reason && consolidateStatus === "skipped"
        ? '<div style="font-size:.75rem;color:#9da7b3;margin-top:.2rem;">' + OC.escapeHTML(String(mbc.reason)) + '</div>'
        : "";
      consolidateValue =
        '<span style="color:#c9d1d9;font-weight:700;">合并 ' + merged + ' 话题</span> · ' +
        '<span style="color:' + statusColour + ';font-size:.9rem;font-weight:600;">' + OC.escapeHTML(statusTok) + '</span>' +
        '<div class="mb-substep-time">' +
          '<div>开始：' + startStr + '</div>' +
          '<div>完成：' + finishStr + '</div>' +
          '<div>耗时：' + durStr + '</div>' +
        '</div>' +
        reasonSuffix;
      const threshold = Number(mbc.threshold || 0);
      consolidateSubtitle = 'new_since_24h ' + Number(mbc.new_since_24h || 0) +
        (threshold > 0 ? ' / threshold ' + threshold : '');
    }
    // Wave 2 (2026-07-21): schedule row on the maintenance card shows
    // the live ``maintenance_schedule`` from the systemd timer. When
    // the timer is absent (``active_state: unknown``), render a quiet
    // ``—`` so operators can spot the gap.
    const msched = h.maintenance_schedule || {};
    let maintenanceScheduleLine;
    if (msched.active_state === "unknown" || !msched.calendar) {
      maintenanceScheduleLine = '<span style="color:#9da7b3;">—</span>';
    } else {
      const lastTriggerStr = msched.last_trigger
        ? fmtUtcToShanghai(msched.last_trigger)
        : "—";
      maintenanceScheduleLine =
        '<span style="color:#c9d1d9;">' + OC.escapeHTML(msched.calendar) + '</span>' +
        ' <span style="color:#9da7b3;font-size:.75rem;">· 上次 ' + OC.escapeHTML(lastTriggerStr) + '</span>';
    }

    // Wave 2 (2026-07-21): the canonical run_id comes from
    // ``last_maintenance_summary.run_id``. We display it as small text
    // on the maintenance card so operators can correlate a UI tick
    // with a log line.
    // Wave 3 (2026-07-21): the canonical run_id renders as a tiny
    // inline chip so the operator can correlate a UI tick with a log
    // line without the identifier eating the card width on mobile.
    const runId = ms.run_id || mbi.run_id || mbc.run_id || "";
    const runIdChip = runId
      ? '<span class="mb-run-id" title="' + OC.escapeHTML(runId) + '">' + OC.escapeHTML(runId) + '</span>'
      : "";
    const ingestRunIdChip = (mbi.run_id && mbi.run_id !== runId)
      ? '<span class="mb-run-id" title="' + OC.escapeHTML(mbi.run_id) + '">子 ' + OC.escapeHTML(mbi.run_id) + '</span>'
      : "";
    const consolidateRunIdChip = (mbc.run_id && mbc.run_id !== runId)
      ? '<span class="mb-run-id" title="' + OC.escapeHTML(mbc.run_id) + '">子 ' + OC.escapeHTML(mbc.run_id) + '</span>'
      : "";

    const cards = [
      ["维护状态", '<span style="color:' + runColor + ';font-weight:700;">' + runStatus + '</span><div style="font-size:.7rem;margin-top:.3rem;">' + lockExplain + '</div>'],
      ["维护计划", maintenanceScheduleLine],
      ["最近运行", lastRunDate ? lastRunDate.toLocaleString() : "从未运行"],
      ["最后成功", lastSuccessAt
        ? lastSuccessAt.toLocaleString()
        : (lastOkDate ? lastOkDate.toLocaleString() + ' <span style="font-size:.7rem;color:#9da7b3;">(日志中 ok 标记)</span>' : "—")],
      ["退出码", summaryExit !== null
        ? '<span style="font-weight:700;color:' + (summaryExit === 0 ? "#7ee787" : "#ff7b72") + ';">' + summaryExit + '</span>'
        : "—"],
      ["失败步骤", summaryFailedStep
        ? '<span style="font-weight:700;color:#ff7b72;">' + OC.escapeHTML(summaryFailedStep) + '</span>'
        : "—"],
      ["文件摄入 chunks", (ms.chunks_scanned || ms.ingested_total) ? '<strong style="font-size:1.3rem;">新增 ' + (ms.ingested_new || 0) + '</strong> / 文件扫描 ' + (ms.chunks_scanned || ms.ingested_total) : "—"],
      ["治理扫描 points", (ms.totals && ms.totals.points_scanned) ? '<strong style="font-size:1.3rem;">' + ms.totals.points_scanned.toLocaleString() + '</strong>' : "—"],
      ["过期待处理", (ms.expired_count || 0) + " 条"],
      ["取代链接", (ms.superseded_links || 0) + " 条"],
      ["最近快照", '<span class="mb-snapshot-name" title="' + OC.escapeHTML(snapName) + '">' + OC.escapeHTML(snapName) + '</span>'],
      ["记忆摄取", ingestValue + (ingestRunIdChip || runIdChip), ingestSubtitle, "span2"],
      ["记忆整合", consolidateValue + (consolidateRunIdChip || runIdChip), consolidateSubtitle, "span2"],
    ];
    let cardsHtml = cards.map(function (pair) {
      const k = pair[0], v = pair[1], n = pair[2], span = pair[3];
      const cls = 'card' + (span === 'span2' ? ' mb-card-wide' : '');
      return '<div class="' + cls + '"><h2>' + k + '</h2><div class="stat">' + v + (n ? ' <small>' + n + '</small>' : '') + '</div></div>';
    }).join("");
    if (runId) {
      cardsHtml += '<div class="card" style="grid-column:1 / -1;"><h2>RUN_ID</h2><div class="stat">' + runIdChip + '</div></div>';
    }
    healthEl.innerHTML = cardsHtml;
  }

  function renderGovernanceCard(h) {
    const target = OC.el("autonomous-governance-cards");
    if (!target) return;
    // The dashboard SSR card already exists; we only fill in extra
    // fields not rendered server-side (RUN_ID, scheduled_at, finished_at,
    // duration, exit_code, mode).
    const ag = h.autonomous_governance || {};
    const setIfPresent = function (selector, value, formatter) {
      const node = document.querySelector(selector);
      if (!node) return;
      if (value === null || value === undefined || value === "") {
        node.textContent = "—";
        return;
      }
      node.textContent = formatter ? formatter(value) : String(value);
    };
    // Wave 6 (2026-07-22): annotate the "计划触发" field with trigger
    // type so operators can distinguish timer-driven runs from manual
    // systemctl starts. When scheduled_at is within 60s of started_at
    // the run was likely timer-triggered; otherwise it was manual.
    const schedAtNode = document.querySelector('[data-governance-card="status"] [data-governance="scheduled-at"]');
    if (schedAtNode) {
      if (!ag.scheduled_at) {
        schedAtNode.textContent = "—";
      } else {
        schedAtNode.textContent = fmtUtcToShanghai(ag.scheduled_at);
        // Determine trigger type
        let triggerType = "";
        if (ag.started_at && ag.scheduled_at) {
          try {
            const schedMs = new Date(ag.scheduled_at).getTime();
            const startMs = new Date(ag.started_at).getTime();
            const diffSec = Math.abs(startMs - schedMs) / 1000;
            if (diffSec < 60) {
              triggerType = " 🕐定时";
            } else {
              triggerType = " 🔧手动";
            }
          } catch (e) { /* ignore parse errors */ }
        }
        if (triggerType) {
          const badge = document.createElement("span");
          badge.style.cssText = "font-size:.7rem;color:#9da7b3;margin-left:.3rem;";
          badge.textContent = triggerType;
          schedAtNode.parentNode.insertBefore(badge, schedAtNode.nextSibling);
        }
      }
    }
    setIfPresent('[data-governance-card="status"] [data-governance="finished-at"]', ag.finished_at, fmtUtcToShanghai);
    setIfPresent('[data-governance-card="status"] [data-governance="duration"]', ag.duration_seconds, function (v) { return v + "s"; });
    setIfPresent('[data-governance-card="status"] [data-governance="exit-code"]', ag.exit_code);
    setIfPresent('[data-governance-card="status"] [data-governance="mode"]', ag.runner_mode);
    // governance_schedule: render the live timer calendar (not
    // ``schedule`` which is the static contract string).
    const gsched = h.governance_schedule || {};
    const schedNode = document.querySelector('[data-governance-card="status"] [data-governance="schedule"]');
    if (schedNode) {
      if (gsched.active_state === "unknown" || !gsched.calendar) {
        schedNode.textContent = "—";
      } else {
        schedNode.textContent = gsched.calendar;
      }
    }
    const schedMeta = document.querySelector('[data-governance-card="status"] [data-governance="schedule-meta"]');
    if (schedMeta) {
      if (!gsched.calendar) {
        schedMeta.textContent = "";
      } else {
        const lastTrig = gsched.last_trigger ? fmtUtcToShanghai(gsched.last_trigger) : "—";
        const nextElapse = gsched.next_elapse ? fmtUtcToShanghai(gsched.next_elapse) : "—";
        schedMeta.textContent = "上次触发 " + lastTrig + " · 下次 " + nextElapse;
      }
    }
    // ``last-run`` is the started_at, not finished_at — the brief pins
    // this so the dashboard never confuses "when did the timer fire"
    // with "when did the work finish".
    setIfPresent('[data-governance-card="status"] [data-governance="last-run"]', ag.started_at || ag.last_run, fmtUtcToShanghai);
    setIfPresent('[data-governance-card="status"] [data-governance="next-run"]', ag.next_scheduled_at || ag.next_run, fmtUtcToShanghai);
    const resultNode = document.querySelector('[data-governance-card="status"] [data-governance="result"]');
    if (resultNode) {
      resultNode.textContent = ag.last_result || "unknown";
      resultNode.style.color = statusColor(ag.last_result);
    }
    const summaryNode = document.querySelector('[data-governance-card="status"] [data-governance="summary"]');
    if (summaryNode) {
      summaryNode.textContent = ag.last_summary || "等待首次治理结果；仅 Memory 内容治理，不物理删除。";
    }
  }

  function renderOverview(h) {
    renderMaintenanceCards(h);
    renderGovernanceCard(h);
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

    renderHealthPill(h);
  }

  function renderHealthPill(h) {
    const pill = document.querySelector(".health-pill");
    if (pill && h && h.total_memories) {
      const pct = Math.round(((h.active || 0) / h.total_memories) * 100);
      pill.textContent = "记忆健康度 " + pct + "%";
      pill.style.borderColor = pct >= 80 ? "#7ee787" : pct >= 50 ? "#ffa657" : "#ff7b72";
    }
  }

  global.OCMemory.renderOverview = renderOverview;
  global.OCMemory.renderHealthPill = renderHealthPill;
  global.OCMemory.TIER_PLAN = TIER_PLAN;
})(typeof window !== "undefined" ? window : globalThis);
