/* OpenClaw Memory OS — Tiers + Duplicates page renderers.
 *
 * Both pages share the same backend-driven, table-only layout.
 */
(function (global) {
  "use strict";
  const OC = global.OCMemory;
  if (!OC) {
    console.error("[tiers_duplicates] missing OCMemory base");
    return;
  }
  if (global.OCMemory.__tiersDuplicatesLoaded) return;
  global.OCMemory.__tiersDuplicatesLoaded = true;

  const ZH = {
    tier: { core: "核心事实", long: "长期偏好", medium: "中期记录", short: "短期记录", working: "临时状态" },
    status: { active: "当前生效", superseded: "已被取代", expired: "已过期", needs_review: "待审核" },
  };

  const TIER_PLAN = {
    core: { n: 1, role: "当前事实", recall: "默认优先", color: "#7ee787" },
    long: { n: 1, role: "长期事实", recall: "默认优先", color: "#7ee787" },
    medium: { n: 2, role: "中期记录", recall: "常规召回", color: "#79c0ff" },
    short: { n: 3, role: "短期记录", recall: "常规召回", color: "#d2a8ff" },
    working: { n: 4, role: "临时状态", recall: "有 expires_at 才召回", color: "#ffa657" },
  };

  function translateTier(name) { return ZH.tier[name] || name; }
  function translateStatus(name) { return ZH.status[name] || name; }

  function tableHTML(head, rows, emptyLabel) {
    if (!rows.length) return '<div class="empty">' + (emptyLabel || "暂无数据") + '</div>';
    return '<table><thead><tr>' + head.map(function (h) { return '<th>' + h + '</th>'; }).join("") + '</tr></thead><tbody>' + rows.map(function (r) { return '<tr>' + r.map(function (c) { return '<td>' + c + '</td>'; }).join("") + '</tr>'; }).join("") + '</tbody></table>';
  }

  function renderTiers(t) {
    if (!t) return;
    const tierTable = OC.el("tierTable");
    if (tierTable) {
      tierTable.innerHTML = tableHTML(
        ["层级", "数字", "角色", "召回优先级", "数量"],
        (t.tiers || []).map(function (r) {
          const plan = TIER_PLAN[r.tier] || { n: "?", role: "—", recall: "—", color: "#9da7b3" };
          return [
            translateTier(r.tier),
            '<span style="color:' + plan.color + ';font-weight:700;">Tier ' + plan.n + '</span>',
            plan.role,
            plan.recall,
            r.count,
          ];
        })
      );
    }
    const statusTable = OC.el("statusTable");
    if (statusTable) {
      statusTable.innerHTML = tableHTML(
        ["状态", "数量"],
        (t.statuses || []).map(function (r) { return [translateStatus(r.status), r.count]; })
      );
    }
  }

  function renderDuplicates(d) {
    const target = OC.el("dupList");
    if (!target) return;
    if (!d || !d.clusters || !d.clusters.length) {
      target.innerHTML = '<div class="empty">未发现近似重复簇。</div>';
      return;
    }
    const controls = '<div class="controls"><button id="consolidateBtn" style="border-color:#7ee787;background:#2ea04322;color:#7ee787;">合并所选重复簇</button></div>';
    target.innerHTML = controls + d.clusters.map(function (c) {
      return '<div class="card" style="margin-bottom:.6rem"><label><input type="checkbox" class="dup-check" data-cluster="' + OC.escapeHTML(c.representative_id) + '"> ' + OC.escapeHTML(c.representative_id) + ' — 得分 ' + OC.escapeHTML(String(c.score)) + '</label><div class="muted">' + OC.escapeHTML(c.rationale) + '</div><div>成员：' + (c.member_ids || []).map(function (id) { return '<code>' + OC.escapeHTML(id) + '</code>'; }).join(", ") + '</div></div>';
    }).join("");
  }

  global.OCMemory.renderTiers = renderTiers;
  global.OCMemory.renderDuplicates = renderDuplicates;
  global.OCMemory.tableHTML = tableHTML;
})(typeof window !== "undefined" ? window : globalThis);
