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
    // The consolidation endpoint takes ``cluster_ids`` (memory IDs to
    // merge into one new memory). We stash the member_ids as a JSON
    // blob on each checkbox so the click handler can collect them
    // without re-querying the API. member_ids always includes the
    // representative as the first entry on the server side, but to be
    // safe we union it client-side.
    const controls = '<div class="controls">' +
      '<button id="consolidateBtn" style="border-color:#7ee787;background:#2ea04322;color:#7ee787;">合并所选重复簇</button>' +
      '<span id="consolidateStatus" class="muted" style="margin-left:.6rem;"></span>' +
      '</div>';
    target.innerHTML = controls + d.clusters.map(function (c) {
      const members = (c.member_ids || []).slice();
      if (c.representative_id && members.indexOf(c.representative_id) === -1) {
        members.unshift(c.representative_id);
      }
      const memberBlob = OC.escapeHTML(JSON.stringify(members));
      return '<div class="card" style="margin-bottom:.6rem">' +
        '<label><input type="checkbox" class="dup-check" data-cluster="' + OC.escapeHTML(c.representative_id) + '" data-cluster-members=\'' + memberBlob + '\'> ' +
        OC.escapeHTML(c.representative_id) + ' — 得分 ' + OC.escapeHTML(String(c.score)) + '</label>' +
        '<div class="muted">' + OC.escapeHTML(c.rationale) + '</div>' +
        '<div>成员：' + (c.member_ids || []).map(function (id) { return '<code>' + OC.escapeHTML(id) + '</code>'; }).join(", ") + '</div>' +
        '</div>';
    }).join("");
    bindConsolidateButton();
  }

  function bindConsolidateButton() {
    const btn = document.getElementById("consolidateBtn");
    if (!btn || btn.dataset.bound) return;
    btn.dataset.bound = "1";
    const status = OC.el("consolidateStatus");
    btn.addEventListener("click", async function () {
      if (!OC || typeof OC.postJSON !== "function") {
        if (status) status.textContent = "❌ 页面脚本未就绪";
        return;
      }
      const checked = Array.prototype.slice.call(
        document.querySelectorAll("#dupList input.dup-check:checked")
      );
      if (!checked.length) {
        OC.setStatus && OC.setStatus(status, "请先勾选要合并的重复簇。", "info");
        if (status && !status.textContent) status.textContent = "请先勾选要合并的重复簇。";
        return;
      }
      // Union the member_ids across all selected clusters so we never
      // submit a duplicate member twice.
      const ids = [];
      const seen = Object.create(null);
      checked.forEach(function (box) {
        let members = [];
        try {
          members = JSON.parse(box.getAttribute("data-cluster-members") || "[]");
        } catch (e) {
          members = [];
        }
        members.forEach(function (mid) {
          if (mid && !seen[mid]) { seen[mid] = true; ids.push(mid); }
        });
      });
      if (!ids.length) {
        if (status) status.textContent = "❌ 所选重复簇没有可合并的成员。";
        return;
      }
      OC.setStatus && OC.setStatus(status, "提交合并分析中…", "info");
      if (status && !OC.setStatus) status.textContent = "提交合并分析中…";
      try {
        const resp = await OC.postJSON("/api/consolidate-duplicates", {
          cluster_ids: ids,
          strategy: "merge",
        });
        if (!resp.ok) {
          OC.setStatus && OC.setStatus(status, "❌ HTTP " + resp.status, "err");
          return;
        }
        const body = await resp.json();
        const consolidation = (body && body.consolidation) || {};
        const consolidatedId = consolidation.consolidated_id || "(?)";
        const merged = (consolidation.merged_member_ids || []).length;
        OC.setStatus && OC.setStatus(
          status,
          "✅ 分析完成 · consolidated_id=" + consolidatedId + " · 合并 " + merged + " 条",
          "ok"
        );
      } catch (e) {
        OC.setStatus && OC.setStatus(status, "❌ " + e.message, "err");
      }
    });
  }

  global.OCMemory.renderTiers = renderTiers;
  global.OCMemory.renderDuplicates = renderDuplicates;
  global.OCMemory.bindConsolidateButton = bindConsolidateButton;
  global.OCMemory.tableHTML = tableHTML;
})(typeof window !== "undefined" ? window : globalThis);
