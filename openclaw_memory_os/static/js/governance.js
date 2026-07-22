/* OpenClaw Memory OS — Memory 自主治理 page renderer.
 *
 * Renders deletion candidates from /api/deletion-candidates. The
 * dashboard language is intentionally *review-only* — the action
 * confirms exclusion (exclusion from the auto-cleanup pool) but
 * never performs a physical delete. This module preserves that
 * contract: button labels + status text use "排除", "确认" or
 * "review-only" wording only; never "delete".
 */
(function (global) {
  "use strict";
  const OC = global.OCMemory;
  if (!OC) {
    console.error("[governance] missing OCMemory base");
    return;
  }
  if (global.OCMemory.__governanceLoaded) return;
  global.OCMemory.__governanceLoaded = true;

  const ZH = {
    tier: { core: "核心事实", long: "长期偏好", medium: "中期记录", short: "短期记录", working: "临时状态" },
    status: { active: "当前生效", superseded: "已被取代", expired: "已过期", needs_review: "待审核" },
  };

  function translateTier(name) { return ZH.tier[name] || name; }
  function translateStatus(name) { return ZH.status[name] || name; }

  function tableHTML(head, rows) {
    if (!rows.length) return '<div class="empty">暂无清理候选</div>';
    return '<table><thead><tr>' + head.map(function (h) { return '<th>' + h + '</th>'; }).join("") + '</tr></thead><tbody>' + rows.map(function (r) { return '<tr>' + r.map(function (c) { return '<td>' + c + '</td>'; }).join("") + '</tr>'; }).join("") + '</tbody></table>';
  }

  function renderDeletion(d) {
    const target = OC.el("delList");
    if (!target) return;

    const candidates = (d && d.candidates) || [];
    const countNode = OC.el("deletionCandidateCount");
    if (countNode) countNode.textContent = String((d && d.count != null) ? d.count : candidates.length);
    const markBtn = OC.el("markAllDel");
    if (!candidates.length) {
      target.innerHTML = '<div class="empty">当前没有清理候选。已过滤核心/长期/高重要性/7 天内新建的记忆。</div>';
      if (markBtn) { markBtn.disabled = true; markBtn.title = "没有候选可标记"; }
      return;
    }
    target.innerHTML = tableHTML(
      ["ID", "层级", "状态", "原因", "建议操作", "操作"],
      candidates.map(function (c) {
        return [
          OC.escapeHTML(c.id),
          translateTier(c.tier),
          translateStatus(c.status),
          OC.escapeHTML(c.reason),
          OC.escapeHTML(c.recommended_action),
          '<span class="muted" title="review-only：当前无持久化排除/删除接口">仅审核</span>',
        ];
      })
    );

    // Bulk confirmation is deliberately local-only. There is currently
    // no persisted "reviewed/excluded" API, and calling an unrelated
    // consolidation endpoint would falsely imply state changed.
    if (markBtn && !markBtn.dataset.bound) {
      markBtn.dataset.bound = "1";
      markBtn.addEventListener("click", function () {
        const status = OC.el("delStatus");
        OC.setStatus(status, "✅ 已查看当前 " + candidates.length + " 条候选（review-only，无持久化变更）", "ok");
      });
    }
  }

  global.OCMemory.renderGovernanceCandidates = renderDeletion;
})(typeof window !== "undefined" ? window : globalThis);
