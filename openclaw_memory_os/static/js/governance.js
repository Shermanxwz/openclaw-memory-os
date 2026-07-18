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
    if (!candidates.length) {
      target.innerHTML = '<div class="empty">当前没有清理候选。已过滤核心/长期/高重要性/7 天内新建的记忆。</div>';
      // Disable button if there are no candidates.
      const markBtn = OC.el("markAllDel");
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
          '<button class="del-mark" data-id="' + OC.escapeHTML(c.id) + '" style="padding:.2rem .5rem;border-radius:4px;border:1px solid #f85149;background:#f8514922;color:#ff7b72;cursor:pointer;font-size:.8rem;">✕ 排除</button>',
        ];
      })
    );

    // Per-row "exclude" button only marks that single id (review-only).
    document.querySelectorAll(".del-mark").forEach(function (btn) {
      btn.addEventListener("click", async function () {
        const id = btn.getAttribute("data-id");
        btn.disabled = true;
        const status = OC.el("delStatus");
        OC.setStatus(status, "排除 " + id + " 中...", "info");
        try {
          const r = await OC.postJSON("/api/consolidate-duplicates", { cluster_ids: [id] });
          OC.setStatus(status, r.ok ? "✅ 已确认排除 (review-only)" : "❌ 服务端错误 " + r.status, r.ok ? "ok" : "err");
        } catch (e) {
          OC.setStatus(status, "❌ 网络错误", "err");
        }
      });
    });

    // Bulk confirmation button: marked candidates are *review-only*;
    // the OS never performs a physical delete from this surface.
    const markBtn = OC.el("markAllDel");
    if (markBtn) {
      markBtn.addEventListener("click", async function () {
        const ids = [];
        document.querySelectorAll("#delList tbody tr td:first-child").forEach(function (td) {
          ids.push(td.textContent || "");
        });
        if (!ids.length) return;
        const status = OC.el("delStatus");
        OC.setStatus(status, "确认中... (" + ids.length + " 条候选 review-only)", "info");
        try {
          const r = await OC.postJSON("/api/consolidate-duplicates", { cluster_ids: ids });
          OC.setStatus(status, r.ok
            ? "✅ 已确认 " + ids.length + " 条候选 (无物理删除)"
            : "❌ 服务端错误 " + r.status,
            r.ok ? "ok" : "err");
        } catch (e) {
          OC.setStatus(status, "❌ 网络错误", "err");
        }
      });
    }
  }

  global.OCMemory.renderGovernanceCandidates = renderDeletion;
})(typeof window !== "undefined" ? window : globalThis);
