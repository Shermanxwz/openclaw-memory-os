/* OpenClaw Memory OS — Memories page renderer.
 *
 * Reads /api/dashboard/memories, partitions by collection and renders
 * a *read-only* table. The actions surface is review-only: "查看" only.
 * The page header makes the read-only contract explicit; no UI control
 * performs a physical delete.
 */
(function (global) {
  "use strict";
  const OC = global.OCMemory;
  if (!OC) {
    console.error("[memories] missing OCMemory base");
    return;
  }
  if (global.OCMemory.__memoriesLoaded) return;
  global.OCMemory.__memoriesLoaded = true;

  const ZH = {
    tier: { core: "核心事实", long: "长期偏好", medium: "中期记录", short: "短期记录", working: "临时状态" },
    status: { active: "当前生效", superseded: "已被取代", expired: "已过期", needs_review: "待审核" },
  };

  function translateTier(name) { return ZH.tier[name] || name; }
  function translateStatus(name) { return ZH.status[name] || name; }

  function badge(status) {
    return '<span class="badge badge-' + OC.escapeHTML(status) + '">' + OC.escapeHTML(ZH.status[status] || status) + '</span>';
  }

  function buildRow(m) {
    const cols = [
      OC.escapeHTML(m.id || "—"),
      translateTier(m.tier),
      badge(m.status),
      OC.escapeHTML(m.collection || m.source || "—"),
      OC.escapeHTML(m.score != null ? String(m.score) : "—"),
      OC.escapeHTML(m.text ? String(m.text).slice(0, 120) : "—"),
      OC.escapeHTML(m.updated_at || m.created_at || "—"),
    ];
    return '<tr>' + cols.map(function (c) { return '<td>' + c + '</td>'; }).join("") + '</tr>';
  }

  function renderCollection(node, name, items) {
    const safeName = OC.escapeHTML(name);
    const table = '<table><thead><tr>' +
      '<th>ID</th><th>层级</th><th>状态</th><th>Collection</th><th>Score</th><th>内容(预览)</th><th>更新时间</th>' +
      '</tr></thead><tbody>' + items.map(buildRow).join("") + '</tbody></table>';
    return '<div class="card" style="margin-bottom:1rem;"><h2>Collection: ' + safeName + ' <small style="color:#9da7b3;font-weight:400;">(' + items.length + ' 条 · read-only)</small></h2>' + table + '</div>';
  }

  function render() {
    const target = OC.el("memories-list");
    const meta = OC.el("memories-meta");
    if (!target) return;
    target.innerHTML = '<div class="empty">读取中...</div>';

    OC.getJSON("/api/dashboard/memories?limit=200")
      .then(function (payload) {
        const memories = (payload && payload.memories) || [];
        const collections = (payload && payload.collections) || [];
        const backend = (payload && payload.backend) || "—";
        const count = (payload && payload.count) || memories.length;
        if (meta) {
          meta.innerHTML = '' +
            '<div class="grid">' +
              '<div class="card"><h2>Backend</h2><div class="stat"><code>' + OC.escapeHTML(backend) + '</code></div></div>' +
              '<div class="card"><h2>Collections</h2><div class="stat">' + OC.escapeHTML(String(collections.length)) + '</div></div>' +
              '<div class="card"><h2>本视图记忆数</h2><div class="stat">' + OC.escapeHTML(String(count)) + '</div></div>' +
              '<div class="card"><h2>策略</h2><div class="stat" style="font-size:1rem;color:#7ee787;">read-only · 无物理删除</div></div>' +
            '</div>';
        }

        if (!memories.length) {
          target.innerHTML = '<div class="empty">未发现记忆。Backend: ' + OC.escapeHTML(backend) + '。</div>';
          return;
        }

        // Group by collection
        const byCollection = {};
        memories.forEach(function (m) {
          const key = m.collection || m.source || "default";
          if (!byCollection[key]) byCollection[key] = [];
          byCollection[key].push(m);
        });

        target.innerHTML = Object.keys(byCollection).sort().map(function (key) {
          return renderCollection(target, key, byCollection[key]);
        }).join("");
      })
      .catch(function (e) {
        target.innerHTML = '<div class="empty">❌ 读取失败: ' + OC.escapeHTML(e.message) + '</div>';
      });
  }

  global.OCMemory.renderMemoriesPage = render;
})(typeof window !== "undefined" ? window : globalThis);
