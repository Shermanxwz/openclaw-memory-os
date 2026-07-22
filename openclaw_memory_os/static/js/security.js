/* OpenClaw Memory OS — Security page renderer.
 *
 * Reads /api/security/sessions, renders the active session table (with
 * current-session marker), recent security events, and wires the
 * "撤销全部其他会话" button to /api/security/sessions/revoke-all.
 */
(function (global) {
  "use strict";
  const OC = global.OCMemory;
  if (!OC) {
    console.error("[security] missing OCMemory base");
    return;
  }
  if (global.OCMemory.__securityLoaded) return;
  global.OCMemory.__securityLoaded = true;

  function statusBadge(revoked, current) {
    if (revoked) return '<span class="badge badge-superseded">已撤销</span>';
    if (current)  return '<span class="badge badge-active">当前</span>';
    return '<span class="badge badge-active">活跃</span>';
  }

  function renderSessions(node, sessions) {
    if (!node) return;
    if (!sessions || !sessions.length) {
      node.innerHTML = '<div class="empty">当前没有活跃会话。</div>';
      return;
    }
    const rows = sessions.map(function (s) {
      return '<tr>' +
        '<td><code>' + OC.escapeHTML(s.fingerprint || "—") + '</code></td>' +
        '<td>' + OC.escapeHTML(s.issued_at || "—") + '</td>' +
        '<td>' + OC.escapeHTML(String(s.max_age || "—")) + 's</td>' +
        '<td>' + statusBadge(!!s.revoked, !!s.current) + '</td>' +
      '</tr>';
    });
    node.innerHTML = '<table><thead><tr>' +
      '<th>Fingerprint</th><th>签发于</th><th>Max-Age</th><th>状态</th>' +
      '</tr></thead><tbody>' + rows.join("") + '</tbody></table>';
  }

  function renderEvents(node, events) {
    if (!node) return;
    if (!events || !events.length) {
      node.innerHTML = '<div class="empty">暂无最近安全事件。</div>';
      return;
    }
    const rows = events.map(function (e) {
      return '<tr>' +
        '<td>' + OC.escapeHTML(e.timestamp || e.ts || "—") + '</td>' +
        '<td><code>' + OC.escapeHTML(e.action || "—") + '</code></td>' +
        '<td>' + OC.escapeHTML(e.detail || "—") + '</td>' +
      '</tr>';
    });
    node.innerHTML = '<table><thead><tr>' +
      '<th>时间</th><th>动作</th><th>详情</th>' +
      '</tr></thead><tbody>' + rows.join("") + '</tbody></table>';
  }

  function formatMaxAge(seconds) {
    // Surface the session cookie ``max_age`` as a human-friendly
    // Chinese label. Live cookie config (auth.py:9) is HttpOnly +
    // Secure + SameSite=Lax; live ``max_age`` comes from each session
    // row, so the dashboard never has to hardcode a literal here.
    const s = Number(seconds);
    if (!isFinite(s) || s <= 0) return "—";
    if (s % 86400 === 0) {
      const days = Math.round(s / 86400);
      return days + " 天";
    }
    if (s % 3600 === 0) {
      const hours = Math.round(s / 3600);
      return hours + " 小时";
    }
    if (s >= 60) {
      const minutes = Math.round(s / 60);
      return minutes + " 分钟";
    }
    return s + " 秒";
  }

  function renderAuthStatus(node, sessions) {
    if (!node) return;
    const active = (sessions || []).filter(function (s) { return !s.revoked; }).length;
    // Derive the displayed max_age from the live session rows; fall
    // back to "—" when the list is empty so the dashboard never lies
    // about a constant value that may be rotated via env at runtime.
    const maxAgeSeconds = (sessions && sessions.length)
      ? sessions[0].max_age
      : null;
    const maxAgeLabel = formatMaxAge(maxAgeSeconds);
    node.innerHTML = '' +
      '<div class="card"><h2>Auth 状态</h2><div class="stat status-ok">已启用</div><small>Password + TOTP · 共享 Token</small></div>' +
      '<div class="card"><h2>CSRF</h2><div class="stat status-ok">已启用</div><small>csrf_token cookie + X-CSRF-Token</small></div>' +
      '<div class="card"><h2>活跃 Session</h2><div class="stat">' + OC.escapeHTML(String(active)) + '</div><small>当前 refresh 在浏览器</small></div>' +
      '<div class="card"><h2>Session</h2><div class="stat">' + OC.escapeHTML(maxAgeLabel) + '</div><small>HttpOnly · Secure · SameSite=Lax</small></div>';
  }

  function bindRevokeAll() {
    const btn = OC.el("revokeAllBtn");
    if (!btn || btn.dataset.bound) return;
    btn.dataset.bound = "1";
    btn.addEventListener("click", async function () {
      const status = OC.el("revokeAllStatus");
      OC.setStatus(status, "撤销全部其他会话中...", "info");
      try {
        const resp = await OC.postJSON("/api/security/sessions/revoke-all", {});
        if (!resp.ok) {
          OC.setStatus(status, "❌ HTTP " + resp.status, "err");
          return;
        }
        const body = await resp.json();
        OC.setStatus(status, "✅ 已撤销 " + OC.escapeHTML(String(body.revoked || 0)) + " 个会话", "ok");
        // Reload to update session table
        render();
      } catch (e) {
        OC.setStatus(status, "❌ " + e.message, "err");
      }
    });
  }

  function render() {
    const sessionsNode = OC.el("security-sessions");
    const eventsNode = OC.el("security-events");
    const authNode = OC.el("security-cards");
    OC.getJSON("/api/security/sessions")
      .then(function (payload) {
        const sessions = (payload && payload.sessions) || [];
        const events = (payload && payload.events) || [];
        renderAuthStatus(authNode, sessions);
        renderSessions(sessionsNode, sessions);
        renderEvents(eventsNode, events);
        bindRevokeAll();
      })
      .catch(function (e) {
        if (sessionsNode) sessionsNode.innerHTML = '<div class="empty">❌ 加载失败: ' + OC.escapeHTML(e.message) + '</div>';
      });
  }

  global.OCMemory.renderSecurityPage = render;
})(typeof window !== "undefined" ? window : globalThis);
