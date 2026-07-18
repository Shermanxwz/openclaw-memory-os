/* OpenClaw Memory OS — Shared static API client (v0.3.0)
 *
 * This module is loaded BEFORE every per-section module. It exposes
 * one IIFE-shaped namespace (``window.OCMemory``) so per-section
 * modules can share helpers without a bundler.
 *
 * Two design rules:
 *   1. The browser-side never touches credentials. Auth is carried by
 *      the HttpOnly session cookie set during /login; JS only echoes
 *      back the CSRF token from the non-HttpOnly ``csrf_token`` cookie.
 *   2. All non-GET requests flow through ``postJSON`` so the CSRF
 *      header is always set on state-changing endpoints.
 */
(function (global) {
  "use strict";

  if (global.OCMemory && global.OCMemory.__shared) {
    return; // already loaded; idempotent guard
  }

  const COOKIE_NAME = "csrf_token";
  const DEFAULT_TIMEOUT_MS = 60_000;

  function getCookie(name) {
    if (!name) return "";
    const target = String(name);
    const pairs = (document.cookie || "").split(";");
    for (const raw of pairs) {
      const trimmed = raw.trim();
      if (!trimmed) continue;
      const idx = trimmed.indexOf("=");
      if (idx < 0) continue;
      if (decodeURIComponent(trimmed.slice(0, idx)) === target) {
        return decodeURIComponent(trimmed.slice(idx + 1));
      }
    }
    return "";
  }

  function csrfHeaders(extra) {
    const headers = Object.assign({ "Content-Type": "application/json" }, extra || {});
    const csrf = getCookie(COOKIE_NAME);
    if (csrf) headers["X-CSRF-Token"] = csrf;
    return headers;
  }

  function authHeaders(extra) {
    return Object.assign({}, extra || {});
  }

  function _withTimeout(url, init, timeoutMs) {
    const ctrl = new AbortController();
    const t = setTimeout(function () { ctrl.abort(); }, Math.max(1, timeoutMs || DEFAULT_TIMEOUT_MS));
    const upstream = fetch(url, Object.assign({}, init, { signal: ctrl.signal }));
    upstream.finally(function () { clearTimeout(t); });
    return upstream;
  }

  async function getJSON(url, opts) {
    opts = opts || {};
    const init = {
      method: "GET",
      credentials: "same-origin",
      headers: authHeaders(opts.headers),
      signal: opts.signal,
    };
    const resp = await _withTimeout(url, init, opts.timeoutMs || DEFAULT_TIMEOUT_MS);
    if (!resp.ok) throw new Error("HTTP " + resp.status + " " + (resp.statusText || ""));
    const ct = resp.headers.get("content-type") || "";
    if (ct.indexOf("application/json") === -1) {
      // Avoid silently swallowing HTML 404 pages as JSON
      throw new Error("Unexpected content-type: " + ct);
    }
    return resp.json();
  }

  async function postJSON(url, body, opts) {
    opts = opts || {};
    const init = {
      method: "POST",
      credentials: "same-origin",
      headers: csrfHeaders(opts.headers),
      body: JSON.stringify(body == null ? {} : body),
      signal: opts.signal,
    };
    return _withTimeout(url, init, opts.timeoutMs || DEFAULT_TIMEOUT_MS);
  }

  function el(id) {
    return document.getElementById(id);
  }

  function escapeHTML(s) {
    return (s == null ? "" : String(s)).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  function formatTimestamp(value) {
    if (!value) return "—";
    const d = new Date(value);
    if (isNaN(d.getTime())) return String(value);
    return d.toLocaleString();
  }

  function setStatus(node, text, kind) {
    if (!node) return;
    node.textContent = text;
    node.dataset.status = kind || "info";
  }

  const exports = {
    __shared: true,
    CSRF_COOKIE: COOKIE_NAME,
    getCookie: getCookie,
    csrfHeaders: csrfHeaders,
    authHeaders: authHeaders,
    getJSON: getJSON,
    postJSON: postJSON,
    el: el,
    escapeHTML: escapeHTML,
    formatTimestamp: formatTimestamp,
    setStatus: setStatus,
  };

  global.OCMemory = exports;

  // Compatibility shims for modules that still reference the old
  // dashboard.js top-level helpers. New code should prefer
  // ``OCMemory.*`` directly.
  if (typeof global.getCookie !== "function") global.getCookie = getCookie;
  if (typeof global.csrfHeaders !== "function") global.csrfHeaders = csrfHeaders;
  if (typeof global.authHeaders !== "function") global.authHeaders = authHeaders;
  if (typeof global.postJSON !== "function") global.postJSON = postJSON;
  if (typeof global.getJSON !== "function") global.getJSON = getJSON;
  if (typeof global.escapeHTML !== "function") global.escapeHTML = escapeHTML;
  if (typeof global.el !== "function") global.el = el;
})(typeof window !== "undefined" ? window : globalThis);
