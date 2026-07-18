"""Authentication for the dashboard.

The OS is intended to be deployed behind a reverse proxy that terminates
TLS. Authentication is layered:

1. **Bearer / cookie token** (``MEMORY_OS_TOKEN``) — the historical,
   backwards-compatible mode. Clients POST ``/login`` with the token in
   a form field (or via ``Authorization: Bearer`` header). The server
   responds with an HttpOnly + Secure + SameSite=Strict session cookie.

2. **Password + TOTP** — when ``MEMORY_OS_PASSWORD`` and
   ``MEMORY_OS_TOTP_SECRET`` are set, the login flow upgrades to a
   two-step challenge: the form asks for a password **and** a 6-digit
   TOTP code. The TOTP implementation is RFC 6238 (HMAC-SHA1, 30s step,
   6 digits) using only the standard library.

3. **CSRF protection** — every state-changing request must carry a
   matching ``csrf_token`` cookie + ``X-CSRF-Token`` header / form
   field. ``/login`` and ``/logout`` issue and verify CSRF tokens via
   the ``csrf_token`` cookie.

4. **Session revocation** — ``/logout`` and ``/api/security/sessions/
   revoke-all`` persist revocations in the :class:`SessionStore` so
   that a revoked cookie token remains rejected across service
   restarts. Active sessions issued before a restart stay valid until
   their ``max_age`` elapses.

Multi-user auth is out of scope; the goal is to keep the project safe
to open-source while remaining useful.

Query-string tokens (``?token=...``) are **not** accepted: they would
land in nginx access logs, browser history, and Referer headers.
"""

from __future__ import annotations

import hashlib
import logging
import hmac
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional, Set

from fastapi import HTTPException, Request, Response, status

from .config import Settings, get_settings
from .sessions import SessionStore

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "memory_os_session"
CSRF_COOKIE_NAME = "csrf_token"
DEFAULT_SESSION_MAX_AGE = 60 * 60 * 12  # 12 hours
TOTP_STEP_SECONDS = 30
TOTP_DIGITS = 6
TOTP_WINDOW = 1  # Accept current step and ±1 step (90-second total window)

# In-process session revocation cache. Used as a fast-path before
# consulting the persistent SessionStore so that a revocation made in
# the same process is visible to the very next request without a SQLite
# round-trip. The persistent copy is the source of truth — the in-
# memory set is just a convenience layer that resets on restart.
_revoked_sessions: Set[str] = set()

# Process-wide SessionStore singleton. Lazy-initialised by
# ``get_session_store``; tests may swap it via
# ``_set_session_store_for_tests``.
_session_store: Optional[SessionStore] = None
_session_store_lock = threading.Lock()


def get_session_store() -> SessionStore:
    """Return the process-wide SessionStore with race-free lazy creation."""
    global _session_store
    if _session_store is None:
        with _session_store_lock:
            if _session_store is None:
                _session_store = SessionStore()
    return _session_store


def close_session_store() -> None:
    """Thread-safe, idempotent close of the process-wide SessionStore.

    Called during FastAPI lifespan shutdown so that the SQLite connection
    is cleanly closed (flushing WAL to the main DB file) before the
    process exits.  Without this, the WAL file can be left in an
    inconsistent state if the process is killed or crashes during
    shutdown, which causes the next startup to see stale or missing
    session data.
    """
    global _session_store
    with _session_store_lock:
        if _session_store is None:
            return
        store = _session_store
        _session_store = None
    try:
        store.close()
    except Exception as e:
        logger.warning("session store close failed: %s", e)


def _set_session_store_for_tests(store: Optional[SessionStore]) -> Optional[SessionStore]:
    """Swap the test SessionStore and return the previous live object.

    Ownership remains with the caller. In particular, the previous store is not
    closed here: tests commonly restore the returned object, and restoring an
    already-closed SQLite connection creates order-dependent failures.
    """
    global _session_store
    with _session_store_lock:
        previous = _session_store
        _session_store = store
    return previous


def _reset_auth_state_for_tests() -> None:
    """Reset all module-level auth state for test isolation.

    Used by conftest.py autouse fixture to prevent cross-test contamination.
    Clears the revoked-sessions cache, closes any cached SessionStore,
    and resets the module singleton so the next access re-initialises
    from a clean slate.
    """
    global _session_store
    _revoked_sessions.clear()
    with _session_store_lock:
        if _session_store is not None:
            try:
                _session_store.close()
            except Exception:
                pass
            _session_store = None


def _session_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def session_max_age() -> int:
    """Return the configured session cookie ``max_age`` in seconds.

    Falls back to :data:`DEFAULT_SESSION_MAX_AGE` when the env var is
    unset, empty, or non-positive.
    """
    import os

    raw = os.environ.get("MEMORY_OS_SESSION_MAX_AGE", "").strip()
    if not raw:
        return DEFAULT_SESSION_MAX_AGE
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SESSION_MAX_AGE
    return value if value > 0 else DEFAULT_SESSION_MAX_AGE


def password_hash(password: str) -> str:
    """Hash a password with Argon2id.

    The returned value is the standard PHC string emitted by
    ``argon2-cffi`` (``$argon2id$v=19$...``). ``verify_password`` keeps
    backward compatibility with legacy ``sha256$`` and plaintext env vars so
    existing deployments can migrate without being locked out.
    """
    from argon2 import PasswordHasher

    return PasswordHasher().hash(password)


def verify_password(submitted: str, settings: Optional[Settings] = None) -> bool:
    """Constant-time comparison of a submitted password.

    Returns ``True`` only when both:

    * :attr:`Settings.memory_os_password` is configured (non-empty), and
    * the submitted password matches the stored hash.

    The stored value may be either an Argon2id PHC hash (preferred), a legacy
    ``sha256$<salt>$<digest>`` hash, or a plaintext password
    (backwards-compatible).
    """
    settings = settings or get_settings()
    expected = (settings.memory_os_password or "").strip()
    if not expected:
        return False
    if not submitted:
        return False
    if expected.startswith("$argon2id$") or expected.startswith("$argon2i$"):
        try:
            from argon2 import PasswordHasher
            from argon2.exceptions import VerificationError, VerifyMismatchError

            PasswordHasher().verify(expected, submitted)
            return True
        except (VerificationError, VerifyMismatchError, Exception):
            return False
    # Support plaintext (backwards compat) and sha256$ legacy hashed values.
    if expected.startswith("sha256$"):
        parts = expected.split("$", 2)
        if len(parts) == 3:
            digest = hashlib.sha256(f"{parts[1]}:{submitted}".encode("utf-8")).hexdigest()
            return hmac.compare_digest(digest, parts[2])
    # Plaintext fallback: constant-time comparison.
    return hmac.compare_digest(submitted, expected)


def _hotp(secret: bytes, counter: int) -> str:
    """RFC 4226 HOTP using HMAC-SHA1.

    Truncates the 20-byte digest to ``TOTP_DIGITS`` decimal digits.
    """
    import struct

    counter_bytes = struct.pack(">Q", counter)
    digest = hmac.new(secret, counter_bytes, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code_int = (
        ((digest[offset] & 0x7F) << 24)
        | ((digest[offset + 1] & 0xFF) << 16)
        | ((digest[offset + 2] & 0xFF) << 8)
        | (digest[offset + 3] & 0xFF)
    )
    return str(code_int % (10 ** TOTP_DIGITS)).zfill(TOTP_DIGITS)


def totp_now(secret: str, *, step: int = TOTP_STEP_SECONDS, now: Optional[float] = None) -> str:
    """Compute the TOTP code for the current time window.

    Accepts both raw and base32-encoded secrets. Time can be pinned via
    ``now`` (seconds since epoch) for deterministic testing.
    """
    if now is None:
        now = time.time()
    counter = int(now // step)
    return _hotp(_decode_totp_secret(secret), counter)


def verify_totp(submitted: str, secret: str, *, now: Optional[float] = None) -> bool:
    """Validate a TOTP code with a small time-skew window (±1 step)."""
    if not submitted or not secret:
        return False
    submitted = submitted.strip()
    if not submitted.isdigit() or len(submitted) != TOTP_DIGITS:
        return False
    if now is None:
        now = time.time()
    counter = int(now // TOTP_STEP_SECONDS)
    key = _decode_totp_secret(secret)
    for offset in range(-TOTP_WINDOW, TOTP_WINDOW + 1):
        candidate = _hotp(key, counter + offset)
        if hmac.compare_digest(candidate, submitted):
            return True
    return False


def _decode_totp_secret(secret: str) -> bytes:
    """Decode a base32 TOTP secret, tolerating whitespace and padding."""
    import base64

    cleaned = "".join(secret.split()).upper()
    # base32 requires padding to a multiple of 8 characters.
    padding = (-len(cleaned)) % 8
    cleaned = cleaned + ("=" * padding)
    try:
        return base64.b32decode(cleaned, casefold=True)
    except Exception as e:
        raise ValueError(f"Invalid TOTP secret (expected base32): {e}") from e


# ---------------------------------------------------------------------------
# CSRF helpers
# ---------------------------------------------------------------------------


def generate_csrf_token() -> str:
    """Return a fresh 32-byte hex CSRF token."""
    return secrets.token_hex(32)


def issue_csrf_cookie(response: Response, token: Optional[str] = None) -> str:
    """Attach the CSRF cookie (non-HttpOnly so JS can read it) and return the token."""
    token = token or generate_csrf_token()
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=token,
        max_age=session_max_age(),
        httponly=False,  # JS must read this to echo it in the X-CSRF-Token header
        secure=True,
        samesite="strict",
        path="/",
    )
    return token


def get_or_create_csrf_cookie(request: Request, response: Response) -> str:
    """Return the existing CSRF cookie, or issue a new one on the response."""
    existing = request.cookies.get(CSRF_COOKIE_NAME)
    if existing:
        return existing
    return issue_csrf_cookie(response)


def verify_csrf(request: Request, submitted: Optional[str] = None) -> bool:
    """Verify the CSRF token from cookie + header/form.

    Returns ``True`` when the submitted token matches the cookie. The
    caller decides whether to raise on failure (this helper is used by
    both the HTML form path and the JSON API path).
    """
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not cookie_token:
        return False
    if submitted is None:
        submitted = request.headers.get("X-CSRF-Token") or request.headers.get("x-csrf-token")
    if not submitted:
        return False
    return hmac.compare_digest(cookie_token, submitted)


def _constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def extract_token(request: Request, settings: Optional[Settings] = None) -> Optional[str]:
    """Pull a bearer token from a trusted source.

    Order:
      1. ``Authorization: Bearer ...`` header (preferred for non-browser clients).
      2. Session cookie ``memory_os_session``.

    Query-string tokens are intentionally NOT accepted.
    """
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth:
        parts = auth.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()

    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if cookie:
        return cookie.strip()

    return None


def _store_rejects_token(provided: str) -> Optional[bool]:
    """Return True for a known invalid session, False for a valid one, else None."""
    store = _session_store
    if store is None or not provided:
        return None
    try:
        if store.is_valid(provided):
            return False
        return True if store.contains(provided) else None
    except Exception:
        return None


def verify_token(provided: Optional[str], settings: Optional[Settings] = None) -> bool:
    """Validate a provided token against configured credentials.

    Authentication backends, in priority order:

    * **Legacy bearer token** (``MEMORY_OS_TOKEN``) — exact match in
      constant time.  This is the *only* value accepted as a raw
      ``Authorization: Bearer`` header.
    * **Session cookie token** — an opaque ID issued by the persistent
      :class:`SessionStore`.  Validated via ``store.is_valid(...)`` so
      that revocations persist across service restarts.

    **Password-as-bearer is removed.**  The password is only accepted
    through the ``/login`` form (with optional TOTP).  This prevents
    long-lived bearer tokens that equal the login password.

    Revoked session tokens are rejected even when they would otherwise
    match the bearer-token ``hmac.compare_digest`` — once an operator
    flips a cookie off the allow-list, it stays off.
    """
    settings = settings or get_settings()
    if not settings.auth_enabled:
        return True
    if not provided:
        return False
    if provided in _revoked_sessions:
        return False
    expected = settings.memory_os_token or ""
    # Bearer-token path: direct hmac compare against MEMORY_OS_TOKEN.
    # Per the v0.3.0.x contract, bearer tokens are NEVER validated by the
    # SessionStore. If the SessionStore happens to have a row that the
    # bearer token matches AND that row is revoked, the revocation wins.
    if expected and _constant_time_eq(provided, expected):
        persisted_state = _store_rejects_token(provided)
        if persisted_state is True:
            return False
        return True
    # Cookie-token path: an opaque session ID issued via
    # ``set_session_cookie`` is validated against the persistent store.
    # ``is_valid`` covers the "still alive & not revoked" check.
    # The store is lazy-initialised here so a fresh process that
    # inherited an empty module-global but shares the on-disk
    # sessions DB (the cross-restart persistence test contract)
    # can still recognise a previously-issued cookie.
    if _session_store is None:
        try:
            get_session_store()
        except Exception:
            # Store init failure must never break the auth path —
            # we just fall through to ``return False`` below.
            pass
    if _session_store is not None:
        try:
            if _session_store.is_valid(provided):
                return True
        except Exception:
            # Never let a store hiccup break the auth path.
            pass
    return False


def revoke_session(token: Optional[str]) -> bool:
    """Persist one session revocation before updating the process cache.

    Unknown tokens (including the configured static bearer token) are not added
    to the cache. Storage errors propagate so logout can fail closed instead of
    presenting success while a stolen cookie remains valid after restart.
    """
    if not token:
        return False
    try:
        revoked = get_session_store().revoke(token)
    except Exception as exc:
        raise RuntimeError("session revocation persistence failed") from exc
    if revoked:
        _revoked_sessions.add(token)
    return revoked


def revoke_all_sessions() -> int:
    """Persist revocation of every known session or raise on storage failure."""
    try:
        return int(get_session_store().revoke_all())
    except Exception as exc:
        raise RuntimeError("session revoke-all persistence failed") from exc


def list_sessions(*, current_token: Optional[str] = None) -> list[dict[str, Any]]:
    """Return non-secret session metadata for the security dashboard.

    Delegates to :meth:`SessionStore.list`. The raw token never leaves
    the store; callers receive a short SHA-256 ``fingerprint`` plus
    issued/expire/revoked fields so the dashboard can show "which
    browser issued this session" without leaking the secret.
    """
    if _session_store is None:
        return []
    try:
        return _session_store.list(current_token=current_token)
    except Exception:
        return []


def is_session_revoked(token: Optional[str]) -> bool:
    """Return True if ``token`` is in the in-process revocation cache.

    Note: this only reflects revocations made in *this* process. The
    authoritative check is :meth:`SessionStore.is_valid`, which
    consults the persistent store.
    """
    return bool(token) and token in _revoked_sessions


def require_auth(request: Request) -> None:
    """FastAPI dependency that enforces token auth."""
    settings = get_settings()
    if not settings.auth_enabled:
        return
    provided = extract_token(request, settings)
    if not verify_token(provided, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def login_html_redirect(request: Request) -> None:
    """Like ``require_auth`` but returns a 401 the login page can render."""
    settings = get_settings()
    if not settings.auth_enabled:
        return
    provided = extract_token(request, settings)
    if not verify_token(provided, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )


def attempt_login(
    password: str = "",
    totp_code: str = "",
    token: str = "",
    recovery_code: str = "",
    settings: Optional[Settings] = None,
) -> bool:
    """Validate a login form submission.

    Modes (priority order):

    1. **Password + TOTP** — when ``MEMORY_OS_PASSWORD`` is set AND the
       v0.3.0.x ``PASSWORD_TOTP_AUTH`` feature flag is enabled (default
       on), both must validate. TOTP is required if
       ``MEMORY_OS_TOTP_SECRET`` is also set; otherwise the password
       alone is enough.
    2. **Password + Recovery Code** — when TOTP is configured but the
       user has lost their device, a one-time recovery code can be
       submitted instead of the TOTP code.  The code is consumed on
       use (cannot be reused).
    3. **Shared bearer token** — when no password is configured (or
       ``PASSWORD_TOTP_AUTH=off`` forces the legacy path), the legacy
       ``token`` field is accepted (backwards compatible).

    v0.3.0.x: ``PASSWORD_TOTP_AUTH=off`` makes the password+TOTP path
    inert even when ``MEMORY_OS_PASSWORD`` is configured, so operators
    can revert to the legacy bearer-token-only flow without rotating
    their env vars.
    """
    settings = settings or get_settings()
    if not settings.auth_enabled:
        return True

    # v0.3.0.x feature flag: force the legacy bearer-token path when
    # the password+TOTP path is disabled. The caller still has to pass
    # the bearer token through ``token`` / ``Authorization``.
    if not bool(getattr(settings, "password_totp_auth", True)):
        return verify_token(token, settings)

    if settings.memory_os_password:
        # If the user submitted a password, validate password (+TOTP if configured).
        if password:
            if not verify_password(password, settings):
                return False
            # TOTP required when secret is configured.
            if settings.memory_os_totp_secret:
                # Accept TOTP code OR one-time recovery code.
                if totp_code and verify_totp(totp_code, settings.memory_os_totp_secret):
                    return True
                if recovery_code and _consume_recovery_code(recovery_code):
                    return True
                return False
            return True
        # No password submitted: fall through to token check below.

    # Fall back to the legacy bearer-token flow (or token-only when
    # password is set but the user submitted a token instead).
    return verify_token(token, settings)


def set_session_cookie(
    response: Response,
    token: str,
    settings: Optional[Settings] = None,
    *,
    max_age: Optional[int] = None,
) -> None:
    """Persist a session before issuing its cookie (fail closed)."""
    settings = settings or get_settings()
    age = max_age if max_age is not None else session_max_age()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session issuance failed.",
        )
    try:
        store = get_session_store()
        store.create(token, int(age), issued_at=datetime.now(timezone.utc))
    except Exception as exc:
        logger.error("session persistence failed; refusing to issue cookie: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session store unavailable.",
        ) from exc
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=age,
        httponly=True,
        secure=True,
        samesite="strict",
        path="/",
    )


def clear_session_cookie(
    response: Response,
    token: Optional[str] = None,
) -> None:
    """Remove the session cookie AND persist revocation of ``token`` (if given).

    Passing ``token`` is optional but recommended: without it, the
    SessionStore cannot be told which row to revoke, so the cookie
    value is only gone client-side. Callers that have the token at
    hand (e.g. the ``/logout`` handler reading from ``request.cookies``)
    should pass it through.
    """
    if token:
        try:
            revoke_session(token)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session revocation unavailable.",
            ) from exc
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(CSRF_COOKIE_NAME, path="/")


# ---------------------------------------------------------------------------
# Login rate limiter
# ---------------------------------------------------------------------------

# Tier 1 — sliding window: 5 failures in 60 s blocks *further* requests
# for 5 minutes. This matches the legacy ``_LOGIN_MAX_ATTEMPTS`` /
# ``_LOGIN_WINDOW_SECONDS`` defaults so existing callers and tests
# continue to behave the same.
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 60

# Tier 2 — escalating lockouts (runbook G0.6):
# * 5 consecutive failures → 5 minute lockout (300 s).
# * 10 consecutive failures → 30 minute lockout (1800 s).
# Once either lockout fires, the IP must wait out the window before
# ANY /login attempt is accepted (HTTP 429). The legacy 5/60 s window
# continues to feed the "consecutive failure" counter that decides
# which tier (or no tier) is active.
_LOGIN_LOCKOUT_5_FAILS_SECONDS = 5 * 60       # 5 minutes
_LOGIN_LOCKOUT_10_FAILS_SECONDS = 30 * 60     # 30 minutes
_LOGIN_CONSECUTIVE_WINDOW_SECONDS = 30 * 60   # 30 minutes — sliding
                                              # window that decides
                                              # whether failures still
                                              # "count" as consecutive


class LoginRateLimiter:
    """In-process sliding-window rate limiter for ``/login``.

    Tracks failed login attempts per client IP.  After
    ``_LOGIN_MAX_ATTEMPTS`` failures within ``_LOGIN_WINDOW_SECONDS``,
    subsequent attempts are rejected with HTTP 429 until the window
    expires.

    Escalating lockouts (runbook G0.6):

    * 5 consecutive failures → 5 minute lockout.
    * 10 consecutive failures → 30 minute lockout.

    ``consecutive failures`` is the count of failures within the last
    ``_LOGIN_CONSECUTIVE_WINDOW_SECONDS`` — older failures decay out
    so that a user who doesn't try for 30 minutes is back to the
    baseline 5-in-60s rate.

    Per-IP privacy: only ``hashlib.sha256(ip)[:16]`` is used as the
    in-process key; the raw client IP is never written to logs or
    storage from this module.

    This is a defence-in-depth measure; the primary brute-force
    protection is the Argon2id password hash + TOTP.
    """

    def __init__(self) -> None:
        # IP-hash → list of failure timestamps (sliding window of
        # ``_LOGIN_WINDOW_SECONDS``).
        self._attempts: dict[str, list[float]] = {}
        # IP-hash → monotonic "currently locked until" timestamp (epoch
        # seconds). Set when an escalating lockout fires and cleared by
        # ``reset`` after a successful login or by ``is_limited`` when
        # the window elapses.
        self._locked_until: dict[str, float] = {}
        # IP-hash → longest recent run of consecutive failures. Not
        # purged automatically; ``record_failure`` reads the rolling
        # window from ``self._attempts`` instead so this attribute is
        # only kept for debugging / introspection.

    @staticmethod
    def _ip_key(client_ip: str) -> str:
        """Hash an IP into a stable in-process bucket key.

        Per runbook G0.6 and G7.x, raw IPs must not appear in logs.
        A 16-char SHA-256 prefix is plenty to disambiguate buckets
        without leaking the underlying IP.
        """
        return hashlib.sha256((client_ip or "").encode("utf-8")).hexdigest()[:16]

    def _purge_window(self, key: str, now: float) -> list[float]:
        """Return the failure window for ``key`` with old entries pruned."""
        window = self._attempts.get(key, [])
        # Prune entries outside BOTH the 60s window (used for the
        # 5/60s limiter) AND the consecutive-window (used by the
        # escalating tier). Pruning to the larger of the two keeps
        # ``record_failure`` cheap.
        window[:] = [
            ts
            for ts in window
            if now - ts < max(_LOGIN_WINDOW_SECONDS, _LOGIN_CONSECUTIVE_WINDOW_SECONDS)
        ]
        self._attempts[key] = window
        return window

    def _consecutive_failures(self, key: str, now: float) -> int:
        """Number of failures for ``key`` within the consecutive window."""
        window = self._purge_window(key, now)
        return sum(
            1
            for ts in window
            if now - ts < _LOGIN_CONSECUTIVE_WINDOW_SECONDS
        )

    def _lockout_seconds_remaining(self, key: str, now: float) -> float:
        """Return seconds left on an active escalating lockout (0 if none)."""
        until = self._locked_until.get(key)
        if not until:
            return 0.0
        remaining = float(until) - float(now)
        if remaining <= 0:
            # Stale lockout — clean up.
            self._locked_until.pop(key, None)
            return 0.0
        return remaining

    def is_limited(self, client_ip: str) -> bool:
        """Return ``True`` if the IP is currently locked out OR has hit
        the sliding-window limit.

        Three independent reasons can flag ``is_limited == True``:

        1. An escalating lockout (5-min or 30-min) is in effect.
        2. The 5-in-60s sliding window has been reached.
        3. (Implicit) ``record_failure`` will raise the next tier
           when the next call comes in.

        Callers should consult ``is_limited`` *before* doing the
        expensive Argon2 verify; ``record_failure`` should be called
        on every failed attempt so the counter advances.
        """
        key = self._ip_key(client_ip)
        now = time.time()
        if self._lockout_seconds_remaining(key, now) > 0:
            return True
        self._purge_window(key, now)
        return len(self._attempts.get(key, [])) >= _LOGIN_MAX_ATTEMPTS

    def lockout_remaining(self, client_ip: str) -> float:
        """Return seconds left on an active escalating lockout (0 if none).

        Exposed for ``/login`` to surface "Try again in 4m 32s"
        hints. Returns 0 when no lockout is in effect.
        """
        return self._lockout_seconds_remaining(self._ip_key(client_ip), time.time())

    def record_failure(self, client_ip: str) -> None:
        """Record a failed login attempt for the given IP.

        Also advances the escalating tier:

        * 5 consecutive failures → 5 minute lockout.
        * 10 consecutive failures → 30 minute lockout.

        Idempotent: redundant failures within an already-active
        lockout will NOT extend the lockout (we never push the
        ``_locked_until`` further out from this method — the only
        way to reset is ``reset``).
        """
        key = self._ip_key(client_ip)
        now = time.time()
        window = self._purge_window(key, now)
        window.append(now)
        # Promote to a tier if we just crossed a threshold.
        consecutive = self._consecutive_failures(key, now)
        if consecutive >= 10:
            existing = self._lockout_seconds_remaining(key, now)
            if existing < _LOGIN_LOCKOUT_10_FAILS_SECONDS:
                self._locked_until[key] = now + _LOGIN_LOCKOUT_10_FAILS_SECONDS
        elif consecutive >= _LOGIN_MAX_ATTEMPTS:
            existing = self._lockout_seconds_remaining(key, now)
            if existing < _LOGIN_LOCKOUT_5_FAILS_SECONDS:
                self._locked_until[key] = now + _LOGIN_LOCKOUT_5_FAILS_SECONDS

    def reset(self, client_ip: str) -> None:
        """Clear the failure counter AND any active lockout for ``client_ip``.

        Called on a successful login so a returning user is not
        punished by their previous (legitimate) failed attempts.
        """
        key = self._ip_key(client_ip)
        self._attempts.pop(key, None)
        self._locked_until.pop(key, None)


# Process-wide rate limiter singleton.
_login_limiter = LoginRateLimiter()


# ---------------------------------------------------------------------------
# One-time recovery codes
# ---------------------------------------------------------------------------

_RECOVERY_CODE_COUNT = 10
_RECOVERY_CODE_LENGTH = 8  # characters, alphanumeric


def generate_recovery_codes(count: int = _RECOVERY_CODE_COUNT) -> list[str]:
    """Generate a set of one-time recovery codes.

    Each code is ``_RECOVERY_CODE_LENGTH`` alphanumeric characters.
    Codes are stored as Argon2id hashes so that a DB leak does not
    expose usable codes.
    """
    import string
    alphabet = string.ascii_uppercase + string.digits
    # Exclude easily confused characters.
    alphabet = "".join(c for c in alphabet if c not in "0O1I")
    codes = []
    for _ in range(count):
        code = "".join(secrets.choice(alphabet) for _ in range(_RECOVERY_CODE_LENGTH))
        codes.append(code)
    return codes


def hash_recovery_code(code: str) -> str:
    """Hash a recovery code with Argon2id for storage."""
    from argon2 import PasswordHasher
    return PasswordHasher().hash(code)


def verify_recovery_code(code: str, stored_hash: str) -> bool:
    """Verify a recovery code against its stored Argon2id hash."""
    from argon2 import PasswordHasher
    from argon2.exceptions import VerificationError, VerifyMismatchError
    try:
        PasswordHasher().verify(stored_hash, code)
        return True
    except (VerificationError, VerifyMismatchError, Exception):
        return False


def _consume_recovery_code(code: str) -> bool:
    """Try to consume a one-time recovery code.

    Looks up the code's Argon2id hash in the SessionStore's ``recovery_codes``
    table.  If found and not yet used, marks it as used and returns ``True``.
    Otherwise returns ``False``.

    This is a linear scan over a small table (≤10 rows), which is fine for
    the expected volume.
    """
    if _session_store is None or not code:
        return False
    try:
        # Ensure the recovery_codes table exists.
        _session_store._conn.execute(
            "CREATE TABLE IF NOT EXISTS recovery_codes ("
            "code_hash TEXT PRIMARY KEY, "
            "used INTEGER NOT NULL DEFAULT 0, "
            "used_at TEXT"
            ")"
        )
        rows = _session_store._conn.execute(
            "SELECT code_hash, used FROM recovery_codes WHERE used = 0"
        ).fetchall()
        for row in rows:
            if verify_recovery_code(code, row["code_hash"]):
                # Mark as used.
                _session_store._conn.execute(
                    "UPDATE recovery_codes SET used = 1, used_at = ? WHERE code_hash = ?",
                    (datetime.now(timezone.utc).isoformat(), row["code_hash"]),
                )
                return True
        return False
    except Exception:
        return False


def store_recovery_codes(codes: list[str]) -> None:
    """Hash and persist a batch of recovery codes into the SessionStore.

    Existing unused codes are cleared first so that only the latest batch
    is active.
    """
    if _session_store is None:
        return
    try:
        _session_store._conn.execute(
            "CREATE TABLE IF NOT EXISTS recovery_codes ("
            "code_hash TEXT PRIMARY KEY, "
            "used INTEGER NOT NULL DEFAULT 0, "
            "used_at TEXT"
            ")"
        )
        # Clear old unused codes.
        _session_store._conn.execute("DELETE FROM recovery_codes WHERE used = 0")
        for code in codes:
            code_hash = hash_recovery_code(code)
            _session_store._conn.execute(
                "INSERT OR IGNORE INTO recovery_codes (code_hash, used) VALUES (?, 0)",
                (code_hash,),
            )
    except Exception:
        pass


def list_recovery_codes() -> list[dict[str, Any]]:
    """Return metadata for all recovery codes (hashes only, never raw codes)."""
    if _session_store is None:
        return []
    try:
        _session_store._conn.execute(
            "CREATE TABLE IF NOT EXISTS recovery_codes ("
            "code_hash TEXT PRIMARY KEY, "
            "used INTEGER NOT NULL DEFAULT 0, "
            "used_at TEXT"
            ")"
        )
        rows = _session_store._conn.execute(
            "SELECT code_hash, used, used_at FROM recovery_codes ORDER BY rowid"
        ).fetchall()
        return [
            {
                "fingerprint": row["code_hash"][:16],
                "used": bool(int(row["used"])),
                "used_at": row["used_at"],
            }
            for row in rows
        ]
    except Exception:
        return []
