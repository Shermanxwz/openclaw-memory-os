"""Runtime configuration loaded from environment variables.

All values are intentionally read at runtime so the same image can be used
in dev, demo, and production with different env files. No secrets are baked
into the image. See ``.env.example`` for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional


def _project_root() -> Path:
    # openclaw_memory_os/config.py -> repo root
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings.

    Attributes:
        memory_os_token: If set, all routes except ``/health`` require either
            ``Authorization: Bearer <token>`` or the browser login form.
        qdrant_url: Optional Qdrant gRPC/HTTP endpoint. When unset or empty,
            the sample backend is used.
        qdrant_collection: Qdrant collection name to query.
        qdrant_api_key: Optional Qdrant API key. Never logged.
        sample_data_path: Path to the bundled sample JSON file used when
            Qdrant is not configured.
        max_recall_results: Hard cap on recall-test result size.
        recency_half_life_days: Days after which a memory's recency boost
            halves.
        superseded_penalty: Multiplier applied to superseded memories.
        expired_penalty: Multiplier applied to expired memories.
        env_file: Optional path to a dotenv-style file to load on startup.
    """

    memory_os_token: Optional[str] = None
    memory_os_password: Optional[str] = None
    memory_os_totp_secret: Optional[str] = None
    qdrant_url: Optional[str] = None
    qdrant_collection: str = "openclaw_memories"
    qdrant_secondary_collections: List[str] = field(default_factory=list)
    qdrant_api_key: Optional[str] = None
    sample_data_path: Path = field(default_factory=lambda: _project_root() / "data" / "sample_memories.json")
    max_recall_results: int = 25
    recency_half_life_days: float = 30.0
    superseded_penalty: float = 0.25
    expired_penalty: float = 0.1
    importance_boost_scale: float = 0.6
    # Recall fallback: when the active-only pass yields fewer than this many
    # hits, automatically expand the search to include superseded memories
    # as lower-priority results. Set to a large value (e.g. 10**9) or
    # RECALL_FALLBACK_SUPERSEDED=off to disable. ``include_superseded=True``
    # in the request always wins and skips the fallback logic.
    recall_fallback_superseded: bool = True
    recall_fallback_superseded_min_results: int = 5

    # --- Feature flags (v0.3.0.x) ---------------------------------------
    # Each flag has a safe default that preserves the previous behaviour
    # so a missing env var never changes runtime semantics. See
    # ``docs/feature-flags.md`` for the contract.

    #: Enable the v0.3.0 unified ``RetrievalEngine`` path
    #: (dense + lexical + RRF + feature rerank + Active-first /
    #: Superseded-fallback). When ``False`` the legacy
    #: ``ranking.build_recall_response`` scorer is used so the operator
    #: can revert to the v0.2.x behaviour without redeploying.
    retrieval_engine_v2: bool = True

    #: Persist recall feedback via the structured
    #: ``recall_feedback_v030`` schema (``recall_runs`` /
    #: ``recall_results`` / ``feedback_events`` tables). When ``False``
    #: the legacy audit-log path is used instead, matching v0.2.x.
    structured_feedback: bool = True

    #: Enable the policy-evolution cycle (candidate search + shadow +
    #: auto-promote / rollback). When ``False`` the evolution endpoints
    #: are still wired but the cycle is a no-op (``status="disabled"``),
    #: matching the pre-v0.3.0 behaviour where policy was static.
    evolution_enabled: bool = True

    #: Run shadow comparison in the background (does not promote). When
    #: ``False``, the evolution cycle skips the shadow stage entirely,
    #: going straight to a deterministic candidate verdict. ``SHADOW_ENABLED=off``
    #: is the safe default for evaluations / CI runs that need
    #: reproducible metrics.
    shadow_enabled: bool = True

    #: Enable password + TOTP authentication. When ``False``, the login
    #: flow degrades to the legacy shared bearer token mode regardless of
    #: whether ``MEMORY_OS_PASSWORD`` / ``MEMORY_OS_TOTP_SECRET`` are set.
    #: The default preserves the current behaviour: when a password is
    #: configured the password+TOTP path is preferred; otherwise the
    #: bearer token path is used. Set ``PASSWORD_TOTP_AUTH=off`` to force
    #: the bearer path even when a password is configured.
    password_totp_auth: bool = True

    env_file: Optional[Path] = None

    @property
    def auth_enabled(self) -> bool:
        """Auth is enabled when a non-empty token *or* password is configured.

        The :attr:`password_totp_auth` flag is *additive*: when
        ``False``, it forces the legacy bearer token path even when a
        password is configured. Bearer-token-only auth (no password)
        remains enabled in either case.
        """
        if self.password_totp_auth:
            return bool(self.memory_os_token or self.memory_os_password)
        # Force the legacy bearer path: token-only when no token is set
        # is still treated as "auth disabled" so the OS stays usable in
        # local dev.
        return bool(self.memory_os_token)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build a :class:`Settings` from process env (and optional dotenv file)."""

    # Lightweight dotenv loader: we don't want to hard-depend on a specific
    # library version. ``python-dotenv`` is still installed for end users.
    env_file = os.environ.get("MEMORY_OS_ENV_FILE")
    env_path: Optional[Path] = None
    if env_file:
        p = Path(env_file)
        if p.exists():
            env_path = p
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Existing process env wins; dotenv only fills gaps.
                os.environ.setdefault(key, value)

    sample_path_env = os.environ.get("MEMORY_OS_SAMPLE_PATH")
    sample_path = Path(sample_path_env) if sample_path_env else _project_root() / "data" / "sample_memories.json"

    return Settings(
        memory_os_token=os.environ.get("MEMORY_OS_TOKEN") or None,
        memory_os_password=os.environ.get("MEMORY_OS_PASSWORD") or None,
        memory_os_totp_secret=os.environ.get("MEMORY_OS_TOTP_SECRET") or None,
        qdrant_url=os.environ.get("QDRANT_URL") or None,
        qdrant_collection=os.environ.get("QDRANT_COLLECTION", "openclaw_memories"),
        qdrant_secondary_collections=[c.strip() for c in os.environ.get("QDRANT_SECONDARY_COLLECTIONS", "").split(",") if c.strip()],
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
        sample_data_path=sample_path,
        max_recall_results=int(os.environ.get("MEMORY_OS_MAX_RECALL", "25")),
        recency_half_life_days=float(os.environ.get("MEMORY_OS_RECENCY_HALF_LIFE", "30")),
        superseded_penalty=float(os.environ.get("MEMORY_OS_SUPERSEDED_PENALTY", "0.25")),
        expired_penalty=float(os.environ.get("MEMORY_OS_EXPIRED_PENALTY", "0.1")),
        importance_boost_scale=float(os.environ.get("MEMORY_OS_IMPORTANCE_BOOST", "0.6")),
        # RECALL_FALLBACK_SUPERSEDED=off / 0 / false / no disables fallback.
        # Default: on. We always honor an explicit request flag
        # (``include_superseded=True``) regardless of this setting.
        recall_fallback_superseded=_env_flag(
            "RECALL_FALLBACK_SUPERSEDED", default="on"
        ),
        recall_fallback_superseded_min_results=int(
            os.environ.get("RECALL_FALLBACK_SUPERSEDED_MIN_RESULTS", "5")
        ),
        # --- Feature flags (v0.3.0.x) -----------------------------------
        retrieval_engine_v2=_env_flag("RETRIEVAL_ENGINE_V2", default="on"),
        structured_feedback=_env_flag("STRUCTURED_FEEDBACK", default="on"),
        evolution_enabled=_env_flag("EVOLUTION_ENABLED", default="on"),
        shadow_enabled=_env_flag("SHADOW_ENABLED", default="on"),
        password_totp_auth=_env_flag("PASSWORD_TOTP_AUTH", default="on"),
        env_file=env_path,
    )


def _env_flag(name: str, *, default: str = "off") -> bool:
    """Parse a boolean-ish env var. Empty / unset falls back to ``default``.

    Accepts the canonical 1/0/true/false/yes/no/on/off spellings in any case.
    Used for settings where the safe default is "off" but we still want a
    human-friendly ``on`` / ``off`` setter.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        raw = default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def reset_settings_cache() -> None:
    """Clear the cached settings. Used by tests and CLI tools."""
    get_settings.cache_clear()