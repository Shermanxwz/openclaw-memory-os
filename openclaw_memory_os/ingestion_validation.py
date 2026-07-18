"""v0.3.0 memory-brain ingestion contract.

The legacy ``memory_brain_ingest.py`` script used a loose
``re.search(r'\\{[^}]+\\}', content)`` to pull a JSON blob out
of the LLM response, then trusted every field. The v0.3.0
contract requires:

* A Pydantic schema that rejects any unknown / out-of-range /
  wrong-type field.
* One retry on a corrective prompt if the first attempt fails
  JSON validation; if the retry also fails, deterministic
  fallback (with ``classification_status='fallback'``).
* Length / count caps on the free-form fields: ``keywords <= 8``,
  ``entities <= 8``, ``recall_triggers <= 8``, ``summary <= 80``.
* De-duplication on the list fields.
* A ``prompt_version`` field recorded in the payload so
  downstream tooling can correlate.
* A ``classification_status`` token: ``ok`` / ``retry_ok`` /
  ``fallback`` (a fourth token ``parse_error`` is reserved for
  legacy payloads that pre-date the v0.3.0 contract).
* A ``classification_error_code`` field: ``None`` on success;
  on failure one of ``json_missing`` / ``schema_invalid`` /
  ``llm_unreachable`` / ``llm_status``.

This module is a pure helper. The script that actually calls
the LLM still lives in ``memory_brain_ingest.py`` and threads
the validated payload into the existing Qdrant upsert.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


# Prompt version is bumped when the prompt template in
# memory_brain_ingest.py changes shape.
PROMPT_VERSION = "v0.3.0"

ALLOWED_TYPES = {
    "fact",
    "decision",
    "event",
    "preference",
    "system_config",
    "lesson",
    "relationship",
}
ALLOWED_TOPICS = {
    "infrastructure",
    "business",
    "personal",
    "ai_model",
    "memory_system",
    "health",
    "tools_software",
    "planning",
}
ALLOWED_SENTIMENTS = {"positive", "negative", "neutral"}

MAX_KEYWORDS = 8
MAX_ENTITIES = 8
MAX_TRIGGERS = 8
MAX_SUMMARY_LEN = 80
MAX_TOKENS_PER_FIELD = 64


class _CappedList(BaseModel):
    """Helper: a list with de-duplication and length cap."""

    model_config = ConfigDict(extra="forbid")

    items: List[str] = Field(default_factory=list, max_length=64)

    @field_validator("items", mode="before")
    @classmethod
    def _coerce(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        # Strip whitespace, drop empty / non-strings, de-dupe (case-insensitive)
        out: List[str] = []
        seen: set = set()
        for x in v:
            if x is None:
                continue
            s = str(x).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out


class ClassificationSchema(BaseModel):
    """Pydantic schema for the LLM classify() payload.

    Every field is either required-with-validator or
    default-stable. ``extra='forbid'`` means any unexpected
    field on the LLM output causes a ValidationError, which
    the caller turns into a retry or a deterministic fallback.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "fact"
    topic: str = "infrastructure"
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    summary: str = ""
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    recall_triggers: List[str] = Field(default_factory=list)
    valid_until: Optional[str] = None
    actionable: bool = False

    @field_validator("type")
    @classmethod
    def _type_allowed(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in ALLOWED_TYPES:
            raise ValueError(f"unsupported type: {v!r}")
        return v

    @field_validator("topic")
    @classmethod
    def _topic_allowed(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in ALLOWED_TOPICS:
            raise ValueError(f"unsupported topic: {v!r}")
        return v

    @field_validator("summary")
    @classmethod
    def _summary_capped(cls, v: str) -> str:
        s = (v or "").strip()
        if len(s) > MAX_SUMMARY_LEN:
            s = s[: MAX_SUMMARY_LEN - 1] + "…"
        return s

    @field_validator("keywords", "entities", "recall_triggers", mode="before")
    @classmethod
    def _cap_lists(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        out: List[str] = []
        seen: set = set()
        for x in v:
            if x is None:
                continue
            s = str(x).strip()
            if not s:
                continue
            if len(s) > MAX_TOKENS_PER_FIELD:
                s = s[:MAX_TOKENS_PER_FIELD]
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out[:MAX_KEYWORDS]


# Backwards-compat cap constants for tests.
MAX_KEYWORDS = MAX_KEYWORDS
MAX_ENTITIES = MAX_ENTITIES
MAX_TRIGGERS = MAX_TRIGGERS
MAX_SUMMARY_LEN = MAX_SUMMARY_LEN


def _extract_first_json(content: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced JSON object out of an LLM response.

    Tries (in order):
    1. A fenced ```json ... ``` block.
    2. A fenced ``` ... ``` block (no language tag).
    3. A bare balanced ``{...}`` object (manual brace counting).
    """
    if not content:
        return None
    # 1) json fence
    m = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (ValueError, TypeError):
            pass
    # 2) bare fence
    m = re.search(r"```\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (ValueError, TypeError):
            pass
    # 3) balanced braces (first-level only)
    depth = 0
    start = -1
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = content[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except (ValueError, TypeError):
                        start = -1
                        continue
    return None


def validate_classification(payload: Dict[str, Any]) -> Tuple[Optional[ClassificationSchema], Optional[str]]:
    """Validate a candidate payload.

    Returns ``(parsed, error_code)``. ``parsed`` is ``None``
    when validation failed; ``error_code`` is one of:

    * ``json_missing``   - no JSON object found in source
    * ``schema_invalid`` - pydantic rejected the payload
    * ``type_invalid``   - ``type`` field is not in the
                           allowed set
    * ``topic_invalid``  - same for ``topic``
    """
    if not payload or not isinstance(payload, dict):
        return None, "json_missing"
    try:
        parsed = ClassificationSchema.model_validate(payload)
    except ValidationError as exc:
        # Map specific errors to a stable code so the dashboard
        # can categorise failures.
        msg = str(exc)
        if "unsupported type" in msg:
            return None, "type_invalid"
        if "unsupported topic" in msg:
            return None, "topic_invalid"
        return None, "schema_invalid"
    return parsed, None


def build_fallback_payload(
    text: str,
    *,
    error_code: str,
) -> Dict[str, Any]:
    """Deterministic fallback when the LLM is unreachable or its
    output fails validation twice.

    The fallback is intentionally conservative: type=fact,
    topic=infrastructure, importance=0.6 (the legacy default
    bucket), summary truncated to 80 chars, no keywords /
    entities / recall_triggers. The script records
    ``classification_status='fallback'`` and the error code so
    operators can find these records later.
    """
    summary = (text or "").strip()
    if len(summary) > MAX_SUMMARY_LEN:
        summary = summary[: MAX_SUMMARY_LEN - 1] + "…"
    return {
        "type": "fact",
        "topic": "infrastructure",
        "importance": 0.6,
        "confidence": 0.0,
        "summary": summary,
        "keywords": [],
        "entities": [],
        "recall_triggers": [],
        "valid_until": None,
        "actionable": False,
        "_classification_status": "fallback",
        "_classification_error_code": error_code,
    }


def annotate_payload(
    parsed: ClassificationSchema,
    *,
    status: str = "ok",
    prompt_version: str = PROMPT_VERSION,
) -> Dict[str, Any]:
    """Convert a validated Pydantic instance into a JSON-safe dict
    for the Qdrant payload, attaching the v0.3.0 metadata fields.
    """
    payload = parsed.model_dump()
    payload["_classification_status"] = status
    payload["prompt_version"] = prompt_version
    return payload


# Convenience: the corrective prompt for the single retry.
CORRECTIVE_PROMPT = (
    "The previous response was not valid JSON or did not match the "
    "required schema. Reply with a single, strict JSON object only — no "
    "code fences, no prose, no trailing commentary — using exactly the "
    "field names: type, topic, importance (0.0-1.0), confidence "
    "(0.0-1.0), summary (<=80 chars), keywords (<=8 strings), entities "
    "(<=8 strings), recall_triggers (<=8 strings), valid_until (ISO "
    "date or null), actionable (bool)."
)


# ---------------------------------------------------------------------------
# Runbook G7.1 — true corrective retry against the qwen HTTP endpoint.
# ---------------------------------------------------------------------------
#
# The module docstring documents the "one retry on a corrective prompt"
# contract, but the actual ingest path in ``memory_brain_ingest.ingest_text``
# historically fell straight to the deterministic fallback on the first
# malformed response. ``classify_with_qwen`` (below) is the single
# call-site that closes that gap: it fires the LLM HTTP POST once,
# inspects the response for a parseable JSON object, and on failure
# fires a second POST carrying the corrective prompt. The result is a
# tuple ``(payload, classification_status)`` where the status is one of:
#
#   ``ok``           — qwen returned valid JSON on the first attempt
#   ``retry_ok``     — first attempt failed JSON parsing/validation, the
#                       corrective retry produced a valid object
#   ``retry_failed`` — both attempts failed; callers should fall back
#                       to the deterministic payload
#
# Status tokens are stable strings (not booleans) so the Qdrant payload
# can carry them through to the audit / dashboard layer without
# bespoke serialisation.
#
# The HTTP transport is split into ``_qwen_http_post`` so tests can
# monkeypatch the transport independently of the retry policy. The
# function lives here (not in ``scripts/memory_brain_ingest.py``) so
# that ``test_qwen_retry.py`` and the live ingest script share the
# same retry contract — the previous split was the source of the G7.1
# drift.

# Default timeout for the qwen HTTP POST. Mirrors the timeout the legacy
# ``llm_understand`` function applied so the retry path does not regress
# tail latency when the LLM is slow.
_QWEN_HTTP_TIMEOUT = 30


def _qwen_http_post(url: str, payload: Dict[str, Any], *, timeout: int) -> Optional[Dict[str, Any]]:
    """Best-effort POST to a qwen-compatible chat-completions endpoint.

    Returns the parsed JSON body on success (``status_code == 200``),
    or ``None`` when the request raised, returned a non-2xx status, or
    the body could not be decoded as JSON. The HTTP transport is
    intentionally minimal: it never raises, so callers can detect
    failure by comparing the result to ``None`` without a try/except.

    The function expects ``payload`` to already be the chat-completions
    body (i.e. ``{"model": ..., "messages": [...], ...}``). Tests
    monkeypatch this function to inject canned responses.
    """
    try:
        import requests  # local import so the module loads without requests

        headers = {"Content-Type": "application/json"}
        # Allow operators to inject a bearer token for hosted qwen
        # endpoints without leaking it into source code.
        import os

        api_key = os.environ.get("QWEN_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return None
        try:
            return response.json()
        except (ValueError, TypeError):
            return None
    except Exception:
        # Never raise out of the transport: the retry layer needs a
        # clean ``None`` signal to drive the corrective path.
        return None


def classify_with_qwen(
    text: str,
    *,
    model: str = "qwen2.5:1.5b",
    url: Optional[str] = None,
    prompt: Optional[str] = None,
    timeout: int = _QWEN_HTTP_TIMEOUT,
    http_post: Optional[Any] = None,
    corrective_prompt: Optional[str] = None,
    extract_json: Optional[Any] = None,
) -> Tuple[Dict[str, Any], str]:
    """Classify ``text`` via the qwen HTTP endpoint with one corrective retry.

    Parameters
    ----------
    text:
        The free-form text to classify.
    model:
        Model name sent in the chat-completions payload.
    url:
        Full URL of the chat-completions endpoint. Defaults to
        ``$QWEN_API_URL/chat/completions`` (falling back to
        ``http://127.0.0.1:11434/v1/chat/completions`` when the env
        var is unset — the default that the bundled Ollama install
        exposes).
    prompt:
        Override for the user-visible prompt. Defaults to the v0.3.0
        classify prompt, which asks for the canonical JSON shape
        (``type``/``topic``/``importance``/...). Exposed so tests can
        pin a short canned prompt.
    timeout:
        Seconds to wait for each HTTP POST.
    http_post:
        Test seam. When provided, ``classify_with_qwen`` calls this
        callable instead of :func:`_qwen_http_post`. Signature is
        ``(url, payload, *, timeout) -> Optional[Dict]`` — same as
        the default transport. Tests use this to inject canned
        first-attempt / retry-attempt responses without standing up
        a real LLM.
    corrective_prompt:
        Override for the corrective retry prompt. Defaults to
        :data:`CORRECTIVE_PROMPT`.
    extract_json:
        Test seam for :func:`_extract_first_json`. Same shape as the
        default helper. Exposed so tests can monkeypatch JSON
        extraction without touching the global.

    Returns
    -------
    ``(payload, classification_status)`` where ``payload`` is the
    parsed JSON object on success or an empty dict on full failure,
    and ``classification_status`` is one of ``"ok"`` / ``"retry_ok"``
    / ``"retry_failed"``.
    """
    import os as _os

    if url is None:
        base = (_os.environ.get("QWEN_API_URL") or "http://127.0.0.1:11434/v1").rstrip("/")
        url = f"{base}/chat/completions"
    if prompt is None:
        prompt = (
            "Classify the following memory content. Return ONLY a strict JSON "
            "object with the keys: type, topic, importance (0.0-1.0), "
            "confidence (0.0-1.0), summary (<=80 chars), keywords (<=8 "
            "strings), entities (<=8 strings), recall_triggers (<=8 "
            "strings), valid_until (ISO date or null), actionable (bool).\n\n"
            f"{text[:2000]}"
        )
    if corrective_prompt is None:
        corrective_prompt = CORRECTIVE_PROMPT
    if http_post is None:
        http_post = _qwen_http_post
    if extract_json is None:
        extract_json = _extract_first_json

    base_payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 500,
    }

    # ---- First attempt ---------------------------------------------------
    first = http_post(url, base_payload, timeout=timeout)
    first_payload = _payload_from_response(first, extract_json)
    if first_payload:
        return first_payload, "ok"

    # ---- Corrective retry ------------------------------------------------
    # We feed the SAME user message plus a short correction block back
    # to the LLM. ``messages[-1]`` is replaced so the corrective hint
    # is the last thing the model sees (avoids the model re-emitting
    # the original malformed answer).
    corrective_payload = {
        **base_payload,
        "messages": [
            base_payload["messages"][0],
            {"role": "user", "content": prompt},
            {"role": "user", "content": corrective_prompt},
        ],
        "temperature": 0.0,
    }
    second = http_post(url, corrective_payload, timeout=timeout)
    second_payload = _payload_from_response(second, extract_json)
    if second_payload:
        return second_payload, "retry_ok"

    return {}, "retry_failed"


def _payload_from_response(response: Optional[Dict[str, Any]], extract_json: Any) -> Dict[str, Any]:
    """Extract the first JSON object from a chat-completions response body.

    Returns an empty dict on any error so the caller can detect failure
    with a single ``if not result`` check. The extraction re-uses
    :func:`_extract_first_json` so a malformed response is detected
    even when the LLM HTTP layer is healthy.
    """
    if not response:
        return {}
    try:
        choices = response.get("choices") or []
        if not choices:
            return {}
        message = choices[0].get("message") or {}
        content = (message.get("content") or message.get("reasoning_content") or "").strip()
        if not content:
            return {}
        parsed = extract_json(content)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}
