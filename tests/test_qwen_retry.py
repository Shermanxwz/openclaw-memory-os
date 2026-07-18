"""Tests for the Runbook G7.1 qwen corrective retry.

The module docstring of ``openclaw_memory_os.ingestion_validation``
documents the "one retry on a corrective prompt" contract, but the
actual ingest path historically fell straight to the deterministic
fallback on the first malformed LLM response. The tests below pin
the corrected behaviour so the drift cannot return.

Coverage:

* ``classify_with_qwen`` returns ``("ok", payload)`` when the first
  HTTP attempt produces a parseable JSON object.
* ``classify_with_qwen`` returns ``("retry_ok", payload)`` when the
  first attempt fails and the corrective retry succeeds.
* ``classify_with_qwen`` returns ``("retry_failed", {})`` when both
  attempts fail.
* The corrective payload contains the original text plus the
  corrective prompt (so a regression that drops either input is
  caught).
* ``memory_brain_ingest.llm_understand`` propagates the retry
  status into the payload's ``_classification_status`` field.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


from openclaw_memory_os.ingestion_validation import (
    CORRECTIVE_PROMPT,
    classify_with_qwen,
)


# ---------------------------------------------------------------------------
# Test seam: a canned ``http_post`` and ``extract_json`` so the unit
# tests can simulate the LLM behaviour without standing up a real
# server. Each test patches ``http_post`` and ``extract_json`` via
# the keyword arguments of ``classify_with_qwen`` — the wrapper was
# designed for exactly this shape so the tests stay close to the
# production code path.
# ---------------------------------------------------------------------------


def _ok_response(payload: dict) -> dict:
    """Build a chat-completions response body wrapping ``payload``.

    The body is plain JSON (no fence) so the test can pass either
    :func:`_identity_extract` (which expects raw JSON) or
    :func:`_extract_first_json` (which strips fences) without
    changing the helper.
    """
    return {
        "id": "test-1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": _to_json(payload),
                },
            }
        ],
    }


def _fenced_ok_response(payload: dict) -> dict:
    r"""Build a chat-completions response body with the JSON wrapped in a ``\`\`\`json fence.

    Mirrors how real LLMs (qwen included) format chat-completions
    output, so the wrapper's :func:`_extract_first_json` path is
    exercised end-to-end. Used by the tests that want to verify the
    wrapper handles the fenced-response shape that the production
    LLM emits.
    """
    return {
        "id": "test-1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "```json\n" + _to_json(payload) + "\n```",
                },
            }
        ],
    }


def _bad_response() -> dict:
    """Build a chat-completions response body with non-JSON content."""
    return {
        "id": "test-1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Sorry, I cannot help with that.",
                },
            }
        ],
    }


def _to_json(payload: dict) -> str:
    import json
    return json.dumps(payload, ensure_ascii=False)


def _identity_extract(content: str):
    """Pass-through extractor used by tests that already pass JSON in."""
    import json
    if not content:
        return None
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Tests for classify_with_qwen
# ---------------------------------------------------------------------------


def test_first_attempt_succeeds_status_ok():
    """First attempt produces valid JSON → status="ok", payload is the parsed dict."""

    payload = {
        "type": "fact",
        "topic": "infrastructure",
        "importance": 0.7,
        "summary": "first-try success",
    }
    seen_payloads: list[dict] = []

    def fake_http(url, body, *, timeout):
        seen_payloads.append(body)
        return _ok_response(payload)

    result, status = classify_with_qwen(
        "hello world",
        url="http://example.test/chat/completions",
        http_post=fake_http,
        extract_json=_identity_extract,
    )

    assert status == "ok"
    assert result == payload
    # Only one HTTP attempt was made; no corrective retry fired.
    assert len(seen_payloads) == 1
    # The payload is a chat-completions body with the user message.
    assert seen_payloads[0]["model"] == "qwen2.5:1.5b"
    assert seen_payloads[0]["messages"][0]["role"] == "user"


def test_first_attempt_fails_retry_succeeds_status_retry_ok():
    """First attempt fails, second attempt succeeds → status="retry_ok"."""

    good = {
        "type": "decision",
        "topic": "business",
        "importance": 0.85,
        "summary": "retry success",
    }
    seen_payloads: list[dict] = []

    def fake_http(url, body, *, timeout):
        seen_payloads.append(body)
        # First call returns garbage; second returns the real payload.
        if len(seen_payloads) == 1:
            return _bad_response()
        return _ok_response(good)

    result, status = classify_with_qwen(
        "please classify",
        url="http://example.test/chat/completions",
        http_post=fake_http,
        extract_json=_identity_extract,
    )

    assert status == "retry_ok"
    assert result == good
    # Exactly two attempts: the original + the corrective retry.
    assert len(seen_payloads) == 2


def test_both_attempts_fail_status_retry_failed():
    """Both attempts fail → status="retry_failed", empty dict."""

    seen_payloads: list[dict] = []

    def fake_http(url, body, *, timeout):
        seen_payloads.append(body)
        return _bad_response()

    result, status = classify_with_qwen(
        "any text",
        url="http://example.test/chat/completions",
        http_post=fake_http,
        extract_json=_identity_extract,
    )

    assert status == "retry_failed"
    assert result == {}
    # Exactly two attempts: the original + the corrective retry. We
    # do NOT keep retrying — the Runbook G7.1 contract is "one
    # retry, no more".
    assert len(seen_payloads) == 2


def test_corrective_prompt_includes_original_context():
    """The retry payload must carry the original text plus the corrective prompt.

    The corrective prompt must appear AFTER the original user
    message (so the model sees the correction as the most recent
    context). The original ``text`` must also be present so the
    model still has the substance of the request.
    """
    seen_payloads: list[dict] = []
    text = "## decide whether to migrate auth to oauth2"

    def fake_http(url, body, *, timeout):
        seen_payloads.append(body)
        if len(seen_payloads) == 1:
            return _bad_response()
        return _ok_response(
            {"type": "decision", "topic": "infrastructure", "summary": "migrate"}
        )

    classify_with_qwen(
        text,
        url="http://example.test/chat/completions",
        http_post=fake_http,
        extract_json=_identity_extract,
    )

    assert len(seen_payloads) == 2
    retry = seen_payloads[1]
    messages = retry["messages"]
    # Three messages: original prompt + original text replay + correction.
    # The exact ordering depends on the wrapper's implementation;
    # we assert that all three pieces of context are present.
    joined = "\n".join(str(m.get("content", "")) for m in messages)
    assert text in joined, "original text missing from corrective payload"
    assert (
        "JSON" in joined
    ), "corrective prompt must mention 'JSON' so the model knows to retry as JSON"
    # The last message should be the corrective hint so the model
    # treats it as the most recent context.
    assert messages[-1].get("content")
    assert (
        "JSON" in messages[-1]["content"]
        or "strict" in messages[-1]["content"]
        or "json" in messages[-1]["content"].lower()
    )
    # The retry should run at temperature 0.0 (deterministic retry).
    assert retry.get("temperature") == 0.0


def test_corrective_prompt_default_uses_module_constant():
    """When no override is given, the module-level CORRECTIVE_PROMPT is used."""
    seen_payloads: list[dict] = []

    def fake_http(url, body, *, timeout):
        seen_payloads.append(body)
        if len(seen_payloads) == 1:
            return _bad_response()
        return _ok_response(
            {"type": "fact", "topic": "infrastructure", "summary": "x"}
        )

    classify_with_qwen(
        "hello",
        url="http://example.test/chat/completions",
        http_post=fake_http,
        extract_json=_identity_extract,
    )

    last = seen_payloads[1]
    last_content = last["messages"][-1]["content"]
    assert last_content == CORRECTIVE_PROMPT


def test_http_transport_returns_none_is_treated_as_failure():
    """``http_post`` returning ``None`` (transport failure) must
    trigger the retry path — the wrapper never conflates transport
    failures with malformed responses.

    Both attempts return ``None`` → ``retry_failed``.
    """
    calls = {"count": 0}

    def always_fail(url, body, *, timeout):
        calls["count"] += 1
        return None

    result, status = classify_with_qwen(
        "test",
        url="http://example.test/chat/completions",
        http_post=always_fail,
        extract_json=_identity_extract,
    )

    assert status == "retry_failed"
    assert result == {}
    assert calls["count"] == 2  # exactly one retry, no more


def test_partial_malformed_response_triggers_retry():
    """An LLM response with parseable text but no JSON inside still triggers the retry."""

    def fake_http(url, body, *, timeout):
        # Content is parseable but contains no JSON object.
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "42"}}
            ]
        }

    calls = {"count": 0}

    def counting_fake(url, body, *, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            return fake_http(url, body, timeout=timeout)
        return _ok_response(
            {"type": "fact", "topic": "infrastructure", "summary": "recovered"}
        )

    result, status = classify_with_qwen(
        "anything",
        url="http://example.test/chat/completions",
        http_post=counting_fake,
        extract_json=_identity_extract,
    )

    assert status == "retry_ok"
    assert result["summary"] == "recovered"


# ---------------------------------------------------------------------------
# Tests for memory_brain_ingest.llm_understand integration
# ---------------------------------------------------------------------------


def _load_memory_brain_ingest_module():
    """Load ``scripts/memory_brain_ingest.py`` as a stand-alone module.

    Mirrors the pattern in ``test_qwen_validation`` so we can
    exercise the private path without polluting ``sys.path`` for
    the rest of the test suite.
    """

    repo_root = Path(__file__).resolve().parent.parent
    ingest_path = repo_root / "scripts" / "memory_brain_ingest.py"
    spec = importlib.util.spec_from_file_location(
        "memory_brain_ingest", ingest_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_qwen_http(module, *, fake_http, fake_extract=None):
    """Install a fake ``classify_with_qwen``-shaped function on the module.

    ``memory_brain_ingest.llm_understand`` looks up
    ``classify_with_qwen`` by name in its own globals, so monkey-
    patching the module attribute is enough to redirect the HTTP
    layer in the integration test below.

    When ``fake_extract`` is ``None`` we fall back to the real
    :func:`_extract_first_json` so the test exercises the
    wrapper's full JSON extraction pipeline (fenced, unfenced,
    bare-object). The caller can still pass an identity extractor
    to keep the test focused on the retry policy alone.
    """
    if fake_extract is None:
        from openclaw_memory_os.ingestion_validation import _extract_first_json
        fake_extract = _extract_first_json

    def fake_classify(text, **kwargs):
        url = kwargs.get("url") or "http://example.test/chat/completions"
        model = kwargs.get("model") or "qwen2.5:1.5b"
        timeout = kwargs.get("timeout", 120)
        prompt = kwargs.get("prompt")
        corrective_prompt = kwargs.get("corrective_prompt")

        base_payload = {
            "model": model,
            "messages": (
                [{"role": "user", "content": prompt}] if prompt else []
            ),
            "temperature": 0.1,
            "max_tokens": 500,
        }
        first = fake_http(url, base_payload, timeout=timeout)
        first_payload = _extract_from(first, fake_extract)
        if first_payload:
            return first_payload, "ok"

        corrective_payload = {
            **base_payload,
            "messages": ([{"role": "user", "content": prompt}] if prompt else [])
            + [
                {"role": "user", "content": corrective_prompt or CORRECTIVE_PROMPT},
            ],
            "temperature": 0.0,
        }
        second = fake_http(url, corrective_payload, timeout=timeout)
        second_payload = _extract_from(second, fake_extract)
        if second_payload:
            return second_payload, "retry_ok"
        return {}, "retry_failed"

    def _extract_from(response, extractor):
        if not response:
            return {}
        try:
            choices = response.get("choices") or []
            content = (
                choices[0].get("message", {}).get("content", "")
                if choices
                else ""
            )
            if not content:
                return {}
            parsed = extractor(content)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    module.classify_with_qwen = fake_classify
    return fake_classify


def test_ingest_text_uses_retry_status(monkeypatch):
    """When ``classify_with_qwen`` fires the corrective retry,
    ``llm_understand`` must propagate ``_classification_status =
    "retry_ok"`` so the audit / dashboard can see the retry.

    We patch ``memory_brain_ingest.classify_with_qwen`` to return
    the canonical ``("retry_ok", payload)`` tuple, then call
    ``llm_understand`` and assert the propagated status.
    """
    module = _load_memory_brain_ingest_module()
    captured = {"called": 0}

    def fake_http(url, body, *, timeout):
        captured["called"] += 1
        # First call: garbage. Second call: real payload.
        if captured["called"] == 1:
            return _bad_response()
        return _ok_response(
            {
                "type": "fact",
                "topic": "infrastructure",
                "summary": "retry-recovered",
                "sentiment": "neutral",
                "recall_triggers": ["trigger"],
                "prerequisite_memories": [],
                "valid_until": None,
                "actionable": False,
                "entities": [],
                "keywords": [],
                "importance": 0.6,
            }
        )

    _patch_qwen_http(module, fake_http=fake_http)
    # Force the module to consult our patched ``classify_with_qwen``
    # in ``llm_understand``.
    monkeypatch.setattr(module, "classify_with_qwen", module.classify_with_qwen)

    out = module.llm_understand(
        "sample memory content for retry test", task="classify"
    )

    assert isinstance(out, dict)
    assert out.get("_classification_status") == "retry_ok"
    # The status must flow through even when the schema passes.
    assert out.get("summary") == "retry-recovered"
    assert out.get("prompt_version") == module.PROMPT_VERSION
    # The retry fired exactly twice (original + corrective).
    assert captured["called"] == 2


def test_ingest_text_retry_failed_propagates_status(monkeypatch):
    """When ``classify_with_qwen`` exhausts both attempts,
    ``llm_understand`` must emit a fallback carrying
    ``_classification_status = "retry_failed"``."""

    module = _load_memory_brain_ingest_module()

    def always_bad(url, body, *, timeout):
        return _bad_response()

    _patch_qwen_http(module, fake_http=always_bad)
    monkeypatch.setattr(module, "classify_with_qwen", module.classify_with_qwen)

    out = module.llm_understand(
        "text that will trigger retry_failed", task="classify"
    )

    assert isinstance(out, dict)
    assert out.get("_classification_status") == "retry_failed"
    # Fallback shape: empty triggers / prerequisite_memories so the
    # downstream Qdrant upsert can carry them without raising.
    assert out.get("recall_triggers") == []
    assert out.get("prerequisite_memories") == []