"""Tests for scripts/_qdrant_helpers.py — the Qdrant writeback helpers.

These tests focus on the P0 fix: ``coerce_point_id`` is the single
chokepoint that ensures integer IDs reach Qdrant as native ints (not
strings) so the /points endpoints stop returning HTTP 400. We don't
hit a real Qdrant instance here — the helper's HTTP code path is
exercised by integration tests in the deploy stack. The behaviour we
*can* cover offline is the ID contract.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"


def _load_helpers():
    spec = importlib.util.spec_from_file_location("qdrant_helpers", SCRIPTS_DIR / "_qdrant_helpers.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_coerce_point_id_digit_string_becomes_int():
    mod = _load_helpers()
    # This is the P0 regression case: previously the helper sent
    # "62" as a string and Qdrant returned 400.
    assert mod.coerce_point_id("62") == 62
    assert isinstance(mod.coerce_point_id("62"), int)


def test_coerce_point_id_int_passthrough():
    mod = _load_helpers()
    assert mod.coerce_point_id(62) == 62
    assert mod.coerce_point_id(0) == 0
    assert mod.coerce_point_id(-42) == -42
    assert isinstance(mod.coerce_point_id(62), int)


def test_coerce_point_id_preserves_uuid():
    mod = _load_helpers()
    uuid = "001225e5-81c9-def5-33de-1fdbe016d825"
    assert mod.coerce_point_id(uuid) == uuid
    assert isinstance(mod.coerce_point_id(uuid), str)


def test_coerce_point_id_preserves_prefixed_string():
    mod = _load_helpers()
    # Prefixed IDs like ``mem-0007`` or hashes must NOT be coerced;
    # they would otherwise become integers and lose the prefix.
    assert mod.coerce_point_id("mem-0007") == "mem-0007"
    assert mod.coerce_point_id("abc123") == "abc123"
    assert mod.coerce_point_id("id-1-2-3") == "id-1-2-3"


def test_coerce_point_id_handles_negative_integers():
    mod = _load_helpers()
    # Qdrant accepts signed 64-bit ints; the helper should match.
    assert mod.coerce_point_id("-42") == -42


def test_coerce_point_id_bool_is_not_treated_as_int():
    """Python bools are a subclass of int; we must avoid coercing them
    to ``1``/``0`` silently because that would corrupt payloads."""
    mod = _load_helpers()
    assert mod.coerce_point_id(True) == "True"
    assert mod.coerce_point_id(False) == "False"


def test_coerce_point_id_none_and_unknown_fall_back_to_string():
    mod = _load_helpers()
    assert mod.coerce_point_id(None) == "None"
    # Tuples / lists / dicts are not real point IDs but the helper
    # should not crash — they stringify predictably.
    assert mod.coerce_point_id((1, 2)) == "(1, 2)"


def test_coerce_point_id_empty_string_passes_through():
    mod = _load_helpers()
    # Empty string is not a digit string, so it should pass through.
    assert mod.coerce_point_id("") == ""


class _FakeHTTPResponse:
    """Minimal stand-in for ``http.client.HTTPResponse``.

    The Qdrant helpers only call ``.read()`` on the returned object, so
    a tiny mock is enough to verify the request body they send.
    """

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:  # pragma: no cover - trivial
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_update_payloads_sends_int_id_not_string(monkeypatch, capsys):
    """P0 regression: a digit-string ID in an update payload must be
    coerced to a native int before the PUT /points body is built.

    Previously the script stringified the id, Qdrant replied with 400,
    and the helper swallowed the error. This test fails if the body
    contains a string id for a pure-numeric input.
    """
    mod = _load_helpers()

    # get_point response: a point with id=42, vector, and a payload.
    point_payload = json.dumps(
        {"result": [{"id": 42, "vector": [0.1, 0.2], "payload": {"tier": "working"}}]}
    ).encode("utf-8")
    # PUT /points/... response: empty success body
    put_response = b'{"result": {"operation_id": 1, "status": "completed"}}'

    responses = [point_payload, put_response]

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        # Snapshot the body sent to Qdrant so the assertion can
        # inspect it.
        if not hasattr(fake_urlopen, "captured"):
            fake_urlopen.captured = []  # type: ignore[attr-defined]
        fake_urlopen.captured.append(  # type: ignore[attr-defined]
            json.loads(req.data.decode("utf-8"))
        )
        return _FakeHTTPResponse(responses.pop(0))

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    # Avoid sleeping in the loop.
    written = mod.update_payloads(
        "openclaw_memory_os",
        [{"id": "42", "status": "superseded", "superseded_by": "43"}],
        sleep_between=0,
    )
    assert written == 1, "expected the upsert to be counted as written"

    # get_point body should carry the int form
    get_body = fake_urlopen.captured[0]  # type: ignore[attr-defined]
    assert get_body["ids"] == [42]
    # upsert body should also carry the int form
    put_body = fake_urlopen.captured[1]  # type: ignore[attr-defined]
    assert put_body["points"][0]["id"] == 42
    assert isinstance(put_body["points"][0]["id"], int)
    # merged payload should include our new fields
    assert put_body["points"][0]["payload"]["status"] == "superseded"
    assert put_body["points"][0]["payload"]["superseded_by"] == "43"


def test_update_payloads_logs_http_400(monkeypatch, capsys):
    """The helper must surface HTTP 400 instead of swallowing it silently."""
    mod = _load_helpers()

    class _HTTP400(urllib_error := __import__("urllib.error", fromlist=["HTTPError"]).HTTPError):
        def __init__(self):
            super().__init__(url="http://test/points", code=400, msg="Bad Request", hdrs=None, fp=io.BytesIO(b'{"status":{"error":"Validation error"}}'))

        def read(self) -> bytes:
            return b'{"status":{"error":"Validation error"}}'

    # First call (get_point) succeeds, second call (PUT) returns 400.
    get_body = json.dumps(
        {"result": [{"id": 1, "vector": [0.0], "payload": {}}]}
    ).encode("utf-8")

    call = {"n": 0}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        call["n"] += 1
        if call["n"] == 1:
            return _FakeHTTPResponse(get_body)
        raise _HTTP400()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    written = mod.update_payloads(
        "openclaw_memory_os",
        [{"id": 1, "status": "expired"}],
        sleep_between=0,
    )
    assert written == 0
    captured = capsys.readouterr()
    assert "HTTP 400" in captured.out
    assert "upsert failed" in captured.out


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
