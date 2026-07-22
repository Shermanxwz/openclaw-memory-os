"""P0-5: privacy-safe logging — no raw content / text previews in JSONL log
files or cluster dumps.

Audit-report-1 found that ``scripts/memory_brain_ingest.py`` wrote
``"preview": content[:500]`` into ``/var/log/openclaw-memory-brain-errors.jsonl``
and ``scripts/dedup_cron.py`` wrote
``"text_preview": (payload.get("content") or payload.get("text") or "")[:200]``
into the cluster JSON dump. Both landed on disk via nightly cron and
were P0 privacy leaks per Runbook G7.4 / privacy-clean contract.

These tests enforce the fix: neither script may emit a ``preview`` /
``text_preview`` field, and each must instead emit a length indicator
(``content_len`` / ``text_len``) so the dashboard / audit can still see
"this memory is 12 KB" without seeing the body.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"

MEMORY_BRAIN_INGEST = SCRIPTS_DIR / "memory_brain_ingest.py"
DEDUP_CRON = SCRIPTS_DIR / "dedup_cron.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_comments_and_strings(src: str) -> str:
    """Return a copy of ``src`` with comments and string literals masked.

    We use this when scanning for the offending ``"preview":`` /
    ``"text_preview":`` keys so that docstring / comment mentions of
    the old name do not produce false positives. The actual code path
    that emits JSON / writes to a log file is always outside string
    literals, so this filter is safe.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return _regex_strip(src)

    masked_chars = list(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for (start, end) in _node_spans(node, src):
                for i in range(start, end):
                    if masked_chars[i] != "\n":
                        masked_chars[i] = " "
    return "".join(masked_chars)


def _node_spans(node: ast.AST, src: str) -> list[tuple[int, int]]:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return []
    lines = src.splitlines(keepends=True)
    start_line, start_col = node.lineno, node.col_offset
    end_line, end_col = node.end_lineno, node.end_col_offset
    start = sum(len(line) for line in lines[: start_line - 1]) + start_col
    end = sum(len(line) for line in lines[: end_line - 1]) + end_col
    return [(start, end)]


def _regex_strip(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"#[^\n]*", "", src)
    return src


def _kwarg_keys(node: ast.Call) -> set[str]:
    keys: set[str] = set()
    for kw in node.keywords:
        if kw.arg is None:
            continue
        keys.add(kw.arg)
    return keys


def _dict_keys(node: ast.Dict) -> set[str]:
    keys: set[str] = set()
    for k in node.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            keys.add(k.value)
    return keys


def _is_len_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
    )


# ---------------------------------------------------------------------------
# 1. memory_brain_ingest.py — no "preview" field, uses "content_len"
# ---------------------------------------------------------------------------

def test_memory_brain_ingest_no_preview_field():
    """The error-log item must not include a ``preview`` key."""
    src = _read_source(MEMORY_BRAIN_INGEST)
    stripped = _strip_comments_and_strings(src)
    assert '"preview"' not in stripped, (
        "scripts/memory_brain_ingest.py still contains a \"preview\" "
        "string literal in code (must be removed — P0-5 privacy leak)."
    )
    assert 'preview":' not in stripped, (
        "scripts/memory_brain_ingest.py still emits a 'preview:' field "
        "in code (P0-5 privacy leak)."
    )
    assert "content[:500]" not in stripped, (
        "scripts/memory_brain_ingest.py still slices `content[:500]` "
        "in code — that was the P0-5 preview leak."
    )


def test_memory_brain_ingest_uses_length_indicator():
    """AST check: ``record_error`` (or any logger call) must emit a
    ``content_len`` field that wraps ``len(content ...)``."""
    src = _read_source(MEMORY_BRAIN_INGEST)
    tree = ast.parse(src)

    found_content_len = False

    # Pass 1: walk Call nodes for kwarg-style length indicators.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if "content_len" in _kwarg_keys(node):
            found_content_len = True
            for kw in node.keywords:
                if kw.arg == "content_len":
                    if not _is_len_call(kw.value):
                        pytest.fail(
                            f"kwarg 'content_len' must be a `len(...)` "
                            f"call, got {ast.dump(kw.value)}"
                        )

    # Pass 2: walk Dict literals for length-indicator keys.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        if "content_len" not in _dict_keys(node):
            continue
        found_content_len = True
        for k, v in zip(node.keys, node.values):
            if (
                isinstance(k, ast.Constant)
                and isinstance(k.value, str)
                and k.value == "content_len"
            ):
                if not _is_len_call(v):
                    pytest.fail(
                        f"dict key 'content_len' must hold a `len(...)` "
                        f"call, got {ast.dump(v)}"
                    )

    assert found_content_len, (
        "scripts/memory_brain_ingest.py does not emit a `content_len` "
        "length indicator. The fix expects `content_len` to replace "
        "the removed `preview` field so the audit log shows the size "
        "of the memory body without showing the body."
    )


# ---------------------------------------------------------------------------
# 2. dedup_cron.py — no "text_preview" field, uses "text_len"
# ---------------------------------------------------------------------------

def test_dedup_cron_no_text_preview_field():
    """The cluster dump must not include a ``text_preview`` key."""
    src = _read_source(DEDUP_CRON)
    stripped = _strip_comments_and_strings(src)
    assert '"text_preview"' not in stripped, (
        "scripts/dedup_cron.py still contains a \"text_preview\" "
        "string literal in code (must be removed — P0-5 privacy leak)."
    )
    assert "text_preview:" not in stripped, (
        "scripts/dedup_cron.py still emits a 'text_preview:' field "
        "in code (P0-5 privacy leak)."
    )
    assert ')[:200]' not in stripped, (
        "scripts/dedup_cron.py still slices text to 200 chars in code "
        "— that was the P0-5 preview leak."
    )


def test_dedup_cron_uses_length_indicator():
    """AST check: the cluster-member dict must include ``text_len``
    wrapping ``len(payload.get(...) ...)``."""
    src = _read_source(DEDUP_CRON)
    tree = ast.parse(src)

    found_text_len = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and "text_len" in _dict_keys(node):
            found_text_len = True
            for k, v in zip(node.keys, node.values):
                if (
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and k.value == "text_len"
                ):
                    if not _is_len_call(v):
                        pytest.fail(
                            f"dict key 'text_len' must hold a `len(...)` "
                            f"call, got {ast.dump(v)}"
                        )
        elif isinstance(node, ast.Call) and "text_len" in _kwarg_keys(node):
            found_text_len = True
            for kw in node.keywords:
                if kw.arg == "text_len":
                    if not _is_len_call(kw.value):
                        pytest.fail(
                            f"kwarg 'text_len' must be a `len(...)` "
                            f"call, got {ast.dump(kw.value)}"
                        )

    assert found_text_len, (
        "scripts/dedup_cron.py does not emit a `text_len` length "
        "indicator. The fix expects `text_len` to replace the removed "
        "`text_preview` field so the cluster dump shows the size of "
        "each memory without showing the body."
    )


# ---------------------------------------------------------------------------
# 3. Cross-script privacy scan — AST-based per-function analysis.
# ---------------------------------------------------------------------------

# Sentinel function names that indicate the enclosing function is a
# log-write helper. If any of these names appear as the function
# enclosing a content/text slice, that's a privacy leak.
LOG_HELPER_NAMES = frozenset({
    "record_error",
    "audit_emit",
    "write_log",
    "write_status",
    "write_summary",
})

# Sentinel call sites that, if they appear in the same function body
# as a content/text slice, indicate the slice is being written to a
# log file (not just used internally for classification).
LOG_WRITE_AST_TOKENS = (
    "open",        # open(...)
    "write_text",  # Path(...).write_text(...)
    "dump",        # json.dump / pickle.dump
    "dumps",       # json.dumps
)

# Sentinel call sites for stdout / cron log output.
LOG_OUTPUT_AST_TOKENS = (
    "print",
    "info",        # logger.info / logging.info
    "warning",     # logger.warning / logging.warning
    "error",       # logger.error / logging.error
    "debug",       # logger.debug / logging.debug
)


def _function_uses_log_token(fn_node: ast.AST) -> bool:
    """Return True if ``fn_node`` references any log-write /
    log-output call."""
    for sub in ast.walk(fn_node):
        if isinstance(sub, ast.Call):
            func = sub.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name is None:
                continue
            if name in LOG_HELPER_NAMES:
                return True
            if name in LOG_WRITE_AST_TOKENS:
                return True
            if name in LOG_OUTPUT_AST_TOKENS:
                # ``print`` alone is too broad (every script uses
                # print for normal progress output). Only flag if
                # the call also has at least one argument that looks
                # like a content slice — which we check at a higher
                # level. Here we just note that the function does
                # use ``print``.
                if name == "print":
                    # Treat as log output only if the function name
                    # itself suggests logging (e.g. ``log_progress``,
                    # ``write_summary``) or if the script writes to a
                    # file elsewhere. To stay tight, we do NOT flag
                    # ``print`` alone; we rely on the write-token
                    # detection below.
                    continue
                return True
    return False


def _function_writes_to_file(fn_node: ast.AST) -> bool:
    """Return True if ``fn_node`` writes to a file via open(...)
    / write_text / json.dump."""
    for sub in ast.walk(fn_node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name in {"open", "write_text", "dump", "dumps"}:
            # Exclude HTTP / urllib.request which is not a file
            # write. Detect by scanning nearby imports — but a
            # simpler proxy: ``open`` is only used as a file open in
            # our codebase, never as an HTTP open.
            return True
        if name in LOG_HELPER_NAMES:
            return True
    return False


def _function_has_preview_slice(
    fn_node: ast.AST, source: str
) -> list[tuple[int, str]]:
    """Return ``[(line_no, snippet), ...]`` for every
    ``content/text/payload[:N]`` slice found in ``fn_node`` with
    ``50 <= N <= 500``.
    """
    findings: list[tuple[int, str]] = []
    lines = source.splitlines()
    for sub in ast.walk(fn_node):
        if not isinstance(sub, ast.Subscript):
            continue
        slc = sub.slice
        if not isinstance(slc, ast.Slice):
            continue
        if slc.lower is not None or slc.upper is None:
            continue
        upper = slc.upper
        if not isinstance(upper, ast.Constant) or not isinstance(upper.value, int):
            continue
        n = upper.value
        if not (50 <= n <= 500):
            continue
        value = sub.value
        target_name = None
        if isinstance(value, ast.Name):
            target_name = value.id
        elif isinstance(value, ast.Attribute):
            target_name = value.attr
        if target_name not in {"content", "text", "payload", "body", "memory"}:
            continue
        line_no = sub.lineno
        snippet = lines[line_no - 1].strip() if line_no <= len(lines) else ""
        findings.append((line_no, snippet))
    return findings


def _function_bodies(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """Yield ``(qualified_name, function_node)`` for every function /
    method defined in the tree. Top-level non-def statements are
    collected under the synthetic ``__main__`` name."""
    out: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, node))
    main_body = [
        s for s in getattr(tree, "body", [])
        if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if main_body:
        synth = ast.Module(body=main_body, type_ignores=[])
        out.append(("__main__", synth))
    return out


def test_no_other_preview_leaks_in_scripts():
    """AST-based per-function scan: catch any preview-style slice
    that lives inside a function which also writes to a file.

    This is a forward-looking guardrail: the brief explicitly limits
    scope to ``memory_brain_ingest.py`` and ``dedup_cron.py``, but we
    want a regression test so any future script that reintroduces a
    preview leak into a log file fails CI immediately and gets
    reported back to Runbook G7.4 review.

    We deliberately use AST analysis rather than regex windowing
    so we don't false-positive on internal classification helpers
    like ``classify_tier`` whose ``text[:200]`` slice is only used
    for substring matching and never written to a log file.
    """
    findings: list[tuple[str, str, int, str]] = []

    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        src = _read_source(path)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        for qname, fn_node in _function_bodies(tree):
            slices = _function_has_preview_slice(fn_node, src)
            if not slices:
                continue
            if not _function_writes_to_file(fn_node):
                continue
            for (line_no, snippet) in slices:
                findings.append((path.name, qname, line_no, snippet))

    assert not findings, (
        "Found new preview-style log leaks in scripts/ — these are "
        "P0-5 regressions and must be reported (do NOT auto-fix in "
        "this P0-5 commit; escalate via audit-report):\n"
        + "\n".join(
            f"  - {s}:{ln}  ({qname})  ::  {snippet}"
            for (s, qname, ln, snippet) in findings
        )
    )


# ---------------------------------------------------------------------------
# 4. Belt-and-braces: AST walk confirms that no logger.* / print call
#    has a kwarg / dict key called "preview" or "text_preview" anywhere
#    in the two changed scripts.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path,banned_keys",
    [
        (MEMORY_BRAIN_INGEST, {"preview"}),
        (DEDUP_CRON, {"text_preview", "preview"}),
    ],
)
def test_no_logger_call_emits_preview_key(path: Path, banned_keys: set[str]):
    """AST-level guard: no function call across the two scripts may
    carry a ``preview`` / ``text_preview`` kwarg or dict key.
    """
    src = _read_source(path)
    tree = ast.parse(src)

    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            kwarg_keys = _kwarg_keys(node)
            for banned in banned_keys:
                if banned in kwarg_keys:
                    offenders.append(
                        f"line {node.lineno}: call kwarg {banned!r}"
                    )
        if isinstance(node, ast.Dict):
            d_keys = _dict_keys(node)
            for banned in banned_keys:
                if banned in d_keys:
                    offenders.append(
                        f"line {node.lineno}: dict key {banned!r}"
                    )

    assert not offenders, (
        f"{path.name} still emits a banned preview key:\n  "
        + "\n  ".join(offenders)
    )
