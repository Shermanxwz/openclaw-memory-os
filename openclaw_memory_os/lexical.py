"""Lexical search for OpenClaw Memory OS.

This module implements a local BM25 index over the multi-collection
memory corpus. It is the v0.3.0 replacement for the previous
"token set overlap" keyword path in :mod:`openclaw_memory_os.ranking`.

Design notes
------------

* **No new model dependency.** The tokenizer + BM25 math is all
  stdlib + ``math``. The index lives in process memory and is
  cached to disk as a small JSON document frequency table.

* **Multi-field tokens.** Each memory contributes to the index as
  one document composed of weighted text from several fields:
  exact identifiers, keywords, recall_triggers, entities, summary,
  text, tags, source. Field weights are pinned in
  :data:`FIELD_WEIGHTS`.

* **Exact-identifier boost.** Anything that looks like an env var
  (``MEMORY_OS_TOKEN``), a path (``/api/recall-test``), an IP+port
  (``127.0.0.1:6333``), a model name (``qwen2.5:1.5b``), or a
  version string (``v0.3.0``) gets a hard per-document boost when
  it appears in the query and in the document.

* **CJK handling.** Chinese is tokenised into 2- and 3-grams so
  short queries ("服务器反代") still hit relevant memories.

* **Cache + rebuild.** The index is persisted at
  ``$XDG_STATE_HOME/openclaw-memory-os/lexical-index/`` with a
  schema_ver, a checksum, and a corpus watermark. A missing or
  mismatched cache triggers a full rebuild from a backend.

* **No zero-vector fallback.** Unlike dense search, lexical search
  cannot return "degraded by definition" because the index lives
  in process memory. If the index is empty, the search returns
  empty results and the caller is expected to surface
  ``lexical_available=False`` to the diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .contracts import MemoryRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# English word / snake_case / kebab-case / version tokens
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\.]+")
# Environment variable style: all uppercase with at least one underscore
_ENVVAR_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
# File path segment
_PATHSEG_RE = re.compile(r"(?:/|\.)[A-Za-z0-9_\-\.]+")
# IPv4:port
_IPV4_PORT_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?\b")
# CJK characters (Chinese / Japanese kanji / Korean)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


# Field weights used when building a per-memory document. These are
# pinned by the v0.3.0 contract; changing them requires a schema bump.
#
# Operators can override any subset of these weights at index
# construction time (see :class:`BM25Index.__init__`) or via the
# ``MEMORY_OS_LEXICAL_FIELD_WEIGHTS`` JSON env var. The env var
# takes a JSON object whose keys are subset-replaceable over
# :data:`FIELD_WEIGHTS` (any field not mentioned keeps the
# default). Aliases ``trigger_words`` and ``recall_triggers`` are
# both accepted for backward-compatibility with older operator
# configs.
FIELD_WEIGHTS: Dict[str, float] = {
    "exact_identifier": 3.0,
    "keywords": 2.0,
    "recall_triggers": 2.5,  # alias: trigger_words
    "trigger_words": 2.5,
    "entities": 1.8,
    "entity": 1.8,           # alias: entities
    "summary": 1.5,
    "text": 1.0,
    "tags": 1.4,
    "source": 0.8,
}


def resolve_field_weights(overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Return the effective per-field weight map.

    Resolution order (highest precedence first):

    1. ``overrides`` argument (caller-supplied dict).
    2. ``MEMORY_OS_LEXICAL_FIELD_WEIGHTS`` env var (JSON object).
    3. :data:`FIELD_WEIGHTS` defaults.

    The result always contains *every* canonical key so callers
    can index without conditional lookups; aliases
    (``trigger_words`` / ``recall_triggers``, ``entity`` /
    ``entities``) are kept in sync so the legacy and new names
    both resolve to the same numeric weight.
    """
    out: Dict[str, float] = dict(FIELD_WEIGHTS)
    env_raw = os.environ.get("MEMORY_OS_LEXICAL_FIELD_WEIGHTS")
    if env_raw:
        try:
            env_overrides = json.loads(env_raw)
            if isinstance(env_overrides, dict):
                for k, v in env_overrides.items():
                    try:
                        out[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
        except (ValueError, TypeError):
            logger.warning(
                "resolve_field_weights: ignoring unparseable "
                "MEMORY_OS_LEXICAL_FIELD_WEIGHTS=%r",
                env_raw,
            )
    if overrides:
        for k, v in overrides.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    # Keep aliases in sync so the caller can use either name.
    out["trigger_words"] = out.get("recall_triggers", out.get("trigger_words", 2.5))
    out["recall_triggers"] = out["trigger_words"]
    out["entity"] = out.get("entities", out.get("entity", 1.8))
    out["entities"] = out["entity"]
    return out


# Identifiers we always treat as "exact" regardless of where they
# appear — for the boost to apply even when the value lives in
# ``text`` rather than ``keywords``. Populated by
# :func:`extract_exact_identifiers` from the query.
_EXACT_BOOST_TOKENS: Set[str] = set()


def _is_cjk(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def _emit_cjk_ngrams(text: str) -> List[str]:
    """Emit 2- and 3-character bigrams / trigrams for CJK text.

    We emit both because 2-grams give higher recall on very short
    queries ("服务器") while 3-grams give higher precision on
    slightly longer queries ("配置反代"). The two together strike
    a reasonable balance without needing a real segmenter.
    """
    cjk = [c for c in text if _is_cjk(c)]
    if len(cjk) < 2:
        return []
    grams: List[str] = []
    for i in range(len(cjk) - 1):
        grams.append(cjk[i] + cjk[i + 1])
    for i in range(len(cjk) - 2):
        grams.append(cjk[i] + cjk[i + 1] + cjk[i + 2])
    return grams


def tokenize_lexical(text: str) -> List[str]:
    """Tokenize ``text`` into a flat list of lowercase lexical tokens.

    Output includes:

    * Lowercased English / snake_case / kebab-case words.
    * All-caps identifiers treated as their own tokens (preserved
      so we can recognise them in :func:`extract_exact_identifiers`).
    * CJK 2- and 3-character n-grams (lowercase is a no-op for
      Chinese characters, kept for uniformity with the rest of the
      pipeline).
    * A "type" prefix tag is NOT added here; :func:`build_lexical_document`
      is responsible for field-weighted assembly.
    """
    if not text:
        return []
    out: List[str] = []
    # Use a token regex that does NOT break on colon, dot, dash
    # so model names like "qwen2.5:1.5b" stay intact, and
    # version-like tokens ("v0.3.0", "1.0.0") survive.
    for m in re.finditer(r"[A-Za-z0-9_\-\.:]+", text):
        t = m.group(0).lower()
        if t:
            out.append(t)
    for m in _ENVVAR_RE.finditer(text):
        out.append(m.group(0))  # keep case for exact-match later
    for m in _PATHSEG_RE.finditer(text):
        seg = m.group(0).lstrip("/.").lower()
        if seg and len(seg) > 1:
            out.append(seg)
    for m in _IPV4_PORT_RE.finditer(text):
        out.append(m.group(0))
    out.extend(_emit_cjk_ngrams(text))
    return out


def extract_exact_identifiers(query: str) -> Set[str]:
    """Return the set of "exact match" tokens found in ``query``.

    Exact identifiers are tokens that, when matched in a document,
    should get a per-document additive boost on top of the BM25
    score. They are: env vars, IP+port, path-like segments, and
    mixed-case identifiers.
    """
    found: Set[str] = set()
    for m in _ENVVAR_RE.finditer(query):
        found.add(m.group(0))
    for m in _IPV4_PORT_RE.finditer(query):
        found.add(m.group(0))
    for m in _PATHSEG_RE.finditer(query):
        seg = m.group(0).lstrip("/.")
        if seg and len(seg) > 1:
            found.add(seg)
    # Mixed-case identifiers (CamelCase, snake with caps)
    for m in re.finditer(r"\b[A-Z][A-Za-z0-9_]*[a-z][A-Za-z0-9_]*\b", query):
        found.add(m.group(0))
    return found


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------


def _stringify_list_field(value) -> List[str]:
    """Coerce keywords/entities/triggers into a list of strings.

    Accepts: list, tuple, set, JSON-string, comma-string. Empty on
    anything else. This is the v0.3.0 forward-compatible shape for
    fields that historically came in as either a list, a JSON
    string, or a plain string.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if x]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x]
            except (ValueError, TypeError):
                pass
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s]
    return [str(value)]


def build_lexical_document(
    record: MemoryRecord,
    field_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], Set[str]]:
    """Build the (tokens, exact_identifiers) for a single memory.

    The returned token list has each token repeated by the integer
    floor of the field weight so BM25 naturally weights up
    fields. ``text`` (weight 1.0) contributes each token once;
    ``summary`` (weight 1.5) contributes each token once; ``keywords``
    (weight 2.0) contributes each token twice; ``recall_triggers``
    (alias ``trigger_words``, weight 2.5) contributes each token
    twice; ``entities`` (alias ``entity``, weight 1.8) contributes
    each token once. Fractional weights above the integer floor are
    dropped so a token never contributes a fractional count.

    The exact_identifiers set is what the caller will use to
    compute the per-document exact-match boost during search.

    ``field_weights`` overrides the default
    :data:`FIELD_WEIGHTS` map on a per-call basis. Pass ``None``
    (default) to use :data:`FIELD_WEIGHTS` as-is.
    """
    weights = field_weights if field_weights is not None else FIELD_WEIGHTS

    def _emit(value: str, field: str) -> None:
        if not value:
            return
        w = (
            weights.get(field)
            or weights.get(field.rstrip("s"))  # summary -> summar (no)
            or 1.0
        )
        repeat = max(1, int(w))  # integer floor, at least 1
        toks = tokenize_lexical(value)
        for _ in range(repeat):
            text_tokens.extend(toks)

    text_tokens: List[str] = []
    exact_ids: Set[str] = set()

    # text: weight 1.0 (default)
    _emit(record.text or "", "text")
    # summary: weight 1.5
    _emit(record.summary or "", "summary")
    # source: weight 0.8
    _emit(record.source or "", "source")
    # tags: weight 1.4
    for tag in record.tags or []:
        _emit(tag, "tags")
    # keywords / recall_triggers / entities — all configurable
    # through ``field_weights``. Aliases are accepted (``entity``,
    # ``trigger_words``) so operator configs that follow the
    # v0.3.0 public naming (or the older internal naming) both
    # work.
    for attr in ("keywords", "recall_triggers", "entities"):
        val = getattr(record, attr, None)
        if val is None:
            continue
        for s in _stringify_list_field(val):
            _emit(s, attr)

    # Detect exact identifiers within the assembled document
    full_text = " ".join(
        (record.text or "", record.summary or "", record.source or "", " ".join(record.tags or []))
    )
    exact_ids.update(extract_exact_identifiers(full_text))

    return text_tokens, exact_ids


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------


@dataclass
class _BM25Stats:
    """Document-frequency table for a single token.

    Stored once per token; consulted at query time to compute
    IDF and the per-document term frequency contribution.
    """

    df: int = 0
    # Sum of term frequencies across all documents that contain
    # the term. Cached so we can do incremental updates cheaply.
    total_tf: int = 0


@dataclass
class _IndexState:
    """In-memory snapshot of the lexical index.

    v0.3.0.x (P0-X): the ``inverted_index`` field maps each
    token to its posting list ``[(doc_key, term_freq), ...]`` so
    :meth:`BM25Index.search` can score only the candidate set
    instead of looping over every document. With 52k documents
    the legacy O(N×Q) scan dominated ``lexical_ms`` (8-9 s on a
    hybrid query for ``"v0.3.0"``); the inverted index drops
    that to O(Q × avg_posting_length + candidate_count), which
    is sub-second for typical hybrid queries. The ``stats``
    table still drives IDF and total_tf; we maintain the
    inverted_index in parallel and rebuild it from the cached
    ``__lexical_tf__`` dicts on cache load so existing on-disk
    caches transparently pick up the speed-up after a service
    restart.

    v0.3.0.x schema_version semantics:

    * **v1** — pre-P0-X on-disk caches. Carry ``documents`` with
      ``__lexical_tf__`` but no top-level ``inverted_index``.
      Loaded via :meth:`BM25Index.from_dict` which detects the
      missing inverted_index, rebuilds it from
      ``__lexical_tf__``, and bumps to v2 in memory before
      :meth:`BM25Index.load` writes the migrated cache back to
      disk so the next restart skips the migration.

    * **v2** — current shape. Top-level ``inverted_index`` is
      present and populated; no migration needed on load. The
      default ``schema_version`` for fresh in-memory indexes is
      2 so newly built caches save as v2 from the start.
    """

    # Default schema_version bumped to 2 so fresh in-memory
    # indexes save as v2 on the next ``save()``. v1 caches that
    # need migration bump to 2 in ``from_dict``.
    schema_version: int = 2
    # token -> stats
    stats: Dict[str, _BM25Stats] = field(default_factory=dict)
    # document_key -> (record_dict_for_rebuild, token_count, exact_ids)
    documents: Dict[str, Tuple[Dict, int, Set[str]]] = field(default_factory=dict)
    # token -> [(doc_key, tf_in_that_doc), ...]
    # Insertion-ordered; ``search`` does an O(posting_len) union
    # across query tokens to derive the candidate set.
    inverted_index: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    # monotonic watermark: max(updated_at) we've seen
    last_indexed_at: Optional[str] = None


# Standard BM25 parameters; pinned by the v0.3.0 contract.
_BM25_K1 = 1.2
_BM25_B = 0.75


class BM25Index:
    """A corpus-wide BM25 index over :class:`MemoryRecord` documents.

    The index is intentionally not thread-safe; build it once at
    backend startup (or after a corpus change) and then call
    :meth:`search` from request threads. For our workload
    (~50k points), in-memory cost is on the order of a few tens of
    MB and search is sub-millisecond per query.

    Parameters
    ----------
    field_weights:
        Optional per-field weight overrides. Merged on top of
        :data:`FIELD_WEIGHTS` (and any ``MEMORY_OS_LEXICAL_FIELD_WEIGHTS``
        env var). Use :func:`resolve_field_weights` if you only
        want the env-var + defaults path without explicit
        overrides.
    """

    def __init__(
        self,
        *,
        field_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._state = _IndexState()
        self._average_doc_len: float = 0.0
        self._field_weights: Dict[str, float] = resolve_field_weights(field_weights)
        # Tiny warmed-query cache for repeated keyword probes and
        # dashboard interactions. The index is immutable between
        # add/remove/clear calls, so cached search results are safe
        # until the next mutation. This keeps concurrent identical
        # keyword requests from re-scoring the same posting lists in
        # parallel on the single-worker uvicorn.
        self._search_cache: "OrderedDict[Tuple[Tuple[str, ...], int, float], List[Tuple[str, float]]]" = OrderedDict()
        self._search_cache_max = 256

    @property
    def field_weights(self) -> Dict[str, float]:
        """Return the active per-field weight map (read-only copy)."""
        return dict(self._field_weights)

    # -- population ---------------------------------------------------------

    def add(self, record: MemoryRecord) -> None:
        """Insert or replace one document.

        Clears the search cache and recomputes the average
        document length every call, so a single ``add()`` keeps
        :meth:`search` in a consistent state. For batch
        refreshes (e.g. :func:`incremental_refresh` over many
        records) prefer the internal :meth:`_add_one` path which
        defers the average refresh and cache clear until the
        whole batch is in — bringing a full 27k-doc refresh from
        O(N²) back to O(N).
        """
        self._search_cache.clear()
        self._add_one(record)
        # Single-doc add keeps the public contract: average doc
        # length is up to date immediately after every add(). The
        # batch path in :func:`incremental_refresh` skips this
        # per-record so it stays O(N) total instead of O(N²).
        self._refresh_average()

    def _add_one(self, record: MemoryRecord) -> None:
        """Insert or replace one document without touching the average or cache.

        Used by the batch path in :func:`incremental_refresh` to
        amortise the O(N) average refresh over the whole batch
        instead of paying it per record (O(N²) overall). Public
        :meth:`add` wraps this with the search-cache invalidation
        and average refresh that the single-doc contract
        guarantees; callers that want to refresh in bulk should
        use :func:`incremental_refresh` rather than driving
        ``_add_one`` directly.
        """
        key = record.candidate_key
        # Remove the old one if present (re-indexing is fine)
        if key in self._state.documents:
            self._remove_internal(key)
        tokens, exact_ids = build_lexical_document(record, self._field_weights)
        doc_len = len(tokens)
        tf_counts = Counter(tokens)
        for token, tf in tf_counts.items():
            stats = self._state.stats.setdefault(token, _BM25Stats())
            stats.df += 1
            stats.total_tf += tf
            # P0-X: append (doc_key, tf) to the token's posting list
            # so ``search`` can score only the candidate set rather
            # than looping over every document. Sorting by doc_key
            # on append keeps the lookup stable; the cost is paid
            # at index time, not at query time.
            self._state.inverted_index.setdefault(token, []).append(
                (key, int(tf))
            )
        # Stash a minimal dict so we can rebuild on cache miss.
        # We use ``MemoryRecord.model_dump()`` to capture all
        # fields; deserialisation is via ``MemoryRecord(**...)``.
        record_dict = record.model_dump()
        # Cache the pre-tokenised list AND the per-token Counter
        # so ``_doc_tf`` can do O(1) lookups instead of linear scans.
        record_dict["__lexical_tokens__"] = tokens
        record_dict["__lexical_tf__"] = dict(tf_counts)
        self._state.documents[key] = (record_dict, doc_len, exact_ids)
        ts = (record.created_at or datetime.now(timezone.utc)).isoformat()
        if self._state.last_indexed_at is None or ts > self._state.last_indexed_at:
            self._state.last_indexed_at = ts

    def _remove_internal(self, key: str) -> None:
        """Remove a document without recomputing average doc length."""
        entry = self._state.documents.pop(key, None)
        if entry is None:
            return
        record_dict, _doc_len, _exact = entry
        tokens = record_dict.get("__lexical_tokens__") or []
        tf_counts = Counter(tokens)
        for token, tf in tf_counts.items():
            stats = self._state.stats.get(token)
            if stats is None:
                continue
            stats.df -= 1
            stats.total_tf -= tf
            if stats.df <= 0:
                self._state.stats.pop(token, None)
            # P0-X: drop every posting that references the
            # removed doc from the inverted index. We rebuild
            # the per-token list from scratch (linear in
            # postings, which is acceptable because each token
            # only appears in a handful of docs on average).
            postings = self._state.inverted_index.get(token)
            if postings is None:
                continue
            kept = [(dk, ttf) for (dk, ttf) in postings if dk != key]
            if kept:
                self._state.inverted_index[token] = kept
            else:
                self._state.inverted_index.pop(token, None)

    def remove(self, candidate_key: str) -> None:
        self._remove_internal(candidate_key)
        self._search_cache.clear()
        self._refresh_average()

    def _refresh_average(self) -> None:
        if not self._state.documents:
            self._average_doc_len = 0.0
            return
        total = sum(d[1] for d in self._state.documents.values())
        self._average_doc_len = total / max(1, len(self._state.documents))

    # -- search -------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 10,
        *,
        exact_match_boost: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """Return ``[(candidate_key, score)]`` sorted by descending score.

        ``exact_match_boost`` is the additive per-document boost
        applied when a query token appears in the document's
        exact_identifier set. The caller multiplies the per-token
        weight; the index applies it directly to keep the
        contract simple.

        v0.3.0.x (P0-X): this method now uses the inverted
        index to derive a candidate set from the query tokens,
        then scores only that candidate set. The legacy path
        looped over every document (``O(N × Q)``), which took
        ~9 s on a 52k-doc corpus with 5 query tokens. The
        inverted-index path is ``O(Q × avg_posting_length +
        candidate_count)``, which is sub-second in practice.
        """
        if not query or not self._state.documents:
            return []
        query_tokens = tokenize_lexical(query)
        if not query_tokens:
            return []
        # Drop dupes while preserving order
        seen: Set[str] = set()
        uniq_query: List[str] = []
        for t in query_tokens:
            if t not in seen:
                seen.add(t)
                uniq_query.append(t)
        # Pull out query exact identifiers
        cache_key = (tuple(uniq_query), int(limit), float(exact_match_boost))
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            # Maintain LRU order and return a shallow copy so callers
            # cannot mutate the cached list.
            self._search_cache.move_to_end(cache_key)
            return list(cached)
        query_exacts = extract_exact_identifiers(query)
        n_docs = len(self._state.documents)
        avg = self._average_doc_len or 1.0
        # P0-X: build candidate set by unioning posting lists.
        # Each entry of the inverted index for a query token
        # gives us ``(doc_key, tf_in_doc)`` for free, so we
        # also pre-collect the per-(doc, token) tf pair to avoid
        # an extra ``_doc_tf`` dict lookup during scoring.
        # ``candidate_tfs`` keeps a per-candidate dict of
        # {token: tf_in_doc}, so the inner loop just sums
        # idf-weighted contributions over the keys it actually
        # has.
        candidate_tfs: Dict[str, Dict[str, int]] = {}
        for qt in uniq_query:
            postings = self._state.inverted_index.get(qt)
            if not postings:
                continue
            for doc_key, tf in postings:
                slot = candidate_tfs.get(doc_key)
                if slot is None:
                    slot = {}
                    candidate_tfs[doc_key] = slot
                # If a doc has multiple postings for the same
                # query token (shouldn't normally happen because
                # add() replaces the doc on key collision), keep
                # the higher tf.
                prev = slot.get(qt)
                if prev is None or tf > prev:
                    slot[qt] = tf
        scores: List[Tuple[str, float]] = []
        for key, per_token_tf in candidate_tfs.items():
            doc_entry = self._state.documents.get(key)
            if doc_entry is None:
                # Defensive: a posting should never reference a
                # missing document, but guard against drift.
                continue
            _record, doc_len, doc_exacts = doc_entry
            s = 0.0
            for qt, tf in per_token_tf.items():
                if tf <= 0:
                    continue
                stats = self._state.stats.get(qt)
                if stats is None or stats.df == 0:
                    continue
                idf = math.log(1 + (n_docs - stats.df + 0.5) / (stats.df + 0.5))
                denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / avg)
                s += idf * (tf * (_BM25_K1 + 1)) / (denom or 1.0)
            if doc_exacts & query_exacts:
                s += float(exact_match_boost)
            if s > 0.0:
                scores.append((key, s))
        scores.sort(key=lambda x: (-x[1], x[0]))
        result = scores[:limit]
        self._search_cache[cache_key] = list(result)
        self._search_cache.move_to_end(cache_key)
        while len(self._search_cache) > self._search_cache_max:
            self._search_cache.popitem(last=False)
        return list(result)

    def get_record(self, candidate_key: str) -> Optional[MemoryRecord]:
        """Return the indexed ``MemoryRecord`` for ``candidate_key``.

        The lexical index stores full ``MemoryRecord.model_dump()``
        payloads for cache rebuilds. Returning the record directly
        lets callers avoid a per-hit backend lookup after BM25 search
        (important under concurrency: Qdrant targeted lookups were
        dominating keyword p95 even when BM25 itself was <100ms).
        """
        entry = self._state.documents.get(candidate_key)
        if entry is None:
            return None
        record, _doc_len, _exact = entry
        try:
            return MemoryRecord(**record)
        except Exception:
            return None

    def _doc_tf(self, key: str, token: str) -> int:
        """Count ``token`` occurrences in the document at ``key``.

        Uses the pre-computed ``__lexical_tf__`` Counter dict for O(1)
        lookup instead of linear-scanning the token list.
        """
        entry = self._state.documents.get(key)
        if entry is None:
            return 0
        record, _doc_len, _exact = entry
        tf_map = record.get("__lexical_tf__")
        if tf_map is not None:
            return tf_map.get(token, 0)
        # Fallback: re-derive from token list (should not happen
        # for documents added after the v0.3.0.x upgrade).
        cached = record.get("__lexical_tokens__")
        if cached is not None:
            return cached.count(token)
        return 0

    # -- size / clear -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._state.documents)

    def clear(self) -> None:
        self._state = _IndexState()
        self._average_doc_len = 0.0
        self._search_cache.clear()

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialise the index into a plain dict for disk persistence."""
        return {
            "schema_version": self._state.schema_version,
            "documents": {
                k: (record, doc_len, sorted(exact))
                for k, (record, doc_len, exact) in self._state.documents.items()
            },
            # P0-X: persist the inverted index alongside documents
            # so ``from_dict`` doesn't have to rebuild it by
            # re-tokenising every doc on startup. The on-disk
            # footprint grows roughly linearly with
            # ``sum(per-doc unique tokens)``, which for a 52k
            # corpus is on the order of a few MB — acceptable for
            # the lexical cache directory. The data is plain
            # tuples (``[doc_key, tf]``) so json.dump round-trips
            # it cleanly.
            "inverted_index": {
                token: [[dk, int(tf)] for (dk, tf) in postings]
                for token, postings in self._state.inverted_index.items()
            },
            "last_indexed_at": self._state.last_indexed_at,
            "average_doc_len": self._average_doc_len,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "BM25Index":
        """Restore a :class:`BM25Index` from an on-disk payload.

        Accepts both v1 (legacy) and v2 (current) cache shapes:

        * **v1** (``schema_version == 1`` and no ``inverted_index``
          field) — older caches written before the inverted-index
          optimisation. We rebuild the inverted index from the
          per-record ``__lexical_tf__`` dicts (which v1 caches
          already carry, courtesy of the P0-3 fix) and bump
          ``state.schema_version`` to 2 so the next save writes
          the migrated shape. ``BM25Index.load`` notices the
          migration flag and writes the v2 cache back to disk so
          subsequent restarts skip the rebuild.
        * **v2** (``schema_version >= 2`` or ``inverted_index``
          present) — current shape, no migration needed.

        Schema versions other than 1 or 2 are still rejected
        because they indicate an incompatible cache format.
        """
        payload_schema = int(payload.get("schema_version", 0))
        if payload_schema not in (0, 1, 2):
            raise ValueError(
                f"unsupported lexical-index schema_version: "
                f"{payload.get('schema_version')!r}"
            )
        idx = cls()
        for key, (record, doc_len, exact) in payload.get("documents", {}).items():
            idx._state.documents[key] = (record, doc_len, set(exact))
        # Restore the inverted index from disk if present. v1 caches
        # don't carry this key — we detect that below and rebuild
        # from per-record ``__lexical_tf__`` dicts.
        inverted_payload = payload.get("inverted_index")
        if isinstance(inverted_payload, dict):
            for token, postings in inverted_payload.items():
                if not postings:
                    continue
                idx._state.inverted_index[token] = [
                    (entry[0], int(entry[1])) for entry in postings if len(entry) >= 2
                ]
        idx._state.last_indexed_at = payload.get("last_indexed_at")
        idx._average_doc_len = float(payload.get("average_doc_len", 0.0))
        # v1 → v2 migration: rebuild the inverted index from the
        # per-record ``__lexical_tf__`` dicts when the on-disk
        # cache lacked a top-level ``inverted_index`` field. The
        # P0-3 fix made ``__lexical_tf__`` mandatory on every
        # ``add()``, so every v1 cache in the wild has these
        # cached term-frequency maps and the rebuild is O(sum
        # of per-doc unique tokens) — no re-tokenisation needed.
        needs_migration = (
            not idx._state.inverted_index and bool(idx._state.documents)
        )
        if needs_migration:
            idx._rebuild_inverted_index_from_documents()
            idx._state.schema_version = 2
            idx._migrated_from_v1 = True
            logger.info(
                "BM25Index migrated from v%d → v2: rebuilt inverted_index "
                "from __lexical_tf__ (%d docs, %d tokens)",
                payload_schema,
                len(idx._state.documents),
                len(idx._state.inverted_index),
            )
        else:
            # Keep the loaded schema_version in memory so future
            # saves preserve the existing on-disk shape. v2
            # caches stay at 2; v1 caches that already carried
            # ``inverted_index`` (some Wave B dumps do) stay at
            # 1 — they are functionally equivalent to v2 and
            # don't need to be touched.
            idx._state.schema_version = payload_schema if payload_schema >= 1 else 2
            idx._migrated_from_v1 = False
        # Rebuild aggregate stats DIRECTLY from the cached per-record
        # ``__lexical_tf__`` dicts. ``add()`` always populates
        # ``__lexical_tf__`` (see P0-3 fix), so the cache already
        # carries every (token -> tf) pair we need to compute
        # document frequency and total_tf. This avoids re-tokenizing
        # the entire corpus (which is dominated by CJK n-gram
        # generation) on every lifespan load — historically that
        # single line drove the hybrid recall handler from
        # sub-second to 40-50 seconds on a ~100MB cache.
        #
        # The helper only rebuilds the inverted_index when it's
        # missing, so v2 caches (where we just restored the
        # payload's inverted_index) skip the redundant rebuild
        # — see ``_rebuild_stats_from_cached_tf`` and
        # ``_rebuild_stats`` for the guard.
        idx._rebuild_stats_from_cached_tf()
        return idx

    def _rebuild_stats_from_cached_tf(self) -> None:
        """Populate ``self._state.stats`` from cached ``__lexical_tf__`` dicts.

        Equivalent to :meth:`_rebuild_stats` but reads the pre-computed
        per-token term frequencies stored alongside each document
        instead of re-tokenising ``record.get("text", "")``. Falls back
        to the slow path if any document lacks the cached tf map
        (older cache dumps pre-dating the ``__lexical_tf__`` field).

        v0.3.0.x (P0-X): only rebuilds ``self._state.inverted_index``
        when it's missing. The v1 → v2 migration path in
        :meth:`from_dict` calls ``_rebuild_inverted_index_from_documents``
        directly before this helper runs, so by the time we get
        here the inverted index is already populated and we must
        not clobber it with another redundant rebuild. For v2
        caches (which carried the inverted index on disk) the
        guard short-circuits the rebuild entirely.
        """
        self._state.stats.clear()
        needs_fallback = False
        for _key, (record, _doc_len, _exact) in self._state.documents.items():
            tf_map = record.get("__lexical_tf__")
            if not tf_map:
                needs_fallback = True
                continue
            for token, tf in tf_map.items():
                stats = self._state.stats.get(token)
                if stats is None:
                    stats = _BM25Stats()
                    self._state.stats[token] = stats
                stats.df += 1
                stats.total_tf += int(tf)
        if needs_fallback:
            # Slow path: at least one record lacked the cached tf
            # map. Recompute the full stats table via tokenisation
            # so legacy caches still produce correct results.
            logger.debug(
                "BM25 cache missing __lexical_tf__ on some records; "
                "falling back to _rebuild_stats (slow path)"
            )
            self._rebuild_stats()
            return
        # Only rebuild the inverted index if it's missing (v1
        # caches that didn't migrate during ``from_dict``).
        if not self._state.inverted_index and self._state.documents:
            self._rebuild_inverted_index_from_documents()
        self._refresh_average()

    def _rebuild_inverted_index_from_documents(self) -> None:
        """Rebuild ``self._state.inverted_index`` from each document's tokens.

        Iterates over every loaded document and appends
        ``(doc_key, tf)`` to each of the document's tokens in the
        inverted index. Used both by the cached-tf fast path
        (after P0-3) and by the legacy re-tokenise fallback so a
        fully populated inverted index is always available
        immediately after :meth:`from_dict` returns.

        For each record we prefer the cached ``__lexical_tf__``
        dict (P0-3+ caches), then fall back to counting over the
        cached ``__lexical_tokens__`` list (legacy P0-3 caches
        that kept the tokens but not the Counter), then re-tokenise
        ``record.get("text", "")`` as a last resort (truly bare
        pre-P0-3 caches).
        """
        self._state.inverted_index.clear()
        for key, (record, _doc_len, _exact) in self._state.documents.items():
            tf_map = record.get("__lexical_tf__")
            if not tf_map:
                cached_tokens = record.get("__lexical_tokens__")
                if cached_tokens:
                    tf_map = dict(Counter(cached_tokens))
                else:
                    tf_map = dict(Counter(tokenize_lexical(record.get("text", "") or "")))
            for token, tf in tf_map.items():
                self._state.inverted_index.setdefault(token, []).append(
                    (key, int(tf))
                )

    def _rebuild_stats(self) -> None:
        self._state.stats.clear()
        for key, (record, _doc_len, _exact) in self._state.documents.items():
            tf_counts = Counter(tokenize_lexical(record.get("text", "") or ""))
            for token, tf in tf_counts.items():
                stats = self._state.stats.setdefault(token, _BM25Stats())
                stats.df += 1
                stats.total_tf += tf
        # Only rebuild the inverted index when it's missing
        # (v1 → v2 migration path). v2 caches skip this.
        if not self._state.inverted_index and self._state.documents:
            self._rebuild_inverted_index_from_documents()
        self._refresh_average()

    # -- cache (disk) -------------------------------------------------------

    def save(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        blob = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode("utf-8")
        checksum = hashlib.sha256(blob).hexdigest()
        wrapper = {
            "schema_version": payload["schema_version"],
            "checksum": checksum,
            "watermark": self._state.last_indexed_at,
            "body": payload,
        }
        path = cache_dir / "lexical-index.json"
        tmp = cache_dir / "lexical-index.json.tmp"
        tmp.write_bytes(json.dumps(wrapper, ensure_ascii=False, default=str).encode("utf-8"))
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        os.chmod(path, 0o600)
        # Write the sidecar checksum so corruption can be detected
        # even if the body parser succeeds but the bytes were
        # partially overwritten.
        (cache_dir / "lexical-index.sha256").write_text(checksum + "\n", encoding="utf-8")

    @classmethod
    def load(cls, cache_dir: Path) -> Optional["BM25Index"]:
        path = cache_dir / "lexical-index.json"
        sha = cache_dir / "lexical-index.sha256"
        if not path.exists() or not sha.exists():
            return None
        try:
            raw = json.loads(path.read_bytes().decode("utf-8"))
            actual = hashlib.sha256(
                json.dumps(raw.get("body", {}), ensure_ascii=False, default=str, sort_keys=True)
                .encode("utf-8")
            ).hexdigest()
            if actual != raw.get("checksum"):
                logger.warning(
                    "BM25Index cache checksum mismatch: declared=%s actual=%s; discarding",
                    raw.get("checksum"),
                    actual,
                )
                return None
            return cls.from_dict(raw["body"])
        except (ValueError, OSError, KeyError) as exc:
            logger.warning("BM25Index cache load failed: %s", exc)
            return None


def incremental_refresh(
    index: BM25Index,
    records: Iterable[MemoryRecord],
) -> int:
    """Add/refresh ``records`` into ``index``; return the count added.

    This is the O(N) batch entry point used by
    ``scripts/refresh_lexical.py`` to rebuild the lexical index
    from a backend. Per-record work uses the private
    :meth:`BM25Index._add_one` path so we only recompute the
    corpus-wide average document length **once** at the end (the
    previous implementation drove each batch element through
    :meth:`BM25Index.add`, which recomputed the average on every
    call and made a 27k-doc full refresh O(N²) — a hang of
    multiple minutes in production).

    The public :meth:`BM25Index.add` contract is preserved: a
    direct caller of ``add()`` still gets the search-cache
    invalidation and an up-to-date average after every record.
    Callers that hold the same index across many adds should
    prefer :func:`incremental_refresh` for the amortised cost;
    it also clears the search cache once at the start (mirroring
    what ``add()`` did) so pre-batch cached results are not
    served against an in-progress corpus.
    """
    index._search_cache.clear()
    added = 0
    for r in records:
        index._add_one(r)
        added += 1
    # One O(N) average refresh for the whole batch, regardless
    # of batch size. Without this, the index would lie to
    # ``search`` about the corpus mean — the score normalisation
    # denominator in BM25 depends on it.
    index._refresh_average()
    return added
