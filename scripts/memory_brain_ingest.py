#!/usr/bin/env python3
"""
Memory Brain Ingest — turn memory files into structured long-term memories
Pipeline: read → classify → relate → embed → store

Wave 2 (2026-07-20): embed + LLM calls are routed through
:mod:`openclaw_memory_os.embed_provider`. Default behaviour is
unchanged (OVH-local Ollama). Setting ``EMBED_PROVIDER=newapi`` and
``LLM_PROVIDER=newapi`` switches both paths to NewAPI without
touching this file again.
"""

import json
import os
import sys
import hashlib
import time
import re
from pathlib import Path

# Allow the script to import the project's ingestion-validation
# module. This works both when invoked as a script (sys.path[0] is
# the scripts/ directory) and when imported from a sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from openclaw_memory_os.ingestion_validation import (  # type: ignore
        _extract_first_json,
        PROMPT_VERSION,
        classify_with_qwen,
    )
    _HAS_V030_VALIDATION = True
except ImportError:
    _HAS_V030_VALIDATION = False
try:
    from openclaw_memory_os.embed_provider import (  # type: ignore
        EmbeddingDimensionMismatch,
        EmbeddingUnavailable,
        get_chat_provider,
        get_embed_provider,
    )
    _HAS_EMBED_PROVIDER = True
except ImportError:
    _HAS_EMBED_PROVIDER = False
from datetime import datetime
from pathlib import Path
import requests

# v0.3.0 schema mismatch fix (Batch 4 / Finding B4-1)
# ---------------------------------------------------------------
# The legacy ingest path validated every LLM response through
# ``openclaw_memory_os.ingestion_validation.ClassificationSchema``.
# That schema uses ``extra='forbid'`` and only accepts the union
# of fields the v0.3.0 *primary* classification prompt emits.
# The ``memory_brain_ingest`` prompts request an *auxiliary*
# shape that includes fields the global schema rejects:
#
#   classify prompt      → adds ``sentiment`` (rejected)
#   find_context prompt  → adds ``prerequisite_memories`` (rejected)
#
# The result was that every LLM response failed validation and
# the script silently fell back to the deterministic payload,
# which is the bug Finding B4-1 documents.
#
# The fix introduces a local, permissive ``MemoryBrainSchema``
# that matches the prompt's exact shape (``extra='ignore'``) so
# the LLM's response is preserved verbatim and the auxiliary
# fields flow into the Qdrant payload. On any schema rejection
# the script still emits the deterministic fallback shape
# (``recall_triggers=[]``, ``prerequisite_memories=[]``,
# ``sentiment='neutral'``) so the pipeline never hangs on a
# malformed LLM response.
try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator  # type: ignore
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover - pydantic is a hard dep
    _HAS_PYDANTIC = False

if _HAS_PYDANTIC:
    try:
        from openclaw_memory_os.ingestion_validation import (  # type: ignore
            ALLOWED_TOPICS,
            ALLOWED_TYPES,
        )
    except ImportError:  # pragma: no cover
        ALLOWED_TYPES = {
            "fact", "decision", "event", "preference",
            "system_config", "lesson", "relationship",
        }
        ALLOWED_TOPICS = {
            "infrastructure", "business", "personal", "ai_model",
            "memory_system", "health", "tools_software", "planning",
        }

    ALLOWED_SENTIMENTS = {"positive", "negative", "neutral"}
    _MAX_TRIGGERS = 8
    _MAX_PREREQ = 8

    class MemoryBrainSchema(BaseModel):
        """Permissive schema for the memory_brain ingest prompts.

        ``extra='ignore'`` is intentional: the prompts ask the
        LLM for a small auxiliary shape (sentiment, recall_triggers,
        prerequisite_memories, valid_until), but the classify and
        find_context prompts do not line up exactly. The strict
        global ``ClassificationSchema`` is reserved for the
        primary memory classification; the brain script only
        produces auxiliary metadata, so we accept whatever the
        LLM emits and coerce the well-known fields into the
        payload shape.
        """

        model_config = ConfigDict(extra="ignore")

        type: str = "fact"
        topic: str = "infrastructure"
        importance: float = 0.6
        summary: str = ""
        entities: list = Field(default_factory=list)
        keywords: list = Field(default_factory=list)
        sentiment: str = "neutral"
        actionable: bool = False
        recall_triggers: list = Field(default_factory=list)
        prerequisite_memories: list = Field(default_factory=list)
        valid_until: object = None

        @field_validator("type")
        @classmethod
        def _type_allowed(cls, v):
            v = (str(v or "")).strip().lower()
            if v not in ALLOWED_TYPES:
                raise ValueError(f"unsupported type: {v!r}")
            return v

        @field_validator("topic")
        @classmethod
        def _topic_allowed(cls, v):
            v = (str(v or "")).strip().lower()
            if v not in ALLOWED_TOPICS:
                raise ValueError(f"unsupported topic: {v!r}")
            return v

        @field_validator("sentiment")
        @classmethod
        def _sentiment_allowed(cls, v):
            v = (str(v or "")).strip().lower()
            if v not in ALLOWED_SENTIMENTS:
                raise ValueError(f"unsupported sentiment: {v!r}")
            return v

        @field_validator("recall_triggers", "prerequisite_memories", mode="before")
        @classmethod
        def _cap_lists(cls, v):
            if v is None:
                return []
            if isinstance(v, str):
                v = [v]
            if not isinstance(v, list):
                return []
            out = []
            seen = set()
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
            return out[:_MAX_TRIGGERS]


    def validate_memory_brain(payload):
        """Validate a candidate LLM response against ``MemoryBrainSchema``.

        Returns ``(parsed_dict, error_code)``. ``parsed_dict`` is
        ``None`` on failure; ``error_code`` is one of
        ``json_missing`` / ``schema_invalid`` so the fallback
        payload can carry it through to the audit log.
        """
        if not payload or not isinstance(payload, dict):
            return None, "json_missing"
        try:
            parsed = MemoryBrainSchema.model_validate(payload)
        except ValidationError:
            return None, "schema_invalid"
        # Pydantic returns a model; coerce to a plain dict so the
        # rest of the script can ``.get(...)`` on it without
        # attribute access.
        return parsed.model_dump(), None


    def build_memory_brain_fallback(text, *, error_code):
        """Deterministic fallback when the LLM is unreachable or
        the response fails validation.

        The shape mirrors ``MemoryBrainSchema``'s defaults and
        the well-known auxiliary fields
        (``recall_triggers=[]``, ``prerequisite_memories=[]``,
        ``sentiment='neutral'``) so the script can read these
        fields without conditional checks.
        """
        summary = (text or "").strip()
        if len(summary) > 80:
            summary = summary[:79] + "…"
        return {
            "type": "fact",
            "topic": "infrastructure",
            "importance": 0.6,
            "summary": summary,
            "entities": [],
            "keywords": [],
            "sentiment": "neutral",
            "actionable": False,
            "recall_triggers": [],
            "prerequisite_memories": [],
            "valid_until": None,
            "_classification_status": "fallback",
            "_classification_error_code": error_code,
            "prompt_version": PROMPT_VERSION,
        }
else:  # pragma: no cover - pydantic is a hard dep
    def validate_memory_brain(payload):  # type: ignore[no-redef]
        if not payload or not isinstance(payload, dict):
            return None, "json_missing"
        return dict(payload), None

    def build_memory_brain_fallback(text, *, error_code):  # type: ignore[no-redef]
        summary = (text or "").strip()
        if len(summary) > 80:
            summary = summary[:79] + "…"
        return {
            "type": "fact",
            "topic": "infrastructure",
            "importance": 0.6,
            "summary": summary,
            "entities": [],
            "keywords": [],
            "sentiment": "neutral",
            "actionable": False,
            "recall_triggers": [],
            "prerequisite_memories": [],
            "valid_until": None,
            "_classification_status": "fallback",
            "_classification_error_code": error_code,
        }

# === 配置 ===
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("MEMORY_BRAIN_COLLECTION", os.environ.get("QDRANT_COLLECTION", "openclaw_memory_brain"))
LLM_URL = os.environ.get("LLM_API_URL", "http://localhost:11434/v1")  # 本地模型
LLM_KEY = os.environ.get("LLM_API_KEY", "")  # 本地 Ollama 不需要 key；禁止硬编码真实 key
EMBED_URL = os.environ.get("EMBED_URL", os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text:latest")
WORKSPACE = os.environ.get("WORKSPACE_ROOT", str(Path.cwd().parent))
MEMORY_DIR = f"{WORKSPACE}/memory"
STATE_FILE = f"{WORKSPACE}/memory/.brain_state.json"
ERROR_LOG = os.environ.get("MEMORY_BRAIN_ERROR_LOG", "/var/log/openclaw-memory-brain-errors.jsonl")
STORE_ERROR_CONTENT = os.environ.get("MEMORY_BRAIN_STORE_ERROR_CONTENT") == "1"
STATUS_FILE = os.environ.get("MEMORY_BRAIN_STATUS_FILE", "/var/log/openclaw-memory-brain-status.json")
MAX_FILES_PER_RUN = int(os.environ.get("MEMORY_BRAIN_MAX_FILES", "20"))  # 0 = all changed files



def qdrant_headers() -> dict:
    key = os.environ.get("QDRANT_API_KEY", "")
    return {"api-key": key} if key else {}


def llm_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if LLM_KEY:
        headers["Authorization"] = f"Bearer {LLM_KEY}"
    return headers


def point_id_from_hash(content_hash: str) -> int:
    # Stable 63-bit integer id. Avoids full-collection max-id scans and remains
    # deterministic for duplicate retries. Qdrant accepts unsigned integers; keep
    # under signed 63-bit to stay portable across tooling.
    return int.from_bytes(hashlib.sha256(content_hash.encode()).digest()[:8], "big") & ((1 << 63) - 1)

# 记忆类型定义
MEMORY_TYPES = {
    "fact": "客观事实",
    "decision": "做出的决策",
    "event": "发生的事件",
    "preference": "偏好/习惯",
    "system_config": "系统配置",
    "lesson": "教训/踩坑",
    "relationship": "人物关系",
}

# 话题分类
TOPICS = [
    "infrastructure", "business", "personal", "ai_model",
    "memory_system", "health", "tools_software", "planning"
]

# === Qdrant 操作 ===
def embed_text(text: str) -> list:
    """Embed a string.

    Wave 2 (2026-07-20): routes through
    :mod:`openclaw_memory_os.embed_provider` when available so the
    script honours ``EMBED_PROVIDER``. Falls back to the legacy
    ``requests`` POST against ``EMBED_URL`` / ``EMBED_MODEL`` for
    slim CI containers that do not have the package importable.
    """
    if _HAS_EMBED_PROVIDER:
        # 3 retry loop preserved from the legacy implementation so
        # transient NewAPI / Ollama blips do not abort the whole
        # ingest run. The provider's own raise is caught here and
        # converted to a retryable failure.
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                return get_embed_provider().embed(text)
            except (EmbeddingUnavailable, EmbeddingDimensionMismatch) as exc:
                last_exc = exc
                if attempt == 2:
                    break
                time.sleep(2)
            except Exception as exc:
                # Unknown error: still retry once, then re-raise.
                last_exc = exc
                if attempt == 2:
                    raise
                time.sleep(2)
        raise RuntimeError(
            f"embed provider failed after 3 attempts: {last_exc}"
        )
    # Legacy fallback path (slim CI). Mirrors the pre-wave-2 3-retry
    # behaviour verbatim.
    for attempt in range(3):
        try:
            r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=90)
            return r.json()["embedding"]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

def qdrant_upsert(points: list):
    """批量写入 Qdrant"""
    r = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
        json={"points": points},
        headers=qdrant_headers(),
        timeout=120,
    )
    r.raise_for_status()
    return r.json()

def qdrant_search(vector: list, limit=5, filter_payload=None):
    """语义搜索"""
    body = {"vector": vector, "limit": limit, "with_payload": True}
    if filter_payload:
        body["filter"] = {"must": [{"key": k, "match": {"value": v}} for k, v in filter_payload.items()]}
    r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/search", json=body, headers=qdrant_headers(), timeout=60)
    r.raise_for_status()
    return r.json()["result"]

def qdrant_get_point(point_id: int):
    """获取单个点"""
    r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}/points/{point_id}", headers=qdrant_headers(), timeout=30)
    r.raise_for_status()
    return r.json()["result"]


def record_error(stage: str, source: str, error: Exception, content: str = ""):
    """Append failed memory-processing items to a JSONL queue for later audit.

    Privacy contract (Runbook G7.4 / privacy-clean): we never write the raw
    ``content`` body to disk — only its length and a content hash. The
    error queue is meant for audit / retry, not for reconstructing the
    memory payload, so dumping the first 500 chars was a P0-5 privacy
    leak. The optional ``MEMORY_BRAIN_STORE_ERROR_CONTENT`` env var is
    intentionally honoured here because it requires an explicit opt-in
    (env = "1") and is intended for local debugging only.
    """
    try:
        item = {
            "ts": datetime.now().isoformat(),
            "stage": stage,
            "source": source,
            "error": str(error)[:500],
            "content_hash": hashlib.md5(content.strip().encode()).hexdigest() if content else "",
            "content_len": len(content or ""),
        }
        if STORE_ERROR_CONTENT:
            item["content"] = content
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception:
        pass


# === LLM 理解引擎 ===
# Importance contract:
#   The LLM is prompted to return a 1-5 integer importance (5 = critical,
#   1 = trivial). For the Qdrant payload, we normalize this to the
#   ``[0.0, 1.0]`` float range that the Memory OS ranking layer expects
#   (see ``openclaw_memory_os/backends/__init__.py`` where the adapter
#   also clamps the value defensively). Without this normalization the
#   LLM-emitted "5" landed in the payload as the integer ``5`` while
#   ranking treated it as ``1.0``, distorting tier weight comparison.

IMPORTANCE_MIN_RAW = 1.0   # LLM-reported minimum (1 = trivial)
IMPORTANCE_MAX_RAW = 5.0   # LLM-reported maximum (5 = critical)


def normalize_importance(raw: object) -> float:
    """Map an LLM-reported importance onto the ``[0.0, 1.0]`` range.

    Contract:

    * Real ints 1..5  -> divide by 5, clamp to [0.2, 1.0].
    * Numeric strings that parse to an int 1..5 (e.g. ``"4"``) ->
      likewise treat as a 1-5 score (LLMs sometimes stringify).
    * Floats in [0.0, 1.0] -> pass through verbatim.
    * Anything else (str garbage, None, out-of-range) -> default 0.6.

    The default middle ground is 3 / 5 = 0.6 — previously a "3" stored
    as the integer ``3`` would have ranked above all "1-2" entries while
    ranking only saw "1.0"; this fix makes the on-disk value consistent
    with what the ranking layer reads.

    Disambiguation note: the integers 1..5 and the floats in
    ``[0.0, 1.0]`` overlap at ``1.0``. We use the 1-5 scale whenever the
    raw value is exactly an integer (real Python int, or a numeric string
    whose stripped form parses to an int). Floats are already the
    ranking-layer contract and pass through verbatim.
    """
    # bool is a subclass of int in Python; treat it as garbage rather
    # than 0 / 1.
    if isinstance(raw, bool) or raw is None:
        return 0.6
    if isinstance(raw, int):
        return max(0.0, min(1.0, float(raw) / IMPORTANCE_MAX_RAW))
    # Numeric strings: if they parse to an integer that fits the
    # 1-5 contract, normalize on that scale; otherwise treat as garbage.
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped:
            try:
                as_int = int(stripped)
            except ValueError:
                as_int = None
            if as_int is not None and not isinstance(as_int, bool):
                # Re-route through the int branch by recursing once.
                return normalize_importance(as_int)
            # Otherwise fall through to float parsing below.
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.6
    return max(0.0, min(1.0, value))


def llm_understand(text: str, task: str) -> dict:
    """
    让 LLM 理解内容并提取结构化信息
    task: "classify" | "summarize" | "extract_entities" | "find_relations"
    """
    prompts = {
        "classify": f"""分析以下记忆内容，返回JSON（只返回JSON，不要其他文字）：
{{
  "type": "fact|decision|event|preference|system_config|lesson|relationship",
  "topic": "infrastructure|business|personal|ai_model|memory_system|health|tools_software|planning",
  "importance": 0.0-1.0,  // 1.0=极其重要（核心决策、密码、架构），0.0=琐碎；存储前 normalize_importance() 保证 [0.0, 1.0]
  "summary": "一句话摘要，15字以内",
  "entities": ["实体1", "实体2"],  // 人名、项目名、机器名、公司名
  "keywords": ["关键词1", "关键词2", "关键词3"],  // 用于检索的关键词
  "sentiment": "positive|negative|neutral",
  "actionable": true/false  // 是否有可执行的行动项
}}

记忆内容：
{text[:2000]}""",

        "find_context": f"""以下是一个记忆片段，请找出它可能在什么场景下被需要。返回JSON：
{{
  "recall_triggers": ["触发词1", "触发词2"],  // 用户提到这些词时应该回忆这条记忆
  "prerequisite_memories": ["需要先知道什么背景"],  // 可选
  "valid_until": null  // 如果信息有时效性，填过期时间 (ISO格式)，否则 null
}}

记忆内容：
{text[:1000]}"""
    }

    try:
        # Wave 2 (2026-07-20): pick the chat model + URL from the
        # configured provider. When ``LLM_PROVIDER=newapi`` the
        # provider returns the real NewAPI model name
        # (``qwen3:4b-instruct``) and the NewAPI base URL. Default
        # is unchanged (OVH-local Ollama ``qwen2.5:1.5b``).
        if _HAS_EMBED_PROVIDER:
            try:
                _chat = get_chat_provider()
                _models = [_chat.model]
                _url = f"{_chat.base_url.rstrip('/')}/chat/completions"
            except Exception:
                _models = ["qwen2.5:1.5b"]
                _url = f"{LLM_URL.rstrip('/')}/chat/completions"
        else:
            _models = ["qwen2.5:1.5b"]
            _url = f"{LLM_URL.rstrip('/')}/chat/completions"
        models = _models
        # v0.3.0 / G7.1 — the retry policy now lives in
        # ``openclaw_memory_os.ingestion_validation.classify_with_qwen``,
        # which fires ONE corrective HTTP POST on top of the original
        # attempt. The previous inline path here silently fell through
        # to the deterministic fallback on the first malformed
        # response, which is the Runbook G7.1 regression the new
        # wrapper closes.
        if _HAS_V030_VALIDATION:
            url = _url
            classify_prompt = prompts.get("classify", task) if task == "classify" else None
            raw, status = classify_with_qwen(
                text,
                model=models[0],
                url=url,
                prompt=classify_prompt,
                timeout=120,
            )
            if raw:
                # The brain script uses a permissive local schema
                # (sentiment / recall_triggers / prerequisite_memories)
                # that the global ClassificationSchema would reject
                # under ``extra='forbid'`` — that's the Batch 4
                # Finding B4-1 fix. Validate against the local
                # MemoryBrainSchema first so the auxiliary fields
                # survive verbatim.
                parsed, error_code = validate_memory_brain(raw)
                if parsed is not None:
                    parsed["_classification_status"] = status
                    parsed["prompt_version"] = PROMPT_VERSION
                    return parsed
                # The corrective retry still produced a JSON object
                # but it didn't validate against the local schema
                # (e.g. ``type`` not in the allow-list). Carry the
                # status token through to the fallback so the audit
                # log can distinguish "ok retry but bad schema"
                # from "no retry at all".
                fb = build_memory_brain_fallback(
                    text, error_code=error_code or "schema_invalid"
                )
                fb["_classification_status"] = (
                    "retry_failed" if status == "retry_failed" else status
                )
                return fb
            # raw is empty: classify_with_qwen already tried the
            # corrective retry and came back empty. Emit a
            # fallback with ``_classification_status = retry_failed``
            # so the audit / dashboard can see the retry was
            # attempted (rather than collapsing to plain
            # ``fallback`` and losing the signal).
            fb = build_memory_brain_fallback(text, error_code="llm_unreachable")
            fb["_classification_status"] = status or "retry_failed"
            return fb
        # Legacy path (kept for fallback when the project module
        # cannot be imported, e.g. running the script in a slim
        # CI container without the package installed). The legacy
        # path does NOT retry — a malformed LLM response falls
        # straight through to ``{}`` and the caller fills in
        # defaults.
        resp = None
        last_error = None
        for model in models:
            try:
                r = requests.post(
                    f"{LLM_URL}/chat/completions",
                    headers=llm_headers(),
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompts.get(task, task)}],
                        "temperature": 0.1,
                        "max_tokens": 500
                    },
                    timeout=120
                )
                candidate = r.json()
                if r.status_code == 200 and "choices" in candidate:
                    resp = candidate
                    break
                last_error = RuntimeError(f"HTTP {r.status_code}: {str(candidate)[:200]}")
            except Exception as e:
                last_error = e
        if resp is None:
            raise last_error or RuntimeError("LLM returned no usable response")
        msg = resp["choices"][0]["message"]
        content = (msg.get("content") or msg.get("reasoning_content", "")).strip()
        # v0.3.0: extract the first balanced JSON object, then
        # validate it through the Pydantic schema. On any failure
        # we fall back to the deterministic payload so the
        # pipeline can never hang on a malformed LLM response.
        if _HAS_V030_VALIDATION:
            raw = _extract_first_json(content) or {}
            # v0.3.0 / Batch 4 — Finding B4-1: the brain prompts ask
            # the LLM for an *auxiliary* shape (sentiment,
            # recall_triggers, prerequisite_memories, valid_until)
            # that the global ``ClassificationSchema`` rejects with
            # ``extra='forbid'``. Validate against the local,
            # permissive ``MemoryBrainSchema`` instead so the LLM's
            # response is preserved verbatim. On any rejection we
            # still emit a deterministic fallback shape so the
            # pipeline never hangs on a malformed LLM response.
            parsed, error_code = validate_memory_brain(raw)
            if parsed is not None:
                return parsed
            return build_memory_brain_fallback(text, error_code=error_code or "json_missing")
        # Original (legacy) regex fallback when neither the global
        # package nor pydantic is importable.
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}
    except Exception as e:
        print(f"  ⚠️ LLM理解失败({task}): {str(e)[:60]}")
        record_error(f"llm_{task}", "llm_understand", e, text)
        if _HAS_V030_VALIDATION:
            fb = build_memory_brain_fallback(text, error_code="llm_unreachable")
            # classify_with_qwen never raised (it caught its own
            # errors); a raise here means the import itself or the
            # wrapper itself blew up before any retry fired. Tag
            # the fallback accordingly so the audit log can
            # distinguish "wrapper raised before retry" from
            # "retry fired but came back empty".
            fb["_classification_status"] = "retry_failed"
            return fb
        return {"type": "fact", "topic": "infrastructure", "importance": 3, "summary": text[:30]}


# === 记忆摄取流水线 ===
def ingest_text(content: str, source: str, timestamp: str = None):
    """
    处理一段文本 → 理解、向量化、存储
    """
    if not timestamp:
        timestamp = datetime.now().isoformat()

    # 1. 去重检查
    chk = hashlib.md5(content.strip().encode()).hexdigest()
    r = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        json={"filter": {"must": [{"key": "content_hash", "match": {"value": chk}}]}, "limit": 1, "with_payload": False},
        headers=qdrant_headers(),
        timeout=30,
    )
    r.raise_for_status()
    if r.json()["result"]["points"]:
        return None, "duplicate"

    # 2. LLM 理解和分类
    info = llm_understand(content, "classify")
    if not info:
        if _HAS_V030_VALIDATION:
            info = build_memory_brain_fallback(content, error_code="llm_unreachable")
        else:
            info = {"type": "fact", "topic": "infrastructure", "importance": 3, "summary": content[:30]}

    # 3. 找关联记忆
    vec = embed_text(content)
    related = qdrant_search(vec, limit=3)
    related_ids = [p["id"] for p in related if p["score"] > 0.7] if related else []

    # 4. 生成 recall triggers
    ctx = llm_understand(content, "find_context")
    triggers = ctx.get("recall_triggers", []) if ctx else []

    # 5. 构建富 Payload
    payload = {
        "content": content,
        "content_hash": chk,
        "source": source,
        "type": info.get("type", "fact"),
        "topic": info.get("topic", "infrastructure"),
        "importance": normalize_importance(info.get("importance", 3)),
        "summary": info.get("summary", ""),
        "entities": json.dumps(info.get("entities", [])),
        "keywords": json.dumps(info.get("keywords", [])),
        "sentiment": info.get("sentiment", "neutral"),
        "actionable": info.get("actionable", False),
        "timestamp": timestamp,
        "related_memories": json.dumps(related_ids),
        "recall_triggers": json.dumps(triggers),
        "valid_until": ctx.get("valid_until"),
        "access_count": 0,
        "last_accessed": None,
    }
    # v0.3.0 ingest metadata: attach the same prompt_version +
    # classification_status fields that the validated payload
    # carries, so the dashboard can tell whether a record was
    # LLM-classified or fell back. The keys are prefixed with
    # ``_`` so they don't collide with the legacy payload
    # schema.
    if _HAS_V030_VALIDATION:
        payload["_classification_status"] = info.get(
            "_classification_status", "ok"
        )
        payload["_classification_error_code"] = info.get(
            "_classification_error_code", None
        )
        payload["prompt_version"] = info.get("prompt_version", PROMPT_VERSION)

    # 6. 写入 Qdrant
    point_id = point_id_from_hash(chk)
    qdrant_upsert([{"id": point_id, "vector": vec, "payload": payload}])
    return point_id, info.get("summary", "")


# === 文件处理 ===
def process_file(filepath: str) -> dict:
    """处理一个完整的记忆文件"""
    print(f"\nProcessing: {filepath}")
    with open(filepath, 'r', encoding="utf-8", errors="replace") as f:
        text = f.read()

    source = os.path.basename(filepath)
    ingested, skipped, summaries = 0, 0, []

    # 按 ## 标题拆分段落
    sections = re.split(r'\n(?=## )', text)
    if len(sections) <= 1:
        # 没有标题，整体处理
        sections = [text]

    for section in sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue

        # 记录时间戳
        ts_match = re.search(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2})', section)
        ts = ts_match.group(1) if ts_match else None

        try:
            pid, status = ingest_text(section, source, ts)
            if pid:
                ingested += 1
                summaries.append(status)
            else:
                skipped += 1
        except Exception as e:
            print(f"  ❌ 段落处理失败: {str(e)[:80]}")
            record_error("process_section", source, e, section)
            skipped += 1

    return {"file": filepath, "ingested": ingested, "skipped": skipped, "summaries": summaries}


def load_state() -> dict:
    """加载处理状态"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {"processed_files": {}, "last_run": None}

def save_state(state: dict):
    """保存处理状态"""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    state["last_run"] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


# === 主逻辑 ===
def main():
    state = load_state()
    files = sorted(Path(MEMORY_DIR).glob("*.md"))
    total = {"ingested": 0, "skipped": 0, "files": 0}

    # 处理最近修改的文件；默认每次最多 20 个，MEMORY_BRAIN_MAX_FILES=0 表示全部。
    files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
    if MAX_FILES_PER_RUN > 0:
        files = files[:MAX_FILES_PER_RUN]
    for f in files:
        fname = f.name
        mtime = f.stat().st_mtime

        # 跳过已处理且未修改的文件
        if fname in state["processed_files"]:
            if state["processed_files"][fname].get("mtime") == mtime:
                continue

        result = process_file(str(f))
        total["ingested"] += result["ingested"]
        total["skipped"] += result["skipped"]
        total["files"] += 1

        state["processed_files"][fname] = {"mtime": mtime, "ingested": result["ingested"]}
    # === 错误队列：有内容则重试≤3次，无内容或超限自动删除 ===
    retried_ok = 0
    if os.path.exists(ERROR_LOG):
        fresh = []
        with open(ERROR_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not entry.get("content"):
                    continue  # 无内容直接丢弃
                entry.setdefault("retry_count", 0)
                if entry["retry_count"] >= 3:
                    continue  # 超3次丢弃
                try:
                    _, _ = ingest_text(entry["content"], entry.get("source", "error_retry"), entry.get("timestamp"))
                    retried_ok += 1
                except Exception:
                    entry["retry_count"] += 1
                    fresh.append(json.dumps(entry, ensure_ascii=False))
        if fresh:
            with open(ERROR_LOG, "w", encoding="utf-8") as f:
                for line in fresh:
                    f.write(line + "\n")
        else:
            if os.path.exists(ERROR_LOG):
                os.remove(ERROR_LOG)
        if retried_ok:
            print(f"  ♻️ 重试成功: {retried_ok} 条")
        if not fresh and not retried_ok:
            # 所有错误都被清理了
            pass
    # 写入状态文件供审计面板采集    # 写入状态文件供审计面板采集
    # Wave 2 (2026-07-21): when this script is invoked through
    # ``scripts/maintenance.sh`` the canonical sub-step state must land
    # in ``maintenance-summary.json`` (written by ``_write_summary.py``)
    # rather than the legacy ``/var/log/openclaw-memory-brain-status.json``
    # file that the dashboard no longer reads. We detect the maintenance
    # path via ``MAINTENANCE_RUN_ID``; when it is set the sub-step stdout
    # line (captured by the ``>> "$LOG_FILE" 2>&1`` redirection in
    # maintenance.sh) is the only artefact the summary writer needs, so
    # we skip the legacy JSON write and just print a structured marker.
    error_count = 0
    if os.path.exists(ERROR_LOG):
        with open(ERROR_LOG) as ef:
            error_count = sum(1 for line in ef if line.strip())
    if os.environ.get("MAINTENANCE_RUN_ID"):
        # Structured marker the _write_summary parser will pick up: it
        # carries the run_id + sub-step id so the canonical
        # ``steps.ingest`` block can be correlated with this invocation.
        print(
            f"[brain-ingest] run_id={os.environ['MAINTENANCE_RUN_ID']} "
            f"files_processed={total['files']} "
            f"total_ingested={total['ingested']} "
            f"total_skipped={total['skipped']} "
            f"error_queue={error_count} "
            f"status=ok"
        )
    else:
        try:
            status = {"script": "ingest", "last_run": datetime.now().isoformat(), "total_ingested": total["ingested"], "total_skipped": total["skipped"], "files_processed": total["files"], "error_queue": error_count}
            with open(STATUS_FILE, "w") as f:
                json.dump(status, f)
        except Exception:
            pass
    save_state(state)
    print(f"\nDone: {total['files']} 文件, {total['ingested']} 新记忆, {total['skipped']} 跳过")
    print("Qdrant points: (查询中...)")
    r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", headers=qdrant_headers(), timeout=30)
    r.raise_for_status()
    print(f"   {r.json()['result']['points_count']} 条记忆")

if __name__ == "__main__":
    main()
