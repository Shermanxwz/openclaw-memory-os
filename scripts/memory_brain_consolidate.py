#!/usr/bin/env python3
"""
Memory Brain Consolidate — compact recent memories into topic summaries
Phases: orient → gather → consolidate → prune

Memory Brain never deletes memories. Stale memories are surfaced for human
review only. The previous MEMORY_BRAIN_ALLOW_DELETE opt-in has been
removed; consolidation only flags stale candidates for human review.

Wave 2 (2026-07-20): embed + LLM calls are routed through
:mod:`openclaw_memory_os.embed_provider`. Setting
``EMBED_PROVIDER=newapi`` / ``LLM_PROVIDER=newapi`` switches both
paths to NewAPI without touching this file again.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from openclaw_memory_os.embed_provider import (  # type: ignore
        EmbeddingDimensionMismatch,
        EmbeddingUnavailable,
        ChatUnavailable,
        get_chat_provider,
        get_embed_provider,
    )
    _HAS_EMBED_PROVIDER = True
except ImportError:
    _HAS_EMBED_PROVIDER = False
import requests

# Reuse the importance-normalization helper from the ingest pipeline so
# consolidated summaries also land in Qdrant with the ``[0.0, 1.0]`` range
# expected by the Memory OS ranking layer (see ingest.normalize_importance).
try:
    from memory_brain_ingest import normalize_importance
except ImportError:
    # When scripts/ is not on sys.path (e.g. running the file directly),
    # fall back to a local mirror of the same helper. The integer / float
    # disambiguation matches the ingest module so a 1.0 float and a 1 int
    # do not collapse to the same value. Numeric-string ints are also
    # routed through the 1-5 scale (LLMs sometimes stringify).
    def normalize_importance(raw):
        if isinstance(raw, bool) or raw is None:
            return 0.6
        if isinstance(raw, int):
            return max(0.0, min(1.0, float(raw) / 5.0))
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped:
                try:
                    as_int = int(stripped)
                except ValueError:
                    as_int = None
                if as_int is not None and not isinstance(as_int, bool):
                    return normalize_importance(as_int)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.6
        return max(0.0, min(1.0, value))

# === 配置 ===
# Wave 2 (2026-07-20) — bugfix 2026-07-21:
# The defaults below now fall through to the wave-2 ``*_PROVIDER_*`` env
# vars before falling back to legacy OVH-Ollama defaults. This keeps the
# script aligned with ``.env``'s NewAPI path (see :mod:`embed_provider`)
# without dropping the offline Ollama fallback for slim CI containers.
#
# Priority for LLM_MODEL:
#   1. explicit ``MEMORY_BRAIN_LLM_MODEL`` (legacy override)
#   2. ``LLM_PROVIDER_MODEL`` (wave-2 NewAPI channel, e.g. qwen3:4b-instruct)
#   3. OLLAMA default ``qwen2.5:1.5b``
#
# Priority for LLM_URL:
#   1. explicit ``MEMORY_BRAIN_LLM_CHAT_URL`` / ``LLM_CHAT_URL`` overrides
#   2. ``LLM_PROVIDER_URL`` (wave-2 NewAPI gateway, /v1 suffix preserved)
#   3. ``OLLAMA_URL`` legacy ``/api/chat`` shape
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("MEMORY_BRAIN_COLLECTION", os.environ.get("QDRANT_COLLECTION", "openclaw_memory_brain"))
LLM_URL = (
    os.environ.get("MEMORY_BRAIN_LLM_CHAT_URL")
    or os.environ.get("LLM_CHAT_URL")
    or os.environ.get("LLM_PROVIDER_URL")
    or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
)
LLM_MODEL = (
    os.environ.get("MEMORY_BRAIN_LLM_MODEL")
    or os.environ.get("LLM_PROVIDER_MODEL")
    or "qwen2.5:1.5b"
)
EMBED_URL = (
    os.environ.get("EMBED_URL")
    or os.environ.get("EMBED_PROVIDER_URL")
    or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/embeddings"
)
EMBED_MODEL = (
    os.environ.get("EMBED_MODEL")
    or os.environ.get("EMBED_PROVIDER_MODEL")
    or "nomic-embed-text:latest"
)
WORKSPACE = os.environ.get("WORKSPACE_ROOT", str(Path.cwd().parent))
MEMORY_FILE = f"{WORKSPACE}/MEMORY.md"
STATE_FILE = f"{WORKSPACE}/memory/.dream_state.json"
STATUS_FILE = os.environ.get("MEMORY_BRAIN_DREAM_STATUS_FILE", "/var/log/openclaw-memory-brain-dream-status.json")

# 整合触发条件（仿 Claude Code）
DREAM_MIN_HOURS = int(os.environ.get("DREAM_MIN_HOURS", "24"))        # 默认每天最多整合一次
DREAM_MIN_NEW_MEMORIES = int(os.environ.get("DREAM_MIN_NEW_MEMORIES", "20"))  # 至少20条新记忆
MEMORY_MAX_LINES = 200      # MEMORY.md 最大行数
MEMORY_MAX_KB = 25          # MEMORY.md 最大 KB
MEMORY_AUTOPRUNE_ENABLED = False  # MEMORY.md 只报警，不自动截断
GATHER_DAYS = int(os.environ.get("DREAM_GATHER_DAYS", "1"))
MAX_MEMORIES_PER_TOPIC = int(os.environ.get("DREAM_MAX_MEMORIES_PER_TOPIC", "20"))
MAX_TOPICS_PER_RUN = int(os.environ.get("DREAM_MAX_TOPICS_PER_RUN", "5"))



def qdrant_headers() -> dict:
    key = os.environ.get("QDRANT_API_KEY", "")
    return {"api-key": key} if key else {}


def stable_summary_id(topic: str, source_ids: list) -> int:
    raw = topic + ":" + ",".join(map(str, sorted(source_ids)))
    import hashlib
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") & ((1 << 63) - 1)

def embed_text(text: str) -> list:
    """Embed a string.

    Wave 2 (2026-07-20): routes through
    :mod:`openclaw_memory_os.embed_provider` when available so the
    script honours ``EMBED_PROVIDER``. Falls back to the legacy
    single-shot ``requests`` POST against ``EMBED_URL`` for slim CI
    containers that do not have the package importable.
    """
    if _HAS_EMBED_PROVIDER:
        try:
            return get_embed_provider().embed(text)
        except (EmbeddingUnavailable, EmbeddingDimensionMismatch) as exc:
            # Mirror the legacy contract: raise so the caller (a
            # single Qdrant upsert) fails loudly and the operator
            # sees the consolidated-summary path as broken rather
            # than silently producing a zero-vector point.
            raise RuntimeError(f"embed provider failed: {exc}") from exc
    # Legacy fallback.
    r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]

def qdrant_search(vector: list, limit=10, filter_payload=None):
    body = {"vector": vector, "limit": limit, "with_payload": True, "with_vector": False}
    if filter_payload:
        body["filter"] = {"must": [{"key": k, "match": {"value": v}} for k, v in filter_payload.items()]}
    r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/search", json=body, headers=qdrant_headers(), timeout=60)
    r.raise_for_status()
    return r.json()["result"]

def qdrant_scroll(limit=50, offset_id=None, filter_payload=None):
    """滚动读取所有点"""
    body = {"limit": limit, "with_payload": True, "with_vector": False}
    if offset_id is not None:
        body["offset"] = offset_id
    if filter_payload:
        body["filter"] = {"must": [{"key": k, "match": {"value": v}} for k, v in filter_payload.items()]}
    r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll", json=body, headers=qdrant_headers(), timeout=60)
    r.raise_for_status()
    return r.json()["result"]

def llm_consolidate(memories: list, task: str) -> str:
    """本地 LLM 执行整合任务（Ollama qwen2.5:1.5b，零token）"""
    memories_text = "\n---\n".join([
        f"[ID:{m['id']}] {m['payload'].get('summary','')}: {m['payload'].get('content','')[:200]}"
        for m in memories[:20]
    ])

    prompts = {
        "summarize": f"""你是一个记忆整合系统。请将以下相关记忆合并成简洁的长久记忆条目（Markdown格式，不超过5行）。
只输出合并后的Markdown内容，不要其他文字。

相关记忆：
{memories_text}""",

        "identify_stale": f"""以下是按时间排列的记忆条目。找出其中：
1. 已经完全过时的信息（如已废弃的配置、已解决且不再需要的问题）
2. 相互矛盾的信息（取最新的为准，标记旧的为过时）

返回JSON：
{{"stale_ids": [ID1, ID2], "conflicts": [{{"keep": ID1, "discard": ID2, "reason": "原因"}}]}}

记忆：
{memories_text}""",

        "check_MEMORY_md": f"""请检查以下当前 MEMORY.md 内容，找出：
1. 重复的内容
2. 已过时的信息
3. 可以合并的相似内容
4. 缺失但应该记录的重要内容（从相关记忆中推断）

当前 MEMORY.md：
{memories_text[:3000]}"""
    }

    try:
        # Wave 2 (2026-07-20): honour the configured chat provider
        # (LLM_PROVIDER env). When ``LLM_PROVIDER=newapi`` the call
        # goes to NewAPI ``/v1/chat/completions`` with model
        # ``qwen3:4b-instruct``; default is OVH Ollama.
        if _HAS_EMBED_PROVIDER:
            try:
                _chat = get_chat_provider()
                _model = _chat.model
                _url = f"{_chat.base_url.rstrip('/')}/chat/completions"
            except Exception:
                _model = LLM_MODEL
                _url = LLM_URL
        else:
            _model = LLM_MODEL
            _url = LLM_URL
        r = requests.post(
            _url,
            json={
                "model": _model,
                "messages": [{"role": "user", "content": prompts.get(task, task)}],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 500}
            },
            timeout=120
        )
        resp = r.json()
        msg = resp.get("message", {})
        return msg.get("content", "").strip()
    except Exception as e:
        return f"ERROR: {e}"


# === Phase 1: Orient — 评估当前状态 ===
def orient():
    """评估记忆系统当前状态"""
    print("🧭 Phase 1: Orient — 评估记忆状态")

    # 总向量数
    r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", headers=qdrant_headers(), timeout=30)
    r.raise_for_status()
    total = r.json()["result"]["points_count"]

    # 最近24h新增
    since = (datetime.now() - timedelta(hours=24)).isoformat()
    qdrant_scroll(limit=1, filter_payload={"type": "fact"})  # placeholder
    # 实际按时间过滤（Qdrant不支持时间过滤，改用滚动统计）
    new_count = 0
    offset = None
    while True:
        batch = qdrant_scroll(limit=100, offset_id=offset)
        for pt in batch["points"]:
            ts = pt["payload"].get("timestamp", "")
            if ts > since:
                new_count += 1
        if not batch.get("next_page_offset"):
            break
        offset = batch["next_page_offset"]

    # MEMORY.md 状态
    mem_lines = 0
    mem_kb = 0
    if os.path.exists(MEMORY_FILE):
        mem_lines = len(open(MEMORY_FILE).readlines())
        mem_kb = os.path.getsize(MEMORY_FILE) / 1024

    print(f"  总记忆: {total} | 24h新增: {new_count} | MEMORY.md: {mem_lines}行/{mem_kb:.1f}KB")

    return {
        "total_points": total,
        "new_since_24h": new_count,
        "mem_lines": mem_lines,
        "mem_kb": mem_kb,
    }


# === Phase 2: Gather — 收集待整合记忆 ===
def gather(state: dict):
    """收集需要整合的记忆"""
    print("📥 Phase 2: Gather — 收集近期记忆")

    # 默认只收集最近 1 天；需要更宽窗口可用 DREAM_GATHER_DAYS 覆盖。
    last_run = (datetime.now() - timedelta(days=GATHER_DAYS)).isoformat()

    # 收集按话题分组的记忆
    topics_map = {}
    offset = None
    while True:
        batch = qdrant_scroll(limit=100, offset_id=offset)
        for pt in batch["points"]:
            ts = pt["payload"].get("timestamp", "")
            if ts > last_run:
                topic = pt["payload"].get("topic") or "untagged"
                if topic not in topics_map:
                    topics_map[topic] = []
                if pt["payload"].get("type") == "consolidated_summary":
                    continue
                topics_map[topic].append(pt)
        if not batch.get("next_page_offset"):
            break
        offset = batch["next_page_offset"]

    for topic, mems in topics_map.items():
        print(f"  {topic}: {len(mems)} 条记忆")

    return topics_map


# === Phase 3: Consolidate — 整合记忆 ===
def consolidate(topics_map: dict):
    """按话题合并记忆，生成摘要"""
    print("🧩 Phase 3: Consolidate — 整合记忆")

    summaries = {}

    topic_items = sorted(topics_map.items(), key=lambda kv: len(kv[1]), reverse=True)[:MAX_TOPICS_PER_RUN]
    for topic, memories in topic_items:
        if len(memories) < 2:
            continue

        # 限制每次整合条数，以免 prompt 超限或 LLM 超时
        memories = memories[:MAX_MEMORIES_PER_TOPIC]
        print(f"  整合 {topic} ({len(memories)}条)...")

        # 合并该话题下的所有近期记忆（不再要求 importance >= 4）
        summary = llm_consolidate(memories, "summarize")
        summaries[topic] = {
            "count": len(memories),
            "summary": summary,
            "source_ids": [m["id"] for m in memories]
        }

        # 存入 Qdrant 作为摘要记忆。ID 按 topic+source_ids 稳定生成，避免每次重复制造 summary。
        vec = embed_text(f"[{topic}] {summary}")
        next_id = stable_summary_id(topic, summaries[topic]["source_ids"])

        payload = {
            "content": f"[Topic Summary: {topic}]\n{summary}",
            "source": "auto-dream-consolidation",
            "type": "consolidated_summary",
            "topic": topic,
            "importance": normalize_importance(5),
            "summary": summary[:100],
            "entities": "[]",
            "keywords": json.dumps([topic]),
            "timestamp": datetime.now().isoformat(),
            "related_memories": json.dumps(summaries[topic]["source_ids"]),
            "access_count": 0,
            "last_accessed": None,
        }
        r = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
            json={"points": [{"id": next_id, "vector": vec, "payload": payload}]},
            headers=qdrant_headers(),
            timeout=120,
        )
        r.raise_for_status()

    return summaries


# === Phase 4: Prune — 裁剪和清理 ===
def prune():
    """裁剪过期/冗余记忆，维护 MEMORY.md"""
    print("✂️ Phase 4: Prune — 裁剪清理")

    # 检查 MEMORY.md 大小
    if os.path.exists(MEMORY_FILE):
        mem_kb = os.path.getsize(MEMORY_FILE) / 1024
        mem_lines = len(open(MEMORY_FILE).readlines())

        if mem_lines > MEMORY_MAX_LINES or mem_kb > MEMORY_MAX_KB:
            print(f"  ⚠️ MEMORY.md 超标: {mem_lines}行/{mem_kb:.1f}KB")
            content = open(MEMORY_FILE).read()

            # 让 LLM 审查
            review = llm_consolidate([{"id": 0, "payload": {"summary": "MEMORY.md", "content": content}}], "check_MEMORY_md")
            print(f"  LLM审查结果: {review[:200]}...")

            if not MEMORY_AUTOPRUNE_ENABLED:
                print("  ℹ️ MEMORY.md 自动裁剪已禁用，仅记录超标提示")
            else:
                # 简单裁剪策略：保留最重要的部分，截断末尾
                with open(MEMORY_FILE) as f:
                    lines = f.readlines()
                # 保留前面重要的内容
                keep_lines = lines[:MEMORY_MAX_LINES]
                with open(MEMORY_FILE + ".bak", 'w') as f:
                    f.writelines(lines)
                with open(MEMORY_FILE, 'w') as f:
                    f.writelines(keep_lines)
                print(f"  已裁剪: {len(lines)} → {len(keep_lines)} 行 (备份: MEMORY.md.bak)")

    # 清理 Qdrant 中标记为过期的记忆
    now = datetime.now().isoformat()
    # Qdrant scroll 找 valid_until 过期的
    stale_ids = []
    offset = None
    while True:
        batch = qdrant_scroll(limit=100, offset_id=offset)
        for pt in batch["points"]:
            vu = pt["payload"].get("valid_until")
            if vu and vu < now:
                stale_ids.append(pt["id"])
        if not batch.get("next_page_offset"):
            break
        offset = batch["next_page_offset"]

    if stale_ids:
        print(f"  ℹ️ 标记 {len(stale_ids)} 条过期记忆等待人工 review（Memory Brain 不删除记忆）")


# === 主逻辑 ===
def load_dream_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_dream": None, "dream_count": 0}

def save_dream_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    state["last_dream"] = datetime.now().isoformat()
    state["dream_count"] = state.get("dream_count", 0) + 1
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def check_trigger(state: dict, orient_result: dict) -> bool:
    """检查是否满足做梦条件"""
    # 距上次不少于24h
    if state["last_dream"]:
        last = datetime.fromisoformat(state["last_dream"])
        hours = (datetime.now() - last).total_seconds() / 3600
        if hours < DREAM_MIN_HOURS - 0.1:  # 给6分钟余量，避免浮点边界
            reason = f"距上次整合仅 {hours:.1f}h < {DREAM_MIN_HOURS}h"
            print(f"  ⏰ {reason}, 跳过")
            return False, reason

    # 至少有5条新记忆
    if orient_result["new_since_24h"] < DREAM_MIN_NEW_MEMORIES:
        reason = f"新增 {orient_result['new_since_24h']} < {DREAM_MIN_NEW_MEMORIES}"
        print(f"  📭 新记忆不足 ({reason}), 跳过")
        return False, reason

    return True, "ready"

def main():
    print("Memory Brain consolidation start")
    print(f"  时间: {datetime.now().isoformat()}")
    print()

    state = load_dream_state()

    # Phase 1
    orient_result = orient()

    # 检查触发条件
    trigger_ok, trigger_reason = check_trigger(state, orient_result)
    # Wave 2 (2026-07-21): when this script is invoked through
    # ``scripts/maintenance.sh`` (or ``scripts/memory_brain.py``) the
    # canonical sub-step state must land in ``maintenance-summary.json``
    # instead of ``/var/log/openclaw-memory-brain-dream-status.json``.
    # The detection is via ``MAINTENANCE_RUN_ID``: when set, the legacy
    # file write is suppressed and a structured stdout marker replaces
    # it so ``scripts/_write_summary.py`` can pick the run up. Manual
    # invocations (no run_id) keep the legacy write so existing
    # operator scripts continue to work.
    maintenance_run_id = os.environ.get("MAINTENANCE_RUN_ID", "")
    if not trigger_ok:
        if maintenance_run_id:
            print(
                f"[brain-consolidate] run_id={maintenance_run_id} "
                f"status=skipped reason={trigger_reason} "
                f"new_since_24h={orient_result.get('new_since_24h')} "
                f"total_points={orient_result.get('total_points')} "
                f"merged_topics=0 threshold={DREAM_MIN_NEW_MEMORIES} "
                f"topics_merged=0"
            )
        else:
            try:
                status = {
                    "script": "consolidate",
                    "last_run": datetime.now().isoformat(),
                    "status": "skipped",
                    "reason": trigger_reason,
                    "topics_merged": 0,
                    "dream_count": state.get("dream_count", 0),
                    "new_since_24h": orient_result.get("new_since_24h"),
                    "total_points": orient_result.get("total_points"),
                }
                with open(STATUS_FILE, "w") as f:
                    json.dump(status, f)
            except Exception:
                pass
        return

    # Phase 2
    topics = gather(state)

    # Phase 3
    summaries = consolidate(topics) if topics else {}

    # Phase 4
    prune()

    # 保存状态
    save_dream_state(state)
    # 写入状态文件
    if maintenance_run_id:
        # Mirror the skipped-path marker so the parser in
        # ``_write_summary.py`` only needs one shape to recognise.
        merged = len(summaries) if summaries else 0
        print(
            f"[brain-consolidate] run_id={maintenance_run_id} "
            f"status=ok topics_merged={merged} merged_topics={merged} "
            f"threshold={DREAM_MIN_NEW_MEMORIES} "
            f"new_since_24h={orient_result.get('new_since_24h')} "
            f"total_points={orient_result.get('total_points')} "
            f"dream_count={state['dream_count']}"
        )
    else:
        try:
            merged = len(summaries) if summaries else 0
            status = {"script": "consolidate", "last_run": datetime.now().isoformat(), "status": "completed", "topics_merged": merged, "dream_count": state["dream_count"]}
            with open(STATUS_FILE, "w") as f:
                json.dump(status, f)
        except Exception:
            pass

    print(f"\nConsolidation done: {len(summaries) if summaries else 0} 个话题整合, 第{state['dream_count']}次")
    r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", headers=qdrant_headers(), timeout=30)
    r.raise_for_status()
    print(f"Qdrant: {r.json()['result']['points_count']} 条记忆")

if __name__ == "__main__":
    main()
