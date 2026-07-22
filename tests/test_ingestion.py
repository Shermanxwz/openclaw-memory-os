"""Tests for the ingestion module (checkpoint/resume, chunking, state management).

These tests use the ingestion module's helpers and the IngestProgress model
without connecting to Qdrant or Ollama (dry-run mode).
"""

from __future__ import annotations

import json
from pathlib import Path


from openclaw_memory_os.ingestion import IngestionManager
from openclaw_memory_os.models import IngestProgress


def test_ingest_progress_model():
    p = IngestProgress(
        checkpoint_id="test_001",
        total_files=5,
        total_chunks=100,
    )
    assert p.status == "running"
    assert p.total_files == 5
    assert p.total_chunks == 100
    assert p.written == 0
    assert p.failed == 0

    # Serialize round-trip
    data = p.model_dump(mode="json")
    p2 = IngestProgress(**data)
    assert p2.checkpoint_id == "test_001"


def test_chunking_no_sections(tmp_path: Path):
    mgr = IngestionManager(workspace_root=tmp_path)
    source = tmp_path / "test.md"
    text = "just a single paragraph without headers"
    chunks = list(mgr._split_by_sections(text, source))
    assert len(chunks) == 1
    assert chunks[0][1] == text


def test_chunking_empty_text(tmp_path: Path):
    mgr = IngestionManager(workspace_root=tmp_path)
    source = tmp_path / "empty.md"
    chunks = list(mgr._split_by_sections("   ", source))
    assert chunks == []


def test_chunking_sections(tmp_path: Path):
    mgr = IngestionManager(workspace_root=tmp_path)
    source = tmp_path / "notes.md"
    text = (
        "## Section One\n\ncontent one with enough text to exceed the minimum threshold of fifty characters easily\n\n"
        "## Section Two\n\ncontent two with also enough text to comfortably pass the minimum threshold of fifty characters\n\n"
        "## Section Three\n\ncontent three with enough text to pass the fifty character minimum threshold easily now\n"
    )
    chunks = list(mgr._split_by_sections(text, source))
    assert len(chunks) >= 3
    # Each chunk should have a unique ID
    ids = set(c[0] for c in chunks)
    assert len(ids) == len(chunks)


def test_chunking_skips_short_sections(tmp_path: Path):
    mgr = IngestionManager(workspace_root=tmp_path)
    source = tmp_path / "short.md"
    text = "## A\n\ntoo short\n\n## B\n\nok this section has enough characters to pass the 50-char threshold so it works\n"
    chunks = list(mgr._split_by_sections(text, source))
    # Only section B should pass
    assert len(chunks) == 1
    assert "enough characters" in chunks[0][1]


def test_chunking_filters_sensitive_content(tmp_path: Path):
    mgr = IngestionManager(workspace_root=tmp_path)
    source = tmp_path / "secrets.md"
    text = "## Public\n\nnormal content here\n\n## Secret\n\nAPI_Key: sk-abcdefghijklmnopqrstuvwxyz0123456789\n"  # privacy-allow: *
    chunks = list(mgr._split_by_sections(text, source))
    # Only section "Public" should remain (or if "## Public" is just a heading, it needs 50 chars)
    for _, chunk in chunks:
        assert "API_Key" not in chunk
        assert "sk-" not in chunk


def test_save_and_load_checkpoint(tmp_path: Path):
    mgr = IngestionManager(workspace_root=tmp_path)
    # Override checkpoint path to a temp location
    mgr.state_dir = tmp_path / "state"
    mgr.state_dir.mkdir(parents=True, exist_ok=True)
    mgr._checkpoint_path = mgr.state_dir / "ingest_checkpoint.json"

    progress = IngestProgress(
        checkpoint_id="cp_test",
        total_files=10,
        total_chunks=50,
        written=25,
        failed=2,
        completed_chunk_ids=["c1", "c2", "c3"],
    )
    mgr._save_checkpoint(progress)

    # Verify file exists
    assert mgr._checkpoint_path.exists()
    data = json.loads(mgr._checkpoint_path.read_text())
    assert data["checkpoint_id"] == "cp_test"
    assert data["written"] == 25

    # Load it back
    loaded = mgr._load_checkpoint()
    assert loaded is not None
    assert loaded.total_chunks == 50
    assert len(loaded.completed_chunk_ids) == 3

    # Clear
    mgr._clear_checkpoint()
    assert not mgr._checkpoint_path.exists()


def test_load_nonexistent_checkpoint(tmp_path: Path):
    mgr = IngestionManager(workspace_root=tmp_path)
    mgr._checkpoint_path = tmp_path / "nonexistent.json"
    assert mgr._load_checkpoint() is None


def test_classify_tier():
    mgr = IngestionManager()
    assert mgr._classify_tier("core_keyword_a: something", "test.md") == "core"
    assert mgr._classify_tier("## 教训\n important lesson", "test.md") == "long"
    assert mgr._classify_tier("in_progress state", "test.md") == "working"
    assert mgr._classify_tier("short note", "memory/2026-01-01.md") == "short"
    assert mgr._classify_tier("normal content here", "test.md") == "medium"


def test_classify_type():
    mgr = IngestionManager()
    assert mgr._classify_type("This is a rule", "test.md") == "rule"
    assert mgr._classify_type("## 教训\n learned something", "test.md") == "lesson"
    assert mgr._classify_type("context info", "test.md") == "context"
    assert mgr._classify_type("in_progress: building", "test.md") == "status"
    assert mgr._classify_type("regular note", "test.md") == "note"


def test_infer_topic_from_stem():
    mgr = IngestionManager()
    topic = mgr._infer_topic("some content", Path("memory/rust_tips.md"))
    assert topic == "rust_tips"


def test_infer_topic_from_heading():
    mgr = IngestionManager()
    topic = mgr._infer_topic("# Project Alpha\n\ncontent", Path("MEMORY.md"))
    assert topic == "Project Alpha"


def test_build_payload_includes_v022_fields(tmp_path: Path):
    mgr = IngestionManager(workspace_root=tmp_path)
    source = tmp_path / "memory" / "test_topic.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    # Text must be >= 50 chars and pass sensitive filter
    text = "## Test\n\nThis is a test content section for memory payload with enough text to pass the threshold\n"
    source.write_text(text)
    chunks = list(mgr._split_by_sections(text, source))
    assert len(chunks) == 1, f"expected 1 chunk, got {len(chunks)}"
    chunk_id, ctext = chunks[0]
    assert len(ctext) >= 50
    payload = mgr._build_payload(ctext, source, chunk_id)
    assert "owner_confirmed" in payload
    assert payload["owner_confirmed"] is False
    assert "line_start" in payload
    assert payload["line_start"] is None
    assert "line_end" in payload
    assert "type" in payload
    assert payload["type"] == "note"
    assert "topic" in payload
    assert payload["topic"] is not None


def test_dry_run_returns_progress(tmp_path: Path):
    """Test that a dry-run ingestion creates correct state without hitting any services."""
    mgr = IngestionManager(workspace_root=tmp_path)
    # Write a memory file
    mem_file = tmp_path / "MEMORY.md"
    mem_file.write_text("## Topic\n\ncontent here for the ingestion test purpose\n")
    progress = mgr.run_ingestion(dry_run=True)
    assert progress.status == "completed"
    assert progress.total_files == 1
    assert progress.total_chunks == 1


def test_list_memory_files(tmp_path: Path):
    mgr = IngestionManager(workspace_root=tmp_path)
    # No files
    assert mgr._list_memory_files() == []

    # Create MEMORY.md
    (tmp_path / "MEMORY.md").write_text("# test")
    files = mgr._list_memory_files()
    assert len(files) == 1
    assert files[0].name == "MEMORY.md"

    # Create memory dir
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "a.md").write_text("# a")
    (mem_dir / "b.md").write_text("# b")
    files = mgr._list_memory_files()
    assert len(files) == 3  # MEMORY.md + a.md + b.md
