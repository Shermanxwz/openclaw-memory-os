from __future__ import annotations

from openclaw_memory_os.backends import QdrantBackend


def test_payload_to_memory_casts_integer_superseded_by_to_string():
    backend = QdrantBackend.__new__(QdrantBackend)
    mem = backend._payload_to_memory(
        62,
        {
            "content": "old memory",
            "source": "memory/2026-07-13.md",
            "status": "superseded",
            "superseded_by": 2122,
        },
    )
    assert mem is not None
    assert mem.id == "62"
    assert mem.superseded_by == "2122"
