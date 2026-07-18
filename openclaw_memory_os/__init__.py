"""OpenClaw Memory OS.

A governance-layer dashboard and recall-testing layer designed to sit in front of
an OpenClaw-style memory store. Ships with a sample-data backend for offline
demo / tests, and an optional Qdrant adapter that activates when
``qdrant-client`` is installed and ``QDRANT_URL`` is set.

This package deliberately does **not** physically delete memories. Deletion
flows are review-only and emit candidate lists for human approval.
"""

from .config import Settings, get_settings
from .models import (
    AuditLogEntry,
    ConsolidationRequest,
    ConsolidationResult,
    FeedbackEntry,
    HealthSummary,
    IngestProgress,
    Memory,
    MemoryStatus,
    MemoryTier,
    RecallHit,
)

__all__ = [
    "Settings",
    "get_settings",
    "Memory",
    "MemoryTier",
    "MemoryStatus",
    "RecallHit",
    "HealthSummary",
    "AuditLogEntry",
    "FeedbackEntry",
    "ConsolidationRequest",
    "ConsolidationResult",
    "IngestProgress",
]

__version__ = "0.3.0"