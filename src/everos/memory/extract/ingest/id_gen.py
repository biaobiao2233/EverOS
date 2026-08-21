"""Deterministic, human-readable ``message_id`` generation.

Format: ``m_<session_id>_<timestamp_ms>_<idx:03d>``.

Human-readable so logs / debugging / md entries stay greppable. Deterministic
so caller retries (same payload) produce the same ids — pipeline merge
naturally dedupes via the message_id PK in ``unprocessed_buffer``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

_IDX_PAD = 3  # caller batches are capped at 500 messages (DTO limit), 3 digits cover it


def gen_message_id(session_id: str, timestamp_ms: int, idx: int) -> str:
    """Return ``m_<session_id>_<timestamp_ms>_<idx:03d>``."""
    return f"m_{session_id}_{timestamp_ms}_{idx:0{_IDX_PAD}d}"


def gen_staged_message_id(
    session_id: str,
    timestamp_ms: int,
    message: dict[str, Any],
    *,
    app_id: str,
    project_id: str,
) -> str:
    """Return a chunk-position-independent id for deferred ingestion."""

    wire = json.dumps(
        {
            "app_id": app_id,
            "project_id": project_id,
            "session_id": session_id,
            "message": message,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(wire.encode("utf-8")).hexdigest()[:24]
    return f"ms_{session_id}_{timestamp_ms}_{digest}"
