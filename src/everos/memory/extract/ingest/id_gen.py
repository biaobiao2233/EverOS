"""Deterministic, human-readable ``message_id`` generation.

Format: ``m_<session_id>_<timestamp_ms>_<idx:03d>``.

Human-readable so logs / debugging / md entries stay greppable. Deterministic
so caller retries (same payload) produce the same ids — pipeline merge
naturally dedupes via the message_id PK in ``unprocessed_buffer``.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, NamedTuple

_IDX_PAD = 3  # caller batches are capped at 500 messages (DTO limit), 3 digits cover it
_SOURCE_RE = re.compile(r"^[a-z0-9_.-]{1,32}$")


class StagedMessageIdentity(NamedTuple):
    """Stable logical identity + content fingerprint for deferred ingestion."""

    receipt_id: str
    idem_key: str
    message_id: str
    source: str
    external_ref: str | None
    revision: int
    payload_sha256: str


def external_idem_key(session_id: str, source: str, external_ref: str) -> str:
    """Build the scoped logical key used by stage + publish."""

    if not _SOURCE_RE.fullmatch(source):
        raise ValueError("invalid staged message source")
    if source.startswith("__"):
        raise ValueError("staged message source uses a reserved namespace")
    external_ref = external_ref.strip()
    if not external_ref or len(external_ref) > 512:
        raise ValueError("external_ref must be a non-empty string")
    return f"{source}:{session_id}:{external_ref}"


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


def staged_message_identity(
    session_id: str,
    message: dict[str, Any],
    *,
    app_id: str,
    project_id: str,
) -> StagedMessageIdentity:
    """Return the durable logical identity for one staged message.

    ``external_ref`` is authoritative when supplied.  The revision is stored
    separately so a higher revision can supersede content while retaining the
    same logical message identity.  Sources that do not own a stable external
    id fall back to a canonical content hash (including event timestamp), which
    is still independent of request batch position.
    """

    source_raw = message.get("source") or "api"
    if not isinstance(source_raw, str) or not _SOURCE_RE.fullmatch(source_raw):
        raise ValueError("invalid staged message source")
    if source_raw.startswith("__"):
        raise ValueError("staged message source uses a reserved namespace")
    source = source_raw

    external_raw = message.get("external_ref")
    if external_raw is None:
        external_ref = None
    elif isinstance(external_raw, str) and external_raw.strip():
        external_ref = external_raw.strip()
    else:
        raise ValueError("external_ref must be a non-empty string")

    revision_raw = message.get("revision", 0)
    if isinstance(revision_raw, bool):
        raise ValueError("revision must be a non-negative integer")
    try:
        revision = int(revision_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("revision must be a non-negative integer") from exc
    if revision < 0:
        raise ValueError("revision must be a non-negative integer")
    if external_ref is None and revision != 0:
        raise ValueError("revision requires external_ref")

    payload = {
        key: value
        for key, value in message.items()
        if key not in {"source", "external_ref", "revision"}
    }
    wire = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    payload_sha256 = hashlib.sha256(wire.encode("utf-8")).hexdigest()

    logical_ref = external_ref if external_ref is not None else payload_sha256
    logical_source = source if external_ref is not None else "__content_hash__"
    idem_key = (
        external_idem_key(session_id, logical_source, logical_ref)
        if external_ref is not None
        else f"{logical_source}:{session_id}:{logical_ref}"
    )
    scope_wire = f"{app_id}\0{project_id}\0{idem_key}"
    digest = hashlib.sha256(scope_wire.encode("utf-8")).hexdigest()
    return StagedMessageIdentity(
        receipt_id=f"evmsg1-{digest}",
        idem_key=idem_key,
        message_id=f"ms_{session_id}_{digest[:24]}",
        source=logical_source,
        external_ref=external_ref,
        revision=revision,
        payload_sha256=payload_sha256,
    )
