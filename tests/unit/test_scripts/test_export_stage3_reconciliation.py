"""Safety tests for the content-free Stage 3 reconciliation exporter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "export_stage3_reconciliation.py"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("_reconcile_exporter", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_is_read_only_and_content_free(tmp_path: Path) -> None:
    db = tmp_path / "system.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE memory_message_receipt (
                receipt_id TEXT PRIMARY KEY,
                app_id TEXT,
                project_id TEXT,
                session_id TEXT,
                idem_key TEXT,
                message_id TEXT,
                source TEXT,
                external_ref TEXT,
                revision INTEGER,
                payload_sha256 TEXT,
                authority_state TEXT,
                authority_ref TEXT
            );
            CREATE TABLE unprocessed_buffer (
                message_id TEXT PRIMARY KEY,
                app_id TEXT,
                project_id TEXT,
                session_id TEXT,
                track TEXT,
                sender_id TEXT,
                sender_name TEXT,
                role TEXT,
                timestamp TEXT,
                text TEXT,
                tool_calls_json TEXT,
                tool_call_id TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO memory_message_receipt VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "receipt-1",
                "chatgpt",
                "solo",
                "chatgpt-business-test",
                "__content_hash__:chatgpt-business-test:hash",
                "message-1",
                "__content_hash__",
                None,
                0,
                "payload-hash",
                "pending_publish",
                None,
            ),
        )
        conn.execute(
            "INSERT INTO unprocessed_buffer VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "message-1",
                "chatgpt",
                "solo",
                "chatgpt-business-test",
                "memorize",
                "user",
                "User",
                "user",
                "2026-08-28 12:00:00.123000",
                "TOP SECRET RAW MESSAGE",
                None,
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    manifest = tmp_path / "expected.json"
    manifest.write_text(
        json.dumps(
            {
                "expected": {
                    "summary": {"app_id": "chatgpt", "project_id": "solo"},
                    "sessions": [{"session_id": "chatgpt-business-test"}],
                }
            }
        ),
        encoding="utf-8",
    )

    before = _sha256(db)
    result = _load_exporter().export_rows(db, manifest)
    after = _sha256(db)

    assert before == after
    assert result["summary"]["receipt_count"] == 1
    assert result["summary"]["buffer_present_count"] == 1
    wire = json.dumps(result, ensure_ascii=False)
    assert "TOP SECRET RAW MESSAGE" not in wire
    row = result["rows"][0]
    assert row["buffer_present"] is True
    assert (
        row["buffer_text_sha256"]
        == hashlib.sha256(b"TOP SECRET RAW MESSAGE").hexdigest()
    )
    assert row["buffer_timestamp_ms"] == 1787918400123
