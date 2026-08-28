"""Export content-free Stage 3 receipt/buffer metadata for reconciliation.

The database is opened in SQLite read-only mode with ``query_only`` enabled.
No message text or content_items_json is selected or emitted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _payload_sha256(value: dict[str, Any]) -> str:
    wire = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def _timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return int(value)
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return int(parsed.timestamp() * 1000)


def _load_sessions(path: Path) -> tuple[str, str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("expected", payload)
    summary = expected.get("summary", {})
    app_id = str(summary.get("app_id") or "chatgpt")
    project_id = str(summary.get("project_id") or "solo")
    sessions = [str(item["session_id"]) for item in expected.get("sessions", [])]
    if not sessions:
        raise RuntimeError("expected manifest contains no recovery sessions")
    if len(set(sessions)) != len(sessions):
        raise RuntimeError("expected manifest contains duplicate session IDs")
    return app_id, project_id, sessions


def _chunks(values: list[str], size: int = 80) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def export_rows(db_path: Path, manifest_path: Path) -> dict[str, Any]:
    app_id, project_id, sessions = _load_sessions(manifest_path)
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        rows: list[dict[str, Any]] = []
        for chunk in _chunks(sessions):
            placeholders = ",".join("?" for _ in chunk)
            query = f"""
                SELECT
                    r.receipt_id,
                    r.app_id,
                    r.project_id,
                    r.session_id,
                    r.idem_key,
                    r.message_id,
                    r.source,
                    r.external_ref,
                    r.revision,
                    r.payload_sha256,
                    r.authority_state,
                    r.authority_ref,
                    CASE WHEN b.message_id IS NULL THEN 0 ELSE 1 END AS buffer_present,
                    b.app_id AS buffer_app_id,
                    b.project_id AS buffer_project_id,
                    b.session_id AS buffer_session_id,
                    b.track AS buffer_track,
                    b.sender_id AS buffer_sender_id,
                    b.sender_name AS buffer_sender_name,
                    b.role AS buffer_role,
                    b.timestamp AS buffer_timestamp,
                    b.text AS buffer_text,
                    b.tool_calls_json AS buffer_tool_calls_json,
                    b.tool_call_id AS buffer_tool_call_id
                FROM memory_message_receipt AS r
                LEFT JOIN unprocessed_buffer AS b
                  ON b.message_id = r.message_id
                WHERE r.app_id = ?
                  AND r.project_id = ?
                  AND r.session_id IN ({placeholders})
                ORDER BY r.session_id, r.idem_key
            """
            params: list[Any] = [app_id, project_id, *chunk]
            for row in conn.execute(query, params):
                item = dict(row)
                item["buffer_present"] = bool(item["buffer_present"])
                raw_timestamp = item.pop("buffer_timestamp", None)
                raw_text = item.pop("buffer_text", None)
                raw_tool_calls = item.pop("buffer_tool_calls_json", None)
                raw_tool_call_id = item.pop("buffer_tool_call_id", None)
                timestamp_ms = _timestamp_ms(raw_timestamp)
                item["buffer_timestamp_ms"] = timestamp_ms
                item["buffer_text_sha256"] = _sha256_text(raw_text)
                if item["buffer_present"] and timestamp_ms is not None:
                    payload = {
                        "sender_id": item["buffer_sender_id"],
                        "sender_name": item["buffer_sender_name"],
                        "role": item["buffer_role"],
                        "timestamp": timestamp_ms,
                        "content": raw_text,
                        "tool_calls": (
                            json.loads(raw_tool_calls) if raw_tool_calls else None
                        ),
                        "tool_call_id": raw_tool_call_id,
                    }
                    item["buffer_payload_sha256"] = _payload_sha256(payload)
                    payload_without_timestamp = {
                        key: value
                        for key, value in payload.items()
                        if key != "timestamp"
                    }
                    item["buffer_payload_without_timestamp_sha256"] = _payload_sha256(
                        payload_without_timestamp
                    )
                else:
                    item["buffer_payload_sha256"] = None
                    item["buffer_payload_without_timestamp_sha256"] = None
                rows.append(item)

        state_counts = Counter(str(row["authority_state"]) for row in rows)
        source_counts = Counter(str(row["source"]) for row in rows)
        buffer_present = sum(1 for row in rows if row["buffer_present"])
        return {
            "schema_version": 1,
            "kind": "ev-ingest-01-stage3-server-export",
            "summary": {
                "app_id": app_id,
                "project_id": project_id,
                "requested_session_count": len(sessions),
                "receipt_count": len(rows),
                "buffer_present_count": buffer_present,
                "authority_state_counts": dict(sorted(state_counts.items())),
                "source_counts": dict(sorted(source_counts.items())),
            },
            "rows": rows,
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = export_rows(args.db, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
