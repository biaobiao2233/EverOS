"""Persistent idempotency ledger for ``/memory/add`` and ``/memory/flush``.

The table stores request fingerprints and small progress receipts only.  Raw
conversation payloads remain in the existing buffer / memcell stores and are
never copied into the operation ledger.
"""

from __future__ import annotations

from typing import Literal

from sqlalchemy import Index

from everos.core.persistence.sqlite import BaseTable, Field

OperationKind = Literal["add", "flush"]
OperationState = Literal["running", "completed", "failed"]
OperationStage = Literal[
    "claimed",
    "memcells_committed",
    "sync_dispatch_completed",
]


class MemoryOperation(BaseTable, table=True):
    """One durable write operation, keyed by a caller-generated id."""

    __tablename__ = "memory_operation"  # type: ignore[assignment]
    __table_args__ = (
        Index(
            "ix_memory_operation_scope_session_state",
            "app_id",
            "project_id",
            "session_id",
            "state",
        ),
    )

    operation_id: str = Field(primary_key=True)
    # SQLModel 0.0.x cannot map ``typing.Literal`` to a SQLite column;
    # service/repository transitions enforce the finite-state contract.
    kind: str
    app_id: str = Field(default="default")
    project_id: str = Field(default="default")
    session_id: str = Field(index=True)
    request_sha256: str
    mode: str = Field(default="chat")
    plan_version: int = Field(default=1)
    hard_token_limit: int = Field(default=65536)
    hard_msg_limit: int = Field(default=500)
    message_count: int = Field(default=0)
    state: str = Field(default="running", index=True)
    stage: str = Field(default="claimed")
    memcell_ids_json: str = Field(default="[]")
    response_json: str | None = Field(default=None)
    error_code: str | None = Field(default=None)
    retryable: bool = Field(default=False)
