"""Durable per-message idempotency and publish-authority receipts.

Raw conversation content stays in ``unprocessed_buffer`` / ``memcell``.
This sidecar intentionally stores only stable identity + progress metadata so
late transport retries can be rejected after the buffer row has been consumed
without duplicating message bodies in another persistence surface.
"""

from __future__ import annotations

from sqlalchemy import Index

from everos.core.persistence.sqlite import BaseTable, Field


class MemoryMessageReceipt(BaseTable, table=True):
    """One durable logical-message receipt.

    ``idem_key`` is unique inside an app/project scope and includes the
    session component. ``message_id`` is the canonical EverOS id used by the
    buffer and MemCell ledger. ``authority_state`` is deliberately a string
    rather than ``Literal`` because SQLModel 0.0.x cannot map Literals.
    """

    __tablename__ = "memory_message_receipt"  # type: ignore[assignment]
    __table_args__ = (
        Index(
            "ux_memory_message_receipt_scope_idem",
            "app_id",
            "project_id",
            "idem_key",
            unique=True,
        ),
        Index(
            "ux_memory_message_receipt_message_id",
            "message_id",
            unique=True,
        ),
        Index(
            "ix_memory_message_receipt_scope_session_state",
            "app_id",
            "project_id",
            "session_id",
            "authority_state",
        ),
    )

    receipt_id: str = Field(primary_key=True)
    app_id: str = Field(default="default")
    project_id: str = Field(default="default")
    session_id: str = Field(index=True)
    idem_key: str
    message_id: str
    source: str = Field(default="api")
    external_ref: str | None = None
    revision: int = Field(default=0)
    payload_sha256: str
    authority_state: str = Field(default="pending_publish", index=True)
    authority_ref: str | None = None
    memcell_ids_json: str = Field(default="[]")
