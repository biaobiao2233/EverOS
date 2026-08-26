"""Durable, authority-neutral lifecycle checkpoint for a boundary tail."""

from __future__ import annotations

from sqlalchemy import Index, UniqueConstraint

from everos.component.utils.datetime import UtcDatetime
from everos.core.persistence.sqlite import BaseTable, Field
from everos.core.persistence.sqlite.base import UtcDateTimeColumn


class BoundaryLifecycle(BaseTable, table=True):
    """One recoverable wait/deadline record per scoped session tail.

    This table never grants publish authority. ``authority_state`` is a
    snapshot used to fail closed if a caller tries to process an unpublished
    tail after restart.
    """

    __tablename__ = "boundary_lifecycle"  # type: ignore[assignment]
    __table_args__ = (
        UniqueConstraint(
            "app_id",
            "project_id",
            "session_id",
            "track",
            name="uq_boundary_lifecycle_scope_session_track",
        ),
        Index("ix_boundary_lifecycle_due", "state", "idle_deadline", "max_deadline"),
    )

    lifecycle_id: str = Field(primary_key=True)
    app_id: str = Field(default="default")
    project_id: str = Field(default="default")
    session_id: str = Field(index=True)
    track: str = Field(default="memorize")
    revision: int = Field(default=0)
    message_digest: str
    message_ids_json: str = "[]"
    should_wait: bool | None = None
    state: str = Field(default="waiting", index=True)
    authority_state: str = Field(default="published", index=True)
    observed_at: UtcDatetime = Field(sa_type=UtcDateTimeColumn)
    idle_deadline: UtcDatetime | None = Field(default=None, sa_type=UtcDateTimeColumn)
    max_deadline: UtcDatetime | None = Field(default=None, sa_type=UtcDateTimeColumn)
    final_requested: bool = False
    consumed: bool = False
