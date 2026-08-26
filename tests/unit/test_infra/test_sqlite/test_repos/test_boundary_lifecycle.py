"""Durable boundary lifecycle checkpoint monotonicity and due selection."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite import BoundaryLifecycle
from everos.infra.persistence.sqlite.repos.boundary_lifecycle import (
    _BoundaryLifecycleRepo,
)


@pytest.fixture
async def repo(tmp_path: Path) -> _BoundaryLifecycleRepo:
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return _BoundaryLifecycleRepo(session_factory=factory)


def _row(*, revision: int, consumed: bool = False) -> BoundaryLifecycle:
    now = dt.datetime.now(dt.UTC)
    return BoundaryLifecycle(
        lifecycle_id="lifecycle-1",
        app_id="app",
        project_id="project",
        session_id="session",
        track="memorize",
        revision=revision,
        message_digest=f"digest-{revision}",
        message_ids_json=f'["message-{revision}"]',
        observed_at=now,
        idle_deadline=now - dt.timedelta(seconds=1),
        max_deadline=now + dt.timedelta(seconds=30),
        consumed=consumed,
        state="consumed" if consumed else "waiting",
    )


async def test_lifecycle_revision_is_monotone_and_consumed_is_terminal(
    repo: _BoundaryLifecycleRepo,
) -> None:
    await repo.upsert(_row(revision=2))
    await repo.upsert(_row(revision=1))
    stored = await repo.get_for_session(
        "session", app_id="app", project_id="project", track="memorize"
    )
    assert stored is not None
    assert stored.revision == 2
    assert stored.message_digest == "digest-2"

    await repo.upsert(_row(revision=2, consumed=True))
    await repo.upsert(_row(revision=2, consumed=False))
    stored = await repo.get_for_session(
        "session", app_id="app", project_id="project", track="memorize"
    )
    assert stored is not None
    assert stored.consumed is True


async def test_lifecycle_due_rows_are_restart_discoverable(
    repo: _BoundaryLifecycleRepo,
) -> None:
    await repo.upsert(_row(revision=1))
    due = await repo.list_due(dt.datetime.now(dt.UTC))
    assert [row.lifecycle_id for row in due] == ["lifecycle-1"]
