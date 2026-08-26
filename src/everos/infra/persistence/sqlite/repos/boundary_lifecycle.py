"""Persistence for recoverable boundary lifecycle checkpoints."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables import BoundaryLifecycle


class _BoundaryLifecycleRepo(RepoBase[BoundaryLifecycle]):
    model = BoundaryLifecycle

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    async def get_for_session(
        self,
        session_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
        track: str = "memorize",
    ) -> BoundaryLifecycle | None:
        async with session_scope(self._factory) as session:
            stmt = select(BoundaryLifecycle).where(
                BoundaryLifecycle.app_id == app_id,
                BoundaryLifecycle.project_id == project_id,
                BoundaryLifecycle.session_id == session_id,
                BoundaryLifecycle.track == track,
            )
            return (await session.execute(stmt)).scalars().first()

    async def upsert(self, row: BoundaryLifecycle) -> None:
        async with session_scope(self._factory) as session:
            current = await session.get(BoundaryLifecycle, row.lifecycle_id)
            if current is None:
                session.add(row)
            else:
                # Revision is the monotone boundary identity. A late replay
                # must never roll a newer wait checkpoint back; likewise an
                # already-consumed checkpoint cannot be resurrected by a
                # duplicate request at the same revision.
                if row.revision < current.revision or (
                    current.consumed and not row.consumed
                ):
                    return
                for field in (
                    "revision",
                    "message_digest",
                    "message_ids_json",
                    "should_wait",
                    "state",
                    "authority_state",
                    "observed_at",
                    "idle_deadline",
                    "max_deadline",
                    "final_requested",
                    "consumed",
                ):
                    setattr(current, field, getattr(row, field))
            await session.commit()

    async def list_due(self, now: dt.datetime) -> list[BoundaryLifecycle]:
        async with session_scope(self._factory) as session:
            stmt = select(BoundaryLifecycle).where(
                BoundaryLifecycle.state == "waiting",
                BoundaryLifecycle.consumed.is_(False),
                (BoundaryLifecycle.max_deadline <= now)
                | (BoundaryLifecycle.idle_deadline <= now),
            )
            return list((await session.execute(stmt)).scalars().all())


boundary_lifecycle_repo = _BoundaryLifecycleRepo()
