"""Read helpers for durable per-message receipts.

Stage/publish/consume mutations intentionally stay in the service transaction
that also changes ``unprocessed_buffer`` / ``memory_operation`` so those
business states cannot diverge across a crash boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables import MemoryMessageReceipt


class _MemoryMessageReceiptRepo(RepoBase[MemoryMessageReceipt]):
    model = MemoryMessageReceipt

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    async def get_by_idem_key(
        self,
        *,
        app_id: str,
        project_id: str,
        idem_key: str,
    ) -> MemoryMessageReceipt | None:
        async with session_scope(self._factory) as session:
            stmt = select(MemoryMessageReceipt).where(
                MemoryMessageReceipt.app_id == app_id,
                MemoryMessageReceipt.project_id == project_id,
                MemoryMessageReceipt.idem_key == idem_key,
            )
            return (await session.execute(stmt)).scalars().first()

    async def map_by_message_ids(
        self,
        message_ids: Sequence[str],
    ) -> dict[str, MemoryMessageReceipt]:
        if not message_ids:
            return {}
        async with session_scope(self._factory) as session:
            stmt = select(MemoryMessageReceipt).where(
                MemoryMessageReceipt.message_id.in_(tuple(message_ids))
            )
            rows = list((await session.execute(stmt)).scalars().all())
        return {row.message_id: row for row in rows}


memory_message_receipt_repo = _MemoryMessageReceiptRepo()
