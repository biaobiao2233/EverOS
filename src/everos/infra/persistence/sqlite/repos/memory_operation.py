"""Repository for the persistent memory-operation idempotency ledger."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables import MemoryOperation


def canonical_json(value: Any) -> str:
    """Return deterministic strict JSON suitable for hashes and receipts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class _MemoryOperationRepo(RepoBase[MemoryOperation]):
    model = MemoryOperation

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    async def claim(self, operation: MemoryOperation) -> tuple[MemoryOperation, bool]:
        """Insert ``operation`` once; return ``(stored_row, was_created)``."""

        values = operation.model_dump()
        async with session_scope(self._factory) as session:
            result = await session.execute(
                sqlite_insert(MemoryOperation)
                .values(**values)
                .on_conflict_do_nothing(index_elements=["operation_id"])
            )
            await session.commit()
            stored = await session.get(MemoryOperation, operation.operation_id)
            if stored is None:  # pragma: no cover - defensive DB invariant
                raise RuntimeError("memory operation claim committed without a row")
            return stored, bool(result.rowcount)

    async def get(self, operation_id: str) -> MemoryOperation | None:
        async with session_scope(self._factory) as session:
            return await session.get(MemoryOperation, operation_id)

    async def mark_running(self, operation_id: str) -> MemoryOperation:
        """Clear a retryable failure before resuming the saved stage."""

        return await self._transition(
            operation_id,
            state="running",
            error_code=None,
            retryable=False,
        )

    async def mark_memcells_committed(
        self,
        operation_id: str,
        memcell_ids: Sequence[str],
    ) -> MemoryOperation:
        return await self._transition(
            operation_id,
            state="running",
            stage="memcells_committed",
            memcell_ids_json=canonical_json(list(memcell_ids)),
            error_code=None,
            retryable=False,
        )

    async def mark_completed(
        self,
        operation_id: str,
        response: Mapping[str, object],
    ) -> MemoryOperation:
        return await self._transition(
            operation_id,
            state="completed",
            stage="sync_dispatch_completed",
            response_json=canonical_json(dict(response)),
            error_code=None,
            retryable=False,
        )

    async def mark_failed(
        self,
        operation_id: str,
        *,
        error_code: str,
        retryable: bool,
    ) -> MemoryOperation:
        return await self._transition(
            operation_id,
            state="failed",
            error_code=error_code,
            retryable=retryable,
        )

    async def _transition(
        self,
        operation_id: str,
        **updates: object,
    ) -> MemoryOperation:
        async with session_scope(self._factory) as session:
            result = await session.execute(
                update(MemoryOperation)
                .where(MemoryOperation.operation_id == operation_id)
                # Completed is an irreversible terminal receipt.  A timeout
                # handler that arrives late must never downgrade it.
                .where(MemoryOperation.state != "completed")
                .values(**updates)
            )
            await session.commit()
            row = await session.get(MemoryOperation, operation_id)
            if row is None:
                raise KeyError(f"memory operation not found: {operation_id}")
            if not result.rowcount and row.state != "completed":
                raise RuntimeError(
                    f"memory operation transition did not apply: {operation_id}"
                )
            return row


memory_operation_repo = _MemoryOperationRepo()
