"""Bounded, durable dispatcher for deferred memory flush operations.

SQLite is the queue of record.  The in-memory queue contains at most one
"check the ledger" wake-up signal, so HTTP traffic can never create an
unbounded number of tasks or concurrent Gemini calls.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.sqlite import memory_operation_repo

from ._session_lock import scoped_session_lock
from .memorize import run_background_flush_operation

logger = get_logger(__name__)


class BackgroundFlushScheduler:
    """Single-worker dispatcher backed by the persistent operation ledger."""

    def __init__(self) -> None:
        self._wake_queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is not None and not self._worker.done():
            return
        self._worker = asyncio.create_task(
            self._run(),
            name="everos-background-flush",
        )
        self.wake()

    async def stop(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker

    def wake(self) -> None:
        """Request a ledger scan without allocating another processing task."""

        if self._wake_queue.full():
            return
        self._wake_queue.put_nowait(None)

    async def _run(self) -> None:
        while True:
            await self._wake_queue.get()
            attempted: set[str] = set()
            while True:
                pending = await memory_operation_repo.list_background_flush_pending()
                operation = next(
                    (item for item in pending if item.operation_id not in attempted),
                    None,
                )
                if operation is None:
                    break
                attempted.add(operation.operation_id)
                try:
                    # A reserved cross-process lock makes the default
                    # concurrency of one global even under multiple uvicorn
                    # workers. The operation's own scoped session lock remains
                    # inside memorize and protects its buffer.
                    async with scoped_session_lock(
                        MemoryRoot.default(),
                        "__background_flush_worker__",
                        app_id="__everos__",
                        project_id="__system__",
                    ):
                        await run_background_flush_operation(operation.operation_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # The normal memorize state machine persists a retryable
                    # failure. Avoid a hot retry loop; the next wake or restart
                    # gives it another bounded attempt.
                    logger.exception(
                        "background_flush_failed",
                        extra={"operation_id": operation.operation_id},
                    )


_scheduler = BackgroundFlushScheduler()


def get_background_flush_scheduler() -> BackgroundFlushScheduler:
    return _scheduler
