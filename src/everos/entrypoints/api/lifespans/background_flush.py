"""Lifecycle wiring for the durable, bounded background flush worker."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from everos.core.lifespan import LifespanProvider
from everos.service.background_flush import get_background_flush_scheduler


class BackgroundFlushLifespanProvider(LifespanProvider):
    """Recover queued work after all extraction dependencies are ready."""

    def __init__(self, order: int = 60) -> None:
        super().__init__(name="background_flush", order=order)

    async def startup(self, app: FastAPI) -> Any:
        scheduler = get_background_flush_scheduler()
        await scheduler.start()
        return scheduler

    async def shutdown(self, app: FastAPI) -> None:
        await get_background_flush_scheduler().stop()
