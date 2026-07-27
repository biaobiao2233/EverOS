"""Application-service facade for the private read-only admin API."""

from __future__ import annotations

from typing import Any

from everos.memory.admin import (
    AdminMemoryFileContentResponse,
    AdminMemoryFilesResponse,
    AdminMemoryKind,
    PipelineStatusResponse,
    get_memory_file_content,
    get_pipeline_status,
    list_memory_files,
)


async def list_files(
    *, kind: AdminMemoryKind | None, query: str | None, page: int, page_size: int
) -> AdminMemoryFilesResponse:
    return await list_memory_files(
        kind=kind, query=query, page=page, page_size=page_size
    )


async def get_file_content(path: str) -> AdminMemoryFileContentResponse:
    return await get_memory_file_content(path)


async def pipeline_status(app: Any) -> PipelineStatusResponse:
    return await get_pipeline_status(app)
