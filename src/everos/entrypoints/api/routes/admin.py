"""Private, bearer-protected read-only endpoints for local administration."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request

from everos.memory.admin import (
    AdminFileNotFoundError,
    AdminFileTooLargeError,
    AdminInvalidPathError,
    AdminMemoryFileContentResponse,
    AdminMemoryFilesResponse,
    AdminMemoryKind,
    PipelineStatusResponse,
)
from everos.service import admin as admin_service

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get("/memory-files", response_model=AdminMemoryFilesResponse)
async def get_memory_files(
    kind: AdminMemoryKind | None = None,
    query: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 50,
) -> AdminMemoryFilesResponse:
    """List safe Markdown-source metadata, using stable path ordering."""
    return await admin_service.list_files(
        kind=kind, query=query, page=page, page_size=page_size
    )


@router.get("/memory-files/content", response_model=AdminMemoryFileContentResponse)
async def get_memory_file_content(
    path: Annotated[str, Query(min_length=1, max_length=1024)],
) -> AdminMemoryFileContentResponse:
    """Return one bounded UTF-8 Markdown source named by a listed path."""
    try:
        return await admin_service.get_file_content(path)
    except AdminInvalidPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AdminFileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AdminFileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(request: Request) -> PipelineStatusResponse:
    """Return observed memory-root, Cascade, and indexing state only."""
    return await admin_service.pipeline_status(request.app)
