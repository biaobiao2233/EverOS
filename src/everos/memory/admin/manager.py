"""Safe filesystem browsing and truthful runtime status for admin clients."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.sqlite import md_change_state_repo

from .dto import (
    AdminMemoryFile,
    AdminMemoryFileContentData,
    AdminMemoryFileContentResponse,
    AdminMemoryFilesData,
    AdminMemoryFilesResponse,
    AdminMemoryKind,
    CascadeStatus,
    IndexingStatus,
    MemoryRootStatus,
    PipelineStatusData,
    PipelineStatusResponse,
    QueueStatus,
)

MAX_CONTENT_BYTES = 1_048_576
"""Largest Markdown source the content endpoint will return (1 MiB)."""

_SYSTEM_DIRECTORIES = frozenset({".index", ".tmp"})
logger = get_logger(__name__)


class AdminInvalidPathError(ValueError):
    """The requested path is not a safe, emitted Markdown relative path."""


class AdminFileNotFoundError(FileNotFoundError):
    """The requested Markdown source no longer exists or is not readable."""


class AdminFileTooLargeError(ValueError):
    """The requested Markdown source exceeds :data:`MAX_CONTENT_BYTES`."""


def _kind_for(relative: PurePosixPath) -> AdminMemoryKind:
    parts = relative.parts
    if "episodes" in parts:
        return AdminMemoryKind.EPISODE
    if relative.name == "user.md":
        return AdminMemoryKind.PROFILE
    if ".atomic_facts" in parts:
        return AdminMemoryKind.ATOMIC_FACT
    if ".foresights" in parts:
        return AdminMemoryKind.FORESIGHT
    if ".cases" in parts:
        return AdminMemoryKind.AGENT_CASE
    if relative.name == "SKILL.md" and "skills" in parts:
        return AdminMemoryKind.AGENT_SKILL
    return AdminMemoryKind.OTHER


def _is_safe_regular_markdown(root: Path, candidate: Path) -> bool:
    """Return whether *candidate* is a non-symlink Markdown file under root."""
    try:
        if not candidate.is_file():
            return False
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
        resolved_candidate.relative_to(resolved_root)
        relative = candidate.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return False
    except (OSError, ValueError):
        return False
    return True


def _is_browsable_relative_path(relative: PurePosixPath) -> bool:
    """Exclude system-managed directories from the Markdown truth-source API."""
    return not any(part in _SYSTEM_DIRECTORIES for part in relative.parts)


def _list_files_sync(
    root: Path,
    kind: AdminMemoryKind | None,
    query: str | None,
    page: int,
    page_size: int,
) -> AdminMemoryFilesResponse:
    if not root.is_dir():
        return AdminMemoryFilesResponse(
            data=AdminMemoryFilesData(
                page=page,
                page_size=page_size,
                total_count=0,
            )
        )
    items: list[AdminMemoryFile] = []
    needle = query.casefold() if query else None
    for candidate in root.rglob("*.md"):
        if not _is_safe_regular_markdown(root, candidate):
            continue
        relative = PurePosixPath(candidate.relative_to(root).as_posix())
        if not _is_browsable_relative_path(relative):
            continue
        file_kind = _kind_for(relative)
        path = relative.as_posix()
        if kind is not None and file_kind != kind:
            continue
        if needle is not None and needle not in path.casefold():
            continue
        stat = candidate.stat()
        items.append(
            AdminMemoryFile(
                path=path,
                kind=file_kind,
                size_bytes=stat.st_size,
                modified_at=dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.UTC),
            )
        )
    items.sort(key=lambda item: item.path)
    start = (page - 1) * page_size
    return AdminMemoryFilesResponse(
        data=AdminMemoryFilesData(
            items=items[start : start + page_size],
            page=page,
            page_size=page_size,
            total_count=len(items),
        )
    )


async def list_memory_files(
    *,
    kind: AdminMemoryKind | None,
    query: str | None,
    page: int,
    page_size: int,
) -> AdminMemoryFilesResponse:
    """List file metadata only; file content is never read on this path."""
    return await asyncio.to_thread(
        _list_files_sync, MemoryRoot.default().root, kind, query, page, page_size
    )


def _resolve_requested_file(root: Path, path: str) -> tuple[Path, PurePosixPath]:
    if not path:
        raise AdminInvalidPathError("invalid memory file path")
    if (
        "\\" in path
        or "\x00" in path
        or PurePosixPath(path).is_absolute()
        or PureWindowsPath(path).is_absolute()
        or bool(PureWindowsPath(path).drive)
    ):
        raise AdminInvalidPathError("invalid memory file path")
    relative = PurePosixPath(path)
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise AdminInvalidPathError("invalid memory file path")
    if relative.suffix != ".md":
        raise AdminInvalidPathError("memory file path must name a .md file")
    if not _is_browsable_relative_path(relative):
        raise AdminInvalidPathError("memory file path is not browsable")
    candidate = root.joinpath(*relative.parts)
    if not _is_safe_regular_markdown(root, candidate):
        raise AdminFileNotFoundError("memory file not found")
    return candidate, relative


def _read_file_sync(root: Path, path: str) -> AdminMemoryFileContentResponse:
    candidate, relative = _resolve_requested_file(root, path)
    try:
        with candidate.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            if stat.st_size > MAX_CONTENT_BYTES:
                raise AdminFileTooLargeError(
                    "memory file exceeds the 1 MiB content limit"
                )
            raw = handle.read(MAX_CONTENT_BYTES + 1)
    except OSError as exc:
        raise AdminFileNotFoundError("memory file not found") from exc
    if len(raw) > MAX_CONTENT_BYTES:
        raise AdminFileTooLargeError("memory file exceeds the 1 MiB content limit")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AdminInvalidPathError("memory file is not valid UTF-8") from exc
    if not _is_safe_regular_markdown(root, candidate):
        raise AdminFileNotFoundError("memory file not found")
    return AdminMemoryFileContentResponse(
        data=AdminMemoryFileContentData(
            path=relative.as_posix(),
            kind=_kind_for(relative),
            size_bytes=len(raw),
            modified_at=dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.UTC),
            content=content,
        )
    )


async def get_memory_file_content(path: str) -> AdminMemoryFileContentResponse:
    """Read one previously listed Markdown source, bounded and UTF-8 only."""
    return await asyncio.to_thread(_read_file_sync, MemoryRoot.default().root, path)


async def get_pipeline_status(app: Any) -> PipelineStatusResponse:
    """Report only state this process can observe; never estimate progress."""
    root = MemoryRoot.default().root
    lifecycle_data = getattr(app.state, "lifespan_data", {})
    cascade = (
        lifecycle_data.get("cascade") if isinstance(lifecycle_data, dict) else None
    )
    sqlite_ready = isinstance(lifecycle_data, dict) and "sqlite" in lifecycle_data
    lancedb_ready = isinstance(lifecycle_data, dict) and "lancedb" in lifecycle_data

    if sqlite_ready:
        try:
            summary = await md_change_state_repo.queue_summary()
            queue = QueueStatus(status="available", **dataclasses.asdict(summary))
        except Exception as exc:
            logger.warning(
                "admin_queue_status_unavailable",
                error_type=type(exc).__name__,
            )
            queue = QueueStatus(
                status="unavailable",
                reason="sqlite storage is unavailable",
            )
    else:
        queue = QueueStatus(
            status="unavailable", reason="sqlite lifespan is unavailable"
        )

    if cascade is None:
        cascade_status = CascadeStatus(
            status="unavailable",
            reason="cascade lifespan is unavailable",
            running=None,
            queue=queue,
        )
    else:
        cascade_status = CascadeStatus(
            status="available",
            running=bool(getattr(cascade, "_started", False)),
            queue=queue,
        )
    if lancedb_ready:
        indexing = IndexingStatus(
            status="available",
            lancedb_directory_exists=root.joinpath(".index", "lancedb").is_dir(),
        )
    else:
        indexing = IndexingStatus(
            status="unavailable",
            reason="lancedb lifespan is unavailable",
            lancedb_directory_exists=None,
        )
    return PipelineStatusResponse(
        data=PipelineStatusData(
            memory_root=MemoryRootStatus(
                status="available" if root.is_dir() else "unavailable",
                reason=None
                if root.is_dir()
                else "memory root directory does not exist",
                path=str(root),
                exists=root.is_dir(),
            ),
            cascade=cascade_status,
            indexing=indexing,
        )
    )
