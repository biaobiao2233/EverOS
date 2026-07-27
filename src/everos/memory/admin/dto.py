"""Strict wire DTOs for the private, read-only admin API."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AdminMemoryKind(StrEnum):
    """Recognised Markdown truth-source kinds, inferred from their path."""

    EPISODE = "episode"
    PROFILE = "profile"
    ATOMIC_FACT = "atomic_fact"
    FORESIGHT = "foresight"
    AGENT_CASE = "agent_case"
    AGENT_SKILL = "agent_skill"
    OTHER = "other"


class AdminMemoryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    kind: AdminMemoryKind
    size_bytes: int = Field(ge=0)
    modified_at: dt.datetime


class AdminMemoryFilesData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AdminMemoryFile] = Field(default_factory=list)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_count: int = Field(ge=0)


class AdminMemoryFilesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: AdminMemoryFilesData


class AdminMemoryFileContentData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    kind: AdminMemoryKind
    size_bytes: int = Field(ge=0)
    modified_at: dt.datetime
    content: str


class AdminMemoryFileContentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: AdminMemoryFileContentData


class QueueStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["available", "unavailable"]
    reason: str | None = None
    pending: int | None = Field(default=None, ge=0)
    done: int | None = Field(default=None, ge=0)
    failed_retryable: int | None = Field(default=None, ge=0)
    failed_permanent: int | None = Field(default=None, ge=0)
    max_lsn: int | None = Field(default=None, ge=0)
    last_processed_lsn: int | None = Field(default=None, ge=0)


class MemoryRootStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["available", "unavailable"]
    reason: str | None = None
    path: str | None = None
    exists: bool | None = None


class CascadeStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["available", "unavailable"]
    reason: str | None = None
    running: bool | None = None
    queue: QueueStatus


class IndexingStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["available", "unavailable"]
    reason: str | None = None
    lancedb_directory_exists: bool | None = None


class PipelineStatusData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_root: MemoryRootStatus
    cascade: CascadeStatus
    indexing: IndexingStatus


class PipelineStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: PipelineStatusData
