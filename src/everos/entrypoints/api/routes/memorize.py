"""POST /api/v1/memory/add and /api/v1/memory/flush.

DTOs follow the v1 API brief (01_v1_api_brief.md §2 / §3). Routes are
thin adapters: validate the DTO, dump to dict, hand to service. No
business logic lives here.

``/flush`` is OSS-only (the cloud edition decides boundary timing
server-side and does not expose this endpoint).
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from everos.core.errors import MultimodalError
from everos.core.observability.tracing import gen_request_id
from everos.service import (
    MemoryMessageConflictError,
    MemoryMessageRecoveryError,
    MemoryOperationConflictError,
    MemoryOperationRecoveryError,
    MemoryOperationStatus,
    get_memory_operation_status,
    memorize,
    publish,
    queue_background_flush,
    stage,
)
from everos.service.background_flush import get_background_flush_scheduler

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


# ── Path-safe identifier ────────────────────────────────────────────────────
# ``app_id`` / ``project_id`` become directory segments under the memory
# root, so they must reject ``.`` and ``..`` (path traversal). The basic
# character whitelist is enforced via ``pattern`` (pydantic_core uses the
# Rust regex engine, which does NOT support lookaround), and the two
# reserved tokens are filtered out with a follow-up ``AfterValidator``.
_PATH_SAFE_CHARSET = r"^[a-zA-Z0-9_.@+-]+$"
_PATH_TRAVERSAL_TOKENS = frozenset({".", ".."})
_PATH_SAFE_RE = re.compile(_PATH_SAFE_CHARSET)
_ADD_OPERATION_ID = r"^evop1-add-[0-9a-f]{64}$"
_STAGE_OPERATION_ID = r"^evop1-stage-[0-9a-f]{64}$"
_PUBLISH_OPERATION_ID = r"^evop1-publish-[0-9a-f]{64}$"
_FLUSH_OPERATION_ID = r"^evop1-flush-[0-9a-f]{64}$"
_STAGE_SOURCE = r"^[a-z0-9_.-]{1,32}$"
_PAYLOAD_SHA256 = r"^[0-9a-f]{64}$"


def _reject_path_traversal(value: str) -> str:
    if value in _PATH_TRAVERSAL_TOKENS:
        raise ValueError("'.' and '..' are reserved (path traversal)")
    if not _PATH_SAFE_RE.match(value):
        raise ValueError(
            "Only alphanumerics, underscore, dot, hyphen, @, and + are allowed"
        )
    return value


PathSafeId = Annotated[str, AfterValidator(_reject_path_traversal)]


# DTOs ────────────────────────────────────────────────────────────────────────


class ToolFunctionDTO(BaseModel):
    name: str
    arguments: str  # JSON string per OpenAI Chat Completions spec


class ToolCallDTO(BaseModel):
    id: str
    type: str = "function"
    function: ToolFunctionDTO


class ContentItemDTO(BaseModel):
    """Content piece (v1 API brief appendix A)."""

    type: Literal["text", "image", "audio", "doc", "pdf", "html", "email"]
    text: str | None = None
    uri: str | None = None
    base64: str | None = None
    ext: str | None = None
    name: str | None = None
    extras: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class MessageItemDTO(BaseModel):
    # sender_id becomes an owner directory on the episode write path.
    sender_id: PathSafeId = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=_PATH_SAFE_CHARSET,
    )
    sender_name: str | None = None
    role: Literal["user", "assistant", "tool"]
    timestamp: int = Field(
        ...,
        gt=0,
        description=(
            "Message event time as Unix epoch in **milliseconds** "
            "(v1 API contract; the algo layer auto-detects sec vs ms "
            "for backward compat but the contract is ms)."
        ),
    )
    content: str | list[ContentItemDTO]
    tool_calls: list[ToolCallDTO] | None = None
    tool_call_id: str | None = None


class StagedMessageItemDTO(MessageItemDTO):
    """Message plus caller-owned logical identity for deferred ingestion."""

    source: str = Field(default="api", pattern=_STAGE_SOURCE)
    external_ref: str | None = Field(default=None, min_length=1, max_length=512)
    revision: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_staged_identity(self) -> StagedMessageItemDTO:
        if self.source.startswith("__"):
            raise ValueError("staged message source uses a reserved namespace")
        if self.source in {"web", "chatgpt"} and self.external_ref is None:
            raise ValueError(f"{self.source} staged messages require external_ref")
        if self.revision != 0 and self.external_ref is None:
            raise ValueError("revision requires external_ref")
        return self


class MemorizeAddRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    app_id: PathSafeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=_PATH_SAFE_CHARSET,
    )
    project_id: PathSafeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=_PATH_SAFE_CHARSET,
    )
    messages: list[MessageItemDTO] = Field(..., min_length=1, max_length=500)
    operation_id: str | None = Field(
        default=None,
        pattern=_ADD_OPERATION_ID,
        description="Optional caller-stable idempotency key.",
    )


class AddResponseData(BaseModel):
    message_count: int
    status: Literal["accumulated", "extracted"]
    operation_id: str | None = None
    replayed: bool | None = None


class MemorizeStageRequest(MemorizeAddRequest):
    messages: list[StagedMessageItemDTO] = Field(..., min_length=1, max_length=500)
    operation_id: str = Field(
        ...,
        pattern=_STAGE_OPERATION_ID,
        description="Caller-stable deferred-ingest idempotency key.",
    )


class StageResponseData(BaseModel):
    message_count: int
    status: Literal["staged"]
    operation_id: str
    replayed: bool
    inserted_count: int = 0
    updated_count: int = 0
    duplicate_count: int = 0
    stale_count: int = 0
    consumed_replay_count: int = 0


class PublishMessageRefDTO(BaseModel):
    source: str | None = Field(default=None, pattern=_STAGE_SOURCE)
    external_ref: str | None = Field(default=None, min_length=1, max_length=512)
    payload_sha256: str | None = Field(default=None, pattern=_PAYLOAD_SHA256)
    revision: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_reference(self) -> PublishMessageRefDTO:
        if self.source is not None and self.source.startswith("__"):
            raise ValueError("staged message source uses a reserved namespace")
        has_external_ref = self.external_ref is not None
        has_payload_hash = self.payload_sha256 is not None
        if has_payload_hash:
            if self.source is not None or has_external_ref:
                raise ValueError(
                    "payload_sha256 publish reference is mutually exclusive with "
                    "source + external_ref"
                )
            if self.revision != 0:
                raise ValueError("payload_sha256 publish reference requires revision 0")
            return self
        if not has_external_ref:
            raise ValueError("publish requires source + external_ref or payload_sha256")
        return self


class MemorizePublishRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    app_id: PathSafeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=_PATH_SAFE_CHARSET,
    )
    project_id: PathSafeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=_PATH_SAFE_CHARSET,
    )
    messages: list[PublishMessageRefDTO] = Field(..., min_length=1, max_length=500)
    authority_ref: str = Field(..., min_length=1, max_length=512)
    operation_id: str = Field(..., pattern=_PUBLISH_OPERATION_ID)


class PublishResponseData(BaseModel):
    message_count: int
    status: Literal["published"]
    operation_id: str
    replayed: bool
    published_count: int = 0
    already_published_count: int = 0
    consumed_count: int = 0


class MemorizeFlushRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    app_id: PathSafeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=_PATH_SAFE_CHARSET,
    )
    project_id: PathSafeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=_PATH_SAFE_CHARSET,
    )
    operation_id: str | None = Field(
        default=None,
        pattern=_FLUSH_OPERATION_ID,
        description="Optional caller-stable idempotency key.",
    )
    background: bool = False

    @model_validator(mode="after")
    def require_operation_id_for_background(self) -> MemorizeFlushRequest:
        if self.background and self.operation_id is None:
            raise ValueError("background flush requires operation_id")
        return self


class FlushResponseData(BaseModel):
    status: Literal["extracted", "no_extraction", "processing"]
    operation_id: str | None = None
    replayed: bool | None = None


class SuccessEnvelope[T](BaseModel):
    """200 wrapper: ``request_id`` sits at the top level, not inside ``data``."""

    request_id: str
    data: T


# Route ──────────────────────────────────────────────────────────────────────


@router.post("/add", response_model_exclude_none=True)
async def add_memory(
    req: Annotated[MemorizeAddRequest, ...],
    request: Request,
) -> SuccessEnvelope[AddResponseData]:
    """Add messages into the user-memory + agent-memory pipelines."""
    request_id = getattr(request.state, "request_id", None) or _gen_request_id()
    try:
        result = await memorize(req.model_dump())
    except MultimodalError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except (MemoryOperationConflictError, MemoryOperationRecoveryError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return SuccessEnvelope(
        request_id=request_id,
        data=AddResponseData(
            message_count=result.message_count,
            status=result.status,
            operation_id=result.operation_id,
            replayed=result.replayed if result.operation_id is not None else None,
        ),
    )


@router.post("/stage", response_model_exclude_none=True)
async def stage_memory(
    req: Annotated[MemorizeStageRequest, ...],
    request: Request,
) -> SuccessEnvelope[StageResponseData]:
    """Durably stage messages without invoking boundary or extraction."""

    request_id = getattr(request.state, "request_id", None) or _gen_request_id()
    try:
        result = await stage(req.model_dump())
    except MultimodalError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except (
        MemoryMessageConflictError,
        MemoryMessageRecoveryError,
        MemoryOperationConflictError,
        MemoryOperationRecoveryError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return SuccessEnvelope(
        request_id=request_id,
        data=StageResponseData(
            message_count=result.message_count,
            status=result.status,
            operation_id=result.operation_id,
            replayed=result.replayed,
            inserted_count=result.inserted_count,
            updated_count=result.updated_count,
            duplicate_count=result.duplicate_count,
            stale_count=result.stale_count,
            consumed_replay_count=result.consumed_replay_count,
        ),
    )


@router.post("/publish", response_model_exclude_none=True)
async def publish_memory(
    req: Annotated[MemorizePublishRequest, ...],
    request: Request,
) -> SuccessEnvelope[PublishResponseData]:
    """Authorize exact staged revisions; performs no extraction."""

    request_id = getattr(request.state, "request_id", None) or _gen_request_id()
    try:
        result = await publish(req.model_dump())
    except (
        MemoryMessageConflictError,
        MemoryMessageRecoveryError,
        MemoryOperationConflictError,
        MemoryOperationRecoveryError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return SuccessEnvelope(
        request_id=request_id,
        data=PublishResponseData(
            message_count=result.message_count,
            status=result.status,
            operation_id=result.operation_id,
            replayed=result.replayed,
            published_count=result.published_count,
            already_published_count=result.already_published_count,
            consumed_count=result.consumed_count,
        ),
    )


@router.post("/flush", response_model_exclude_none=True)
async def flush_memory(
    req: Annotated[MemorizeFlushRequest, ...],
    request: Request,
) -> SuccessEnvelope[FlushResponseData]:
    """Force boundary detection over the current ``session_id`` buffer.

    [OSS-only] — cloud edition decides boundary timing server-side and
    does not expose this endpoint.
    """
    request_id = getattr(request.state, "request_id", None) or _gen_request_id()
    try:
        if req.background:
            result = await queue_background_flush(
                {
                    "session_id": req.session_id,
                    "app_id": req.app_id,
                    "project_id": req.project_id,
                    "operation_id": req.operation_id,
                }
            )
            get_background_flush_scheduler().wake()
            return SuccessEnvelope(
                request_id=request_id,
                data=FlushResponseData(
                    status="processing",
                    operation_id=result.operation_id,
                    replayed=result.replayed,
                ),
            )
        result = await memorize(
            {
                "session_id": req.session_id,
                "app_id": req.app_id,
                "project_id": req.project_id,
                "messages": [],
                "operation_id": req.operation_id,
            },
            is_final=True,
        )
    except (MemoryOperationConflictError, MemoryOperationRecoveryError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # service's ``accumulated`` = nothing to flush (buffer was empty);
    # ``extracted`` = at least one cell carved out.
    status: Literal["extracted", "no_extraction"] = (
        "extracted" if result.status == "extracted" else "no_extraction"
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=FlushResponseData(
            status=status,
            operation_id=result.operation_id,
            replayed=result.replayed if result.operation_id is not None else None,
        ),
    )


@router.get(
    "/operations/{operation_id}",
    response_model_exclude_none=True,
)
async def memory_operation_status(
    operation_id: str,
    request: Request,
) -> SuccessEnvelope[MemoryOperationStatus]:
    """Inspect a write receipt without returning conversation content."""

    if not re.fullmatch(
        r"evop1-(?:add|stage|publish|flush)-[0-9a-f]{64}", operation_id
    ):
        raise HTTPException(status_code=422, detail="invalid operation id")
    request_id = getattr(request.state, "request_id", None) or _gen_request_id()
    try:
        status = await get_memory_operation_status(operation_id)
    except MemoryOperationRecoveryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if status is None:
        raise HTTPException(status_code=404, detail="operation not found")
    return SuccessEnvelope(request_id=request_id, data=status)


def _gen_request_id() -> str:
    """Fallback request id when no middleware set one."""
    return gen_request_id()
