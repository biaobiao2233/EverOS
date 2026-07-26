"""Memorize use case — ingest + boundary + durable pipeline dispatch.

End-to-end orchestration:

    POST /api/v1/memory/add { session_id, messages[] }
        → ingest.process → IngestResult
        → _boundary.prepare_cells(mode=settings.memorize.mode) → cells
        → UserMemoryPipeline.run(cells, ...)
        → AgentMemoryPipeline.run(cells, ...) if mode == "agent"
        → merge outcome.status → {message_count, status}

The boundary stage owns buffer / merge / boundary / tail — so the same
``cells`` feed both pipelines in agent mode (chat mode skips the agent
pipeline entirely).

Lazy singletons: writer / loader / pipelines / LLM client are all
constructed on first use (service module imports run before lifespan
resolves the memory-root and reads env vars).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel

from everos.component.llm import get_llm_client
from everos.config import load_settings
from everos.core.errors import MultimodalError
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.engine import OfflineEngine
from everos.infra.persistence.markdown import EpisodeWriter
from everos.infra.persistence.sqlite import (
    MemoryOperation,
    canonical_json,
    memcell_repo,
    memory_operation_repo,
)
from everos.memory import IngestResult, MemCell
from everos.memory.extract.ingest import process as ingest_process
from everos.memory.extract.pipeline import (
    AgentMemoryPipeline,
    UserMemoryPipeline,
)
from everos.memory.prompt_slots import PromptLoader
from everos.memory.strategies import (
    extract_agent_case,
    extract_agent_skill,
    extract_atomic_facts,
    extract_foresight,
    extract_user_profile,
    trigger_profile_clustering,
    trigger_skill_clustering,
)
from everos.service._boundary import BoundaryOutcome, prepare_cells
from everos.service._session_lock import scoped_session_lock

logger = get_logger(__name__)


class MemorizeResult(BaseModel):
    """What memorize returns to the caller (route serialises it)."""

    message_count: int
    status: Literal["accumulated", "extracted"]
    operation_id: str | None = None
    replayed: bool = False


class MemoryOperationStatus(BaseModel):
    """Safe, content-free view exposed by the operation query endpoint."""

    operation_id: str
    kind: Literal["add", "flush"]
    app_id: str
    project_id: str
    session_id: str
    state: Literal["running", "completed", "failed"]
    stage: Literal["claimed", "memcells_committed", "sync_dispatch_completed"]
    memcell_count: int
    retryable: bool
    error_code: str | None = None
    response: dict[str, Any] | None = None


class MemoryOperationConflictError(ValueError):
    """The caller reused an operation id for a different request."""


class MemoryOperationRecoveryError(RuntimeError):
    """The ledger references incomplete or inconsistent persisted state."""


# Lazy singletons ────────────────────────────────────────────────────────────


_episode_writer: EpisodeWriter | None = None
_prompt_loader: PromptLoader | None = None
_user_pipeline: UserMemoryPipeline | None = None
_agent_pipeline: AgentMemoryPipeline | None = None
_ome_engine: OfflineEngine | None = None


def _config_root() -> Path:
    """Return the directory holding bundled prompt slots (``config/``)."""
    # ``src/everos/config/`` ships in the wheel alongside this service module.
    return Path(__file__).resolve().parent.parent / "config"


def _get_episode_writer() -> EpisodeWriter:
    global _episode_writer
    if _episode_writer is None:
        _episode_writer = EpisodeWriter(MemoryRoot.default())
    return _episode_writer


def _get_prompt_loader() -> PromptLoader:
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader(_config_root())
    return _prompt_loader


def _get_user_pipeline() -> UserMemoryPipeline:
    global _user_pipeline
    if _user_pipeline is None:
        _user_pipeline = UserMemoryPipeline(
            episode_writer=_get_episode_writer(),
            prompt_loader=_get_prompt_loader(),
            llm_client=get_llm_client(),
            engine=_get_engine(),
        )
    return _user_pipeline


def _get_agent_pipeline() -> AgentMemoryPipeline:
    global _agent_pipeline
    if _agent_pipeline is None:
        _agent_pipeline = AgentMemoryPipeline(engine=_get_engine())
    return _agent_pipeline


def _get_engine() -> OfflineEngine:
    """Return the singleton OfflineEngine; constructed + registered on first call.

    Lifecycle (start/stop) is wired by ``OmeLifespanProvider``.
    """
    global _ome_engine
    if _ome_engine is None:
        root = MemoryRoot.default()
        jobstore_path = root.ome_db
        jobstore_path.parent.mkdir(parents=True, exist_ok=True)
        engine = OfflineEngine(
            config=OMEConfig(
                jobstore_path=jobstore_path,
                config_path=root.ome_config,
            )
        )
        engine.register(extract_atomic_facts)
        engine.register(extract_foresight)
        engine.register(extract_agent_case)
        engine.register(trigger_skill_clustering)
        engine.register(extract_agent_skill)
        engine.register(trigger_profile_clustering)
        engine.register(extract_user_profile)
        _ome_engine = engine
    return _ome_engine


# Public entry ───────────────────────────────────────────────────────────────


async def memorize(
    payload: dict[str, Any],
    *,
    is_final: bool = False,
) -> MemorizeResult:
    """Execute one add/flush cycle with optional durable idempotency.

    Args:
        payload: ``{"session_id", "messages": [...], "operation_id"?}`` —
            entrypoints DTO dumped to a dict.  When ``operation_id`` is
            supplied, the request is claimed in SQLite before any business
            state changes and may safely be retried after interruption.
        is_final: ``True`` only for flush (algo guarantees ``tail=[]``).

    Concurrency: serialised per scoped ``session_id`` via
    :func:`everos.service._session_lock.scoped_session_lock`. The lock
    spans the entire read-merge-boundary-write cycle so concurrent /add
    calls on the same session cannot lose-update each other's tail.
    An outer ``asyncio.timeout`` (configured by
    ``settings.memorize.session_lock_timeout_seconds``) ensures a stuck
    LLM cannot hold the lock indefinitely — on timeout the task is
    cancelled and ``async with`` auto-releases the lock.
    """
    request_payload = dict(payload)
    operation_id_raw = request_payload.pop("operation_id", None)
    operation_id = str(operation_id_raw) if operation_id_raw is not None else None
    kind: Literal["add", "flush"] = "flush" if is_final else "add"
    session_id = str(request_payload["session_id"])
    app_id = str(request_payload.get("app_id") or "default")
    project_id = str(request_payload.get("project_id") or "default")
    request_sha256 = _request_sha256(kind, request_payload)
    settings = load_settings()
    configured_mode = settings.memorize.mode
    configured_boundary = settings.boundary_detection
    response_message_count = len(request_payload.get("messages", []))

    operation: MemoryOperation | None = None
    if operation_id is not None:
        _validate_operation_id(operation_id, kind)
        operation, _ = await memory_operation_repo.claim(
            MemoryOperation(
                operation_id=operation_id,
                kind=kind,
                app_id=app_id,
                project_id=project_id,
                session_id=session_id,
                request_sha256=request_sha256,
                mode=configured_mode,
                plan_version=1,
                hard_token_limit=configured_boundary.hard_token_limit,
                hard_msg_limit=configured_boundary.hard_msg_limit,
                message_count=response_message_count,
            )
        )
        _validate_operation(
            operation,
            kind=kind,
            app_id=app_id,
            project_id=project_id,
            session_id=session_id,
            request_sha256=request_sha256,
        )
        if operation.state == "completed":
            return _result_from_completed_operation(operation, replayed=True)
        if operation.state == "failed" and not operation.retryable:
            raise MemoryOperationRecoveryError(
                f"operation {operation_id!r} failed permanently "
                f"({operation.error_code or 'unknown'})"
            )

    # The cross-process session lock is the execution lease.  A duplicate
    # request that times out while merely waiting for that lease must not
    # write a failure receipt over the request that is actually running.
    owns_execution = False
    try:
        async with asyncio.timeout(settings.memorize.session_lock_timeout_seconds):
            async with scoped_session_lock(
                MemoryRoot.default(),
                session_id,
                app_id=app_id,
                project_id=project_id,
            ):
                if operation_id is not None:
                    current = await memory_operation_repo.get(operation_id)
                    if current is None:  # pragma: no cover - DB invariant
                        raise MemoryOperationRecoveryError(
                            f"operation disappeared after claim: {operation_id!r}"
                        )
                    _validate_operation(
                        current,
                        kind=kind,
                        app_id=app_id,
                        project_id=project_id,
                        session_id=session_id,
                        request_sha256=request_sha256,
                    )
                    if current.state == "completed":
                        return _result_from_completed_operation(current, replayed=True)
                    if current.state == "failed":
                        if not current.retryable:
                            raise MemoryOperationRecoveryError(
                                f"operation {operation_id!r} is not retryable"
                            )
                        current = await memory_operation_repo.mark_running(operation_id)
                    operation = current
                    owns_execution = True

                if operation is not None:
                    mode, hard_token_limit, hard_msg_limit = _operation_plan(operation)
                    response_message_count = operation.message_count
                else:
                    mode = configured_mode
                    hard_token_limit = configured_boundary.hard_token_limit
                    hard_msg_limit = configured_boundary.hard_msg_limit

                result = await _memorize_locked(
                    request_payload,
                    mode=mode,
                    hard_token_limit=hard_token_limit,
                    hard_msg_limit=hard_msg_limit,
                    response_message_count=response_message_count,
                    is_final=is_final,
                    operation=operation,
                )
                if operation_id is not None:
                    await memory_operation_repo.mark_completed(
                        operation_id,
                        {
                            "message_count": result.message_count,
                            "status": result.status,
                        },
                    )
                    result.operation_id = operation_id
                return result
    except Exception as exc:
        if (
            operation_id is not None
            and owns_execution
            and not isinstance(exc, MemoryOperationConflictError)
        ):
            retryable = not isinstance(
                exc, (MultimodalError, MemoryOperationRecoveryError)
            )
            try:
                await memory_operation_repo.mark_failed(
                    operation_id,
                    error_code=type(exc).__name__,
                    retryable=retryable,
                )
            except Exception:  # pragma: no cover - preserve original failure
                logger.exception(
                    "memory_operation_failure_receipt_failed",
                    extra={"operation_id": operation_id},
                )
        raise


async def _memorize_locked(
    payload: dict[str, Any],
    *,
    mode: Literal["chat", "agent"],
    hard_token_limit: int,
    hard_msg_limit: int,
    response_message_count: int,
    is_final: bool,
    operation: MemoryOperation | None,
) -> MemorizeResult:
    """Inner critical section — runs under the per-session lock."""
    ingested = await ingest_process(payload)
    if operation is not None and operation.stage == "memcells_committed":
        boundary = await _restore_boundary(operation, ingested)
    elif operation is not None and operation.stage == "sync_dispatch_completed":
        raise MemoryOperationRecoveryError(
            "operation has a completed stage without a completed receipt"
        )
    else:
        boundary = await prepare_cells(
            ingested,
            mode=mode,
            is_final=is_final,
            llm_client=get_llm_client(),
            prompt_loader=_get_prompt_loader(),
            hard_token_limit=hard_token_limit,
            hard_msg_limit=hard_msg_limit,
            operation_id=operation.operation_id if operation is not None else None,
        )

    if not boundary.cells:
        # Nothing went past the boundary stage — no pipelines to dispatch.
        return MemorizeResult(
            message_count=response_message_count,
            status=_merge_status(boundary.status, "skipped"),
        )

    user_outcome = await _get_user_pipeline().run(
        ingested,
        cells=boundary.cells,
        memcell_ids=boundary.memcell_ids,
        per_cell_all_senders=boundary.per_cell_all_senders,
    )
    if mode == "agent":
        # User output is committed first.  Agent derivation must never get
        # ahead of the canonical Episode path when a process is interrupted.
        agent_outcome = await _get_agent_pipeline().run(
            ingested,
            cells=boundary.cells,
            memcell_ids=boundary.memcell_ids,
        )
        merged_status = _merge_status(user_outcome.status, agent_outcome.status)
    else:
        merged_status = _merge_status(user_outcome.status, "skipped")

    return MemorizeResult(
        message_count=response_message_count,
        status=merged_status,
    )


async def get_memory_operation_status(
    operation_id: str,
) -> MemoryOperationStatus | None:
    """Return a content-free status receipt for one write operation."""

    operation = await memory_operation_repo.get(operation_id)
    if operation is None:
        return None
    memcell_ids = _decode_string_list(operation.memcell_ids_json, field="memcell_ids")
    response: dict[str, Any] | None = None
    if operation.state == "completed":
        if (
            operation.stage != "sync_dispatch_completed"
            or operation.response_json is None
        ):
            raise MemoryOperationRecoveryError(
                "completed operation has an inconsistent receipt"
            )
        decoded = json.loads(operation.response_json)
        if not isinstance(decoded, dict):
            raise MemoryOperationRecoveryError("operation response is not an object")
        response = decoded
    elif operation.response_json is not None:
        raise MemoryOperationRecoveryError(
            "incomplete operation unexpectedly has a response receipt"
        )
    return MemoryOperationStatus(
        operation_id=operation.operation_id,
        kind=operation.kind,
        app_id=operation.app_id,
        project_id=operation.project_id,
        session_id=operation.session_id,
        state=operation.state,
        stage=operation.stage,
        memcell_count=len(memcell_ids),
        retryable=operation.retryable,
        error_code=operation.error_code,
        response=response,
    )


def _request_sha256(kind: str, payload: dict[str, Any]) -> str:
    wire = canonical_json({"kind": kind, "payload": payload})
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def _validate_operation_id(operation_id: str, kind: str) -> None:
    prefix = f"evop1-{kind}-"
    suffix = operation_id.removeprefix(prefix)
    if (
        not operation_id.startswith(prefix)
        or len(suffix) != 64
        or any(ch not in "0123456789abcdef" for ch in suffix)
    ):
        raise ValueError(f"invalid {kind} operation id")


def _operation_plan(
    operation: MemoryOperation,
) -> tuple[Literal["chat", "agent"], int, int]:
    """Validate and return the immutable execution plan captured at claim."""

    if operation.plan_version != 1:
        raise MemoryOperationRecoveryError(
            f"unsupported operation plan version: {operation.plan_version}"
        )
    if operation.mode not in ("chat", "agent"):
        raise MemoryOperationRecoveryError("operation has an invalid memory mode")
    if operation.hard_token_limit <= 0 or operation.hard_msg_limit <= 0:
        raise MemoryOperationRecoveryError("operation has invalid boundary limits")
    if operation.message_count < 0:
        raise MemoryOperationRecoveryError("operation has invalid message_count")
    return (
        cast(Literal["chat", "agent"], operation.mode),
        operation.hard_token_limit,
        operation.hard_msg_limit,
    )


def _validate_operation(
    operation: MemoryOperation,
    *,
    kind: str,
    app_id: str,
    project_id: str,
    session_id: str,
    request_sha256: str,
) -> None:
    expected = (kind, app_id, project_id, session_id, request_sha256)
    actual = (
        operation.kind,
        operation.app_id,
        operation.project_id,
        operation.session_id,
        operation.request_sha256,
    )
    if actual != expected:
        raise MemoryOperationConflictError(
            f"operation id {operation.operation_id!r} was already used for "
            "a different request"
        )


def _result_from_completed_operation(
    operation: MemoryOperation, *, replayed: bool
) -> MemorizeResult:
    if operation.response_json is None:
        raise MemoryOperationRecoveryError(
            "completed operation has no response receipt"
        )
    try:
        response = json.loads(operation.response_json)
        result = MemorizeResult.model_validate(response)
    except (ValueError, TypeError) as exc:
        raise MemoryOperationRecoveryError(
            "completed operation has an invalid response receipt"
        ) from exc
    result.operation_id = operation.operation_id
    result.replayed = replayed
    return result


async def _restore_boundary(
    operation: MemoryOperation,
    ingested: IngestResult,
) -> BoundaryOutcome:
    memcell_ids = _decode_string_list(operation.memcell_ids_json, field="memcell_ids")
    if not memcell_ids:
        raise MemoryOperationRecoveryError(
            "memcells_committed operation has no memcell ids"
        )
    rows = await memcell_repo.find_by_ids(memcell_ids)
    if len(rows) != len(memcell_ids):
        raise MemoryOperationRecoveryError("one or more operation memcells are missing")

    cells: list[MemCell] = []
    per_cell_message_ids: list[list[str]] = []
    per_cell_all_senders: list[list[str]] = []
    for row in rows:
        if (
            row.app_id != operation.app_id
            or row.project_id != operation.project_id
            or row.session_id != operation.session_id
        ):
            raise MemoryOperationRecoveryError(
                f"memcell {row.memcell_id!r} does not match operation scope"
            )
        try:
            cells.append(MemCell.model_validate_json(row.payload_json))
            per_cell_message_ids.append(
                _decode_string_list(row.message_ids_json, field="message_ids")
            )
            per_cell_all_senders.append(
                _decode_string_list(row.sender_ids_json, field="sender_ids")
            )
        except (ValueError, TypeError) as exc:
            raise MemoryOperationRecoveryError(
                f"memcell {row.memcell_id!r} has an invalid recovery payload"
            ) from exc

    return BoundaryOutcome(
        cells=cells,
        memcell_ids=memcell_ids,
        per_cell_message_ids=per_cell_message_ids,
        per_cell_all_senders=per_cell_all_senders,
        status="extracted",
        message_count=operation.message_count,
    )


def _decode_string_list(raw: str, *, field: str) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MemoryOperationRecoveryError(f"invalid {field} receipt") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise MemoryOperationRecoveryError(f"invalid {field} receipt")
    return value


def _merge_status(
    user: Literal["accumulated", "extracted", "skipped"],
    agent: Literal["accumulated", "extracted", "skipped"],
) -> Literal["accumulated", "extracted"]:
    """Either ``extracted`` wins; otherwise ``accumulated``."""
    if user == "extracted" or agent == "extracted":
        return "extracted"
    return "accumulated"
