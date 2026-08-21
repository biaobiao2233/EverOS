"""Boundary stage — shared upstream step for the dual-pipeline memorize flow.

Owns the buffer / merge / boundary / tail-persistence sequence so the same
``cells`` feed both :class:`everos.memory.extract.pipeline.UserMemoryPipeline`
and :class:`everos.memory.extract.pipeline.AgentMemoryPipeline` (the
latter only runs when ``mode == "agent"``).

Mode dispatch:

- ``"chat"``  → :func:`everalgo.boundary.detect_boundaries` on a filtered
  ``ChatMessage`` list (tool rows / assistant-with-tool_calls dropped).
- ``"agent"`` → :class:`everalgo.agent_memory.AgentBoundaryDetector` on the
  full ``ConversationItem`` list (tool rows preserved).

Both paths share a single unprocessed-buffer track (``"memorize"``) because
boundary detection is single-pass; switching mode requires a fresh service
process (see ``settings.memorize.mode``).

The boundary stage also owns the **sqlite ``memcell`` ledger**: each cell
gets exactly one row regardless of mode (since the algorithm produces one
canonical cell). Downstream pipelines (user + agent) reference the same
``memcell_id``; PK collisions used to occur when each pipeline tried to
insert its own row per cell.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Literal, NamedTuple

from everalgo.agent_memory import AgentBoundaryDetector
from everalgo.boundary import detect_boundaries
from everalgo.types import (
    ChatMessage,
    ConversationItem,
    MemCell,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)
from everalgo.types import ToolCall as AlgoToolCall
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from everos.component.utils.datetime import from_timestamp, to_timestamp_ms
from everos.core.observability.logging import get_logger
from everos.core.persistence.sqlite import session_scope
from everos.infra.persistence.sqlite import (
    ConversationStatus,
    Memcell,
    MemoryOperation,
    UnprocessedBuffer,
    conversation_status_repo,
    get_session_factory,
    unprocessed_buffer_repo,
)
from everos.memory import CanonicalMessage, IngestResult, ToolCall

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

    from everos.memory.prompt_slots import PromptLoader

logger = get_logger(__name__)

_TRACK = "memorize"
"""Shared track used for both the unprocessed-buffer and the memcell
ledger — boundary detection is mode-dispatched but single-pass, so it
does not need per-pipeline separation."""

_RAW_TYPE_BY_MODE: dict[str, str] = {
    "chat": "Conversation",
    "agent": "AgentTrajectory",
}


Mode = Literal["chat", "agent"]
Status = Literal["accumulated", "extracted", "skipped"]


class BoundaryOutcome(NamedTuple):
    """Result handed to the dual pipelines.

    Lists are parallel: index ``i`` describes cell ``i``.
    ``memcell_ids`` are minted here and shared across both pipelines
    (Episode.parent_id / UserPipelineStarted.memcell_id both reference
    the same id — single sqlite ``memcell`` row per cell).
    ``message_count`` is the count of fresh (newly-arrived, post-filter)
    canonical rows from this call; the response DTO reads it directly.
    """

    cells: list[MemCell]
    memcell_ids: list[str]
    per_cell_message_ids: list[list[str]]
    per_cell_all_senders: list[list[str]]
    status: Status
    message_count: int


async def prepare_cells(
    ingested: IngestResult,
    *,
    mode: Mode,
    is_final: bool,
    llm_client: LLMClient | None,
    prompt_loader: PromptLoader,
    hard_token_limit: int,
    hard_msg_limit: int,
    operation_id: str | None = None,
) -> BoundaryOutcome:
    """Run the boundary stage end-to-end and persist tail back to buffer."""
    app_id = ingested.app_id
    project_id = ingested.project_id
    fresh = _filter_for_mode(ingested.messages, mode)
    if not fresh and not is_final:
        return _empty_outcome(status="skipped", message_count=0)

    buffer_rows = await unprocessed_buffer_repo.list_for_track(
        ingested.session_id, _TRACK, app_id=app_id, project_id=project_id
    )
    buffered = [_row_to_canonical(r) for r in buffer_rows]
    merged = _merge_dedupe_sort(buffered, fresh)
    if not merged:
        return _empty_outcome(status="accumulated", message_count=0)

    # Need a role=user anchor for downstream episode extraction; assistant-
    # only / tool-only batches sit in the buffer until a user message lands.
    if not is_final and not any(m.role == "user" for m in merged):
        await _replace_buffer(ingested.session_id, merged, app_id, project_id)
        await _touch_last_message_ts(ingested.session_id, merged, app_id, project_id)
        return _empty_outcome(status="accumulated", message_count=len(fresh))

    if llm_client is None:
        await _replace_buffer(ingested.session_id, merged, app_id, project_id)
        logger.warning(
            "memorize_no_llm_client",
            extra={"session_id": ingested.session_id, "buffered": len(merged)},
        )
        return _empty_outcome(status="skipped", message_count=len(fresh))

    boundary_prompt = prompt_loader.load("boundary_detection")
    cells, tail = await _detect(
        merged,
        mode=mode,
        llm_client=llm_client,
        prompt=boundary_prompt,
        is_final=is_final,
        hard_token_limit=hard_token_limit,
        hard_msg_limit=hard_msg_limit,
    )

    if not cells:
        # boundary returned an empty cells set → roll the merged slice
        # back into the buffer (algo says it's still mid-conversation).
        await _replace_buffer(ingested.session_id, merged, app_id, project_id)
        await _touch_last_message_ts(ingested.session_id, merged, app_id, project_id)
        return _empty_outcome(status="accumulated", message_count=len(fresh))

    memcell_ids = [_mint_memcell_id() for _ in cells]
    per_cell_message_ids = _split_messages_per_cell(merged, cells)
    per_cell_all_senders = [_unique_all_senders(c) for c in cells]

    # Write one memcell row per cell (shared across user / agent pipelines).
    # MemCell has no single owner — multi-user dialogue slices stay owner-
    # agnostic. Per-user fan-out (Episode / AtomicFact / Foresight / Profile)
    # happens downstream via ``sender_ids``.
    raw_type = _RAW_TYPE_BY_MODE[mode]
    rows = [
        _build_memcell_row(
            cell=cell,
            memcell_id=memcell_id,
            session_id=ingested.session_id,
            app_id=app_id,
            project_id=project_id,
            raw_type=raw_type,
            message_ids=per_cell_message_ids[i],
            sender_ids=per_cell_all_senders[i],
        )
        for i, (cell, memcell_id) in enumerate(zip(cells, memcell_ids, strict=True))
    ]
    last_cell_ts = max((cell.timestamp for cell in cells), default=0)
    tail_canonical = _slice_tail(merged, tail)
    await _commit_cells_and_tail(
        rows=rows,
        session_id=ingested.session_id,
        app_id=app_id,
        project_id=project_id,
        tail=tail_canonical,
        last_cell_ts=last_cell_ts,
        operation_id=operation_id,
    )

    return BoundaryOutcome(
        cells=cells,
        memcell_ids=memcell_ids,
        per_cell_message_ids=per_cell_message_ids,
        per_cell_all_senders=per_cell_all_senders,
        status="extracted",
        message_count=len(fresh),
    )


async def stage_messages(
    ingested: IngestResult,
    *,
    operation_id: str,
) -> int:
    """Reliably merge one deferred upload batch without running extraction.

    The idempotent buffer insert and the operation's terminal receipt share one SQLite
    transaction.  A caller therefore never observes ``staged`` unless every
    canonical row is durable, and a crash cannot leave a committed buffer with
    an ambiguous operation receipt.
    """

    app_id = ingested.app_id
    project_id = ingested.project_id
    fresh = list(ingested.messages)
    async with session_scope(get_session_factory()) as session:
        if fresh:
            rows = [_canonical_to_row(message, app_id, project_id) for message in fresh]
            await session.execute(
                sqlite_insert(UnprocessedBuffer)
                .values([row.model_dump() for row in rows])
                .on_conflict_do_nothing(index_elements=["message_id"])
            )

            status_stmt = select(ConversationStatus).where(
                ConversationStatus.app_id == app_id,
                ConversationStatus.project_id == project_id,
                ConversationStatus.session_id == ingested.session_id,
                ConversationStatus.track == _TRACK,
            )
            status = (await session.execute(status_stmt)).scalars().first()
            last_message_ts = max(message.timestamp for message in fresh)
            if status is None:
                status = ConversationStatus(
                    app_id=app_id,
                    project_id=project_id,
                    session_id=ingested.session_id,
                    track=_TRACK,
                    last_message_ts=last_message_ts,
                )
                session.add(status)
            elif (
                status.last_message_ts is None
                or last_message_ts > status.last_message_ts
            ):
                status.last_message_ts = last_message_ts

        operation = await session.get(MemoryOperation, operation_id)
        if operation is None:
            raise KeyError(f"memory operation not found: {operation_id}")
        if operation.state == "completed":
            raise RuntimeError(
                "stage operation unexpectedly completed while executing: "
                f"{operation_id}"
            )
        operation.state = "completed"
        operation.stage = "messages_staged"
        operation.response_json = json.dumps(
            {"message_count": len(fresh), "status": "staged"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        operation.error_code = None
        operation.retryable = False
        await session.commit()
    return len(fresh)


# ── Mode-specific filter ──────────────────────────────────────────────────


def _filter_for_mode(
    msgs: list[CanonicalMessage], mode: Mode
) -> list[CanonicalMessage]:
    """Chat mode drops tool rows; agent mode keeps everything."""
    if mode == "chat":
        return [m for m in msgs if m.role in ("user", "assistant") and not m.tool_calls]
    return list(msgs)


# ── Boundary dispatch ─────────────────────────────────────────────────────


_BOUNDARY_MAX_ATTEMPTS = 3


async def _detect(
    merged: list[CanonicalMessage],
    *,
    mode: Mode,
    llm_client: LLMClient,
    prompt: str,
    is_final: bool,
    hard_token_limit: int,
    hard_msg_limit: int,
) -> tuple[list[MemCell], list[ConversationItem]]:
    # Retry on ValueError to absorb transient LLM JSON-parse failures from
    # the everalgo boundary detector; non-ValueError errors propagate.
    last_err: ValueError | None = None
    for attempt in range(_BOUNDARY_MAX_ATTEMPTS):
        try:
            if mode == "chat":
                chat_msgs = [_to_chat_message(m) for m in merged]
                result = await detect_boundaries(
                    chat_msgs,
                    llm=llm_client,
                    prompt=prompt,
                    is_final=is_final,
                    hard_token_limit=hard_token_limit,
                    hard_msg_limit=hard_msg_limit,
                )
                return list(result.cells), list(result.tail)
            # Agent mode — facade does filter→detect→remap to preserve tool
            # items. AgentBoundaryDetector intentionally does not expose hard
            # limits; the boundary primitive's defaults apply.
            items = [_to_conversation_item(m) for m in merged]
            detector = AgentBoundaryDetector(llm=llm_client)
            result = await detector.adetect(items, is_final=is_final, prompt=prompt)
            return list(result.cells), list(result.tail)
        except ValueError as err:
            last_err = err
            logger.warning(
                "boundary_detect_retry",
                extra={
                    "attempt": attempt + 1,
                    "max_attempts": _BOUNDARY_MAX_ATTEMPTS,
                    "mode": mode,
                    "error": str(err),
                },
            )
    assert last_err is not None
    raise last_err


# ── CanonicalMessage → algo wire types ────────────────────────────────────


def _to_chat_message(m: CanonicalMessage) -> ChatMessage:
    return ChatMessage(
        id=m.message_id,
        role=m.role,  # type: ignore[arg-type]
        sender_id=m.sender_id,
        sender_name=m.sender_name,
        content=m.text,
        timestamp=to_timestamp_ms(m.timestamp),
    )


def _to_conversation_item(m: CanonicalMessage) -> ConversationItem:
    """Map one canonical row to one ``ConversationItem`` (1:1).

    Dispatch rules — order matters:

    1. ``role="tool"`` (paired with a ``tool_call_id``) → :class:`ToolCallResult`.
    2. ``role="assistant"`` carrying non-empty ``tool_calls`` →
       :class:`ToolCallRequest`; the optional ``content`` text rides along.
    3. ``role`` in {``"user"``, ``"assistant"``} (text-only) →
       :class:`ChatMessage`.

    Caller is expected to provide well-formed inputs (no orphan tool rows,
    no role≠tool with ``tool_call_id``). The fall-through case logs and
    raises so unexpected shapes don't silently corrupt the cell index map.
    """
    ts_ms = to_timestamp_ms(m.timestamp)
    if m.role == "tool" and m.tool_call_id:
        return ToolCallResult(
            tool_call_id=m.tool_call_id,
            content=m.text,
            timestamp=ts_ms,
        )
    if m.role == "assistant" and m.tool_calls:
        return ToolCallRequest(
            tool_calls=[
                AlgoToolCall(
                    id=tc.id,
                    function=ToolCallFunction(
                        name=tc.function.get("name", ""),
                        arguments=tc.function.get("arguments", ""),
                    ),
                )
                for tc in m.tool_calls
            ],
            timestamp=ts_ms,
            content=m.text or None,
            sender_id=m.sender_id,
            sender_name=m.sender_name,
        )
    if m.role in ("user", "assistant"):
        return ChatMessage(
            id=m.message_id,
            role=m.role,  # type: ignore[arg-type]
            sender_id=m.sender_id,
            sender_name=m.sender_name,
            content=m.text,
            timestamp=ts_ms,
        )
    # Orphan tool row or unexpected role — break loudly; corrupting the
    # cell→message index map silently is worse than a 5xx.
    raise ValueError(
        f"cannot map canonical row to ConversationItem: role={m.role!r} "
        f"message_id={m.message_id!r} has_tool_call_id={m.tool_call_id is not None}"
    )


# ── Buffer + status helpers ───────────────────────────────────────────────


async def _commit_cells_and_tail(
    *,
    rows: list[Memcell],
    session_id: str,
    app_id: str,
    project_id: str,
    tail: list[CanonicalMessage],
    last_cell_ts: int,
    operation_id: str | None,
) -> None:
    """Atomically commit MemCells, consume the buffer, and checkpoint an op.

    Previously these were three independent transactions.  A pipeline error
    could therefore leave committed MemCells plus a consumed buffer without a
    durable receipt telling a retry what to resume.  Keeping the business rows
    and the operation stage in one SQLite transaction closes that ambiguity.
    """

    tail_rows = [_canonical_to_row(message, app_id, project_id) for message in tail]
    async with session_scope(get_session_factory()) as session:
        session.add_all(rows)

        if last_cell_ts:
            stmt = select(ConversationStatus).where(
                ConversationStatus.app_id == app_id,
                ConversationStatus.project_id == project_id,
                ConversationStatus.session_id == session_id,
                ConversationStatus.track == _TRACK,
            )
            status = (await session.execute(stmt)).scalars().first()
            if status is None:
                status = ConversationStatus(
                    app_id=app_id,
                    project_id=project_id,
                    session_id=session_id,
                    track=_TRACK,
                    last_memcell_ts=from_timestamp(last_cell_ts),
                )
                session.add(status)
            else:
                status.last_memcell_ts = from_timestamp(last_cell_ts)

        await session.execute(
            delete(UnprocessedBuffer).where(
                UnprocessedBuffer.app_id == app_id,
                UnprocessedBuffer.project_id == project_id,
                UnprocessedBuffer.session_id == session_id,
                UnprocessedBuffer.track == _TRACK,
            )
        )
        if tail_rows:
            session.add_all(tail_rows)

        if operation_id is not None:
            operation = await session.get(MemoryOperation, operation_id)
            if operation is None:
                raise KeyError(f"memory operation not found: {operation_id}")
            operation.stage = "memcells_committed"
            operation.memcell_ids_json = json.dumps(
                [row.memcell_id for row in rows],
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )

        await session.commit()


async def _replace_buffer(
    session_id: str,
    rows: list[CanonicalMessage],
    app_id: str,
    project_id: str,
) -> None:
    await unprocessed_buffer_repo.replace(
        session_id,
        _TRACK,
        [_canonical_to_row(m, app_id, project_id) for m in rows],
        app_id=app_id,
        project_id=project_id,
    )


async def _touch_last_message_ts(
    session_id: str,
    merged: list[CanonicalMessage],
    app_id: str,
    project_id: str,
) -> None:
    await conversation_status_repo.touch_last_message_ts(
        session_id,
        _TRACK,
        max(m.timestamp for m in merged),
        app_id=app_id,
        project_id=project_id,
    )


def _canonical_to_row(
    m: CanonicalMessage, app_id: str, project_id: str
) -> UnprocessedBuffer:
    return UnprocessedBuffer(
        message_id=m.message_id,
        app_id=app_id,
        project_id=project_id,
        session_id=m.session_id,
        track=_TRACK,
        sender_id=m.sender_id,
        sender_name=m.sender_name,
        role=m.role,
        timestamp=m.timestamp,
        content_items_json=json.dumps(m.content_items),
        text=m.text,
        tool_calls_json=(
            json.dumps([tc.model_dump() for tc in m.tool_calls])
            if m.tool_calls
            else None
        ),
        tool_call_id=m.tool_call_id,
    )


def _row_to_canonical(r: UnprocessedBuffer) -> CanonicalMessage:
    tool_calls: list[ToolCall] | None = None
    if r.tool_calls_json:
        tool_calls = [ToolCall.model_validate(d) for d in json.loads(r.tool_calls_json)]
    content_items = json.loads(r.content_items_json) if r.content_items_json else []
    # ``r.timestamp`` is UtcDatetime — the BaseTable load-event hook
    # re-attaches ``tzinfo=UTC`` on ORM hydrate, so no defensive coercion
    # is needed here.
    return CanonicalMessage(
        message_id=r.message_id,
        session_id=r.session_id,
        sender_id=r.sender_id,
        sender_name=r.sender_name,
        role=r.role,  # type: ignore[arg-type]
        timestamp=r.timestamp,
        content_items=content_items,
        text=r.text,
        tool_calls=tool_calls,
        tool_call_id=r.tool_call_id,
    )


# ── Merge / split / sender helpers ────────────────────────────────────────


def _merge_dedupe_sort(
    buffered: list[CanonicalMessage],
    new: list[CanonicalMessage],
) -> list[CanonicalMessage]:
    """Dedupe transport aliases; sort by ``(timestamp, message_id)``.

    Deferred ``/stage`` uses a chunk-position-independent content hash while
    the legacy ``/add`` route retains its historical index-based message id.
    During a rolling client/server upgrade the same text row can therefore
    arrive once through each route with different ids. Keep the first
    persisted row, but also compare the canonical message body so that route
    aliases cannot create duplicate MemCells.
    """

    seen_ids: set[str] = set()
    seen_transports: dict[str, set[str]] = {}
    merged: list[CanonicalMessage] = []
    for message in (*buffered, *new):
        if message.message_id in seen_ids:
            continue
        identity = json.dumps(
            message.model_dump(mode="json", exclude={"message_id"}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        transport = (
            "stage"
            if message.message_id.startswith("ms_")
            else "legacy_add"
            if message.message_id.startswith("m_")
            else "other"
        )
        prior_transports = seen_transports.setdefault(identity, set())
        if (
            transport == "stage"
            and "legacy_add" in prior_transports
            or transport == "legacy_add"
            and "stage" in prior_transports
        ):
            continue
        seen_ids.add(message.message_id)
        prior_transports.add(transport)
        merged.append(message)
    return sorted(merged, key=lambda m: (m.timestamp, m.message_id))


def _slice_tail(
    merged: list[CanonicalMessage],
    tail: list[ConversationItem],
) -> list[CanonicalMessage]:
    """The tail is a trailing slice of ``merged`` (per algo contract)."""
    n = len(tail)
    if n == 0:
        return []
    return merged[-n:]


def _split_messages_per_cell(
    merged: list[CanonicalMessage],
    cells: list[MemCell],
) -> list[list[str]]:
    """Map each cell index → list of everos message_ids.

    The boundary stage maintains a 1:1 ordering between canonical rows and
    items handed to algo, so we walk ``merged`` left-to-right consuming
    ``len(cell.items)`` rows per cell.
    """
    result: list[list[str]] = []
    ptr = 0
    for cell in cells:
        n = len(cell.items)
        result.append([merged[ptr + i].message_id for i in range(n)])
        ptr += n
    return result


def _unique_all_senders(cell: MemCell) -> list[str]:
    """Distinct sender_ids in a cell, preserving first-occurrence order.

    ``ToolCallResult`` does not carry a ``sender_id`` (tool runners are not
    speakers); ``getattr`` keeps the helper agnostic to the item variant.
    """
    senders: list[str] = []
    for item in cell.items:
        sid = getattr(item, "sender_id", None)
        if sid and sid not in senders:
            senders.append(sid)
    return senders


def _build_memcell_row(
    *,
    cell: MemCell,
    memcell_id: str,
    session_id: str,
    app_id: str,
    project_id: str,
    raw_type: str,
    message_ids: list[str],
    sender_ids: list[str],
) -> Memcell:
    return Memcell(
        memcell_id=memcell_id,
        app_id=app_id,
        project_id=project_id,
        session_id=session_id,
        track=_TRACK,
        raw_type=raw_type,
        message_ids_json=json.dumps(message_ids),
        sender_ids_json=json.dumps(sender_ids),
        payload_json=cell.model_dump_json(),
        timestamp=from_timestamp(cell.timestamp),
    )


def _mint_memcell_id() -> str:
    """Generate an everos-owned memcell identifier."""
    return f"mc_{uuid.uuid4().hex[:12]}"


def _empty_outcome(*, status: Status, message_count: int) -> BoundaryOutcome:
    return BoundaryOutcome(
        cells=[],
        memcell_ids=[],
        per_cell_message_ids=[],
        per_cell_all_senders=[],
        status=status,
        message_count=message_count,
    )
