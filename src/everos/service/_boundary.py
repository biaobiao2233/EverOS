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

from everos.component.utils.datetime import from_timestamp, to_timestamp_ms
from everos.core.observability.logging import get_logger
from everos.core.persistence.sqlite import session_scope
from everos.infra.persistence.sqlite import (
    ConversationStatus,
    Memcell,
    MemoryMessageReceipt,
    MemoryOperation,
    UnprocessedBuffer,
    conversation_status_repo,
    get_session_factory,
    memory_message_receipt_repo,
    unprocessed_buffer_repo,
)
from everos.memory import CanonicalMessage, IngestResult, ToolCall
from everos.memory.extract.ingest.id_gen import (
    external_idem_key,
    staged_message_identity,
)
from everos.service._deferred_errors import (
    MemoryMessageConflictError,
    MemoryMessageRecoveryError,
)

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
_AUTH_PENDING = "pending_publish"
_AUTH_PUBLISHED = "published"
_AUTH_CONSUMED = "consumed"


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


class StageMutationSummary(NamedTuple):
    """Content-free counters for one atomic stage transaction."""

    message_count: int
    inserted_count: int
    updated_count: int
    duplicate_count: int
    stale_count: int
    consumed_replay_count: int


class PublishMutationSummary(NamedTuple):
    """Content-free counters for one explicit publish transaction."""

    message_count: int
    published_count: int
    already_published_count: int
    consumed_count: int


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
    eligible_rows, protected_rows = await _partition_buffer_rows_by_authority(
        buffer_rows
    )
    protected = [_row_to_canonical(r) for r in protected_rows]
    # A legacy /add carrying the same logical row as an unpublished /stage
    # must not bypass the server-side authority gate during a rolling client
    # migration. Pending staged rows therefore act as a dedupe fence even
    # though they are invisible to boundary/extraction.
    fresh = _drop_fresh_aliases_of_protected(protected, fresh)
    buffered = [_row_to_canonical(r) for r in eligible_rows]
    merged = _merge_dedupe_sort(buffered, fresh)
    if not merged:
        return _empty_outcome(status="accumulated", message_count=0)

    # Need a role=user anchor for downstream episode extraction; assistant-
    # only / tool-only batches sit in the buffer until a user message lands.
    if not is_final and not any(m.role == "user" for m in merged):
        await _replace_buffer(
            ingested.session_id,
            merged,
            app_id,
            project_id,
            protected=protected,
        )
        await _touch_last_message_ts(ingested.session_id, merged, app_id, project_id)
        return _empty_outcome(status="accumulated", message_count=len(fresh))

    if llm_client is None:
        await _replace_buffer(
            ingested.session_id,
            merged,
            app_id,
            project_id,
            protected=protected,
        )
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
        await _replace_buffer(
            ingested.session_id,
            merged,
            app_id,
            project_id,
            protected=protected,
        )
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
        protected=protected,
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
    raw_messages: list[dict[str, object]],
) -> StageMutationSummary:
    """Reliably merge one deferred upload batch without running extraction.

    Per-message receipts make idempotency independent of request chunking and
    survive successful buffer consumption.  The buffer mutation, receipt
    mutation, and terminal batch operation receipt share one transaction.
    """

    app_id = ingested.app_id
    project_id = ingested.project_id
    fresh = list(ingested.messages)
    if len(raw_messages) != len(fresh):
        raise MemoryMessageRecoveryError(
            "stage raw/canonical message counts do not match"
        )

    staged_items = []
    seen_idem: set[str] = set()
    for raw, canonical in zip(raw_messages, fresh, strict=True):
        identity = staged_message_identity(
            ingested.session_id,
            dict(raw),
            app_id=app_id,
            project_id=project_id,
        )
        if identity.idem_key in seen_idem:
            raise MemoryMessageConflictError(
                "stage request contains the same logical message more than once"
            )
        seen_idem.add(identity.idem_key)
        if canonical.message_id != identity.message_id:
            raise MemoryMessageRecoveryError(
                "stage canonical message id does not match durable identity"
            )
        staged_items.append((raw, canonical, identity))

    inserted_count = 0
    updated_count = 0
    duplicate_count = 0
    stale_count = 0
    consumed_replay_count = 0
    active_messages: list[CanonicalMessage] = []

    async with session_scope(get_session_factory()) as session:
        for _raw, canonical, identity in staged_items:
            receipt_stmt = select(MemoryMessageReceipt).where(
                MemoryMessageReceipt.app_id == app_id,
                MemoryMessageReceipt.project_id == project_id,
                MemoryMessageReceipt.idem_key == identity.idem_key,
            )
            receipt = (await session.execute(receipt_stmt)).scalars().first()

            if receipt is None:
                orphan = await session.get(UnprocessedBuffer, identity.message_id)
                if orphan is not None:
                    raise MemoryMessageRecoveryError(
                        "staged message buffer row exists without durable receipt"
                    )
                session.add(
                    MemoryMessageReceipt(
                        receipt_id=identity.receipt_id,
                        app_id=app_id,
                        project_id=project_id,
                        session_id=ingested.session_id,
                        idem_key=identity.idem_key,
                        message_id=identity.message_id,
                        source=identity.source,
                        external_ref=identity.external_ref,
                        revision=identity.revision,
                        payload_sha256=identity.payload_sha256,
                        authority_state=_AUTH_PENDING,
                    )
                )
                session.add(_canonical_to_row(canonical, app_id, project_id))
                inserted_count += 1
                active_messages.append(canonical)
                continue

            if (
                receipt.session_id != ingested.session_id
                or receipt.message_id != identity.message_id
                or receipt.source != identity.source
                or receipt.external_ref != identity.external_ref
            ):
                raise MemoryMessageRecoveryError(
                    "staged message receipt identity does not match request"
                )
            if receipt.authority_state not in {
                _AUTH_PENDING,
                _AUTH_PUBLISHED,
                _AUTH_CONSUMED,
            }:
                raise MemoryMessageRecoveryError(
                    f"invalid message authority state: {receipt.authority_state!r}"
                )

            if identity.revision < receipt.revision:
                stale_count += 1
                continue

            if identity.revision == receipt.revision:
                if identity.payload_sha256 != receipt.payload_sha256:
                    raise MemoryMessageConflictError(
                        "same staged message revision has a different payload"
                    )
                if receipt.authority_state == _AUTH_CONSUMED:
                    consumed_replay_count += 1
                    continue
                buffered = await session.get(UnprocessedBuffer, receipt.message_id)
                if buffered is None:
                    raise MemoryMessageRecoveryError(
                        "active staged receipt is missing its buffer row"
                    )
                if _canonical_transport_identity(
                    _row_to_canonical(buffered)
                ) != _canonical_transport_identity(canonical):
                    raise MemoryMessageRecoveryError(
                        "active staged buffer payload does not match its receipt"
                    )
                duplicate_count += 1
                active_messages.append(canonical)
                continue

            # Higher revisions supersede in-place but never inherit publish
            # authority from an older payload.
            await session.merge(_canonical_to_row(canonical, app_id, project_id))
            receipt.revision = identity.revision
            receipt.payload_sha256 = identity.payload_sha256
            receipt.authority_state = _AUTH_PENDING
            receipt.authority_ref = None
            receipt.memcell_ids_json = "[]"
            updated_count += 1
            active_messages.append(canonical)

        if active_messages:
            status_stmt = select(ConversationStatus).where(
                ConversationStatus.app_id == app_id,
                ConversationStatus.project_id == project_id,
                ConversationStatus.session_id == ingested.session_id,
                ConversationStatus.track == _TRACK,
            )
            status = (await session.execute(status_stmt)).scalars().first()
            last_message_ts = max(message.timestamp for message in active_messages)
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
        summary = StageMutationSummary(
            message_count=len(fresh),
            inserted_count=inserted_count,
            updated_count=updated_count,
            duplicate_count=duplicate_count,
            stale_count=stale_count,
            consumed_replay_count=consumed_replay_count,
        )
        operation.state = "completed"
        operation.stage = "messages_staged"
        operation.response_json = json.dumps(
            {
                "message_count": summary.message_count,
                "status": "staged",
                "inserted_count": summary.inserted_count,
                "updated_count": summary.updated_count,
                "duplicate_count": summary.duplicate_count,
                "stale_count": summary.stale_count,
                "consumed_replay_count": summary.consumed_replay_count,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        operation.error_code = None
        operation.retryable = False
        await session.commit()
    return summary


async def publish_messages(
    *,
    session_id: str,
    app_id: str,
    project_id: str,
    items: list[dict[str, object]],
    authority_ref: str,
    operation_id: str,
) -> PublishMutationSummary:
    """Authorize exact staged revisions without invoking extraction."""

    refs: list[tuple[str, int]] = []
    seen_idem: set[str] = set()
    for item in items:
        source = item.get("source") or "api"
        external_ref = item.get("external_ref")
        revision = item.get("revision", 0)
        if not isinstance(source, str) or not isinstance(external_ref, str):
            raise MemoryMessageConflictError(
                "publish requires source + external_ref for every message"
            )
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise MemoryMessageConflictError(
                "publish revision must be a non-negative integer"
            )
        idem_key = external_idem_key(session_id, source, external_ref)
        if idem_key in seen_idem:
            raise MemoryMessageConflictError(
                "publish request contains the same logical message more than once"
            )
        seen_idem.add(idem_key)
        refs.append((idem_key, revision))

    published_count = 0
    already_published_count = 0
    consumed_count = 0
    async with session_scope(get_session_factory()) as session:
        for idem_key, revision in refs:
            stmt = select(MemoryMessageReceipt).where(
                MemoryMessageReceipt.app_id == app_id,
                MemoryMessageReceipt.project_id == project_id,
                MemoryMessageReceipt.idem_key == idem_key,
            )
            receipt = (await session.execute(stmt)).scalars().first()
            if receipt is None:
                raise MemoryMessageConflictError(
                    "publish references a message that has not been staged"
                )
            if receipt.session_id != session_id:
                raise MemoryMessageRecoveryError(
                    "publish receipt session does not match request"
                )
            if receipt.revision != revision:
                raise MemoryMessageConflictError(
                    "publish must reference the exact currently staged revision"
                )

            if receipt.authority_state == _AUTH_PENDING:
                buffered = await session.get(UnprocessedBuffer, receipt.message_id)
                if buffered is None:
                    raise MemoryMessageRecoveryError(
                        "pending publish receipt is missing its buffer row"
                    )
                receipt.authority_state = _AUTH_PUBLISHED
                receipt.authority_ref = authority_ref
                published_count += 1
            elif receipt.authority_state == _AUTH_PUBLISHED:
                if receipt.authority_ref != authority_ref:
                    raise MemoryMessageConflictError(
                        "message was already published under a different authority"
                    )
                already_published_count += 1
            elif receipt.authority_state == _AUTH_CONSUMED:
                if receipt.authority_ref != authority_ref:
                    raise MemoryMessageConflictError(
                        "consumed message has a different publish authority"
                    )
                consumed_count += 1
            else:
                raise MemoryMessageRecoveryError(
                    f"invalid message authority state: {receipt.authority_state!r}"
                )

        operation = await session.get(MemoryOperation, operation_id)
        if operation is None:
            raise KeyError(f"memory operation not found: {operation_id}")
        if operation.state == "completed":
            raise RuntimeError(
                "publish operation unexpectedly completed while executing: "
                f"{operation_id}"
            )
        summary = PublishMutationSummary(
            message_count=len(refs),
            published_count=published_count,
            already_published_count=already_published_count,
            consumed_count=consumed_count,
        )
        operation.state = "completed"
        operation.stage = "messages_published"
        operation.response_json = json.dumps(
            {
                "message_count": summary.message_count,
                "status": "published",
                "published_count": summary.published_count,
                "already_published_count": summary.already_published_count,
                "consumed_count": summary.consumed_count,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        operation.error_code = None
        operation.retryable = False
        await session.commit()
    return summary


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
    protected: list[CanonicalMessage],
    last_cell_ts: int,
    operation_id: str | None,
) -> None:
    """Atomically commit MemCells, consume the buffer, and checkpoint an op.

    Previously these were three independent transactions.  A pipeline error
    could therefore leave committed MemCells plus a consumed buffer without a
    durable receipt telling a retry what to resume.  Keeping the business rows
    and the operation stage in one SQLite transaction closes that ambiguity.
    """

    replacement = _dedupe_messages_by_id([*protected, *tail])
    replacement_rows = [
        _canonical_to_row(message, app_id, project_id) for message in replacement
    ]
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
        if replacement_rows:
            session.add_all(replacement_rows)

        # A staged receipt remains durable after its raw buffer row is
        # consumed. This is what makes a same-revision retry after a successful
        # flush a true no-op instead of another extraction opportunity.
        consumed_to_memcells: dict[str, list[str]] = {}
        for row in rows:
            message_ids = json.loads(row.message_ids_json)
            if not isinstance(message_ids, list) or not all(
                isinstance(message_id, str) for message_id in message_ids
            ):
                raise MemoryMessageRecoveryError(
                    f"memcell {row.memcell_id!r} has invalid message ids"
                )
            for message_id in message_ids:
                consumed_to_memcells.setdefault(message_id, []).append(row.memcell_id)

        if consumed_to_memcells:
            receipt_stmt = select(MemoryMessageReceipt).where(
                MemoryMessageReceipt.message_id.in_(tuple(consumed_to_memcells))
            )
            receipts = list((await session.execute(receipt_stmt)).scalars().all())
            for receipt in receipts:
                if receipt.authority_state != _AUTH_PUBLISHED:
                    raise MemoryMessageRecoveryError(
                        "boundary attempted to consume a staged message without "
                        "published authority"
                    )
                receipt.authority_state = _AUTH_CONSUMED
                receipt.memcell_ids_json = json.dumps(
                    consumed_to_memcells[receipt.message_id],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )

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
    *,
    protected: list[CanonicalMessage] | None = None,
) -> None:
    replacement = _dedupe_messages_by_id([*(protected or []), *rows])
    await unprocessed_buffer_repo.replace(
        session_id,
        _TRACK,
        [_canonical_to_row(m, app_id, project_id) for m in replacement],
        app_id=app_id,
        project_id=project_id,
    )


async def _partition_buffer_rows_by_authority(
    rows: list[UnprocessedBuffer],
) -> tuple[list[UnprocessedBuffer], list[UnprocessedBuffer]]:
    """Split extraction-visible rows from unpublished staged rows.

    Legacy rows have no receipt and remain visible for backward compatibility.
    ``consumed`` + buffer presence is an impossible state; fail closed rather
    than risking duplicate extraction.
    """

    receipts = await memory_message_receipt_repo.map_by_message_ids(
        [row.message_id for row in rows]
    )
    eligible: list[UnprocessedBuffer] = []
    protected: list[UnprocessedBuffer] = []
    for row in rows:
        receipt = receipts.get(row.message_id)
        if receipt is None:
            # Pre-Stage3 deferred rows also use the ``ms_`` transport prefix
            # but have no authority receipt.  Treat them as unpublished and
            # preserve them fail-closed.  Ordinary legacy /add rows (``m_``)
            # and pre-existing test/import rows remain backward compatible.
            if row.message_id.startswith("ms_"):
                protected.append(row)
            else:
                eligible.append(row)
            continue
        if (
            receipt.app_id != row.app_id
            or receipt.project_id != row.project_id
            or receipt.session_id != row.session_id
        ):
            raise MemoryMessageRecoveryError(
                "message receipt scope does not match its buffer row"
            )
        if receipt.authority_state == _AUTH_PUBLISHED:
            eligible.append(row)
        elif receipt.authority_state == _AUTH_PENDING:
            protected.append(row)
        elif receipt.authority_state == _AUTH_CONSUMED:
            raise MemoryMessageRecoveryError(
                "consumed message receipt still has a live buffer row"
            )
        else:
            raise MemoryMessageRecoveryError(
                f"invalid message authority state: {receipt.authority_state!r}"
            )
    return eligible, protected


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
        identity = _canonical_transport_identity(message)
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


def _drop_fresh_aliases_of_protected(
    protected: list[CanonicalMessage],
    fresh: list[CanonicalMessage],
) -> list[CanonicalMessage]:
    """Prevent legacy /add from bypassing pending stage authority."""

    if not protected or not fresh:
        return fresh
    protected_identities = {
        _canonical_transport_identity(message) for message in protected
    }
    return [
        message
        for message in fresh
        if _canonical_transport_identity(message) not in protected_identities
    ]


def _canonical_transport_identity(message: CanonicalMessage) -> str:
    return json.dumps(
        message.model_dump(mode="json", exclude={"message_id"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _dedupe_messages_by_id(
    messages: list[CanonicalMessage],
) -> list[CanonicalMessage]:
    """Preserve first occurrence while rebuilding one buffer slice."""

    seen: set[str] = set()
    result: list[CanonicalMessage] = []
    for message in messages:
        if message.message_id in seen:
            continue
        seen.add(message.message_id)
        result.append(message)
    return sorted(result, key=lambda m: (m.timestamp, m.message_id))


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
