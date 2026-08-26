"""Boundary wait/deadline state machine.

Authority is evaluated before lifecycle decisions.  A timer can choose when
to retry an already-authorized tail, but it can never turn
``pending_publish`` into ``published``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from everos.component.utils.datetime import ensure_utc, get_utc_now
from everos.infra.persistence.sqlite import (
    BoundaryLifecycle,
    boundary_lifecycle_repo,
    memory_message_receipt_repo,
)

from .observability import observe_lifecycle_decision


class BoundaryLifecycleState(StrEnum):
    WAITING = "waiting"
    READY = "ready"
    BLOCKED_PENDING_PUBLISH = "blocked_pending_publish"
    CONSUMED = "consumed"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class LifecyclePolicy:
    idle_timeout_seconds: float = 3.0
    max_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.idle_timeout_seconds <= 0 or self.max_delay_seconds <= 0:
            raise ValueError("lifecycle deadlines must be positive")
        if self.idle_timeout_seconds > self.max_delay_seconds:
            raise ValueError("idle timeout cannot exceed max delay")


@dataclass(frozen=True, slots=True)
class BoundaryLifecycleRecord:
    lifecycle_id: str
    app_id: str
    project_id: str
    session_id: str
    track: str
    revision: int
    message_digest: str
    message_ids: tuple[str, ...]
    should_wait: bool | None
    observed_at: dt.datetime
    idle_deadline: dt.datetime | None
    max_deadline: dt.datetime | None
    authority_state: str
    state: BoundaryLifecycleState = BoundaryLifecycleState.WAITING
    final_requested: bool = False
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class BoundaryLifecycleDecision:
    action: str
    reason: str
    record: BoundaryLifecycleRecord


_DEFAULT_POLICY = LifecyclePolicy()


def evaluate_boundary_lifecycle(
    record: BoundaryLifecycleRecord,
    *,
    now: dt.datetime | None = None,
    policy: LifecyclePolicy = _DEFAULT_POLICY,
    explicit_final: bool = False,
    session_end: bool = False,
    force_flush: bool = False,
    authority_state: str | None = None,
) -> BoundaryLifecycleDecision:
    """Return ``wait`` or ``process`` without mutating authority.

    The order is deliberate: terminal/authority checks precede all timer and
    force checks, so pending publish and consumed receipts fail closed.
    """

    now = _aware(now or get_utc_now())
    authority = authority_state or record.authority_state
    if record.consumed or record.state == BoundaryLifecycleState.CONSUMED:
        return _decision(
            record,
            now=now,
            action="skip",
            reason="consumed",
            state=BoundaryLifecycleState.CONSUMED,
        )
    if authority == "pending_publish":
        return _decision(
            record,
            now=now,
            action="block",
            reason="pending_publish",
            state=BoundaryLifecycleState.BLOCKED_PENDING_PUBLISH,
        )
    if authority not in {"published", "legacy"}:
        return _decision(
            record,
            now=now,
            action="block",
            reason="unknown_authority",
            state=BoundaryLifecycleState.BLOCKED_PENDING_PUBLISH,
        )
    if explicit_final or session_end or force_flush or record.final_requested:
        return _decision(
            record,
            now=now,
            action="process",
            reason="explicit_or_session_final",
            state=BoundaryLifecycleState.READY,
            final_requested=True,
        )
    if record.should_wait is False:
        return _decision(
            record,
            now=now,
            action="process",
            reason="detector_says_no_wait",
            state=BoundaryLifecycleState.READY,
        )
    if record.max_deadline is not None and now >= _aware(record.max_deadline):
        return _decision(
            record,
            now=now,
            action="process",
            reason="max_delay_elapsed",
            state=BoundaryLifecycleState.READY,
        )
    if record.idle_deadline is not None and now >= _aware(record.idle_deadline):
        return _decision(
            record,
            now=now,
            action="process",
            reason="idle_timeout_elapsed",
            state=BoundaryLifecycleState.READY,
        )
    return _decision(
        record,
        now=now,
        action="wait",
        reason="tail_not_ready",
        state=BoundaryLifecycleState.WAITING,
    )


class BoundaryLifecycleStore:
    """Async adapter that makes restart recovery explicit and testable."""

    async def load(
        self,
        session_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
        track: str = "memorize",
    ) -> BoundaryLifecycleRecord | None:
        row = await boundary_lifecycle_repo.get_for_session(
            session_id, app_id=app_id, project_id=project_id, track=track
        )
        return _from_row(row) if row is not None else None

    async def save(self, record: BoundaryLifecycleRecord) -> None:
        await boundary_lifecycle_repo.upsert(
            BoundaryLifecycle(
                lifecycle_id=record.lifecycle_id,
                app_id=record.app_id,
                project_id=record.project_id,
                session_id=record.session_id,
                track=record.track,
                revision=record.revision,
                message_digest=record.message_digest,
                message_ids_json=json.dumps(
                    list(record.message_ids), separators=(",", ":"), sort_keys=True
                ),
                should_wait=record.should_wait,
                state=record.state.value,
                authority_state=record.authority_state,
                observed_at=_aware(record.observed_at),
                idle_deadline=(
                    _aware(record.idle_deadline) if record.idle_deadline else None
                ),
                max_deadline=(
                    _aware(record.max_deadline) if record.max_deadline else None
                ),
                final_requested=record.final_requested,
                consumed=record.consumed,
            )
        )

    async def list_due(
        self, now: dt.datetime | None = None
    ) -> list[BoundaryLifecycleRecord]:
        rows = await boundary_lifecycle_repo.list_due(_aware(now or get_utc_now()))
        due: list[BoundaryLifecycleRecord] = []
        for row in rows:
            record = _from_row(row)
            if record.state == BoundaryLifecycleState.BLOCKED_PENDING_PUBLISH:
                # Publish commits receipt authority before the separate
                # lifecycle checkpoint can be re-armed. If the process dies
                # in that narrow window, surface the durable row on the next
                # scheduler scan and repair the checkpoint before dispatching
                # it. Pending or incomplete receipts remain fail-closed.
                receipts = await memory_message_receipt_repo.map_by_message_ids(
                    record.message_ids
                )
                if not record.message_ids or len(receipts) != len(record.message_ids):
                    continue
                if not all(
                    receipt.authority_state in {"published", "consumed"}
                    for receipt in receipts.values()
                ):
                    continue
                record = _replace(
                    record,
                    authority_state="published",
                    state=BoundaryLifecycleState.WAITING,
                )
                await self.save(record)
            due.append(record)
        return due


def build_lifecycle_record(
    *,
    app_id: str,
    project_id: str,
    session_id: str,
    track: str,
    message_ids: Sequence[str],
    revision: int,
    should_wait: bool | None,
    authority_state: str,
    observed_at: dt.datetime | None = None,
    policy: LifecyclePolicy = _DEFAULT_POLICY,
    consumed: bool = False,
) -> BoundaryLifecycleRecord:
    observed = _aware(observed_at or get_utc_now())
    ids = tuple(message_ids)
    digest = hashlib.sha256(
        json.dumps(list(ids), separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    lifecycle_id = hashlib.sha256(
        f"{app_id}\0{project_id}\0{session_id}\0{track}".encode()
    ).hexdigest()
    return BoundaryLifecycleRecord(
        lifecycle_id=lifecycle_id,
        app_id=app_id,
        project_id=project_id,
        session_id=session_id,
        track=track,
        revision=revision,
        message_digest=digest,
        message_ids=ids,
        should_wait=should_wait,
        observed_at=observed,
        idle_deadline=observed + dt.timedelta(seconds=policy.idle_timeout_seconds),
        max_deadline=observed + dt.timedelta(seconds=policy.max_delay_seconds),
        authority_state=authority_state,
        state=(
            BoundaryLifecycleState.CONSUMED
            if consumed
            else BoundaryLifecycleState.WAITING
        ),
        consumed=consumed,
    )


def _replace(
    record: BoundaryLifecycleRecord, **changes: object
) -> BoundaryLifecycleRecord:
    data = {
        name: getattr(record, name)
        for name in BoundaryLifecycleRecord.__dataclass_fields__
    }
    data.update(changes)
    return BoundaryLifecycleRecord(**data)


def _decision(
    record: BoundaryLifecycleRecord,
    *,
    now: dt.datetime,
    action: str,
    reason: str,
    state: BoundaryLifecycleState,
    final_requested: bool | None = None,
) -> BoundaryLifecycleDecision:
    changes: dict[str, object] = {"state": state}
    if final_requested is not None:
        changes["final_requested"] = final_requested
    updated = _replace(record, **changes)
    elapsed = max(0.0, (now - _aware(record.observed_at)).total_seconds())
    observe_lifecycle_decision(action=action, reason=reason, elapsed_seconds=elapsed)
    return BoundaryLifecycleDecision(action, reason, updated)


def _from_row(row: BoundaryLifecycle) -> BoundaryLifecycleRecord:
    try:
        ids = json.loads(row.message_ids_json)
    except json.JSONDecodeError as exc:
        raise ValueError("boundary lifecycle message_ids_json is invalid") from exc
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("boundary lifecycle message_ids_json is invalid")
    try:
        state = BoundaryLifecycleState(row.state)
    except ValueError as exc:
        raise ValueError(f"unknown boundary lifecycle state: {row.state!r}") from exc
    return BoundaryLifecycleRecord(
        lifecycle_id=row.lifecycle_id,
        app_id=row.app_id,
        project_id=row.project_id,
        session_id=row.session_id,
        track=row.track,
        revision=row.revision,
        message_digest=row.message_digest,
        message_ids=tuple(ids),
        should_wait=row.should_wait,
        observed_at=_aware(row.observed_at),
        idle_deadline=_aware(row.idle_deadline) if row.idle_deadline else None,
        max_deadline=_aware(row.max_deadline) if row.max_deadline else None,
        authority_state=row.authority_state,
        state=state,
        final_requested=row.final_requested,
        consumed=row.consumed,
    )


def _aware(value: dt.datetime) -> dt.datetime:
    normalized = ensure_utc(value)
    assert normalized is not None
    return normalized
