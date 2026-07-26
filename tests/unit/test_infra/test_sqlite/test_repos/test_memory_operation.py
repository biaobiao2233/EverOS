"""Persistence contract for the memory write-operation ledger."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite import MemoryOperation
from everos.infra.persistence.sqlite.repos.memory_operation import (
    _MemoryOperationRepo,
    canonical_json,
)


@pytest.fixture
async def repo(tmp_path: Path) -> _MemoryOperationRepo:
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return _MemoryOperationRepo(session_factory=factory)


def _operation(*, request_sha256: str = "a" * 64) -> MemoryOperation:
    return MemoryOperation(
        operation_id="evop1-add-" + "1" * 64,
        kind="add",
        app_id="codex",
        project_id="solo",
        session_id="session-1",
        request_sha256=request_sha256,
    )


async def test_claim_is_atomic_and_replay_preserves_original(
    repo: _MemoryOperationRepo,
) -> None:
    first, created = await repo.claim(_operation())
    assert created is True
    assert first.request_sha256 == "a" * 64

    conflicting, created = await repo.claim(_operation(request_sha256="b" * 64))
    assert created is False
    assert conflicting.request_sha256 == "a" * 64


async def test_concurrent_claim_has_exactly_one_creator(
    repo: _MemoryOperationRepo,
) -> None:
    results = await asyncio.gather(*(repo.claim(_operation()) for _ in range(20)))

    assert sum(created for _row, created in results) == 1
    assert {row.request_sha256 for row, _created in results} == {"a" * 64}


async def test_transitions_store_strict_receipts(repo: _MemoryOperationRepo) -> None:
    operation, _ = await repo.claim(_operation())
    operation = await repo.mark_memcells_committed(
        operation.operation_id, ["mc_b", "mc_a"]
    )
    assert operation.stage == "memcells_committed"
    assert json.loads(operation.memcell_ids_json) == ["mc_b", "mc_a"]

    operation = await repo.mark_failed(
        operation.operation_id,
        error_code="JSONDecodeError",
        retryable=True,
    )
    assert operation.state == "failed"
    assert operation.stage == "memcells_committed"
    assert operation.retryable is True

    operation = await repo.mark_running(operation.operation_id)
    assert operation.state == "running"
    assert operation.error_code is None

    operation = await repo.mark_completed(
        operation.operation_id,
        {"status": "extracted", "message_count": 2},
    )
    assert operation.state == "completed"
    assert operation.stage == "sync_dispatch_completed"
    assert operation.response_json == '{"message_count":2,"status":"extracted"}'


async def test_transition_missing_operation_is_loud(repo: _MemoryOperationRepo) -> None:
    with pytest.raises(KeyError, match="not found"):
        await repo.mark_failed(
            "evop1-flush-" + "0" * 64,
            error_code="missing",
            retryable=False,
        )


async def test_completed_receipt_cannot_be_downgraded_by_late_failure(
    repo: _MemoryOperationRepo,
) -> None:
    operation, _ = await repo.claim(_operation())
    await repo.mark_completed(
        operation.operation_id,
        {"status": "extracted", "message_count": 1},
    )
    final = await repo.mark_failed(
        operation.operation_id,
        error_code="TimeoutError",
        retryable=True,
    )
    assert final.state == "completed"
    assert final.stage == "sync_dispatch_completed"
    assert final.error_code is None


def test_canonical_json_is_stable_unicode_and_rejects_nonfinite_numbers() -> None:
    left = canonical_json({"z": "中文", "a": [2, 1]})
    right = canonical_json({"a": [2, 1], "z": "中文"})
    assert left == right == '{"a":[2,1],"z":"中文"}'
    with pytest.raises(ValueError):
        canonical_json({"score": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"score": float("inf")})
