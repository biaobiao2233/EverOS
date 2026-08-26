"""Additive schema + read contract for durable message receipts."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.core.persistence.sqlite import session_scope
from everos.infra.persistence.sqlite import MemoryMessageReceipt
from everos.infra.persistence.sqlite.repos.memory_message_receipt import (
    _MemoryMessageReceiptRepo,
)


@pytest.fixture
async def receipt_env(
    tmp_path: Path,
) -> tuple[_MemoryMessageReceiptRepo, object]:
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return _MemoryMessageReceiptRepo(session_factory=factory), factory


def _receipt(*, receipt_id: str = "evmsg1-" + "1" * 64) -> MemoryMessageReceipt:
    return MemoryMessageReceipt(
        receipt_id=receipt_id,
        app_id="codex",
        project_id="solo",
        session_id="s1",
        idem_key="web:s1:gpt-msg-1",
        message_id="ms_s1_" + "a" * 24,
        source="web",
        external_ref="gpt-msg-1",
        revision=2,
        payload_sha256="b" * 64,
    )


async def test_create_all_adds_receipt_table_and_repo_can_read(
    receipt_env: tuple[_MemoryMessageReceiptRepo, object],
) -> None:
    repo, factory = receipt_env
    row = _receipt()
    async with session_scope(factory) as session:
        session.add(row)
        await session.commit()

    stored = await repo.get_by_idem_key(
        app_id="codex",
        project_id="solo",
        idem_key="web:s1:gpt-msg-1",
    )
    assert stored is not None
    assert stored.message_id == row.message_id
    mapped = await repo.map_by_message_ids([row.message_id])
    assert list(mapped) == [row.message_id]
    assert mapped[row.message_id].authority_state == "pending_publish"


async def test_scope_idem_and_message_id_are_unique(
    receipt_env: tuple[_MemoryMessageReceiptRepo, object],
) -> None:
    _repo, factory = receipt_env
    first = _receipt()
    async with session_scope(factory) as session:
        session.add(first)
        await session.commit()

    same_idem = _receipt(receipt_id="evmsg1-" + "2" * 64)
    same_idem.message_id = "ms_s1_" + "c" * 24
    with pytest.raises(IntegrityError):
        async with session_scope(factory) as session:
            session.add(same_idem)
            await session.commit()

    same_message_id = _receipt(receipt_id="evmsg1-" + "3" * 64)
    same_message_id.idem_key = "web:s1:gpt-msg-3"
    with pytest.raises(IntegrityError):
        async with session_scope(factory) as session:
            session.add(same_message_id)
            await session.commit()
