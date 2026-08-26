"""End-to-end memorize integration tests.

Drives ``service.memorize.memorize()`` with a ``FakeLLMClient`` so the
full chain (ingest → boundary → user / agent pipeline → md + OME emit)
runs without real LLM calls. Each test isolates state by:

- redirecting ``MemoryRoot.default()`` to a ``tmp_path``
- resetting service-layer lazy singletons
- starting / stopping a per-test ``OfflineEngine``
- patching ``get_llm_client`` (boundary + strategies) onto a fake

OME strategies (atomic / foresight) are silenced via ``mock_aextract`` so
this test focuses on the synchronous boundary + pipeline + md path —
strategy dispatch correctness already has its own coverage in
``test_ome_strategies_integration.py``.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sqlite3
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from sqlmodel import SQLModel

from everos.core.persistence import MarkdownReader, MemoryRoot
from everos.memory.search.dto import FilterNode
from everos.memory.search.manager import SearchManager
from everos.service.memorize import (
    BackgroundFlushResult,
    MemorizeResult,
    MemoryMessageConflictError,
    MemoryOperationConflictError,
    PublishResult,
    StageResult,
    get_memory_operation_status,
    memorize,
    publish,
    queue_background_flush,
    stage,
)

# ---------------------------------------------------------------------------
# Canned LLM responses
# ---------------------------------------------------------------------------


def _boundary_response(boundaries: list[int]) -> str:
    """Build a ``detect_boundaries`` JSON response (algo schema)."""
    payload = {
        "reasoning": "test",
        "boundaries": boundaries,
        "should_wait": False,
    }
    return json.dumps(payload)


def _episode_response(title: str = "Test Subject", content: str = "Test body") -> str:
    """Build an ``EpisodeExtractor`` JSON response (algo schema)."""
    return json.dumps(
        {
            "title": title,
            "summary": "Faithful test summary",
            "content": content,
        }
    )


def _make_fake_llm(
    boundary_responses: list[list[int]] | None = None,
    *,
    episode_title: str = "Test Subject",
    episode_content: str = "Test body",
) -> FakeLLMClient:
    """Build a ``FakeLLMClient`` that dispatches by prompt fingerprint.

    Pops one ``boundaries=...`` from ``boundary_responses`` per boundary
    prompt seen; every episode prompt returns the same canned
    ``{title, content}``.
    """
    boundary_queue: list[list[int]] = list(boundary_responses or [])

    def handler(messages: list[LLMChatMessage], **_: Any) -> ChatResponse:
        prompt = messages[0].content
        if "boundaries" in prompt.lower() or "memcell" in prompt.lower():
            cuts = boundary_queue.pop(0) if boundary_queue else []
            return ChatResponse(content=_boundary_response(cuts), model="fake")
        # Fall through to episode (also catches atomic/foresight prompts —
        # they'll return success-but-empty in their mocked extractor below).
        return ChatResponse(
            content=_episode_response(episode_title, episode_content),
            model="fake",
        )

    return FakeLLMClient(handler=handler)


# ---------------------------------------------------------------------------
# Shared setup fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def memorize_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[..., AsyncMock]]:
    """Yield a builder that configures a clean memorize environment.

    Usage::

        async def test_x(memorize_env):
            await memorize_env(mode="chat", fake_llm=_make_fake_llm([...]))
            outcome = await memorize({"session_id": "s", "messages": [...]})

    The builder must be called exactly once per test (it primes singletons
    + starts the OME engine). Teardown stops the engine and disposes the
    sqlite engine.
    """
    monkeypatch.setattr(
        MemoryRoot, "default", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)

    svc = importlib.import_module("everos.service.memorize")
    af_mod = importlib.import_module("everos.memory.strategies.extract_atomic_facts")
    fs_mod = importlib.import_module("everos.memory.strategies.extract_foresight")
    client_mod = importlib.import_module("everos.component.llm.client")

    # Reset singletons.
    for attr in (
        "_episode_writer",
        "_prompt_loader",
        "_user_pipeline",
        "_agent_pipeline",
        "_ome_engine",
    ):
        monkeypatch.setattr(svc, attr, None, raising=False)
    monkeypatch.setattr(client_mod, "_llm_client", None, raising=False)
    monkeypatch.setattr(af_mod, "_writer", None, raising=False)
    monkeypatch.setattr(fs_mod, "_writer", None, raising=False)

    started: dict[str, Any] = {"engine": None, "sqlite_engine": None}

    async def _setup(
        *,
        mode: str = "chat",
        fake_llm: FakeLLMClient,
        hard_token_limit: int = 65536,
        hard_msg_limit: int = 500,
    ) -> None:
        # Provide a non-None API key + base_url so get_llm_client doesn't
        # raise; we replace the cached singleton with our fake right after.
        monkeypatch.setenv("EVEROS_MEMORIZE__MODE", mode)
        monkeypatch.setenv("EVEROS_LLM__API_KEY", "fake-key")
        monkeypatch.setenv("EVEROS_LLM__BASE_URL", "https://fake.example.com")
        monkeypatch.setenv(
            "EVEROS_BOUNDARY_DETECTION__HARD_TOKEN_LIMIT", str(hard_token_limit)
        )
        monkeypatch.setenv(
            "EVEROS_BOUNDARY_DETECTION__HARD_MSG_LIMIT", str(hard_msg_limit)
        )
        from everos.config import load_settings

        load_settings.cache_clear()

        # Replace the cached client singleton with our fake so get_llm_client
        # returns the fake on subsequent calls.
        monkeypatch.setattr(client_mod, "_llm_client", fake_llm)

        # Build sqlite schema.
        from everos.infra.persistence.sqlite import dispose_engine, get_engine

        db_engine = get_engine()
        async with db_engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        started["sqlite_engine"] = (get_engine, dispose_engine)

        # Mock the OME extractors so the async strategy chain is a no-op
        # (the strategy itself still runs; it just sees no facts/foresights).
        mock_af = AsyncMock(return_value=[])
        mock_fs = AsyncMock(return_value=[])
        monkeypatch.setattr(
            af_mod,
            "AtomicFactExtractor",
            lambda *a, **k: type("M", (), {"aextract": mock_af})(),
        )
        monkeypatch.setattr(
            fs_mod,
            "ForesightExtractor",
            lambda *a, **k: type("M", (), {"aextract": mock_fs})(),
        )

        engine = svc._get_engine()
        await engine.start()
        started["engine"] = engine

    yield _setup

    if started["engine"] is not None:
        await started["engine"].stop()
    if started["sqlite_engine"] is not None:
        _, dispose = started["sqlite_engine"]
        await dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(
    role: str,
    content: str,
    *,
    sender_id: str = "u_alice",
    timestamp: int = 1_700_000_000_000,
    tool_calls: list[dict] | None = None,
    tool_call_id: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sender_id": sender_id,
        "role": role,
        "content": content,
        "timestamp": timestamp,
    }
    if tool_calls is not None:
        out["tool_calls"] = tool_calls
    if tool_call_id is not None:
        out["tool_call_id"] = tool_call_id
    return out


def _user(content: str, ts: int, *, sender: str = "u_alice") -> dict[str, Any]:
    return _msg("user", content, sender_id=sender, timestamp=ts)


def _assistant(content: str, ts: int, *, sender: str = "assistant") -> dict[str, Any]:
    return _msg("assistant", content, sender_id=sender, timestamp=ts)


def _staged(
    message: dict[str, Any],
    external_ref: str,
    *,
    revision: int = 0,
    source: str = "web",
) -> dict[str, Any]:
    return {
        **message,
        "source": source,
        "external_ref": external_ref,
        "revision": revision,
    }


def _memcell_rows(tmp_path: Path) -> list[sqlite3.Row]:
    db = tmp_path / ".index" / "sqlite" / "system.db"
    if not db.is_file():
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM memcell ORDER BY timestamp"))
    finally:
        conn.close()


def _buffer_count(tmp_path: Path) -> int:
    db = tmp_path / ".index" / "sqlite" / "system.db"
    if not db.is_file():
        return 0
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM unprocessed_buffer WHERE track='memorize'"
        ).fetchone()[0]
    finally:
        conn.close()


def _buffer_texts(tmp_path: Path) -> list[str | None]:
    db = tmp_path / ".index" / "sqlite" / "system.db"
    if not db.is_file():
        return []
    conn = sqlite3.connect(db)
    try:
        return [
            row[0]
            for row in conn.execute(
                "SELECT text FROM unprocessed_buffer "
                "WHERE track='memorize' ORDER BY timestamp, message_id"
            )
        ]
    finally:
        conn.close()


def _receipt_rows(tmp_path: Path) -> list[sqlite3.Row]:
    db = tmp_path / ".index" / "sqlite" / "system.db"
    if not db.is_file():
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return list(
            conn.execute("SELECT * FROM memory_message_receipt ORDER BY idem_key")
        )
    finally:
        conn.close()


def _episode_paths(tmp_path: Path) -> list[Path]:
    base = tmp_path / "default_app" / "default_project" / "users"
    return sorted(base.rglob("episode-*.md"))


# ---------------------------------------------------------------------------
# Happy path baseline
# ---------------------------------------------------------------------------


async def test_chat_baseline_two_msgs_one_cell(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
) -> None:
    """2 messages → flush forces them into 1 cell + 1 Episode + 1 memcell row."""
    fake = _make_fake_llm(boundary_responses=[[]])  # no internal cuts
    await memorize_env(mode="chat", fake_llm=fake)

    payload = {
        "session_id": "test_chat_1",
        "messages": [
            _user("hello", 1_700_000_000_000),
            _assistant("hi there", 1_700_000_001_000),
        ],
    }
    result = await memorize(payload, is_final=True)

    assert isinstance(result, MemorizeResult)
    assert result.status == "extracted"
    assert result.message_count == 2

    rows = _memcell_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["track"] == "memorize"
    assert rows[0]["raw_type"] == "Conversation"
    # MemCell has no single owner — sender_ids carries the participants.
    assert "u_alice" in json.loads(rows[0]["sender_ids_json"])

    assert _buffer_count(tmp_path) == 0

    md_files = _episode_paths(tmp_path)
    assert len(md_files) == 1
    body = md_files[0].read_text()
    assert "Test Subject" in body
    assert "Test body" in body


# ---------------------------------------------------------------------------
# Input-shape boundary cases (6)
# ---------------------------------------------------------------------------


async def test_empty_batch_non_final_is_skipped(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """``messages=[]`` + ``is_final=False`` → skipped, no side effects."""
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())

    result = await memorize(
        {"session_id": "test_empty_nonfinal", "messages": []}, is_final=False
    )
    assert result.status == "accumulated"
    assert result.message_count == 0
    assert _memcell_rows(tmp_path) == []
    assert _episode_paths(tmp_path) == []


async def test_empty_batch_final_drains_empty_buffer(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """``messages=[]`` + ``is_final=True`` on virgin session → no cells, no md."""
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())

    result = await memorize(
        {"session_id": "test_empty_final", "messages": []}, is_final=True
    )
    assert result.status == "accumulated"
    assert _memcell_rows(tmp_path) == []
    assert _episode_paths(tmp_path) == []


async def test_assistant_only_batch_accumulates(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """No role=user message → boundary stage parks everything in buffer."""
    fake = _make_fake_llm(boundary_responses=[])  # no LLM call expected
    await memorize_env(mode="chat", fake_llm=fake)

    result = await memorize(
        {
            "session_id": "test_asst_only",
            "messages": [
                _assistant("hi", 1_700_000_000_000),
                _assistant("anyone here?", 1_700_000_001_000),
            ],
        },
        is_final=False,
    )
    assert result.status == "accumulated"
    assert _memcell_rows(tmp_path) == []
    assert _buffer_count(tmp_path) == 2  # parked in buffer


async def test_single_user_message_accumulates(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Single user msg → boundary returns no cells (need conversation) → buffer it."""
    fake = _make_fake_llm(boundary_responses=[[]])  # boundary called, no cuts
    await memorize_env(mode="chat", fake_llm=fake)

    result = await memorize(
        {
            "session_id": "test_single",
            "messages": [_user("hello?", 1_700_000_000_000)],
        },
        is_final=False,
    )
    assert result.status == "accumulated"
    assert _memcell_rows(tmp_path) == []
    assert _buffer_count(tmp_path) == 1


async def test_chat_mode_filters_tool_messages(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Chat mode drops ``role=tool`` + assistant-with-tool_calls pre-boundary."""
    fake = _make_fake_llm(boundary_responses=[[]])
    await memorize_env(mode="chat", fake_llm=fake)

    result = await memorize(
        {
            "session_id": "test_chat_filter",
            "messages": [
                _user("debug this", 1_700_000_000_000),
                _msg(
                    "assistant",
                    "calling tool",
                    timestamp=1_700_000_001_000,
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "x", "arguments": "{}"},
                        }
                    ],
                ),
                _msg(
                    "tool",
                    "result",
                    sender_id="tool",
                    timestamp=1_700_000_002_000,
                    tool_call_id="c1",
                ),
                _assistant("here's the answer", 1_700_000_003_000),
            ],
        },
        is_final=True,
    )
    # After filter: 1 user + 1 assistant text = 2 msgs → 1 cell on flush.
    assert result.status == "extracted"
    rows = _memcell_rows(tmp_path)
    assert len(rows) == 1
    ids = json.loads(rows[0]["message_ids_json"])
    assert len(ids) == 2  # tool + assistant-with-tool_calls dropped


async def test_duplicate_message_id_dedup_across_adds(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Same message replayed across two ``/add`` calls is deduped by message_id."""
    fake = _make_fake_llm(boundary_responses=[[], []])  # 2 boundary calls, both empty
    await memorize_env(mode="chat", fake_llm=fake)

    # message_id is derived from (session_id, ts_ms, idx); same payload twice
    # produces the same id, so the second add should be a no-op insert.
    payload = {
        "session_id": "test_dedup",
        "messages": [
            _user("hi", 1_700_000_000_000),
            _assistant("hi back", 1_700_000_001_000),
        ],
    }
    await memorize(payload, is_final=False)
    await memorize(payload, is_final=False)  # replay
    await memorize({"session_id": "test_dedup", "messages": []}, is_final=True)

    rows = _memcell_rows(tmp_path)
    assert len(rows) == 1
    ids = json.loads(rows[0]["message_ids_json"])
    assert len(ids) == 2  # not 4 — dedup worked
    assert len(set(ids)) == 2  # unique


# ---------------------------------------------------------------------------
# Hard-limit cases (2)
# ---------------------------------------------------------------------------


async def test_hard_msg_limit_force_split(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Exceeding ``hard_msg_limit`` triggers a force-split before the LLM call."""
    fake = _make_fake_llm(boundary_responses=[[]])  # LLM call after force-split
    # hard_msg_limit=3 → batch of 5 msgs forces ~1 split before LLM.
    await memorize_env(
        mode="chat", fake_llm=fake, hard_msg_limit=3, hard_token_limit=10_000
    )

    msgs = [
        _user(f"u{i}", 1_700_000_000_000 + i * 1000, sender="u_alice")
        if i % 2 == 0
        else _assistant(f"a{i}", 1_700_000_000_000 + i * 1000)
        for i in range(5)
    ]
    result = await memorize(
        {"session_id": "test_hardmsg", "messages": msgs}, is_final=True
    )
    assert result.status == "extracted"
    rows = _memcell_rows(tmp_path)
    # Force-split + LLM final → at least 2 cells (force + remaining).
    assert len(rows) >= 2


async def test_hard_token_limit_force_split(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Exceeding ``hard_token_limit`` triggers a force-split (token-based)."""
    fake = _make_fake_llm(boundary_responses=[[]])
    # Very small token budget → even tiny content triggers force-split.
    await memorize_env(
        mode="chat", fake_llm=fake, hard_msg_limit=500, hard_token_limit=20
    )

    msgs = [
        _user("a" * 200, 1_700_000_000_000, sender="u_alice"),
        _assistant("b" * 200, 1_700_000_001_000),
        _user("c" * 200, 1_700_000_002_000, sender="u_alice"),
        _assistant("d" * 200, 1_700_000_003_000),
    ]
    result = await memorize(
        {"session_id": "test_hardtok", "messages": msgs}, is_final=True
    )
    assert result.status == "extracted"
    assert len(_memcell_rows(tmp_path)) >= 2


# ---------------------------------------------------------------------------
# Flush state-machine cases (4)
# ---------------------------------------------------------------------------


async def test_flush_on_virgin_session_is_noop(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Flush a session that never received ``/add`` — should not crash."""
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())

    result = await memorize(
        {"session_id": "test_virgin_flush", "messages": []}, is_final=True
    )
    assert result.status == "accumulated"
    assert _memcell_rows(tmp_path) == []


async def test_add_then_flush_then_add(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """After flush drains the buffer, a follow-up ``/add`` still works."""
    fake = _make_fake_llm(boundary_responses=[[], []])
    await memorize_env(mode="chat", fake_llm=fake)

    sid = "test_add_flush_add"
    await memorize(
        {
            "session_id": sid,
            "messages": [
                _user("first", 1_700_000_000_000),
                _assistant("ack", 1_700_000_001_000),
            ],
        },
        is_final=False,
    )
    await memorize({"session_id": sid, "messages": []}, is_final=True)

    rows_after_flush_1 = len(_memcell_rows(tmp_path))
    assert rows_after_flush_1 == 1

    # Second turn after the flush.
    await memorize(
        {
            "session_id": sid,
            "messages": [
                _user("second turn", 1_700_000_010_000),
                _assistant("ok", 1_700_000_011_000),
            ],
        },
        is_final=True,
    )
    assert len(_memcell_rows(tmp_path)) == 2  # cumulative


async def test_consecutive_flushes_second_is_noop(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Flush twice in a row — second call finds empty buffer, no-ops."""
    fake = _make_fake_llm(boundary_responses=[[]])
    await memorize_env(mode="chat", fake_llm=fake)

    sid = "test_double_flush"
    await memorize(
        {
            "session_id": sid,
            "messages": [
                _user("hi", 1_700_000_000_000),
                _assistant("ok", 1_700_000_001_000),
            ],
        },
        is_final=False,
    )
    res1 = await memorize({"session_id": sid, "messages": []}, is_final=True)
    res2 = await memorize({"session_id": sid, "messages": []}, is_final=True)

    assert res1.status == "extracted"
    assert res2.status == "accumulated"  # nothing left
    assert len(_memcell_rows(tmp_path)) == 1


async def test_flush_drains_assistant_only_buffer(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Buffer with only assistant messages: flush still forces them into a cell."""
    fake = _make_fake_llm(boundary_responses=[[]])
    await memorize_env(mode="chat", fake_llm=fake)

    sid = "test_asst_then_flush"
    # Two assistant-only adds → both park in buffer.
    await memorize(
        {
            "session_id": sid,
            "messages": [_assistant("a1", 1_700_000_000_000)],
        },
        is_final=False,
    )
    await memorize(
        {
            "session_id": sid,
            "messages": [_assistant("a2", 1_700_000_001_000)],
        },
        is_final=False,
    )
    assert _buffer_count(tmp_path) == 2

    # Add a user message + flush — boundary should now run.
    result = await memorize(
        {
            "session_id": sid,
            "messages": [_user("anyone there?", 1_700_000_002_000)],
        },
        is_final=True,
    )
    assert result.status == "extracted"
    assert _buffer_count(tmp_path) == 0


# ---------------------------------------------------------------------------
# Multi-session cases (2)
# ---------------------------------------------------------------------------


async def test_two_sessions_are_isolated(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Two session_ids share the engine but their buffers / cells stay separate."""
    fake = _make_fake_llm(boundary_responses=[[], []])  # 1 per session
    await memorize_env(mode="chat", fake_llm=fake)

    await memorize(
        {
            "session_id": "sess_A",
            "messages": [
                _user("hi from A", 1_700_000_000_000, sender="u_alice"),
                _assistant("ack A", 1_700_000_001_000),
            ],
        },
        is_final=True,
    )
    await memorize(
        {
            "session_id": "sess_B",
            "messages": [
                _user("hi from B", 1_700_000_010_000, sender="u_bob"),
                _assistant("ack B", 1_700_000_011_000),
            ],
        },
        is_final=True,
    )

    rows = _memcell_rows(tmp_path)
    assert len(rows) == 2
    sessions = sorted(r["session_id"] for r in rows)
    assert sessions == ["sess_A", "sess_B"]
    # MemCell has no single owner — sender_ids carries who participated.
    senders = {r["session_id"]: json.loads(r["sender_ids_json"]) for r in rows}
    assert "u_alice" in senders["sess_A"]
    assert "u_bob" in senders["sess_B"]


async def test_same_session_multi_add_concatenates(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Multiple adds on the same session accumulate in one buffer until flushed."""
    fake = _make_fake_llm(boundary_responses=[[], [], []])
    await memorize_env(mode="chat", fake_llm=fake)

    sid = "test_multi_add"
    for i in range(3):
        await memorize(
            {
                "session_id": sid,
                "messages": [
                    _user(f"u{i}", 1_700_000_000_000 + i * 2000),
                    _assistant(f"a{i}", 1_700_000_001_000 + i * 2000),
                ],
            },
            is_final=False,
        )
    # Buffer should have 6 messages now (no boundary cuts).
    assert _buffer_count(tmp_path) == 6

    result = await memorize({"session_id": sid, "messages": []}, is_final=True)
    assert result.status == "extracted"
    rows = _memcell_rows(tmp_path)
    assert len(rows) == 1  # one cell from the flush
    ids = json.loads(rows[0]["message_ids_json"])
    assert len(ids) == 6  # all 6 messages folded in


# ---------------------------------------------------------------------------
# Durable operation ledger
# ---------------------------------------------------------------------------


async def test_operation_resumes_after_memcell_commit_without_duplicate(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await memorize_env(mode="chat", fake_llm=_make_fake_llm(boundary_responses=[[]]))
    svc = importlib.import_module("everos.service.memorize")
    real_pipeline = svc._get_user_pipeline()

    class _FailOnce:
        calls = 0

        async def run(self, *args: Any, **kwargs: Any):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated interruption after boundary commit")
            return await real_pipeline.run(*args, **kwargs)

    flaky = _FailOnce()
    monkeypatch.setattr(svc, "_get_user_pipeline", lambda: flaky)
    operation_id = "evop1-flush-" + "a" * 64
    payload = {
        "operation_id": operation_id,
        "session_id": "op_resume",
        "messages": [
            _user("remember this", 1_700_000_000_000),
            _assistant("ack", 1_700_000_001_000),
        ],
    }

    with pytest.raises(RuntimeError, match="simulated interruption"):
        await memorize(payload, is_final=True)
    assert len(_memcell_rows(tmp_path)) == 1
    failed = await get_memory_operation_status(operation_id)
    assert failed is not None
    assert failed.state == "failed"
    assert failed.stage == "memcells_committed"
    assert failed.retryable is True

    resumed = await memorize(payload, is_final=True)
    assert resumed.operation_id == operation_id
    assert resumed.replayed is False
    assert len(_memcell_rows(tmp_path)) == 1
    assert len(_episode_paths(tmp_path)) == 1
    parsed = await MarkdownReader.read(_episode_paths(tmp_path)[0])
    assert len(parsed.entries) == 1

    replay = await memorize(payload, is_final=True)
    assert replay.operation_id == operation_id
    assert replay.replayed is True
    assert replay.message_count == resumed.message_count
    assert len(_memcell_rows(tmp_path)) == 1
    parsed = await MarkdownReader.read(_episode_paths(tmp_path)[0])
    assert len(parsed.entries) == 1


async def test_operation_id_conflict_is_rejected(
    memorize_env: Callable[..., Any],
) -> None:
    await memorize_env(mode="chat", fake_llm=_make_fake_llm(boundary_responses=[[]]))
    operation_id = "evop1-flush-" + "b" * 64
    first = {
        "operation_id": operation_id,
        "session_id": "op_conflict",
        "messages": [_user("first", 1_700_000_000_000)],
    }
    await memorize(first, is_final=True)

    changed = {
        **first,
        "messages": [_user("changed", 1_700_000_000_000)],
    }
    with pytest.raises(MemoryOperationConflictError, match="different request"):
        await memorize(changed, is_final=True)


async def test_concurrent_same_operation_converges_to_one_result(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
) -> None:
    await memorize_env(mode="chat", fake_llm=_make_fake_llm(boundary_responses=[[]]))
    payload = {
        "operation_id": "evop1-flush-" + "c" * 64,
        "session_id": "op_concurrent",
        "messages": [
            _user("one request", 1_700_000_000_000),
            _assistant("one result", 1_700_000_001_000),
        ],
    }

    first, second = await asyncio.gather(
        memorize(payload, is_final=True),
        memorize(payload, is_final=True),
    )
    assert sorted([first.replayed, second.replayed]) == [False, True]
    assert len(_memcell_rows(tmp_path)) == 1
    assert len(_episode_paths(tmp_path)) == 1


async def test_waiter_timeout_does_not_fail_the_active_operation_receipt(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await memorize_env(mode="chat", fake_llm=_make_fake_llm(boundary_responses=[[]]))
    monkeypatch.setenv("EVEROS_MEMORIZE__SESSION_LOCK_TIMEOUT_SECONDS", "0.05")
    from everos.config import load_settings
    from everos.service._session_lock import scoped_session_lock

    load_settings.cache_clear()
    operation_id = "evop1-flush-" + "d" * 64
    payload = {
        "operation_id": operation_id,
        "session_id": "op_waiter_timeout",
        "messages": [
            _user("lease owner", 1_700_000_000_000),
            _assistant("ack", 1_700_000_001_000),
        ],
    }

    async with scoped_session_lock(MemoryRoot.default(), "op_waiter_timeout"):
        with pytest.raises(TimeoutError):
            await memorize(payload, is_final=True)

    waiting = await get_memory_operation_status(operation_id)
    assert waiting is not None
    assert waiting.state == "running"
    assert waiting.stage == "claimed"
    assert waiting.error_code is None

    monkeypatch.setenv("EVEROS_MEMORIZE__SESSION_LOCK_TIMEOUT_SECONDS", "5")
    load_settings.cache_clear()
    completed = await memorize(payload, is_final=True)
    assert completed.operation_id == operation_id
    assert completed.status == "extracted"
    assert len(_memcell_rows(tmp_path)) == 1


# ---------------------------------------------------------------------------
# Deferred stage + durable background flush
# ---------------------------------------------------------------------------


async def test_stage_is_llm_free_idempotent_and_content_free(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())
    svc = importlib.import_module("everos.service.memorize")
    monkeypatch.setattr(
        svc,
        "get_llm_client",
        lambda: (_ for _ in ()).throw(AssertionError("stage called the LLM")),
    )
    operation_id = "evop1-stage-" + "1" * 64
    payload = {
        "operation_id": operation_id,
        "session_id": "stage_no_llm",
        "messages": [
            _user("upload first", 1_700_000_000_000),
            _assistant("stored", 1_700_000_001_000),
        ],
    }

    first = await stage(payload)
    replay = await stage(payload)

    assert isinstance(first, StageResult)
    assert first.status == "staged"
    assert first.message_count == 2
    assert first.replayed is False
    assert replay.replayed is True
    assert _buffer_count(tmp_path) == 2
    receipt = await get_memory_operation_status(operation_id)
    assert receipt is not None
    assert receipt.kind == "stage"
    assert receipt.state == "completed"
    assert receipt.stage == "messages_staged"
    assert receipt.message_count == 2
    assert receipt.memcell_count == 0


async def test_stage_conflict_and_overlap_rechunk_dedupe(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
) -> None:
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())
    first_message = _user("one", 1_700_000_000_000)
    overlap = _assistant("two", 1_700_000_001_000)
    last_message = _user("three", 1_700_000_002_000)
    first = {
        "operation_id": "evop1-stage-" + "2" * 64,
        "session_id": "stage_overlap",
        "messages": [first_message, overlap],
    }
    second = {
        "operation_id": "evop1-stage-" + "3" * 64,
        "session_id": "stage_overlap",
        # The overlapping message moved from batch index 1 to index 0.
        "messages": [overlap, last_message],
    }
    await stage(first)
    await stage(second)
    assert _buffer_count(tmp_path) == 3

    changed = {
        **first,
        "messages": [_user("changed", 1_700_000_000_000)],
    }
    with pytest.raises(MemoryOperationConflictError, match="different request"):
        await stage(changed)


async def test_stage_then_legacy_add_dedupes_transport_aliases(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
) -> None:
    await memorize_env(
        mode="chat",
        fake_llm=_make_fake_llm(boundary_responses=[[]]),
    )
    messages = [
        _user("same row through two routes", 1_700_000_000_000),
        _assistant("must remain one copy", 1_700_000_001_000),
    ]
    await stage(
        {
            "operation_id": "evop1-stage-" + "6" * 64,
            "session_id": "stage_add_alias",
            "messages": messages,
        }
    )

    result = await memorize(
        {
            "session_id": "stage_add_alias",
            "messages": messages,
        }
    )

    assert result.status == "accumulated"
    assert _buffer_count(tmp_path) == 2


async def test_pending_publish_blocks_flush_then_consumes_and_late_replay_is_noop(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
) -> None:
    await memorize_env(
        mode="chat",
        fake_llm=_make_fake_llm(boundary_responses=[[]]),
    )
    sid = "publish_gate"
    messages = [
        _staged(_user("durable user fact", 1_700_000_000_000), "msg-user"),
        _staged(_assistant("ack", 1_700_000_001_000), "msg-assistant"),
    ]

    staged = await stage(
        {
            "operation_id": "evop1-stage-" + "a" * 64,
            "session_id": sid,
            "messages": messages,
        }
    )
    assert staged.inserted_count == 2
    assert staged.duplicate_count == 0
    assert _buffer_count(tmp_path) == 2

    blocked = await memorize({"session_id": sid, "messages": []}, is_final=True)
    assert blocked.status == "accumulated"
    assert _buffer_count(tmp_path) == 2
    assert _memcell_rows(tmp_path) == []
    assert {row["authority_state"] for row in _receipt_rows(tmp_path)} == {
        "pending_publish"
    }

    publish_payload = {
        "operation_id": "evop1-publish-" + "a" * 64,
        "session_id": sid,
        "authority_ref": "ledger:commit-publish-gate",
        "messages": [
            {"source": "web", "external_ref": "msg-user", "revision": 0},
            {
                "source": "web",
                "external_ref": "msg-assistant",
                "revision": 0,
            },
        ],
    }
    published = await publish(publish_payload)
    assert published.published_count == 2
    assert published.already_published_count == 0
    publish_replay = await publish(publish_payload)
    assert publish_replay.replayed is True
    publish_receipt = await get_memory_operation_status(publish_payload["operation_id"])
    assert publish_receipt is not None
    assert publish_receipt.kind == "publish"
    assert publish_receipt.stage == "messages_published"
    assert publish_receipt.state == "completed"

    flushed = await memorize({"session_id": sid, "messages": []}, is_final=True)
    assert flushed.status == "extracted"
    assert _buffer_count(tmp_path) == 0
    assert len(_memcell_rows(tmp_path)) == 1
    assert {row["authority_state"] for row in _receipt_rows(tmp_path)} == {"consumed"}

    late = await stage(
        {
            "operation_id": "evop1-stage-" + "b" * 64,
            "session_id": sid,
            "messages": messages,
        }
    )
    assert late.consumed_replay_count == 2
    assert late.inserted_count == 0
    assert _buffer_count(tmp_path) == 0
    again = await memorize({"session_id": sid, "messages": []}, is_final=True)
    assert again.status == "accumulated"
    assert len(_memcell_rows(tmp_path)) == 1


async def test_external_identity_rechunk_dedupes_across_stage_operations(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
) -> None:
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())
    sid = "external_rechunk"
    one = _staged(_user("one", 1_700_000_000_000), "ext-one")
    overlap = _staged(_assistant("two", 1_700_000_001_000), "ext-two")
    three = _staged(_user("three", 1_700_000_002_000), "ext-three")

    first = await stage(
        {
            "operation_id": "evop1-stage-" + "c" * 64,
            "session_id": sid,
            "messages": [one, overlap],
        }
    )
    second = await stage(
        {
            "operation_id": "evop1-stage-" + "d" * 64,
            "session_id": sid,
            "messages": [overlap, three],
        }
    )

    assert first.inserted_count == 2
    assert second.inserted_count == 1
    assert second.duplicate_count == 1
    assert _buffer_count(tmp_path) == 3
    assert len(_receipt_rows(tmp_path)) == 3


async def test_concurrent_stage_operations_share_one_message_receipt(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
) -> None:
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())
    sid = "concurrent_message_receipt"
    message = _staged(
        _user("one logical upload", 1_700_000_000_000),
        "concurrent-ext-1",
    )

    first, second = await asyncio.gather(
        stage(
            {
                "operation_id": "evop1-stage-" + "7" * 64,
                "session_id": sid,
                "messages": [message],
            }
        ),
        stage(
            {
                "operation_id": "evop1-stage-" + "6" * 64,
                "session_id": sid,
                "messages": [message],
            }
        ),
    )

    assert sorted(
        [
            (first.inserted_count, first.duplicate_count),
            (second.inserted_count, second.duplicate_count),
        ]
    ) == [(0, 1), (1, 0)]
    assert _buffer_count(tmp_path) == 1
    assert len(_receipt_rows(tmp_path)) == 1


async def test_unpublished_stage_rows_are_hidden_from_search_context(
    memorize_env: Callable[..., Any],
) -> None:
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())
    sid = "search_authority_gate"
    await stage(
        {
            "operation_id": "evop1-stage-" + "9" * 64,
            "session_id": sid,
            "messages": [
                _staged(
                    _user("not searchable before publish", 1_700_000_000_000),
                    "search-hidden",
                )
            ],
        }
    )

    manager = object.__new__(SearchManager)
    req = SimpleNamespace(
        filters=FilterNode.model_validate({"session_id": sid}),
        app_id="default",
        project_id="default",
    )
    assert await manager._load_unprocessed(req) == []

    await publish(
        {
            "operation_id": "evop1-publish-" + "9" * 64,
            "session_id": sid,
            "authority_ref": "ledger:search-visible",
            "messages": [
                {"source": "web", "external_ref": "search-hidden", "revision": 0}
            ],
        }
    )
    visible = await manager._load_unprocessed(req)
    assert len(visible) == 1
    assert visible[0].content == "not searchable before publish"


async def test_pre_stage3_ms_row_without_receipt_fails_closed(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
) -> None:
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())
    sid = "legacy_staged_without_receipt"
    await stage(
        {
            "operation_id": "evop1-stage-" + "8" * 64,
            "session_id": sid,
            "messages": [_user("old staged payload", 1_700_000_000_000)],
        }
    )
    db = tmp_path / ".index" / "sqlite" / "system.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("DELETE FROM memory_message_receipt")
        conn.commit()
    finally:
        conn.close()

    blocked = await memorize({"session_id": sid, "messages": []}, is_final=True)
    assert blocked.status == "accumulated"
    assert _buffer_count(tmp_path) == 1
    assert _memcell_rows(tmp_path) == []

    manager = object.__new__(SearchManager)
    req = SimpleNamespace(
        filters=FilterNode.model_validate({"session_id": sid}),
        app_id="default",
        project_id="default",
    )
    assert await manager._load_unprocessed(req) == []


async def test_revision_supersede_resets_authority_and_conflicts_fail_closed(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
) -> None:
    await memorize_env(
        mode="chat",
        fake_llm=_make_fake_llm(boundary_responses=[[]]),
    )
    sid = "revision_gate"
    rev0 = _staged(_user("version zero", 1_700_000_000_000), "editable", revision=0)
    rev1 = _staged(_user("version one", 1_700_000_001_000), "editable", revision=1)

    await stage(
        {
            "operation_id": "evop1-stage-" + "e" * 64,
            "session_id": sid,
            "messages": [rev0],
        }
    )
    await publish(
        {
            "operation_id": "evop1-publish-" + "e" * 64,
            "session_id": sid,
            "authority_ref": "ledger:commit-rev0",
            "messages": [{"source": "web", "external_ref": "editable", "revision": 0}],
        }
    )
    assert _receipt_rows(tmp_path)[0]["authority_state"] == "published"

    superseded = await stage(
        {
            "operation_id": "evop1-stage-" + "f" * 64,
            "session_id": sid,
            "messages": [rev1],
        }
    )
    assert superseded.updated_count == 1
    receipt = _receipt_rows(tmp_path)[0]
    assert receipt["revision"] == 1
    assert receipt["authority_state"] == "pending_publish"
    assert receipt["authority_ref"] is None
    assert _buffer_texts(tmp_path) == ["version one"]

    blocked = await memorize({"session_id": sid, "messages": []}, is_final=True)
    assert blocked.status == "accumulated"
    assert len(_memcell_rows(tmp_path)) == 0

    stale = await stage(
        {
            "operation_id": "evop1-stage-" + "1" * 63 + "0",
            "session_id": sid,
            "messages": [rev0],
        }
    )
    assert stale.stale_count == 1

    conflicting_rev1 = _staged(
        _user("different payload same revision", 1_700_000_001_000),
        "editable",
        revision=1,
    )
    with pytest.raises(MemoryMessageConflictError, match="different payload"):
        await stage(
            {
                "operation_id": "evop1-stage-" + "2" * 64,
                "session_id": sid,
                "messages": [conflicting_rev1],
            }
        )

    with pytest.raises(MemoryMessageConflictError, match="exact currently staged"):
        await publish(
            {
                "operation_id": "evop1-publish-" + "f" * 64,
                "session_id": sid,
                "authority_ref": "ledger:wrong-revision",
                "messages": [
                    {"source": "web", "external_ref": "editable", "revision": 0}
                ],
            }
        )

    final_publish = await publish(
        {
            "operation_id": "evop1-publish-" + "1" * 63 + "0",
            "session_id": sid,
            "authority_ref": "ledger:commit-rev1",
            "messages": [{"source": "web", "external_ref": "editable", "revision": 1}],
        }
    )
    assert final_publish.published_count == 1
    flushed = await memorize({"session_id": sid, "messages": []}, is_final=True)
    assert flushed.status == "extracted"
    assert len(_memcell_rows(tmp_path)) == 1
    receipt = _receipt_rows(tmp_path)[0]
    assert receipt["revision"] == 1
    assert receipt["authority_state"] == "consumed"


async def test_background_flush_is_durable_and_recovered_on_scheduler_start(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
) -> None:
    await memorize_env(
        mode="chat",
        fake_llm=_make_fake_llm(boundary_responses=[[]]),
    )
    staged_messages = [
        _staged(
            _user("remember after restart", 1_700_000_000_000),
            "background-user",
        ),
        _staged(
            _assistant("ack", 1_700_000_001_000),
            "background-assistant",
        ),
    ]
    await stage(
        {
            "operation_id": "evop1-stage-" + "4" * 64,
            "session_id": "background_recovery",
            "messages": staged_messages,
        }
    )
    published = await publish(
        {
            "operation_id": "evop1-publish-" + "4" * 64,
            "session_id": "background_recovery",
            "authority_ref": "ledger:commit-background-1",
            "messages": [
                {"source": "web", "external_ref": "background-user", "revision": 0},
                {
                    "source": "web",
                    "external_ref": "background-assistant",
                    "revision": 0,
                },
            ],
        }
    )
    assert isinstance(published, PublishResult)
    assert published.published_count == 2
    flush_id = "evop1-flush-" + "4" * 64
    queued = await queue_background_flush(
        {
            "operation_id": flush_id,
            "session_id": "background_recovery",
            "app_id": "default",
            "project_id": "default",
        }
    )
    assert isinstance(queued, BackgroundFlushResult)
    assert queued.status == "processing"
    assert queued.replayed is False
    pending = await get_memory_operation_status(flush_id)
    assert pending is not None
    assert pending.state == "running"
    assert pending.stage == "queued"

    # Model a process that died after claiming work but before completion.
    # A fresh scheduler must recover both queued and processing receipts.
    from everos.infra.persistence.sqlite import memory_operation_repo

    await memory_operation_repo.mark_processing(flush_id)
    interrupted = await get_memory_operation_status(flush_id)
    assert interrupted is not None
    assert interrupted.stage == "processing"

    # Constructing a scheduler here models a fresh service process. Repeated
    # wake-ups still feed one bounded worker and one durable operation.
    from everos.service.background_flush import BackgroundFlushScheduler

    scheduler = BackgroundFlushScheduler()
    await scheduler.start()
    for _ in range(100):
        scheduler.wake()
        completed = await get_memory_operation_status(flush_id)
        if completed is not None and completed.state == "completed":
            break
        await asyncio.sleep(0.01)
    await scheduler.stop()

    completed = await get_memory_operation_status(flush_id)
    assert completed is not None
    assert completed.state == "completed"
    assert completed.stage == "sync_dispatch_completed"
    assert len(_memcell_rows(tmp_path)) == 1
    assert len(_episode_paths(tmp_path)) == 1
    assert _buffer_count(tmp_path) == 0


async def test_background_flush_duplicate_claim_replays_without_new_job(
    memorize_env: Callable[..., Any],
) -> None:
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())
    payload = {
        "operation_id": "evop1-flush-" + "5" * 64,
        "session_id": "background_duplicate",
        "app_id": "default",
        "project_id": "default",
    }
    first = await queue_background_flush(payload)
    replay = await queue_background_flush(payload)
    assert first.replayed is False
    assert replay.replayed is True
    receipt = await get_memory_operation_status(payload["operation_id"])
    assert receipt is not None
    assert receipt.stage == "queued"


async def test_failed_retryable_flush_does_not_starve_newer_work(
    memorize_env: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())
    from everos.infra.persistence.sqlite import memory_operation_repo
    from everos.service import background_flush as background_flush_module
    from everos.service.background_flush import BackgroundFlushScheduler

    old_id = "evop1-flush-" + "7" * 64
    new_id = "evop1-flush-" + "8" * 64
    for operation_id, session_id in (
        (old_id, "old_retryable"),
        (new_id, "new_queued"),
    ):
        await queue_background_flush(
            {
                "operation_id": operation_id,
                "session_id": session_id,
                "app_id": "default",
                "project_id": "default",
            }
        )
    await memory_operation_repo.mark_failed(
        old_id,
        error_code="TemporaryProviderError",
        retryable=True,
    )

    calls: list[str] = []

    async def fake_run(operation_id: str) -> MemorizeResult:
        calls.append(operation_id)
        if operation_id == old_id:
            raise RuntimeError("still unavailable")
        await memory_operation_repo.mark_completed(
            operation_id,
            {"message_count": 0, "status": "accumulated"},
        )
        return MemorizeResult(message_count=0, status="accumulated")

    monkeypatch.setattr(
        background_flush_module,
        "run_background_flush_operation",
        fake_run,
    )
    scheduler = BackgroundFlushScheduler()
    await scheduler.start()
    for _ in range(100):
        current = await get_memory_operation_status(new_id)
        if current is not None and current.state == "completed":
            break
        await asyncio.sleep(0.01)
    await scheduler.stop()

    old_receipt = await get_memory_operation_status(old_id)
    new_receipt = await get_memory_operation_status(new_id)
    assert calls == [old_id, new_id]
    assert old_receipt is not None and old_receipt.state == "failed"
    assert new_receipt is not None and new_receipt.state == "completed"
