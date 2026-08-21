"""Unit & regression tests for extract_user_profile strategy.

Uses dynamic lossless batching and FakeLLM prompt interception.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.types import ChatMessage, MemCell
from everalgo.types import Profile as AlgoProfile

from everos.core.persistence import MemoryRoot
from everos.infra.ome.testing import FakeStrategyContext
from everos.infra.persistence.markdown import (
    ProfileReader,
    UserProfileFrontmatter,
)
from everos.memory.events import ProfileClusterUpdated
from everos.memory.strategies._partition_locks import _reset_for_tests
from everos.memory.strategies.extract_user_profile import (
    DEFAULT_MAX_PROMPT_CHARS,
    _persist_profile,
    extract_user_profile,
    split_memcell_losslessly,
)


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


class FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    """Fake LLM client that intercepts chat messages and returns valid JSON."""

    def __init__(self, response_factory=None) -> None:
        self.captured_prompts: list[str] = []
        self.captured_messages: list[list[Any]] = []
        self.response_factory = response_factory

    async def chat(self, messages: list[Any], **kwargs: Any) -> FakeLLMResponse:
        self.captured_messages.append(messages)
        content = str(messages[0].content)
        self.captured_prompts.append(content)

        if self.response_factory:
            res = self.response_factory(content, len(self.captured_prompts))
            if isinstance(res, str):
                return FakeLLMResponse(res)
            return res

        if (
            "Compaction strategies" in content
            or "compact the profile" in content
            or "compact_note" in content
        ):
            resp = {
                "explicit_info": [
                    {
                        "category": "preference",
                        "description": "user likes python",
                    }
                ],
                "implicit_traits": [{"trait": "analytical"}],
            }
        elif "operations" in content or "=== explicit_info ===" in content:
            resp = {
                "operations": [
                    {
                        "action": "add",
                        "type": "explicit_info",
                        "data": {
                            "category": "preference",
                            "description": "user likes python",
                        },
                    }
                ]
            }
        else:
            resp = {
                "explicit_info": [
                    {
                        "category": "preference",
                        "description": "user likes python",
                    }
                ],
                "implicit_traits": [{"trait": "analytical"}],
            }
        return FakeLLMResponse(json.dumps(resp))


def _event(
    *,
    owner_id: str = "u_alice",
    memcell_id: str = "mc_aaaaaaaaaaa1",
    cluster_id: str = "cl_user00000001",
    app_id: str = "codex",
    project_id: str = "solo",
) -> ProfileClusterUpdated:
    return ProfileClusterUpdated(
        owner_id=owner_id,
        memcell_id=memcell_id,
        cluster_id=cluster_id,
        app_id=app_id,
        project_id=project_id,
    )


def _algo_cluster(
    cluster_id: str = "cl_user00000001",
    members: list[str] | None = None,
    last_ts: int = 1000,
) -> AlgoCluster:
    return AlgoCluster(
        id=cluster_id,
        members=members or ["mc_aaaaaaaaaaa1"],
        count=len(members or ["mc_aaaaaaaaaaa1"]),
        last_ts=last_ts,
        centroid=np.zeros(128),
    )


def _memcell_row(
    memcell_id: str,
    sender_id: str = "u_alice",
    ts_ms: int = 1000,
    content: str = "hello world",
    app_id: str = "codex",
    project_id: str = "solo",
) -> Any:
    mc = MemCell(
        items=[
            ChatMessage(
                id=f"{memcell_id}_m1",
                role="user",
                content=content,
                timestamp=ts_ms,
                sender_id=sender_id,
            ),
        ],
        timestamp=ts_ms,
    )

    class FakeRow:
        def __init__(self) -> None:
            self.memcell_id = memcell_id
            self.app_id = app_id
            self.project_id = project_id
            self.payload_json = mc.model_dump_json()

    return FakeRow()


# ── Section 1: Real ProfileWriter Integration (P0-1) ─────────────────────


@pytest.mark.asyncio
async def test_real_persist_profile_integration_with_real_profile_writer(
    tmp_path: Path,
) -> None:
    """P0-1: Real _persist_profile() succeeds without illegal schema keyword."""
    mr = MemoryRoot(root=tmp_path)
    profile = AlgoProfile(
        owner_id="user",
        summary="Automated verification specialist.",
        timestamp=2000,
        explicit_info=[{"category": "Role", "description": "Backend engineer"}],
        implicit_traits=[{"trait": "Safety", "description": "Demands strict testing"}],
    )
    profile.profile_watermark_memcell_ids = ["mc_01", "mc_02"]

    with patch(
        "everos.memory.strategies.extract_user_profile.MemoryRoot.default",
        return_value=mr,
    ):
        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        mod._writer = None  # Reset singleton to pick up patched MemoryRoot

        await _persist_profile(
            profile,
            owner_id="user",
            app_id="codex",
            project_id="solo",
        )

    # 1. Verify exact target file path on disk
    target_file = tmp_path / "codex" / "solo" / "users" / "user" / "user.md"
    assert target_file.exists(), f"Expected profile file at {target_file}"

    # 2. Re-read using real ProfileReader
    reader = ProfileReader(root=mr)
    read_res = await reader.read(
        "user",
        schema=UserProfileFrontmatter,
        app_id="codex",
        project_id="solo",
    )
    assert read_res is not None
    fm, body = read_res

    assert fm.user_id == "user"
    assert fm.app_id == "codex"
    assert fm.project_id == "solo"
    assert fm.profile_timestamp_ms == 2000
    assert fm.profile_watermark_memcell_ids == ["mc_01", "mc_02"]
    assert fm.summary == "Automated verification specialist."
    assert body.strip() == "Automated verification specialist."
    assert fm.explicit_info == [{"category": "Role", "description": "Backend engineer"}]
    assert fm.implicit_traits == [
        {"trait": "Safety", "description": "Demands strict testing"}
    ]


# ── Section 2: Compaction Hard Budget Guard ──────────────────────────────


@pytest.mark.asyncio
async def test_p0_compaction_prompt_budget_exceeded_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _compact() generates >40k prompt, BoundedProfileLLMClient fails closed."""
    # Profile with 25 explicit + 22 implicit = 47 items (> 45 threshold)
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="Initial summary",
        explicit_info=[
            {"category": f"Cat{i}", "description": "fact " * 30} for i in range(25)
        ],
        implicit_traits=[
            {"trait": f"Trait{i}", "description": "trait " * 30} for i in range(22)
        ],
        profile_timestamp_ms=1000,
    )

    mc = MemCell(
        items=[
            ChatMessage(
                id="m_new",
                role="user",
                content="Triggering update with new facts",
                timestamp=1100,
                sender_id="u_alice",
            )
        ],
        timestamp=1100,
    )

    class FakeRow:
        def __init__(self) -> None:
            self.memcell_id = "mc_new"
            self.app_id = "codex"
            self.project_id = "solo"
            self.payload_json = mc.model_dump_json()

    cluster = _algo_cluster(cluster_id="cl_new", members=["mc_new"], last_ts=1100)

    # Factory returning 15 ops -> total items becomes 62 (>45), triggering compact
    # And the merged items are so large that compact prompt exceeds 40k chars!
    def response_factory(prompt: str, call_num: int) -> str:
        return json.dumps(
            {
                "operations": [
                    {
                        "action": "add",
                        "type": "explicit_info",
                        "data": {
                            "category": f"HugeCat_{i}",
                            "description": "giant_trait_content_" * 150,
                        },
                    }
                    for i in range(15)
                ]
            }
        )

    fake_llm = FakeLLM(response_factory=response_factory)

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[FakeRow()])
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock()

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        with pytest.raises(
            ValueError, match="Profile LLM prompt length.*exceeds hard budget"
        ):
            await extract_user_profile(
                _event(cluster_id="cl_new"), FakeStrategyContext()
            )

    # Provider delegate NEVER received >40k prompt!
    for p in fake_llm.captured_prompts:
        assert len(p) <= DEFAULT_MAX_PROMPT_CHARS
    # Watermark write was NEVER called
    mock_writer_cls.return_value.write.assert_not_called()


# ── Section 3: Late-Arriving Same-Timestamp MemCell & Cursor (P0-2) ───────


@pytest.mark.asyncio
async def test_p0_late_same_timestamp_after_watermark_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P0-2: Late cell B (ts=2000) arriving after watermark 2000 commit is merged."""
    # Initial state on disk: Profile at watermark 2000 with cursor ['mc_a']
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="Summary after A",
        profile_timestamp_ms=2000,
        profile_watermark_memcell_ids=["mc_a"],
    )

    # Late cluster with member B (ts=2000)
    cluster_b = _algo_cluster(cluster_id="cl_b_late", members=["mc_b"], last_ts=2000)
    row_b = _memcell_row(
        "mc_b", sender_id="u_alice", ts_ms=2000, content="FACT_FROM_LATE_B"
    )

    fake_llm = FakeLLM()
    persisted_frontmatters: list[UserProfileFrontmatter] = []

    async def mock_write(owner_id, *, frontmatter, body, **kwargs):
        persisted_frontmatters.append(frontmatter)

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster_b])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[row_b])
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock(side_effect=mock_write)

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(
            _event(cluster_id="cl_b_late", memcell_id="mc_b"),
            FakeStrategyContext(),
        )

    # 1. FakeLLM executed with B's content
    assert len(fake_llm.captured_prompts) == 1
    assert "FACT_FROM_LATE_B" in fake_llm.captured_prompts[0]

    # 2. Profile persisted again with same timestamp 2000 and unioned cursor
    assert len(persisted_frontmatters) == 1
    final_fm = persisted_frontmatters[0]
    assert final_fm.profile_timestamp_ms == 2000
    assert final_fm.profile_watermark_memcell_ids == ["mc_a", "mc_b"]


@pytest.mark.asyncio
async def test_p0_late_same_timestamp_failure_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P0-2: Late cell B fails LLM -> cursor stays ['mc_a']; retry re-processes B."""
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="Summary after A",
        profile_timestamp_ms=2000,
        profile_watermark_memcell_ids=["mc_a"],
    )

    cluster_b = _algo_cluster(cluster_id="cl_b_late", members=["mc_b"], last_ts=2000)
    row_b = _memcell_row(
        "mc_b", sender_id="u_alice", ts_ms=2000, content="FACT_FROM_LATE_B"
    )

    # Run 1: LLM fails
    fake_llm_fail = FakeLLM(
        response_factory=lambda p, c: (_ for _ in ()).throw(
            RuntimeError("Provider 500")
        )
    )
    persisted_frontmatters: list[UserProfileFrontmatter] = []

    async def mock_write(owner_id, *, frontmatter, body, **kwargs):
        persisted_frontmatters.append(frontmatter)

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm_fail,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster_b])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[row_b])
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock(side_effect=mock_write)

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        with pytest.raises(RuntimeError, match="Provider 500"):
            await extract_user_profile(
                _event(cluster_id="cl_b_late", memcell_id="mc_b"),
                FakeStrategyContext(),
            )

    # Zero writes on disk
    assert len(persisted_frontmatters) == 0

    # Run 2: Full Retry succeeds
    fake_llm_success = FakeLLM()
    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm_success,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster_b])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[row_b])
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock(side_effect=mock_write)

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(
            _event(cluster_id="cl_b_late", memcell_id="mc_b"),
            FakeStrategyContext(),
        )

    assert len(persisted_frontmatters) == 1
    assert persisted_frontmatters[0].profile_watermark_memcell_ids == [
        "mc_a",
        "mc_b",
    ]


@pytest.mark.asyncio
async def test_p0_cross_timestamp_partial_next_group_not_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P0: Mixed batch with partial next group does NOT commit watermark."""
    # Durable profile at T0=1000 with cursor=['mc_t0']
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="Initial summary",
        profile_timestamp_ms=1000,
        profile_watermark_memcell_ids=["mc_t0"],
    )

    # Group 1500: mc_old (small cell)
    # Group 2000: mc_a (~20k chars), mc_b (~20k chars)
    row_old = _memcell_row(
        "mc_old", sender_id="u_alice", ts_ms=1500, content="OLD_FACT_1500"
    )
    row_a = _memcell_row(
        "mc_a", sender_id="u_alice", ts_ms=2000, content="A_CONTENT_" * 2500
    )
    row_b = _memcell_row(
        "mc_b", sender_id="u_alice", ts_ms=2000, content="B_CONTENT_" * 2500
    )

    cluster_1500 = _algo_cluster(cluster_id="cl_1500", members=["mc_old"], last_ts=1500)
    cluster_2000 = _algo_cluster(
        cluster_id="cl_2000", members=["mc_a", "mc_b"], last_ts=2000
    )

    # Run 1: Batch 1 has [OLD, A], Batch 2 has [B]. FakeLLM fails on batch 2!
    def response_factory_run1(prompt: str, call_num: int) -> str:
        if call_num == 2:
            raise RuntimeError("LLM network failure on batch 2")
        return json.dumps(
            {
                "operations": [
                    {
                        "action": "add",
                        "type": "explicit_info",
                        "data": {
                            "category": "Role",
                            "description": "Engineer",
                        },
                    }
                ]
            }
        )

    fake_llm_run1 = FakeLLM(response_factory=response_factory_run1)
    persisted_fms_run1: list[UserProfileFrontmatter] = []

    async def mock_write_run1(owner_id, *, frontmatter, body, **kwargs):
        persisted_fms_run1.append(frontmatter)

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm_run1,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(
            return_value=[cluster_1500, cluster_2000]
        )
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[row_old, row_a, row_b])
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock(side_effect=mock_write_run1)

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        with pytest.raises(RuntimeError, match="LLM network failure on batch 2"):
            await extract_user_profile(
                _event(cluster_id="cl_2000", memcell_id="mc_b"),
                FakeStrategyContext(),
            )

    # In Run 1, Batch 1 processed [OLD, A] (last item A is not group final)
    # Batch 2 failed on B.
    # Therefore, ProfileWriter.write MUST NOT have been called!
    assert len(persisted_fms_run1) == 0

    # Run 2: Full Retry succeeds
    fake_llm_run2 = FakeLLM()
    persisted_fms_run2: list[UserProfileFrontmatter] = []

    async def mock_write_run2(owner_id, *, frontmatter, body, **kwargs):
        persisted_fms_run2.append(frontmatter)

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm_run2,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(
            return_value=[cluster_1500, cluster_2000]
        )
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[row_old, row_a, row_b])
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock(side_effect=mock_write_run2)

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(
            _event(cluster_id="cl_2000", memcell_id="mc_b"),
            FakeStrategyContext(),
        )

    # In Run 2:
    # Batch 1 processed [OLD, A], Batch 2 processed [B].
    # Exactly 1 durable write occurred when group 2000 completed!
    assert len(persisted_fms_run2) == 1
    final_fm = persisted_fms_run2[0]
    assert final_fm.profile_timestamp_ms == 2000
    assert final_fm.profile_watermark_memcell_ids == ["mc_a", "mc_b"]
    all_prompts = "".join(fake_llm_run2.captured_prompts)
    assert "OLD_FACT_1500" in all_prompts
    assert "A_CONTENT_" in all_prompts
    assert "B_CONTENT_" in all_prompts


@pytest.mark.asyncio
async def test_p0_stale_same_timestamp_already_processed_fast_no_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P0-2: Event for processed cursor member fast no-ops without memcell query."""
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="Summary",
        profile_timestamp_ms=2000,
        profile_watermark_memcell_ids=["mc_a", "mc_b"],
    )
    cluster = _algo_cluster(
        cluster_id="cl_same", members=["mc_a", "mc_b"], last_ts=2000
    )

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_memcell_repo.find_by_ids = AsyncMock()
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        # Trigger event for mc_a (already processed)
        await extract_user_profile(
            _event(cluster_id="cl_same", memcell_id="mc_a"),
            FakeStrategyContext(),
        )

        # memcell_repo MUST NOT be called!
        mock_memcell_repo.find_by_ids.assert_not_called()


@pytest.mark.asyncio
async def test_p0_newer_timestamp_resets_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P0-2: Moving from T=2000 to T=3000 resets cursor to only T=3000 members."""
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="Summary",
        profile_timestamp_ms=2000,
        profile_watermark_memcell_ids=["mc_a", "mc_b"],
    )
    cluster = _algo_cluster(
        cluster_id="cl_3000", members=["mc_c", "mc_d"], last_ts=3000
    )
    rows = [
        _memcell_row("mc_c", sender_id="u_alice", ts_ms=3000, content="FACT_C"),
        _memcell_row("mc_d", sender_id="u_alice", ts_ms=3000, content="FACT_D"),
    ]

    fake_llm = FakeLLM()
    persisted_frontmatters: list[UserProfileFrontmatter] = []

    async def mock_write(owner_id, *, frontmatter, body, **kwargs):
        persisted_frontmatters.append(frontmatter)

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=rows)
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock(side_effect=mock_write)

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(
            _event(cluster_id="cl_3000", memcell_id="mc_c"),
            FakeStrategyContext(),
        )

    assert len(persisted_frontmatters) == 1
    fm = persisted_frontmatters[0]
    assert fm.profile_timestamp_ms == 3000
    # Cursor contains ONLY the 3000 group, mc_a/mc_b reset!
    assert fm.profile_watermark_memcell_ids == ["mc_c", "mc_d"]


@pytest.mark.asyncio
async def test_p0_legacy_frontmatter_without_cursor_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P0-2: Legacy profile without cursor does not replay historical <= watermark."""
    legacy_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="Legacy summary",
        profile_timestamp_ms=2000,
        # profile_watermark_memcell_ids is default []
    )

    cluster_old = _algo_cluster(cluster_id="cl_old", members=["mc_old"], last_ts=2000)
    cluster_fresh = _algo_cluster(
        cluster_id="cl_fresh", members=["mc_fresh"], last_ts=2500
    )

    rows = [
        _memcell_row("mc_old", sender_id="u_alice", ts_ms=2000, content="OLD_FACT"),
        _memcell_row("mc_fresh", sender_id="u_alice", ts_ms=2500, content="FRESH_FACT"),
    ]

    fake_llm = FakeLLM()
    persisted_frontmatters: list[UserProfileFrontmatter] = []

    async def mock_write(owner_id, *, frontmatter, body, **kwargs):
        persisted_frontmatters.append(frontmatter)

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(
            return_value=[cluster_old, cluster_fresh]
        )
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=rows)
        mock_reader_cls.return_value.read = AsyncMock(return_value=[legacy_fm])
        mock_writer_cls.return_value.write = AsyncMock(side_effect=mock_write)

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(
            _event(cluster_id="cl_fresh", memcell_id="mc_fresh"),
            FakeStrategyContext(),
        )

    assert len(fake_llm.captured_prompts) == 1
    # OLD_FACT (ts=2000 <= legacy watermark) was NOT replayed!
    assert "OLD_FACT" not in fake_llm.captured_prompts[0]
    assert "FRESH_FACT" in fake_llm.captured_prompts[0]
    assert persisted_frontmatters[0].profile_timestamp_ms == 2500
    assert persisted_frontmatters[0].profile_watermark_memcell_ids == ["mc_fresh"]


# ── Section 4: Cross-Cluster A/B Global Fresh Processing ──────────────────


@pytest.mark.asyncio
async def test_p0_cross_cluster_a_b_no_skip_and_no_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When event A arrives, fresh cluster B is NOT skipped."""
    cluster_a = _algo_cluster(
        cluster_id="cl_a", members=["mc_a1", "mc_a2"], last_ts=2000
    )
    cluster_b = _algo_cluster(cluster_id="cl_b", members=["mc_b1"], last_ts=1500)

    rows = [
        _memcell_row("mc_b1", sender_id="u_alice", ts_ms=1500, content="FACT_B"),
        _memcell_row("mc_a1", sender_id="u_alice", ts_ms=1800, content="FACT_A1"),
        _memcell_row("mc_a2", sender_id="u_alice", ts_ms=2000, content="FACT_A2"),
    ]

    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="Prior summary",
        profile_timestamp_ms=1000,
        profile_watermark_memcell_ids=["mc_init"],
    )
    fake_llm = FakeLLM()
    persisted_frontmatters: list[UserProfileFrontmatter] = []

    async def mock_write(owner_id, *, frontmatter, body, **kwargs):
        persisted_frontmatters.append(frontmatter)

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(
            return_value=[cluster_a, cluster_b]
        )
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=rows)
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock(side_effect=mock_write)

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        # Trigger Event A (cl_a)
        await extract_user_profile(_event(cluster_id="cl_a"), FakeStrategyContext())

        all_prompts = "".join(fake_llm.captured_prompts)
        assert "FACT_B" in all_prompts
        assert "FACT_A1" in all_prompts
        assert "FACT_A2" in all_prompts
        assert persisted_frontmatters[-1].profile_timestamp_ms == 2000

        # Stale event B -> should fast no-op
        mock_memcell_repo.find_by_ids.reset_mock()
        mock_reader_cls.return_value.read = AsyncMock(
            return_value=[persisted_frontmatters[-1]]
        )

        await extract_user_profile(
            _event(cluster_id="cl_b", memcell_id="mc_b1"), FakeStrategyContext()
        )
        mock_memcell_repo.find_by_ids.assert_not_called()


# ── Section 5: Same-Timestamp Group Atomic Commit ─────────────────────────


@pytest.mark.asyncio
async def test_p0_same_timestamp_multi_batch_atomic_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same timestamp group: watermark commits ONLY on group end."""
    mc1 = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="CONTENT_ONE_" * 2100,
                timestamp=2000,
                sender_id="u_alice",
            )
        ],
        timestamp=2000,
    )
    mc2 = MemCell(
        items=[
            ChatMessage(
                id="m2",
                role="user",
                content="CONTENT_TWO_" * 2100,
                timestamp=2000,
                sender_id="u_alice",
            )
        ],
        timestamp=2000,
    )

    class FakeRow1:
        def __init__(self) -> None:
            self.memcell_id = "mc1"
            self.app_id = "codex"
            self.project_id = "solo"
            self.payload_json = mc1.model_dump_json()

    class FakeRow2:
        def __init__(self) -> None:
            self.memcell_id = "mc2"
            self.app_id = "codex"
            self.project_id = "solo"
            self.payload_json = mc2.model_dump_json()

    cluster = _algo_cluster(
        cluster_id="cl_same_ts", members=["mc1", "mc2"], last_ts=2000
    )
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="Prior summary",
        profile_timestamp_ms=1000,
    )
    fake_llm = FakeLLM()
    written_watermarks: list[int] = []

    async def mock_write(owner_id, *, frontmatter, body, **kwargs):
        written_watermarks.append(frontmatter.profile_timestamp_ms)

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[FakeRow1(), FakeRow2()])
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock(side_effect=mock_write)

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(
            _event(cluster_id="cl_same_ts"), FakeStrategyContext()
        )

    # 2 LLM calls (1 per batch), but exactly 1 disk write when the group finished!
    assert len(fake_llm.captured_prompts) == 2
    for p in fake_llm.captured_prompts:
        assert len(p) <= DEFAULT_MAX_PROMPT_CHARS
    assert written_watermarks == [2000]


@pytest.mark.asyncio
async def test_p0_same_timestamp_failure_and_retry_full_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same timestamp group fails at batch 2: watermark stays 1000; retry succeeds."""
    mc1 = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="CONTENT_ONE_" * 2100,
                timestamp=2000,
                sender_id="u_alice",
            )
        ],
        timestamp=2000,
    )
    mc2 = MemCell(
        items=[
            ChatMessage(
                id="m2",
                role="user",
                content="CONTENT_TWO_" * 2100,
                timestamp=2000,
                sender_id="u_alice",
            )
        ],
        timestamp=2000,
    )

    class FakeRow1:
        def __init__(self) -> None:
            self.memcell_id = "mc1"
            self.app_id = "codex"
            self.project_id = "solo"
            self.payload_json = mc1.model_dump_json()

    class FakeRow2:
        def __init__(self) -> None:
            self.memcell_id = "mc2"
            self.app_id = "codex"
            self.project_id = "solo"
            self.payload_json = mc2.model_dump_json()

    cluster = _algo_cluster(
        cluster_id="cl_fail_retry", members=["mc1", "mc2"], last_ts=2000
    )
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="Prior summary",
        profile_timestamp_ms=1000,
    )

    # Run 1: Batch 2 raises timeout
    def response_factory_run1(prompt: str, call_num: int) -> str:
        if call_num == 2:
            raise RuntimeError("LLM network timeout on batch 2")
        return json.dumps(
            {
                "operations": [
                    {
                        "action": "add",
                        "type": "explicit_info",
                        "data": {"description": "fact 1"},
                    }
                ]
            }
        )

    fake_llm_run1 = FakeLLM(response_factory=response_factory_run1)
    written_watermarks: list[int] = []

    async def mock_write(owner_id, *, frontmatter, body, **kwargs):
        written_watermarks.append(frontmatter.profile_timestamp_ms)

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm_run1,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[FakeRow1(), FakeRow2()])
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock(side_effect=mock_write)

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        with pytest.raises(RuntimeError, match="LLM network timeout on batch 2"):
            await extract_user_profile(
                _event(cluster_id="cl_fail_retry"), FakeStrategyContext()
            )

    # Zero writes on disk! Watermark remains at 1000
    assert written_watermarks == []

    # Run 2: Full retry with successful LLM
    fake_llm_run2 = FakeLLM()
    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm_run2,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[FakeRow1(), FakeRow2()])
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock(side_effect=mock_write)

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(
            _event(cluster_id="cl_fail_retry"), FakeStrategyContext()
        )

    # Both batches re-run and succeed; watermark committed to 2000 once
    assert len(fake_llm_run2.captured_prompts) == 2
    assert written_watermarks == [2000]


@pytest.mark.asyncio
async def test_p0_12_same_timestamp_cells_across_multiple_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """12 cells with same timestamp spanning batches commit watermark once at end."""
    cells = [
        MemCell(
            items=[
                ChatMessage(
                    id=f"m_{i:02d}",
                    role="user",
                    content=f"SAME_TS_CELL_{i:02d}_CONTENT_" * 300,
                    timestamp=3000,
                    sender_id="u_alice",
                )
            ],
            timestamp=3000,
        )
        for i in range(12)
    ]

    class FakeRow:
        def __init__(self, idx: int) -> None:
            self.memcell_id = f"mc_{idx:02d}"
            self.app_id = "codex"
            self.project_id = "solo"
            self.payload_json = cells[idx].model_dump_json()

    cluster = _algo_cluster(
        cluster_id="cl_12",
        members=[f"mc_{i:02d}" for i in range(12)],
        last_ts=3000,
    )
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="Prior summary",
        profile_timestamp_ms=1000,
    )
    fake_llm = FakeLLM()
    written_watermarks: list[int] = []

    async def mock_write(owner_id, *, frontmatter, body, **kwargs):
        written_watermarks.append(frontmatter.profile_timestamp_ms)

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_memcell_repo.find_by_ids = AsyncMock(
            return_value=[FakeRow(i) for i in range(12)]
        )
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock(side_effect=mock_write)

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(_event(cluster_id="cl_12"), FakeStrategyContext())

    assert len(fake_llm.captured_prompts) >= 3
    for p in fake_llm.captured_prompts:
        assert len(p) <= DEFAULT_MAX_PROMPT_CHARS
    assert written_watermarks == [3000]


# ── Section 6: Enhanced Testing Evidence (600 Markers & Giant Exact Hash) ─


@pytest.mark.asyncio
async def test_case_a_600_unique_marker_messages_zero_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """600 unique marker messages: all 600 appear exactly once, every prompt <=40k."""
    cell = MemCell(
        items=[
            ChatMessage(
                id=f"m_{i:04d}",
                role="user",
                content=f"UNIQUE_MESSAGE_{i:04d} payload text for test",
                timestamp=1000 + i,
                sender_id="u_alice",
            )
            for i in range(600)
        ],
        timestamp=2000,
    )

    class FakeRow:
        def __init__(self) -> None:
            self.memcell_id = "mc_600"
            self.app_id = "codex"
            self.project_id = "solo"
            self.payload_json = cell.model_dump_json()

    cluster = _algo_cluster(cluster_id="cl_600", members=["mc_600"], last_ts=2000)
    fake_llm = FakeLLM()

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[FakeRow()])
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock()

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(_event(cluster_id="cl_600"), FakeStrategyContext())

    assert len(fake_llm.captured_prompts) >= 2
    for p in fake_llm.captured_prompts:
        assert len(p) <= DEFAULT_MAX_PROMPT_CHARS

    combined_prompts = "".join(fake_llm.captured_prompts)
    for i in range(600):
        marker = f"UNIQUE_MESSAGE_{i:04d}"
        count = combined_prompts.count(marker)
        assert count == 1, f"Marker {marker} occurred {count} times (expected 1)"


@pytest.mark.asyncio
async def test_case_c_giant_memcell_head_mid_tail_exact_hash_lossless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Giant MemCell: validates exact reconstructed SHA256 of message slices."""
    head_content = "HEAD_DATA_START_" + ("H" * 29000) + "_HEAD_DATA_END"
    mid_content = "MID_DATA_START_" + ("M" * 29000) + "_MID_DATA_END"
    tail_content = "TAIL_DATA_START_" + ("T" * 29000) + "_TAIL_DATA_END"

    head_hash = hashlib.sha256(head_content.encode()).hexdigest()
    mid_hash = hashlib.sha256(mid_content.encode()).hexdigest()
    tail_hash = hashlib.sha256(tail_content.encode()).hexdigest()

    giant_cell = MemCell(
        items=[
            ChatMessage(
                id="m_head",
                role="user",
                content=head_content,
                timestamp=1100,
                sender_id="u_alice",
            ),
            ChatMessage(
                id="m_mid",
                role="user",
                content=mid_content,
                timestamp=1200,
                sender_id="u_alice",
            ),
            ChatMessage(
                id="m_tail",
                role="user",
                content=tail_content,
                timestamp=1300,
                sender_id="u_alice",
            ),
        ],
        timestamp=2000,
    )

    class FakeRow:
        def __init__(self) -> None:
            self.memcell_id = "mc_giant"
            self.app_id = "codex"
            self.project_id = "solo"
            self.payload_json = giant_cell.model_dump_json()

    cluster = _algo_cluster(cluster_id="cl_giant", members=["mc_giant"], last_ts=2000)
    fake_llm = FakeLLM()

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[FakeRow()])
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock()

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(_event(cluster_id="cl_giant"), FakeStrategyContext())

    assert len(fake_llm.captured_prompts) >= 3
    for p in fake_llm.captured_prompts:
        assert len(p) <= DEFAULT_MAX_PROMPT_CHARS

    # Extract all message slice payloads across all prompts
    extracted_fragments: list[str] = []
    pattern = re.compile(
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] "
        r"u_alice\(user_id:u_alice\): (.*?)(?=\n\n|\n\[|\n【|$)",
        re.DOTALL,
    )
    for p in fake_llm.captured_prompts:
        matches = pattern.findall(p)
        extracted_fragments.extend(matches)

    full_reconstructed_text = "".join(extracted_fragments)

    # Extract the exact reconstructed strings for HEAD, MID, TAIL
    head_reconstructed = full_reconstructed_text[: len(head_content)]
    mid_reconstructed = full_reconstructed_text[
        len(head_content) : len(head_content) + len(mid_content)
    ]
    tail_reconstructed = full_reconstructed_text[len(head_content) + len(mid_content) :]

    assert len(head_reconstructed) == len(head_content)
    assert len(mid_reconstructed) == len(mid_content)
    assert len(tail_reconstructed) == len(tail_content)

    assert hashlib.sha256(head_reconstructed.encode()).hexdigest() == head_hash
    assert hashlib.sha256(mid_reconstructed.encode()).hexdigest() == mid_hash
    assert hashlib.sha256(tail_reconstructed.encode()).hexdigest() == tail_hash


# ── Preserved Regression Tests ───────────────────────────────────────────


def test_case_e_unsplittable_metadata_exceeds_budget_fails_closed() -> None:
    """ChatMessage with 45k sender_id fails closed without truncating metadata."""
    huge_sender_id = "user_" + ("x" * 45_000)
    huge_cell = MemCell(
        items=[
            ChatMessage(
                id="m_huge_meta",
                role="user",
                content="short content",
                timestamp=1000,
                sender_id=huge_sender_id,
            )
        ],
        timestamp=1000,
    )

    with pytest.raises(
        ValueError, match="ChatMessage metadata length.*exceeds available"
    ):
        split_memcell_losslessly(huge_cell, max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS)


@pytest.mark.asyncio
async def test_case_h_trae_exclusion_advances_watermark_without_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When candidate MemCells are from 'trae', watermark advances safely."""
    cluster = _algo_cluster(
        cluster_id="cl_trae", members=["mc_t1", "mc_t2"], last_ts=2000
    )
    rows = [
        _memcell_row("mc_t1", sender_id="u_alice", ts_ms=1500, app_id="trae"),
        _memcell_row("mc_t2", sender_id="u_alice", ts_ms=2000, app_id="trae"),
    ]
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="Prior summary",
        profile_timestamp_ms=1000,
    )
    fake_llm = FakeLLM()

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=rows)
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock()

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(_event(cluster_id="cl_trae"), FakeStrategyContext())

    assert len(fake_llm.captured_prompts) == 0
    assert mock_writer_cls.return_value.write.call_count == 1
    fm_arg = mock_writer_cls.return_value.write.call_args[1]["frontmatter"]
    assert fm_arg.profile_timestamp_ms == 2000


@pytest.mark.asyncio
async def test_init_extract_passes_old_profile_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no profile exists on disk, old_profile=None is passed."""
    cluster = _algo_cluster(last_ts=1000)
    rows = [_memcell_row("mc_aaaaaaaaaaa1", ts_ms=1000)]
    fake_llm = FakeLLM()

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=rows)
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock()

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(_event(), FakeStrategyContext())

    assert len(fake_llm.captured_prompts) == 1
    assert "=== explicit_info ===" not in fake_llm.captured_prompts[0]


@pytest.mark.asyncio
async def test_update_extract_preserves_explicit_and_implicit_traits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UPDATE mode preserves explicit_info and implicit_traits in prompt and disk."""
    cluster = _algo_cluster(last_ts=2000)
    rows = [_memcell_row("mc_aaaaaaaaaaa1", ts_ms=1500)]
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="prior summary",
        explicit_info=[{"category": "interests", "description": "embedded systems"}],
        implicit_traits=[{"trait": "terse"}],
        profile_timestamp_ms=1000,
    )
    fake_llm = FakeLLM()

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=rows)
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_writer_cls.return_value.write = AsyncMock()

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(_event(), FakeStrategyContext())

    assert len(fake_llm.captured_prompts) == 1
    prompt = fake_llm.captured_prompts[0]
    assert "embedded systems" in prompt
    assert "terse" in prompt


# ── Partition Locks ──────────────────────────────────────────────────────


async def _run_serialisation_probe(
    owner_a: str, owner_b: str, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Drive two extract_user_profile runs and record entry/exit order."""
    log: list[str] = []

    async def mock_aextract(_memcells, *, sender_id, **_kwargs):
        log.append(f"enter:{sender_id}")
        await asyncio.sleep(0.02)
        log.append(f"leave:{sender_id}")
        return AlgoProfile(
            owner_id=sender_id,
            summary="summary",
            timestamp=1000,
        )

    cluster_a = _algo_cluster(cluster_id="cl_a", members=["mc_a"], last_ts=1000)
    cluster_b = _algo_cluster(cluster_id="cl_b", members=["mc_b"], last_ts=2000)

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileExtractor"
        ) as mock_extractor_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(
            return_value=[cluster_a, cluster_b]
        )
        mock_memcell_repo.find_by_ids = AsyncMock(
            side_effect=lambda ids: [
                _memcell_row(
                    i,
                    sender_id="sender",
                    ts_ms=1000 if i == "mc_a" else 2000,
                )
                for i in ids
            ]
        )
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock()
        mock_extractor_cls.return_value.aextract = mock_aextract

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_reader", None, raising=False)
        monkeypatch.setattr(mod, "_writer", None, raising=False)

        await asyncio.gather(
            extract_user_profile(
                _event(owner_id=owner_a, cluster_id="cl_a"),
                FakeStrategyContext(),
            ),
            extract_user_profile(
                _event(owner_id=owner_b, cluster_id="cl_b"),
                FakeStrategyContext(),
            ),
        )
    return log


@pytest.mark.asyncio
async def test_partition_lock_serialises_runs_on_same_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs sharing owner_id must not overlap critical sections."""
    log = await _run_serialisation_probe("u_alice", "u_alice", monkeypatch)
    assert log in (
        ["enter:u_alice", "leave:u_alice", "enter:u_alice", "leave:u_alice"],
    )


@pytest.mark.asyncio
async def test_partition_lock_lets_different_owners_run_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs on distinct owner_id must run in parallel."""
    log = await _run_serialisation_probe("u_alice", "u_bob", monkeypatch)
    assert log.index("enter:u_alice") < log.index("leave:u_bob")
    assert log.index("enter:u_bob") < log.index("leave:u_alice")
