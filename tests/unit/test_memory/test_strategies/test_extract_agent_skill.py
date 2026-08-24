"""Tests for :func:`extract_agent_skill`.

Mocked seams: ``cluster_repo`` (sqlite), ``agent_case_repo`` /
``agent_skill_repo`` (LanceDB), ``get_embedder`` (component),
``AgentSkillExtractor`` (algo), ``AgentSkillWriter`` / ``AgentSkillReader``
(md). Each retry-class exception (cluster missing / case exists nowhere)
bubbles up so OME's ``max_retries`` machinery catches the race instead of
the strategy implementing its own backoff loop. The cascade-lag scenario
itself must NOT retry: a case durably present in markdown is rescued from
md and the run proceeds (regression tests below).

LanceDB repo behaviour itself (predicate isolation, cosine ranking,
``_distance`` stripping) lives under
``tests/unit/test_infra/test_lancedb/test_repos/``; strategy tests only
verify routing decisions and orchestration glue.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.types import AgentSkill as AlgoAgentSkill

from everos.component.embedding import (
    EmbeddingError,
    EmbeddingNotConfiguredError,
)
from everos.core.persistence import MemoryRoot
from everos.infra.ome.testing import FakeStrategyContext
from everos.infra.persistence.markdown import (
    AgentCaseReader,
    AgentCaseWriter,
    AgentSkillFrontmatter,
    AgentSkillReader,
    AgentSkillWriter,
)
from everos.memory.events import SkillClusterUpdated
from everos.memory.strategies._partition_locks import _reset_for_tests
from everos.memory.strategies.extract_agent_skill import (
    MAX_SKILLS_IN_PROMPT,
    MAX_SUPPORTING_CASES,
    _CaseNotYetIndexedError,
    _ClusterMissingError,
    _collect_supporting_entry_ids,
    _reap_renamed_skills,
    _resolve_query_vector,
    _select_existing_skills,
    _select_supporting_cases,
    _skill_rejection_reason,
    extract_agent_skill,
)

mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


@pytest.fixture
def md_root(tmp_path: Path) -> MemoryRoot:
    return MemoryRoot(tmp_path)


def _install_md_stores(monkeypatch: pytest.MonkeyPatch, root: MemoryRoot) -> None:
    """Bind every md singleton the strategy lazily builds to ``root``."""
    monkeypatch.setattr(mod, "_writer", AgentSkillWriter(root=root), raising=False)
    monkeypatch.setattr(mod, "_reader", AgentSkillReader(root), raising=False)
    monkeypatch.setattr(mod, "_case_reader", AgentCaseReader(root), raising=False)


def _event(
    *,
    cluster_id: str = "cl_xxxxxxxxxxx1",
    case_entry_id: str = "ac_20260517_0001",
    agent_id: str = "agent_42",
) -> SkillClusterUpdated:
    return SkillClusterUpdated(
        case_entry_id=case_entry_id,
        cluster_id=cluster_id,
        agent_id=agent_id,
    )


def _algo_cluster(
    *,
    cluster_id: str = "cl_xxxxxxxxxxx1",
    members: list[str] | None = None,
) -> AlgoCluster:
    return AlgoCluster(
        id=cluster_id,
        centroid=np.zeros(1024, dtype=np.float32),
        count=len(members or ["ac_20260517_0001"]),
        last_ts=1_700_000_000_000,
        preview=[],
        members=members or ["ac_20260517_0001"],
    )


def _lance_case(
    entry_id: str,
    *,
    quality_score: float = 0.8,
    timestamp: _dt.datetime | None = None,
    vector: list[float] | None = None,
    task_intent: str | None = None,
) -> MagicMock:
    """Stand-in for a LanceDB AgentCase row (only fields the strategy reads)."""
    case = MagicMock()
    case.entry_id = entry_id
    case.timestamp = timestamp or _dt.datetime(2026, 5, 17, tzinfo=_dt.UTC)
    case.task_intent = (
        task_intent if task_intent is not None else f"intent of {entry_id}"
    )
    case.approach = f"approach of {entry_id}"
    case.quality_score = quality_score
    case.key_insight = ""
    case.vector = vector or []
    return case


def _frontmatter(
    name: str,
    *,
    agent_id: str = "a",
    cluster_id: str | None = "cl_x",
    source_case_ids: list[str] | None = None,
    confidence: float = 0.5,
    maturity_score: float = 0.5,
) -> AgentSkillFrontmatter:
    return AgentSkillFrontmatter(
        id=f"{agent_id}_{name}",
        agent_id=agent_id,
        name=name,
        description=f"desc {name}",
        confidence=confidence,
        maturity_score=maturity_score,
        source_case_ids=source_case_ids or [],
        cluster_id=cluster_id,
    )


def _reader_stub(fms: list[AgentSkillFrontmatter]) -> MagicMock:
    """Reader double: ``list_by_cluster`` returns each frontmatter paired
    with a synthetic body — mirroring the real ``AgentSkillReader``, which
    returns ``(frontmatter, body)`` pairs directly rather than requiring a
    second, name-based read to hydrate ``content``."""
    reader = MagicMock()
    reader.list_by_cluster = AsyncMock(
        return_value=[(fm, f"body of {fm.name}") for fm in fms]
    )
    return reader


def _lance_skill_row(name: str) -> MagicMock:
    """Stand-in for a LanceDB AgentSkill ranking row (only ``.name`` is read)."""
    row = MagicMock()
    row.name = name
    return row


def _algo_skill(
    name: str = "summarise_doc",
    *,
    description: str | None = None,
    content: str = "full body of the skill",
) -> AlgoAgentSkill:
    return AlgoAgentSkill(
        id="dummyuuid",
        cluster_id="",  # caller will post-stamp
        name=name,
        description=description or f"how to {name}",
        content=content,
        confidence=0.7,
        maturity_score=0.5,
        source_case_ids=["ac_20260517_0001"],
    )


# ── strategy meta + retry-class errors ───────────────────────────────────


async def test_strategy_meta_is_attached() -> None:
    meta = extract_agent_skill._ome_strategy_meta  # type: ignore[attr-defined]
    assert meta.name == "extract_agent_skill"
    assert SkillClusterUpdated in meta.trigger.on
    assert meta.emits == frozenset()
    assert meta.max_retries == 3


async def test_raises_when_cluster_missing_for_retry() -> None:
    """No cluster row yet — OME will retry the run."""
    with patch(
        "everos.memory.strategies.extract_agent_skill.cluster_repo"
    ) as mock_repo:
        mock_repo.get_with_members = AsyncMock(return_value=None)
        with pytest.raises(_ClusterMissingError):
            await extract_agent_skill(_event(), FakeStrategyContext())


async def test_raises_when_target_case_exists_nowhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case is neither in LanceDB nor in markdown — a genuine same-tick
    race between the md append landing and this run. Retry-class error so
    OME catches up."""
    monkeypatch.setattr(
        mod, "_case_reader", MagicMock(find_structured=AsyncMock(return_value=None))
    )
    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_case_repo"
        ) as mock_case_repo,
    ):
        mock_cluster_repo.get_with_members = AsyncMock(return_value=_algo_cluster())
        mock_case_repo.find_by_owner_entry = AsyncMock(return_value=None)
        with pytest.raises(_CaseNotYetIndexedError):
            await extract_agent_skill(_event(), FakeStrategyContext())


# ── cascade-lag rescue (markdown durable source > LanceDB projection) ────


async def _seed_case_md(writer: AgentCaseWriter, *, intent: str, approach: str) -> str:
    """Append one AgentCase entry through the production write path.

    Mirrors ``extract_agent_case._agent_case_to_entry_body``'s field layout.
    Returns the stored marker id (``ac_<date>_<seq>``).
    """
    inline: dict[str, object] = {
        "owner_id": "agent_42",
        "session_id": "s1",
        "timestamp": "2026-05-17T00:00:00+00:00",
        "parent_type": "memcell",
        "parent_id": "mc_a",
        "quality_score": 0.82,
    }
    sections = {"TaskIntent": intent, "Approach": approach}
    append = await writer.append_entry_once(
        "agent_42",
        parent_id="mc_a",
        inline=inline,
        sections=sections,
        date=_dt.date(2026, 5, 17),
    )
    return append.entries[0].marker_id


async def test_cascade_lag_rescues_case_from_markdown(
    monkeypatch: pytest.MonkeyPatch, md_root: MemoryRoot
) -> None:
    """Regression for the production dead-letter: the AgentCase exists
    durably in markdown but its LanceDB projection lags behind (cascade
    backlog). The run must proceed off the md body instead of raising
    ``_CaseNotYetIndexedError`` into the DLQ after exhausting retries.

    Seeds the daily-log entry through the real ``AgentCaseWriter``, points
    the LanceDB repo at ``None`` (the lagging projection), and drives the
    whole strategy: the algo receives the md-derived target, and the skill
    lands on disk.
    """
    _install_md_stores(monkeypatch, md_root)
    marker_id = await _seed_case_md(
        AgentCaseWriter(root=md_root),
        intent="fix the autoreloader",
        approach="restart the watcher process",
    )
    # First entry of the 2026-05-17 bucket: seq 1, zero-padded to 8 digits.
    assert marker_id == "ac_20260517_00000001"

    emitted = [_algo_skill(name="fix_autoreloader")]
    prompt_loader = MagicMock()
    prompt_loader.load.side_effect = lambda name: f"prompt:{name}"
    monkeypatch.setattr(mod, "_prompt_loader", prompt_loader, raising=False)

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_case_repo"
        ) as mock_case_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_skill_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_skill.AgentSkillExtractor"
        ) as mock_extractor_cls,
    ):
        mock_cluster_repo.get_with_members = AsyncMock(
            return_value=_algo_cluster(members=[marker_id])
        )
        # The lagging projection: nothing indexed yet.
        mock_case_repo.find_by_owner_entry = AsyncMock(return_value=None)
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=[])

        async def _fail_topk(*_a: object, **_kw: object) -> list[object]:
            raise AssertionError(
                "no ranking expected: md enumeration found no existing skills"
            )

        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock(
            side_effect=_fail_topk
        )
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=emitted)

        await extract_agent_skill(
            _event(case_entry_id=marker_id), FakeStrategyContext()
        )

    # The algo got the md-rescued case body, not a crash.
    target_arg = mock_extractor_cls.return_value.aextract.call_args.args[0]
    assert target_arg.id == marker_id
    assert target_arg.task_intent == "fix the autoreloader"
    assert target_arg.approach == "restart the watcher process"
    assert target_arg.quality_score == 0.82
    assert target_arg.timestamp == int(
        _dt.datetime(2026, 5, 17, tzinfo=_dt.UTC).timestamp() * 1000
    )

    # And the skill was durably written.
    assert (
        md_root.agents_dir()
        / "agent_42"
        / "skills"
        / "skill_fix_autoreloader"
        / "SKILL.md"
    ).is_file()


async def test_rescue_degrades_on_partial_md_fields(
    monkeypatch: pytest.MonkeyPatch,
    md_root: MemoryRoot,
) -> None:
    """A hand-edited / partially-written md entry degrades to safe empties
    (with a warning) rather than reintroducing the dead-letter the rescue
    exists to prevent."""
    _install_md_stores(monkeypatch, md_root)
    marker_id = (
        (
            await AgentCaseWriter(root=md_root).append_entry_once(
                "agent_42",
                parent_id="mc_b",
                inline={
                    "owner_id": "agent_42",
                    "session_id": "s1",
                    "parent_type": "memcell",
                    "parent_id": "mc_b",
                },
                sections={},  # no TaskIntent / Approach / quality / timestamp
                date=_dt.date(2026, 5, 17),
            )
        )
        .entries[0]
        .marker_id
    )

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_case_repo"
        ) as mock_case_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_skill_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_skill.AgentSkillExtractor"
        ) as mock_extractor_cls,
    ):
        mock_cluster_repo.get_with_members = AsyncMock(
            return_value=_algo_cluster(members=[marker_id])
        )
        mock_case_repo.find_by_owner_entry = AsyncMock(return_value=None)
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=[])

        async def _fail_topk(*_a: object, **_kw: object) -> list[object]:
            raise AssertionError("no ranking expected")

        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock(
            side_effect=_fail_topk
        )
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=[])

        await extract_agent_skill(
            _event(case_entry_id=marker_id), FakeStrategyContext()
        )

    target_arg = mock_extractor_cls.return_value.aextract.call_args.args[0]
    assert target_arg.id == marker_id
    assert target_arg.task_intent == ""
    assert target_arg.quality_score == 0.0


# ── end-to-end orchestration (mocked) ────────────────────────────────────


async def test_extracts_and_persists_with_cluster_id_stamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end (mocked): extractor emits skills → writer stamps cluster_id."""
    target = _lance_case("ac_20260517_0001", vector=[0.1] * 1024)
    supporting = [_lance_case("ac_20260517_0000")]
    existing_fm = _frontmatter(
        "old_skill",
        agent_id="agent_42",
        cluster_id="cl_xxxxxxxxxxx1",
        source_case_ids=["ac_20260517_0000"],
    )
    emitted = [_algo_skill(name="summarise_doc"), _algo_skill(name="batch_then_synth")]
    prompt_loader = MagicMock()
    prompt_loader.load.side_effect = lambda name: f"prompt:{name}"
    monkeypatch.setattr(mod, "_prompt_loader", prompt_loader, raising=False)

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_case_repo"
        ) as mock_case_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_skill_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_skill.AgentSkillExtractor"
        ) as mock_extractor_cls,
        patch(
            "everos.memory.strategies.extract_agent_skill.AgentSkillWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.get_with_members = AsyncMock(
            return_value=_algo_cluster(members=["ac_20260517_0000", "ac_20260517_0001"])
        )
        mock_case_repo.find_by_owner_entry = AsyncMock(return_value=target)
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=supporting)

        # Small cluster path: md enumeration ≤ K → everything used as-is;
        # no ranking round trip.
        async def _fail_topk(*_a: object, **_kw: object) -> list[object]:
            raise AssertionError("ranking pointless on a fully-inclusive set")

        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock(
            side_effect=_fail_topk
        )
        monkeypatch.setattr(mod, "_reader", _reader_stub([existing_fm]), raising=False)
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=emitted)
        mock_writer_cls.return_value.write_main = AsyncMock(return_value=None)
        mock_writer_cls.return_value.delete_skill = AsyncMock(return_value=False)
        monkeypatch.setattr(mod, "_writer", None, raising=False)

        await extract_agent_skill(_event(), FakeStrategyContext())

    extractor_call = mock_extractor_cls.return_value.aextract.call_args
    target_arg = extractor_call.args[0]
    assert target_arg.id == "ac_20260517_0001"
    assert target_arg.task_intent == "intent of ac_20260517_0001"
    existing_arg = extractor_call.kwargs["existing_relevant_skills"]
    assert [s.name for s in existing_arg] == ["old_skill"]
    assert existing_arg[0].id == "agent_42_old_skill"
    assert existing_arg[0].content == "body of old_skill"
    assert [c.id for c in extractor_call.kwargs["supporting_cases"]] == [
        "ac_20260517_0000"
    ]
    assert extractor_call.kwargs["prompt_success"] == "prompt:agent_skill_success"
    assert extractor_call.kwargs["prompt_failure"] == "prompt:agent_skill_failure"

    write_calls = mock_writer_cls.return_value.write_main.call_args_list
    assert len(write_calls) == 2
    for call, expected in zip(write_calls, emitted, strict=True):
        agent_id_arg, skill_name_arg = call.args
        fm = call.kwargs["frontmatter"]
        assert agent_id_arg == "agent_42"
        assert skill_name_arg == expected.name
        assert fm.cluster_id == "cl_xxxxxxxxxxx1"
        assert fm.name == expected.name
        assert fm.confidence == expected.confidence
        assert call.kwargs["body"] == expected.content


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Use sshpass -p hunter2 ssh host", "sshpass_plaintext_password"),
        ("ssh -o StrictHostKeyChecking=no host", "ssh_host_verification_disabled"),
        ("Run chmod 777 /srv/app", "world_writable_permissions"),
        ("Set verify=False for the request", "tls_verification_disabled"),
        ("Token sk-" + ("a" * 32), "openai_style_secret"),
    ],
)
def test_skill_safety_gate_rejects_high_confidence_patterns(
    content: str, expected: str
) -> None:
    assert _skill_rejection_reason(_algo_skill(content=content)) == expected


def test_skill_safety_gate_allows_safe_placeholder_and_verification() -> None:
    skill = _algo_skill(
        content=(
            "## Steps\n1. Read the token from <TOKEN_ENV>.\n"
            "2. Verify the SSH host fingerprint before connecting."
        )
    )
    assert _skill_rejection_reason(skill) is None


def test_persisted_skill_survives_traversal_shaped_llm_name() -> None:
    """A traversal-shaped LLM name is sanitized *before* frontmatter
    construction — the read-side validator must never turn the security
    fix into a dead-letter DoS."""
    raw = "../" * 8 + "tmp/pwned"
    skill = _algo_skill(name=raw)
    sanitized = AgentSkillFrontmatter.sanitize_skill_name(skill.name)

    fm = AgentSkillFrontmatter(
        id=f"agent_42_{sanitized}",
        agent_id="agent_42",
        name=sanitized,
        description=skill.description,
        confidence=skill.confidence,
        maturity_score=skill.maturity_score,
        source_case_ids=list(skill.source_case_ids),
        cluster_id="cl1",
    )
    assert "/" not in fm.name
    assert fm.name == sanitized


# ── _select_existing_skills routing (cluster size × vector availability) ─


async def test_select_existing_skills_small_cluster_returns_all_md_skills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``len(md) ≤ K`` short-circuits — no ranking needed, no LanceDB call."""
    target = _lance_case("ac_001", vector=[0.5] * 1024)
    fms = [_frontmatter(f"s{i}") for i in range(3)]
    monkeypatch.setattr(mod, "_reader", _reader_stub(fms), raising=False)

    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
    ) as mock_repo:

        async def _fail_topk(*_a: object, **_kw: object) -> list[object]:
            raise AssertionError("ranking pointless on a fully-inclusive set")

        mock_repo.find_topk_relevant_in_cluster = AsyncMock(side_effect=_fail_topk)

        got = await _select_existing_skills(
            agent_id="a",
            cluster_id="cl_x",
            target=target,
            app_id="default",
            project_id="default",
        )

    assert [s.name for s in got] == [f"s{i}" for i in range(3)]
    assert [s.content for s in got] == [f"body of s{i}" for i in range(3)]


async def test_select_existing_skills_large_cluster_with_vector_ranks_via_lancedb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``len(md) > K`` and the target carries a vector → cosine top-K over
    LanceDB, hydrated from md; stale LanceDB rows are skipped and md
    backfills the remainder."""
    target = _lance_case("ac_001", vector=[0.5] * 1024)
    fms = [_frontmatter(f"s{i}") for i in range(MAX_SKILLS_IN_PROMPT + 5)]
    monkeypatch.setattr(mod, "_reader", _reader_stub(fms), raising=False)

    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
    ) as mock_repo:
        # Ranking proposes two live skills plus one stale row (indexed once,
        # deleted from md since).
        mock_repo.find_topk_relevant_in_cluster = AsyncMock(
            return_value=[
                _lance_skill_row("s9"),
                _lance_skill_row("ghost"),
                _lance_skill_row("s0"),
            ]
        )

        got = await _select_existing_skills(
            agent_id="a",
            cluster_id="cl_x",
            target=target,
            app_id="default",
            project_id="default",
        )

    # Ranked winners first (stale row dropped), then md-order backfill to K.
    expected = ["s9", "s0"] + [f"s{i}" for i in range(1, 9)]
    assert [s.name for s in got] == expected
    call_kwargs = mock_repo.find_topk_relevant_in_cluster.await_args.kwargs
    assert call_kwargs["query_vector"] == [0.5] * 1024
    assert call_kwargs["top_k"] == MAX_SKILLS_IN_PROMPT


async def test_select_existing_skills_large_cluster_recomputes_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``len(md) > K`` but case has no vector → re-embed ``task_intent`` on the fly."""
    target = _lance_case("ac_001", vector=[], task_intent="how to summarise docs")
    fms = [_frontmatter(f"s{i}") for i in range(MAX_SKILLS_IN_PROMPT + 2)]
    ranked = [_lance_skill_row(f"s{i}") for i in range(MAX_SKILLS_IN_PROMPT)]
    fresh_vec = [0.42] * 1024
    monkeypatch.setattr(mod, "_reader", _reader_stub(fms), raising=False)

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=fresh_vec)

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_embedder",
            return_value=mock_embedder,
        ),
    ):
        mock_repo.find_topk_relevant_in_cluster = AsyncMock(return_value=ranked)

        got = await _select_existing_skills(
            agent_id="a",
            cluster_id="cl_x",
            target=target,
            app_id="default",
            project_id="default",
        )

    mock_embedder.embed.assert_awaited_once_with("how to summarise docs")
    assert len(got) == MAX_SKILLS_IN_PROMPT
    call_kwargs = mock_repo.find_topk_relevant_in_cluster.await_args.kwargs
    assert call_kwargs["query_vector"] == fresh_vec


async def test_select_existing_skills_falls_back_to_md_order_when_embed_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``len(md) > K`` + no vector + embedder fails → md ordering capped at K."""
    fms = [_frontmatter(f"s{i}") for i in range(MAX_SKILLS_IN_PROMPT + 3)]
    target = _lance_case("ac_001", vector=[], task_intent="how to summarise docs")
    monkeypatch.setattr(mod, "_reader", _reader_stub(fms), raising=False)

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(side_effect=EmbeddingError("provider down"))

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_embedder",
            return_value=mock_embedder,
        ),
    ):

        async def _fail_topk(*_a: object, **_kw: object) -> list[object]:
            raise AssertionError("no query vector → no ranking call")

        mock_repo.find_topk_relevant_in_cluster = AsyncMock(side_effect=_fail_topk)

        got = await _select_existing_skills(
            agent_id="a",
            cluster_id="cl_x",
            target=target,
            app_id="default",
            project_id="default",
        )

    assert [s.name for s in got] == [f"s{i}" for i in range(MAX_SKILLS_IN_PROMPT)]


async def test_select_existing_skills_no_vector_no_intent_falls_back_to_md_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The md-rescue path carries no vector; an empty intent leaves nothing
    to embed either — degrade to md order, never raise."""
    fms = [_frontmatter(f"s{i}") for i in range(MAX_SKILLS_IN_PROMPT + 1)]
    target = _lance_case("ac_001", vector=[], task_intent="")
    monkeypatch.setattr(mod, "_reader", _reader_stub(fms), raising=False)

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_embedder"
        ) as mock_get_embedder,
    ):

        async def _fail_topk(*_a: object, **_kw: object) -> list[object]:
            raise AssertionError("no query vector → no ranking call")

        mock_repo.find_topk_relevant_in_cluster = AsyncMock(side_effect=_fail_topk)

        got = await _select_existing_skills(
            agent_id="a",
            cluster_id="cl_x",
            target=target,
            app_id="default",
            project_id="default",
        )

    assert [s.name for s in got] == [f"s{i}" for i in range(MAX_SKILLS_IN_PROMPT)]
    mock_get_embedder.assert_not_called()


async def test_existing_skills_reaches_llm_for_skill_whose_directory_has_a_space(
    monkeypatch: pytest.MonkeyPatch, md_root: MemoryRoot
) -> None:
    """End-to-end regression for the ``list_by_cluster`` drop bug: a
    ``skill_My Skill/`` directory written outside the writer (raw space,
    never sanitized) must reach ``existing_relevant_skills`` with non-empty
    ``content`` — enumeration and hydration happen in one path-based pass,
    with no second, name-based read to drop it one layer downstream.
    """
    _install_md_stores(monkeypatch, md_root)

    skill_dir = md_root.agents_dir() / "a1" / "skills" / "skill_My Skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "id: a1_My Skill\n"
        "type: agent_skill\n"
        "agent_id: a1\n"
        "track: agent\n"
        "name: My Skill\n"
        "description: d\n"
        "confidence: 0.5\n"
        "maturity_score: 0.5\n"
        "cluster_id: cl1\n"
        "---\n"
        "The real skill body.\n",
        encoding="utf-8",
    )

    captured: dict[str, list] = {}

    async def spy_aextract(
        target, *, existing_relevant_skills, supporting_cases, **_kw
    ):
        captured["existing"] = list(existing_relevant_skills)
        return []

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_skill_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_case_repo"
        ) as mock_case_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_skill.AgentSkillExtractor"
        ) as mock_extractor_cls,
    ):
        mock_cluster_repo.get_with_members = AsyncMock(
            return_value=_algo_cluster(cluster_id="cl1", members=["c0", "c1"])
        )

        async def _fail_topk(*_a: object, **_kw: object) -> list[object]:
            raise AssertionError(
                "must not be reached: cluster is within MAX_SKILLS_IN_PROMPT"
            )

        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock(
            side_effect=_fail_topk
        )
        mock_case_repo.find_by_owner_entry = AsyncMock(return_value=_lance_case("c1"))
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=[])
        mock_extractor_cls.return_value.aextract = spy_aextract

        await extract_agent_skill(
            _event(cluster_id="cl1", agent_id="a1", case_entry_id="c1"),
            FakeStrategyContext(),
        )

    assert len(captured["existing"]) == 1
    hydrated = captured["existing"][0]
    assert hydrated.name == "My Skill"
    assert hydrated.content == "The real skill body."


# ── _resolve_query_vector layered fallback ───────────────────────────────


async def test_resolve_query_vector_prefers_persisted_vector() -> None:
    """When ``target.vector`` is set, reuse it; never call the embedder."""
    target = mod._lance_to_target(_lance_case("ac_001", vector=[0.3] * 1024))
    with patch(
        "everos.memory.strategies.extract_agent_skill.get_embedder"
    ) as mock_get_embedder:
        got = await _resolve_query_vector(target)
    assert got == [0.3] * 1024
    mock_get_embedder.assert_not_called()


async def test_resolve_query_vector_returns_empty_when_no_text_either() -> None:
    """No persisted vector + no task_intent → ``[]`` (no policy here)."""
    target = mod._lance_to_target(_lance_case("ac_001", vector=[], task_intent=""))
    with patch(
        "everos.memory.strategies.extract_agent_skill.get_embedder"
    ) as mock_get_embedder:
        got = await _resolve_query_vector(target)
    assert got == []
    mock_get_embedder.assert_not_called()


async def test_resolve_query_vector_swallows_embedder_not_configured() -> None:
    """Missing embedder config is a deployment issue, not a strategy fault."""
    target = mod._lance_to_target(_lance_case("ac_001", vector=[], task_intent="hello"))
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(
        side_effect=EmbeddingNotConfiguredError("no api key")
    )
    with patch(
        "everos.memory.strategies.extract_agent_skill.get_embedder",
        return_value=mock_embedder,
    ):
        got = await _resolve_query_vector(target)
    assert got == []


# ── _select_supporting_cases ranking + cap ───────────────────────────────


async def test_select_supporting_cases_ranks_by_quality_then_timestamp() -> None:
    """Hydrated cases sort ``(quality_score desc, timestamp desc)``."""
    skills = [
        _algo_skill(name="s1", content="c"),
    ]
    skills[0].source_case_ids = ["ac_a", "ac_b", "ac_c"]
    case_a = _lance_case(
        "ac_a",
        quality_score=0.4,
        timestamp=_dt.datetime(2026, 5, 1, tzinfo=_dt.UTC),
    )
    case_b = _lance_case(
        "ac_b",
        quality_score=0.9,
        timestamp=_dt.datetime(2026, 5, 1, tzinfo=_dt.UTC),
    )
    case_c = _lance_case(
        "ac_c",
        quality_score=0.9,
        timestamp=_dt.datetime(2026, 5, 10, tzinfo=_dt.UTC),
    )

    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_case_repo"
    ) as mock_case_repo:
        # Order intentionally scrambled to prove the strategy sorts.
        mock_case_repo.find_by_owner_entries = AsyncMock(
            return_value=[case_a, case_b, case_c]
        )

        got = await _select_supporting_cases(
            skills,
            agent_id="a",
            exclude_entry_id="ac_target",
            app_id="default",
            project_id="default",
        )

    assert [c.entry_id for c in got] == ["ac_c", "ac_b", "ac_a"]


async def test_select_supporting_cases_caps_at_max_supporting() -> None:
    """Hydrated set is truncated to ``MAX_SUPPORTING_CASES``."""
    ids = [f"ac_{i:03d}" for i in range(MAX_SUPPORTING_CASES + 3)]
    skills = [_algo_skill(name="s1", content="c")]
    skills[0].source_case_ids = ids
    hydrated = [
        _lance_case(eid, quality_score=0.5 + 0.01 * i) for i, eid in enumerate(ids)
    ]

    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_case_repo"
    ) as mock_case_repo:
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=hydrated)
        got = await _select_supporting_cases(
            skills,
            agent_id="a",
            exclude_entry_id="ac_target",
            app_id="default",
            project_id="default",
        )

    assert len(got) == MAX_SUPPORTING_CASES


async def test_select_supporting_cases_skips_repo_when_no_lineage_ids() -> None:
    """No usable source ids → ``[]`` without a repo round trip."""
    skills = [_algo_skill(name="s1", content="c")]
    skills[0].source_case_ids = []
    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_case_repo"
    ) as mock_case_repo:
        mock_case_repo.find_by_owner_entries = AsyncMock()
        got = await _select_supporting_cases(
            skills,
            agent_id="a",
            exclude_entry_id="ac_target",
            app_id="default",
            project_id="default",
        )
    assert got == []
    mock_case_repo.find_by_owner_entries.assert_not_awaited()


# ── _collect_supporting_entry_ids dedup + exclude ────────────────────────


def test_collect_supporting_entry_ids_dedups_and_excludes_target() -> None:
    """Source ids fold across skills; duplicates and the target id drop out."""
    skill_a = MagicMock()
    skill_a.source_case_ids = ["ac_a", "ac_b", "ac_target"]
    skill_b = MagicMock()
    skill_b.source_case_ids = ["ac_b", "ac_c"]  # ac_b duplicates skill_a's lineage
    skill_empty = MagicMock()
    skill_empty.source_case_ids = []

    got = _collect_supporting_entry_ids(
        [skill_a, skill_b, skill_empty], exclude="ac_target"
    )
    assert got == ["ac_a", "ac_b", "ac_c"]


def test_collect_supporting_entry_ids_handles_empty_input() -> None:
    """No skills → no supporting cases."""
    assert _collect_supporting_entry_ids([], exclude="ac_anything") == []


# ── partition lock (agent_id-level serialisation) ────────────────────────


async def _run_serialisation_probe(
    agent_id_run_a: str, agent_id_run_b: str
) -> list[str]:
    """Drive two extract_agent_skill runs and record their critical-section order.

    Mocks every I/O seam so the only async work inside the locked region
    is a tiny ``asyncio.sleep`` masquerading as the LLM call. The returned
    log is the strict enter/leave sequence both runs go through.
    """
    log: list[str] = []

    async def mock_aextract(case, **_kwargs):
        log.append(f"enter:{case.id}")
        await asyncio.sleep(0.01)
        log.append(f"leave:{case.id}")
        return []

    target_a = _lance_case("ac_run_a", vector=[0.1] * 1024)
    target_b = _lance_case("ac_run_b", vector=[0.1] * 1024)

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_case_repo"
        ) as mock_case_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_skill_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_skill.AgentSkillExtractor"
        ) as mock_extractor_cls,
        patch("everos.memory.strategies.extract_agent_skill.AgentSkillWriter"),
        patch.object(
            mod,
            "_reader",
            _reader_stub([]),
        ),
    ):
        mock_cluster_repo.get_with_members = AsyncMock(
            return_value=_algo_cluster(members=["ac_run_a", "ac_run_b"])
        )
        mock_case_repo.find_by_owner_entry = AsyncMock(
            side_effect=lambda owner, entry, **_kw: (
                target_a if entry == "ac_run_a" else target_b
            )
        )
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=[])

        async def _fail_topk(*_a: object, **_kw: object) -> list[object]:
            raise AssertionError("no existing skills → no ranking call")

        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock(
            side_effect=_fail_topk
        )
        mock_extractor_cls.return_value.aextract = mock_aextract
        await asyncio.gather(
            extract_agent_skill(
                _event(agent_id=agent_id_run_a, case_entry_id="ac_run_a"),
                FakeStrategyContext(),
            ),
            extract_agent_skill(
                _event(agent_id=agent_id_run_b, case_entry_id="ac_run_b"),
                FakeStrategyContext(),
            ),
        )
    return log


async def test_partition_lock_serialises_runs_on_same_agent() -> None:
    """Two runs sharing ``agent_id`` must not overlap critical sections."""
    log = await _run_serialisation_probe("agent_42", "agent_42")
    assert log in (
        ["enter:ac_run_a", "leave:ac_run_a", "enter:ac_run_b", "leave:ac_run_b"],
        ["enter:ac_run_b", "leave:ac_run_b", "enter:ac_run_a", "leave:ac_run_a"],
    )


async def test_partition_lock_lets_different_agents_run_in_parallel() -> None:
    """Runs on distinct ``agent_id`` must overlap (no false serialisation)."""
    log = await _run_serialisation_probe("agent_42", "agent_43")
    assert log.index("enter:ac_run_a") < log.index("leave:ac_run_b")
    assert log.index("enter:ac_run_b") < log.index("leave:ac_run_a")


# ── rename reconciliation (orphan directories) ──────────────────────────


def _identified_algo_skill(skill_id: str, name: str) -> AlgoAgentSkill:
    """Like :func:`_algo_skill` but with an explicit id — rename
    reconciliation keys off identity, so these tests must control it."""
    return AlgoAgentSkill(
        id=skill_id,
        cluster_id="cl1",
        name=name,
        description="d",
        content="body",
        confidence=0.8,
        maturity_score=0.5,
        source_case_ids=["case_a"],
    )


async def _write_skill(writer: AgentSkillWriter, name: str) -> Path:
    fm = AgentSkillFrontmatter(
        id=f"agent_42_{name}",
        agent_id="agent_42",
        name=name,
        description="d",
        confidence=0.8,
        maturity_score=0.5,
        cluster_id="cl1",
    )
    path = await writer.write_main("agent_42", name, frontmatter=fm, body="body")
    return path.parent


async def test_reap_removes_the_directory_a_rename_left_behind(
    tmp_path: Path,
) -> None:
    """An update that renames a skill must not leave its old directory.

    everalgo's ``_apply_update`` keeps ``prior.id`` while changing the
    name, so the emitted skill is written under a new directory and the
    old one would survive carrying the same ``cluster_id``. That is not a
    cosmetic leak: the next extraction's ``existing_relevant_skills`` come
    from the markdown enumeration, so the orphan returns as a duplicate of
    a skill the LLM already renamed, which is how ``add``-instead-of-
    ``update`` full-replace clobbering gets back in. Uses a real writer on
    a real tmp_path — the property under test is that the directory is
    gone from the filesystem.
    """
    writer = AgentSkillWriter(MemoryRoot(tmp_path))
    old_dir = await _write_skill(writer, "fix_django")
    new_dir = await _write_skill(writer, "fix_django_autoreload")
    assert old_dir.is_dir() and new_dir.is_dir()

    await _reap_renamed_skills(
        writer,
        {"agent_42_fix_django": "fix_django_autoreload"},
        existing_skills=[_identified_algo_skill("agent_42_fix_django", "fix_django")],
        agent_id="agent_42",
        app_id="default",
        project_id="default",
    )

    assert not old_dir.exists()
    assert (new_dir / "SKILL.md").is_file()


async def test_reap_keeps_a_prior_name_another_emitted_skill_claimed(
    tmp_path: Path,
) -> None:
    """Never delete a directory this same batch just wrote.

    With two ops in one extraction — rename ``a`` → ``b`` while a second
    op writes ``a`` — reaping ``a`` by prior name would remove a file
    written moments earlier in the same loop. The claimed-name guard is
    what prevents the reap from undoing its own caller.
    """
    writer = AgentSkillWriter(MemoryRoot(tmp_path))
    dir_a = await _write_skill(writer, "alpha")
    await _write_skill(writer, "beta")

    await _reap_renamed_skills(
        writer,
        {"agent_42_alpha": "beta", "other_id": "alpha"},
        existing_skills=[_identified_algo_skill("agent_42_alpha", "alpha")],
        agent_id="agent_42",
        app_id="default",
        project_id="default",
    )

    assert dir_a.is_dir()


async def test_reap_ignores_newly_added_skills(tmp_path: Path) -> None:
    """A fresh ``add`` carries a uuid4 id absent from the enumerated set.

    Identity is the only thing that survives a rename — ``_apply_update``
    preserves ``prior.id`` while ``_apply_add`` mints a new one — so an
    id that never appeared in ``existing_skills`` cannot be a rename, and
    nothing may be deleted on its account.
    """
    writer = AgentSkillWriter(MemoryRoot(tmp_path))
    kept = await _write_skill(writer, "existing_skill")

    await _reap_renamed_skills(
        writer,
        {"3f2a9c1e4b6d47f8a0c5e9b2d7143a6f": "brand_new_skill"},
        existing_skills=[
            _identified_algo_skill("agent_42_existing_skill", "existing_skill")
        ],
        agent_id="agent_42",
        app_id="default",
        project_id="default",
    )

    assert kept.is_dir()
