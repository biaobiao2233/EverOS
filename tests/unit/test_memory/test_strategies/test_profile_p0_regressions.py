"""Independent regression suite demonstrating P0 failure modes and fixes."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from everalgo.types import ChatMessage, MemCell
from typer.testing import CliRunner

from everos.entrypoints.cli.commands import profile as profile_mod
from everos.infra.persistence.markdown import UserProfileFrontmatter
from everos.memory.strategies.extract_user_profile import (
    DEFAULT_MAX_PROMPT_CHARS,
    _to_algo_profile,
    _to_frontmatter,
    plan_next_step,
    prepare_pending_items,
    render_full_prompt,
    split_memcell_losslessly,
)


def test_p0_1_prefix_expansion_failure_and_fix() -> None:
    """P0-1: 600 messages x 30 chars causes prefix expansion > 40k.

    Fixed by lossless sub-cell splitting.
    """
    mc = MemCell(
        items=[
            ChatMessage(
                id=f"m_{i:04d}",
                role="user",
                content="prefers python and fast tests",
                timestamp=1000 + i,
                sender_id="u_alice",
            )
            for i in range(600)
        ],
        timestamp=2000,
    )

    # 1. Failure demonstration: Raw prompt violates 40k ceiling
    raw_prompt = render_full_prompt([mc], None)
    assert len(raw_prompt) > 40_000, f"Expected > 40k, got {len(raw_prompt)}"

    # 2. Fix demonstration: Lossless splitting bounds all sub-cells
    sub_cells = split_memcell_losslessly(mc, max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS)
    assert len(sub_cells) >= 2
    assert sum(len(sc.items) for sc in sub_cells) == 600
    for sc in sub_cells:
        p = render_full_prompt([sc], None)
        assert len(p) <= DEFAULT_MAX_PROMPT_CHARS


def test_p0_2_explicit_implicit_traits_round_trip_preserved() -> None:
    """P0-2: explicit_info & implicit_traits must be 100% preserved during roundtrip."""
    fm = UserProfileFrontmatter(
        id="profile_user",
        user_id="user",
        summary="User summary",
        explicit_info=[
            {"category": "Environment", "description": "Windows 11 and WSL2"},
            {"category": "Network", "description": "Beijing-Korea Relay"},
        ],
        implicit_traits=[
            {"trait": "Precise", "description": "Prefers automated verification"},
            {"trait": "Safety", "description": "Requires backups before ops"},
        ],
        profile_timestamp_ms=1000,
    )

    algo_p = _to_algo_profile(fm)
    assert len(getattr(algo_p, "explicit_info", [])) == 2
    assert len(getattr(algo_p, "implicit_traits", [])) == 2

    # Render for update contains both
    rendered_update = render_full_prompt([], algo_p)
    assert "Windows 11 and WSL2" in rendered_update
    assert "Requires backups before ops" in rendered_update

    # Re-serialize to frontmatter
    fm_new = _to_frontmatter(algo_p, owner_id="user", app_id="codex", project_id="solo")
    assert fm_new.explicit_info == fm.explicit_info
    assert fm_new.implicit_traits == fm.implicit_traits


def test_p0_timestamp_group_tagging_and_multi_batch_commit() -> None:
    """P0: Timestamp group tagging correctly holds watermark across batches."""
    mc1 = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="A" * 25_000,
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
                content="B" * 25_000,
                timestamp=2000,
                sender_id="u_alice",
            )
        ],
        timestamp=2000,
    )

    pending = prepare_pending_items([("mc1", mc1), ("mc2", mc2)])
    assert len(pending) == 2
    assert pending[0][3] is False  # mc1 is NOT group final
    assert pending[1][3] is True  # mc2 IS group final

    batch_1, pending = plan_next_step(
        pending, None, max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS
    )
    assert batch_1.completed_watermark is None, (
        "Batch 1 must hold watermark because timestamp group is not complete"
    )

    batch_2, pending = plan_next_step(
        pending, None, max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS
    )
    assert batch_2.completed_watermark == 2000, (
        "Batch 2 completes the timestamp group and advances watermark"
    )


def test_p0_cross_timestamp_mixed_batch_watermark_held() -> None:
    """P0: A batch with complete group 1500 and partial group 2000 holds watermark."""
    mc_old = MemCell(
        items=[
            ChatMessage(
                id="m_old",
                role="user",
                content="OLD" * 10,
                timestamp=1500,
                sender_id="u_alice",
            )
        ],
        timestamp=1500,
    )
    mc_a = MemCell(
        items=[
            ChatMessage(
                id="m_a",
                role="user",
                content="A" * 25_000,
                timestamp=2000,
                sender_id="u_alice",
            )
        ],
        timestamp=2000,
    )
    mc_b = MemCell(
        items=[
            ChatMessage(
                id="m_b",
                role="user",
                content="B" * 25_000,
                timestamp=2000,
                sender_id="u_alice",
            )
        ],
        timestamp=2000,
    )

    pending = prepare_pending_items(
        [("mc_old", mc_old), ("mc_a", mc_a), ("mc_b", mc_b)]
    )
    assert len(pending) == 3
    # mc_old is group final for 1500
    assert pending[0][3] is True
    # mc_a is NOT group final for 2000
    assert pending[1][3] is False
    # mc_b IS group final for 2000
    assert pending[2][3] is True

    # Batch 1 takes mc_old + mc_a (fits within 40k)
    batch_1, pending = plan_next_step(
        pending, None, max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS
    )
    assert len(batch_1.memcells) == 2
    # Because mc_a is not group final, watermark MUST BE None
    assert batch_1.completed_watermark is None
    assert batch_1.completed_memcell_ids == []

    # Batch 2 takes mc_b
    batch_2, pending = plan_next_step(
        pending, None, max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS
    )
    assert len(batch_2.memcells) == 1
    # Because mc_b is group final, watermark commits to 2000
    assert batch_2.completed_watermark == 2000
    assert batch_2.completed_memcell_ids == ["mc_a", "mc_b"]


def test_p0_3_offline_recovery_refuses_when_active_service_detected() -> None:
    """P0-3: When EverOS service is active, recovery fails closed with exit code 1."""
    with patch(
        "everos.entrypoints.cli.commands.profile.is_everos_service_active",
        return_value=True,
    ):
        result = CliRunner().invoke(
            profile_mod.app,
            [
                "recover",
                "--owner-id",
                "user",
                "--app-id",
                "codex",
                "--project-id",
                "solo",
            ],
        )
        assert result.exit_code == 1
        assert "Error: EverOS server is currently active" in result.output


def test_p0_4_status_requires_explicit_partitions_and_protects_privacy() -> None:
    """P0-4 & P1-2: Enforce explicit partitions and never leak summary."""
    # 1. Missing partition fails
    res_err = CliRunner().invoke(profile_mod.app, ["status"])
    assert res_err.exit_code != 0

    # 2. Privacy output check
    existing_fm = UserProfileFrontmatter(
        id="profile_user",
        user_id="user",
        summary="CONFIDENTIAL SUMMARY THAT MUST NEVER APPEAR IN OUTPUT",
        profile_timestamp_ms=1000,
    )
    with (
        patch(
            "everos.entrypoints.cli.commands.profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.entrypoints.cli.commands.profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.entrypoints.cli.commands.profile.get_unprocessed_memcells_for_owner",
            return_value=([], 0, 0),
        ),
    ):
        mock_reader_cls.return_value.read = AsyncMock(
            return_value=(
                existing_fm,
                "CONFIDENTIAL SUMMARY THAT MUST NEVER APPEAR IN OUTPUT",
            )
        )
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[])

        res = CliRunner().invoke(
            profile_mod.app,
            [
                "status",
                "--owner-id",
                "user",
                "--app-id",
                "codex",
                "--project-id",
                "solo",
            ],
        )
        assert res.exit_code == 0
        assert "CONFIDENTIAL SUMMARY" not in res.stdout
        assert "Profile Status for 'user' (app=codex, project=solo):" in res.stdout
