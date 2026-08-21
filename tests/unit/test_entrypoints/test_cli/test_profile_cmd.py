"""Unit tests for ``everos profile`` CLI command group."""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from everalgo.types import ChatMessage, MemCell
from everalgo.types import Profile as AlgoProfile
from typer.testing import CliRunner

from everos.core.persistence import MemoryRoot
from everos.core.persistence.locking import LockError
from everos.entrypoints.cli.commands import profile as profile_mod
from everos.infra.persistence.markdown import UserProfileFrontmatter
from everos.infra.persistence.sqlite.tables import (  # noqa: F401
    Cluster,
    ClusterMember,
    Memcell,
)


def _sample_memcell(memcell_id: str, ts: int, sender_id: str = "user") -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id=f"{memcell_id}_m1",
                role="user",
                content=f"message content for {memcell_id}",
                timestamp=ts,
                sender_id=sender_id,
            ),
        ],
        timestamp=ts,
    )


def test_app_registers_status_and_recover() -> None:
    names = {cmd.name for cmd in profile_mod.app.registered_commands}
    assert names == {"status", "recover"}


def test_help_exits_zero() -> None:
    result = CliRunner().invoke(profile_mod.app, ["--help"])
    assert result.exit_code == 0
    assert "status" in result.stdout
    assert "recover" in result.stdout


def test_status_command_requires_explicit_partitions() -> None:
    result = CliRunner().invoke(profile_mod.app, ["status"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "Usage:" in result.output


def test_status_command_output_is_privacy_safe() -> None:
    existing_fm = UserProfileFrontmatter(
        id="profile_user",
        user_id="user",
        summary="SECRET PRIVATE SUMMARY THAT MUST NOT BE PRINTED",
        profile_timestamp_ms=1000,
    )
    mock_cell = _sample_memcell("mc_1", 1500)

    with (
        patch(
            "everos.entrypoints.cli.commands.profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.entrypoints.cli.commands.profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.entrypoints.cli.commands.profile.get_unprocessed_memcells_for_owner",
            return_value=([("mc_1", mock_cell)], 0, 0),
        ),
    ):
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[])

        result = CliRunner().invoke(
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
        assert result.exit_code == 0
        assert "Profile Status for 'user' (app=codex, project=solo):" in result.stdout
        assert "Current Watermark:     1000" in result.stdout
        assert "Pending MemCells:      1" in result.stdout
        assert "SECRET PRIVATE SUMMARY" not in result.stdout


def test_recover_dry_run_is_readonly_and_privacy_safe() -> None:
    existing_fm = UserProfileFrontmatter(
        id="profile_user",
        user_id="user",
        summary="SECRET PRIVATE SUMMARY",
        profile_timestamp_ms=1000,
    )
    mock_cells = [_sample_memcell(f"mc_{i}", 1100 + i * 10) for i in range(30)]

    with (
        patch(
            "everos.entrypoints.cli.commands.profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.entrypoints.cli.commands.profile.get_unprocessed_memcells_for_owner",
            return_value=(
                [(f"mc_{i}", cell) for i, cell in enumerate(mock_cells)],
                0,
                0,
            ),
        ),
    ):
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])

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
                "--dry-run",
                "--batch-size",
                "15",
            ],
        )
        assert result.exit_code == 0
        assert "=== DRY-RUN Profile Recovery for 'user' ===" in result.stdout
        assert "App / Project:         codex / solo" in result.stdout
        assert "Pending MemCells:      30" in result.stdout
        assert "Estimated Batches:     2" in result.stdout
        assert "Batch 01/02:" in result.stdout
        assert "Batch 02/02:" in result.stdout
        assert "SECRET PRIVATE SUMMARY" not in result.stdout


def test_recover_dry_run_real_sqlite_engine_and_immutability(
    tmp_path: Path,
) -> None:
    """Verifies dry-run leaves real SQLite DB bit-for-bit identical."""
    import numpy as np
    from sqlalchemy import create_engine
    from sqlmodel import SQLModel

    mr = MemoryRoot(root=tmp_path)
    mr.system_db.parent.mkdir(parents=True, exist_ok=True)
    db_file = mr.system_db

    sync_engine = create_engine(f"sqlite:///{db_file.as_posix()}")
    SQLModel.metadata.create_all(sync_engine)
    sync_engine.dispose()

    conn = sqlite3.connect(db_file)
    now_str = "2026-01-01T00:00:00Z"
    cell = _sample_memcell("mc_1", 2000, sender_id="user")
    centroid_bytes = np.zeros(128, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO cluster VALUES "
        "(?, ?, 'cl_1', 'codex', 'solo', 'user', 'user', "
        "'user_memory', ?, 1, 2000, '[]')",
        (now_str, now_str, centroid_bytes),
    )
    conn.execute(
        "INSERT INTO cluster_member VALUES (?, ?, 'cl_1', 'mc_1', 'memcell', ?)",
        (now_str, now_str, now_str),
    )
    conn.execute(
        "INSERT INTO memcell VALUES "
        "(?, ?, 'mc_1', 'codex', 'solo', 'sess_1', 'user_memory', "
        "'chat', '[]', '[]', ?, ?)",
        (now_str, now_str, cell.model_dump_json(), now_str),
    )
    conn.commit()
    conn.close()

    hash_before = hashlib.sha256(db_file.read_bytes()).hexdigest()

    with (
        patch(
            "everos.entrypoints.cli.commands.profile.MemoryRoot.default",
            return_value=mr,
        ),
        patch(
            "everos.entrypoints.cli.commands.profile.ProfileReader.read",
            return_value=None,
        ),
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
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "=== DRY-RUN Profile Recovery for 'user' ===" in result.stdout
        assert "Pending MemCells:      1" in result.stdout

    hash_after = hashlib.sha256(db_file.read_bytes()).hexdigest()
    assert hash_before == hash_after, "SQLite file was modified during dry-run!"
    assert not (mr.system_db.parent / "system.db-wal").exists()
    assert not (mr.system_db.parent / "system.db-shm").exists()


def test_recover_fails_closed_when_active_service_detected() -> None:
    """When EverOS server service is active, recover fails closed with exit code 1."""
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
        assert "Please stop the EverOS service before recovery." in result.output


def test_recover_fails_closed_when_process_lock_held() -> None:
    """When another process holds memory_root_lock, recover fails closed."""
    with (
        patch(
            "everos.entrypoints.cli.commands.profile.is_everos_service_active",
            return_value=False,
        ),
        patch(
            "everos.entrypoints.cli.commands.profile.memory_root_lock",
            side_effect=LockError("lock already held"),
        ),
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
        assert "Error: EverOS server is currently active" in result.output or (
            "lock" in result.output
        )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX fcntl subprocess locking test",
)
def test_linux_real_subprocess_lock_refusal(tmp_path: Path) -> None:
    """On Linux, verifies CLI refusal when subprocess holds fcntl lock."""
    mr = MemoryRoot(root=tmp_path)
    lock_file = mr.lock_file
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    # Spawn worker holding fcntl lock
    code = f"""
import fcntl, time, os
fd = os.open("{lock_file}", os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
print("LOCKED", flush=True)
time.sleep(5)
"""
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    try:
        # Wait until locked
        assert proc.stdout is not None
        line = proc.stdout.readline().decode().strip()
        assert line == "LOCKED"

        with patch(
            "everos.entrypoints.cli.commands.profile.MemoryRoot.default",
            return_value=mr,
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
            assert "lock" in result.output or "active" in result.output
    finally:
        proc.kill()
        proc.wait()


def test_recover_execution_commits_watermark_when_lock_acquired() -> None:
    existing_fm = UserProfileFrontmatter(
        id="profile_user",
        user_id="user",
        summary="User profile summary",
        profile_timestamp_ms=1000,
    )
    mock_cells = [
        _sample_memcell("mc_1", 1500),
        _sample_memcell("mc_2", 2000),
    ]
    new_profile = AlgoProfile(
        owner_id="user",
        summary="Updated profile",
        timestamp=2000,
    )

    with (
        patch(
            "everos.entrypoints.cli.commands.profile.is_everos_service_active",
            return_value=False,
        ),
        patch(
            "everos.entrypoints.cli.commands.profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.entrypoints.cli.commands.profile.get_unprocessed_memcells_for_owner",
            return_value=(
                [("mc_1", mock_cells[0]), ("mc_2", mock_cells[1])],
                0,
                0,
            ),
        ),
        patch(
            "everos.entrypoints.cli.commands.profile.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.entrypoints.cli.commands.profile.ProfileExtractor"
        ) as mock_extractor_cls,
        patch(
            "everos.entrypoints.cli.commands.profile._persist_profile",
            new=AsyncMock(),
        ) as mock_persist,
    ):
        mock_reader_cls.return_value.read = AsyncMock(return_value=[existing_fm])
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)

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
        assert result.exit_code == 0
        assert "Starting Profile Recovery for 'user'" in result.stdout
        assert "Recovery Complete" in result.stdout
        assert mock_persist.call_count >= 1


def test_recover_rejects_prompt_budget_above_hard_limit() -> None:
    """P1: CLI rejects --max-prompt-chars above hard 40000 limit with exit code 1."""
    # 1. 40001 must be rejected
    res_40001 = CliRunner().invoke(
        profile_mod.app,
        [
            "recover",
            "--owner-id",
            "user",
            "--app-id",
            "codex",
            "--project-id",
            "solo",
            "--max-prompt-chars",
            "40001",
        ],
    )
    assert res_40001.exit_code == 1
    assert "exceeds hard safety limit of 40000" in res_40001.output

    # 2. 50000 must be rejected
    res_50000 = CliRunner().invoke(
        profile_mod.app,
        [
            "recover",
            "--owner-id",
            "user",
            "--app-id",
            "codex",
            "--project-id",
            "solo",
            "--max-prompt-chars",
            "50000",
        ],
    )
    assert res_50000.exit_code == 1
    assert "exceeds hard safety limit of 40000" in res_50000.output

    # 3. 100000 must be rejected
    res_100000 = CliRunner().invoke(
        profile_mod.app,
        [
            "recover",
            "--owner-id",
            "user",
            "--app-id",
            "codex",
            "--project-id",
            "solo",
            "--max-prompt-chars",
            "100000",
        ],
    )
    assert res_100000.exit_code == 1
    assert "exceeds hard safety limit of 40000" in res_100000.output

    # 4. BoundedProfileLLMClient directly raises ValueError for >40000
    from everos.memory.strategies.extract_user_profile import (
        BoundedProfileLLMClient,
    )

    with pytest.raises(ValueError, match="cannot exceed hard safety limit of 40000"):
        BoundedProfileLLMClient(object(), max_prompt_chars=40001)

    # Valid <= 40000 values succeed
    client_40k = BoundedProfileLLMClient(object(), max_prompt_chars=40000)
    assert client_40k._max_prompt_chars == 40000

    client_30k = BoundedProfileLLMClient(object(), max_prompt_chars=30000)
    assert client_30k._max_prompt_chars == 30000
