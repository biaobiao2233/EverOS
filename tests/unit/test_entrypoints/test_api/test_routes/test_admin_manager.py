"""Focused Linux unit tests for the private admin filesystem/status manager."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

try:
    import fcntl  # noqa: F401
except ImportError:
    pytest.skip(
        "EverOS production persistence requires POSIX fcntl",
        allow_module_level=True,
    )

from everos.config import load_settings
from everos.infra.persistence.sqlite import md_change_state_repo
from everos.infra.persistence.sqlite.repos.md_change_state import QueueSummary
from everos.memory.admin.dto import AdminMemoryKind
from everos.memory.admin.manager import (
    AdminFileNotFoundError,
    AdminFileTooLargeError,
    AdminInvalidPathError,
    _is_browsable_relative_path,
    _kind_for,
    _list_files_sync,
    _read_file_sync,
    _resolve_requested_file,
    get_pipeline_status,
)


@pytest.fixture
def memory_root(tmp_path: Path) -> Path:
    root = tmp_path / "memory-root"
    root.mkdir()
    return root


def test_kind_classification_and_system_directory_exclusion() -> None:
    assert _kind_for(PurePosixPath("episodes/ep.md")) == AdminMemoryKind.EPISODE
    assert _kind_for(PurePosixPath("user.md")) == AdminMemoryKind.PROFILE
    assert (
        _kind_for(PurePosixPath(".atomic_facts/fact.md")) == AdminMemoryKind.ATOMIC_FACT
    )
    assert _kind_for(PurePosixPath(".foresights/item.md")) == AdminMemoryKind.FORESIGHT
    assert _kind_for(PurePosixPath(".cases/case.md")) == AdminMemoryKind.AGENT_CASE
    assert (
        _kind_for(PurePosixPath("skills/example/SKILL.md"))
        == AdminMemoryKind.AGENT_SKILL
    )
    assert _kind_for(PurePosixPath("notes.md")) == AdminMemoryKind.OTHER
    assert _is_browsable_relative_path(PurePosixPath("episodes/ep.md"))
    assert not _is_browsable_relative_path(PurePosixPath(".index/schema.md"))
    assert not _is_browsable_relative_path(PurePosixPath(".tmp/scratch.md"))


def test_list_is_sorted_filtered_paginated_and_metadata_only(
    memory_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episodes = memory_root / "episodes"
    episodes.mkdir()
    (episodes / "b.md").write_text("b", encoding="utf-8")
    (episodes / "a.md").write_text("a", encoding="utf-8")
    (memory_root / "user.md").write_text("profile", encoding="utf-8")
    (memory_root / "ignored.txt").write_text("ignored", encoding="utf-8")
    hidden = memory_root / ".index"
    hidden.mkdir()
    (hidden / "schema.md").write_text("hidden", encoding="utf-8")

    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: pytest.fail("listing read file content"),
    )
    response = _list_files_sync(
        memory_root,
        kind=None,
        query=None,
        page=1,
        page_size=2,
    )
    assert [item.path for item in response.data.items] == [
        "episodes/a.md",
        "episodes/b.md",
    ]
    assert response.data.total_count == 3

    episode_response = _list_files_sync(
        memory_root,
        kind=AdminMemoryKind.EPISODE,
        query="B.MD",
        page=1,
        page_size=50,
    )
    assert [item.path for item in episode_response.data.items] == ["episodes/b.md"]


def test_missing_root_returns_an_explicit_empty_page(tmp_path: Path) -> None:
    response = _list_files_sync(
        tmp_path / "missing",
        kind=None,
        query=None,
        page=3,
        page_size=10,
    )
    assert response.data.items == []
    assert response.data.page == 3
    assert response.data.total_count == 0


@pytest.mark.parametrize(
    "invalid_path",
    [
        "",
        "\x00user.md",
        "../user.md",
        "episodes/../user.md",
        "/etc/passwd.md",
        "//server/share/user.md",
        "C:/user.md",
        "C:user.md",
        r"\\server\share\user.md",
        r"episodes\ep.md",
        ".index/schema.md",
        ".tmp/scratch.md",
        "config.json",
    ],
)
def test_invalid_content_paths_are_rejected(
    memory_root: Path,
    invalid_path: str,
) -> None:
    with pytest.raises(AdminInvalidPathError):
        _resolve_requested_file(memory_root, invalid_path)


def test_literal_percent_filename_round_trips_without_recursive_decoding(
    memory_root: Path,
) -> None:
    percent_file = memory_root / "file%2e.md"
    percent_file.write_text("# percent", encoding="utf-8")

    candidate, relative = _resolve_requested_file(memory_root, "file%2e.md")
    assert candidate == percent_file
    assert relative.as_posix() == "file%2e.md"
    assert _read_file_sync(memory_root, "file%2e.md").data.content == "# percent"

    with pytest.raises(AdminFileNotFoundError):
        _resolve_requested_file(memory_root, "%2e%2e/user.md")


def test_symlink_file_and_symlink_ancestor_are_rejected(
    memory_root: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.md"
    secret.write_text("secret", encoding="utf-8")
    (memory_root / "linked.md").symlink_to(secret)
    (memory_root / "linked-dir").symlink_to(outside, target_is_directory=True)

    with pytest.raises(AdminFileNotFoundError):
        _resolve_requested_file(memory_root, "linked.md")
    with pytest.raises(AdminFileNotFoundError):
        _resolve_requested_file(memory_root, "linked-dir/secret.md")

    listed = _list_files_sync(
        memory_root,
        kind=None,
        query=None,
        page=1,
        page_size=100,
    )
    assert listed.data.items == []


def test_content_read_is_byte_bounded_and_utf8_only(
    memory_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = memory_root / "exact.md"
    exact.write_bytes(b"x" * 1_048_576)
    exact_response = _read_file_sync(memory_root, "exact.md")
    assert exact_response.data.size_bytes == 1_048_576

    oversized = memory_root / "oversized.md"
    oversized.write_bytes(b"x" * 1_048_577)
    with pytest.raises(AdminFileTooLargeError):
        _read_file_sync(memory_root, "oversized.md")

    original_fstat = os.fstat

    def report_stale_small_size(fd: int):
        stat = original_fstat(fd)
        return SimpleNamespace(st_size=1, st_mtime=stat.st_mtime)

    monkeypatch.setattr(os, "fstat", report_stale_small_size)
    with pytest.raises(AdminFileTooLargeError):
        _read_file_sync(memory_root, "oversized.md")

    invalid = memory_root / "invalid.md"
    invalid.write_bytes(b"\x80\x81\xfe\xff")
    with pytest.raises(AdminInvalidPathError):
        _read_file_sync(memory_root, "invalid.md")


async def test_pipeline_status_is_truthful_and_hides_queue_exception(
    memory_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(memory_root))
    load_settings.cache_clear()

    async def broken_queue_summary():
        raise RuntimeError("sensitive sqlite detail")

    monkeypatch.setattr(
        md_change_state_repo,
        "queue_summary",
        broken_queue_summary,
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            lifespan_data={
                "sqlite": object(),
                "cascade": SimpleNamespace(_started=True),
                "lancedb": object(),
            }
        )
    )
    response = await get_pipeline_status(app)
    assert response.data.memory_root.status == "available"
    assert response.data.cascade.running is True
    assert response.data.cascade.queue.status == "unavailable"
    assert response.data.cascade.queue.reason == "sqlite storage is unavailable"
    assert "sensitive" not in response.model_dump_json()


async def test_pipeline_queue_dataclass_and_missing_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(missing))
    load_settings.cache_clear()

    async def queue_summary():
        return QueueSummary(
            pending=2,
            done=3,
            failed_retryable=1,
            failed_permanent=0,
            max_lsn=9,
            last_processed_lsn=8,
        )

    monkeypatch.setattr(md_change_state_repo, "queue_summary", queue_summary)
    app = SimpleNamespace(
        state=SimpleNamespace(
            lifespan_data={
                "sqlite": object(),
                "cascade": SimpleNamespace(_started=False),
            }
        )
    )
    response = await get_pipeline_status(app)
    assert response.data.memory_root.status == "unavailable"
    assert response.data.memory_root.exists is False
    assert response.data.cascade.queue.pending == 2
    assert response.data.cascade.queue.done == 3
    assert response.data.indexing.status == "unavailable"
