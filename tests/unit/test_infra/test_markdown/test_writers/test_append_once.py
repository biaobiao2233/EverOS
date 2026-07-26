"""Natural-key durability tests for derived-memory daily logs."""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path

import pytest

from everos.core.persistence import MarkdownReader, MemoryRoot
from everos.infra.persistence.markdown import AtomicFactWriter, EpisodeWriter
from everos.infra.persistence.markdown.writers.base import DailyLogCorruptionError


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    root = MemoryRoot(tmp_path)
    root.ensure()
    return root


def _inline(parent_id: str, *, owner_id: str = "u1") -> dict[str, object]:
    return {
        "owner_id": owner_id,
        "session_id": "s1",
        "timestamp": "2026-05-15T10:00:00+00:00",
        "parent_type": "memcell",
        "parent_id": parent_id,
    }


async def test_single_retry_returns_original_id_and_content(
    memory_root: MemoryRoot,
) -> None:
    writer = EpisodeWriter(memory_root)
    bucket = dt.date(2026, 5, 15)
    first = await writer.append_entry_once(
        "u1",
        parent_id="mc_once",
        inline=_inline("mc_once"),
        sections={"Content": "first output wins"},
        date=bucket,
    )
    replay = await writer.append_entry_once(
        "u1",
        parent_id="mc_once",
        inline=_inline("mc_once"),
        sections={"Content": "drifted retry"},
        date=bucket,
    )

    assert first.created is True
    assert replay.created is False
    assert replay.entry_ids == first.entry_ids
    assert replay.entries[0].sections["Content"] == "first output wins"
    parsed = await MarkdownReader.read(writer.path_for("u1", bucket))
    assert len(parsed.entries) == 1


async def test_batch_retry_is_one_append_once_unit(memory_root: MemoryRoot) -> None:
    writer = AtomicFactWriter(memory_root)
    bucket = dt.date(2026, 5, 15)
    original = [(_inline("mc_batch"), {"Fact": f"fact {idx}"}) for idx in range(3)]
    first = await writer.append_entries_once(
        "u1", original, parent_id="mc_batch", date=bucket
    )
    replay = await writer.append_entries_once(
        "u1",
        [(_inline("mc_batch"), {"Fact": "different count and content"})],
        parent_id="mc_batch",
        date=bucket,
    )

    assert first.created is True
    assert replay.created is False
    assert replay.entry_ids == first.entry_ids
    assert [item.sections["Fact"] for item in replay.entries] == [
        "fact 0",
        "fact 1",
        "fact 2",
    ]


async def test_legacy_processing_date_is_found_before_source_date_write(
    memory_root: MemoryRoot,
) -> None:
    writer = EpisodeWriter(memory_root)
    legacy_date = dt.date(2026, 7, 13)
    source_date = dt.date(2025, 2, 1)
    legacy_id = await writer.append_entry(
        "u1",
        inline=_inline("mc_legacy"),
        sections={"Content": "legacy processing-day output"},
        date=legacy_date,
    )

    replay = await writer.append_entry_once(
        "u1",
        parent_id="mc_legacy",
        inline=_inline("mc_legacy"),
        sections={"Content": "must not be appended"},
        date=source_date,
    )

    assert replay.created is False
    assert replay.entry_ids == (legacy_id,)
    assert not writer.path_for("u1", source_date).exists()


async def test_legacy_marker_width_is_preserved_for_downstream_references(
    memory_root: MemoryRoot,
) -> None:
    writer = EpisodeWriter(memory_root)
    legacy_date = dt.date(2026, 7, 13)
    source_date = dt.date(2025, 2, 1)
    generated = await writer.append_entry(
        "u1",
        inline=_inline("mc_short_marker"),
        sections={"Content": "legacy short marker"},
        date=legacy_date,
    )
    path = writer.path_for("u1", legacy_date)
    short_marker = "ep_20260713_0001"
    path.write_text(
        path.read_text("utf-8").replace(generated.format(), short_marker),
        "utf-8",
    )

    replay = await writer.append_entry_once(
        "u1",
        parent_id="mc_short_marker",
        inline=_inline("mc_short_marker"),
        sections={"Content": "must not be appended"},
        date=source_date,
    )

    assert replay.created is False
    assert replay.entries[0].marker_id == short_marker
    assert replay.entries[0].entry_id.format() != short_marker


async def test_same_parent_in_multiple_daily_files_fails_closed(
    memory_root: MemoryRoot,
) -> None:
    writer = AtomicFactWriter(memory_root)
    parent_id = "mc_cross_date_duplicate"
    for bucket in (dt.date(2026, 7, 12), dt.date(2026, 7, 13)):
        await writer.append_entry(
            "u1",
            inline=_inline(parent_id),
            sections={"Fact": f"duplicate in {bucket}"},
            date=bucket,
        )

    with pytest.raises(DailyLogCorruptionError, match="multiple daily files"):
        await writer.append_entries_once(
            "u1",
            [(_inline(parent_id), {"Fact": "must not be appended"})],
            parent_id=parent_id,
            date=dt.date(2025, 2, 1),
        )


async def test_two_writer_instances_converge_on_one_entry(
    memory_root: MemoryRoot,
) -> None:
    bucket = dt.date(2026, 5, 15)
    writers = [EpisodeWriter(memory_root), EpisodeWriter(memory_root)]

    async def invoke(index: int):
        writer = writers[index % 2]
        return await writer.append_entry_once(
            "u1",
            parent_id="mc_concurrent",
            inline=_inline("mc_concurrent"),
            sections={"Content": f"proposal {index}"},
            date=bucket,
        )

    results = await asyncio.gather(*(invoke(i) for i in range(20)))
    assert sum(result.created for result in results) == 1
    assert len({result.entry_ids for result in results}) == 1
    parsed = await MarkdownReader.read(writers[0].path_for("u1", bucket))
    assert len(parsed.entries) == 1


async def test_malformed_marker_fails_closed(memory_root: MemoryRoot) -> None:
    writer = EpisodeWriter(memory_root)
    bucket = dt.date(2026, 5, 15)
    await writer.append_entry_once(
        "u1",
        parent_id="mc_good",
        inline=_inline("mc_good"),
        sections={"Content": "good"},
        date=bucket,
    )
    path = writer.path_for("u1", bucket)
    path.write_text(path.read_text("utf-8") + "<!-- entry:broken -->\n", "utf-8")

    with pytest.raises(DailyLogCorruptionError, match="malformed entry markers"):
        await writer.append_entry_once(
            "u1",
            parent_id="mc_new",
            inline=_inline("mc_new"),
            sections={"Content": "must not write"},
            date=bucket,
        )
