"""LanceDB stall-hardening: deadlines, mutex, rebuild-in-place, husk sweep.

Covers the repo-layer guarantees the cascade daemon leans on:

- every read / write / maintenance critical section is bounded; expiry
  raises :class:`VectorStoreBusyError` (retryable) instead of hanging;
- the per-table write lock covers acquisition *and* body, so neither
  waiting nor holding can be unbounded (optimize / prune / rebuild /
  row-writes mutually exclude);
- ``rebuild_indexes`` replaces indices **in place** (lancedb 0.30.2
  supports ``create_index(replace=True)``) and only drops indices whose
  columns left the BM25 set;
- ``prune`` passes ``cleanup_older_than`` + ``delete_unverified=False``
  (cross-process safe) and sweeps empty index-dir husks outside the
  lock;
- the husk sweep can never lose data (rmdir refuses non-empty dirs).

Deadline constants are monkey-patched small so tests stay fast; the
mechanism under test (``asyncio.timeout`` wrapping acquisition + body)
is independent of the budget size.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import time
from pathlib import Path
from typing import ClassVar

import pytest

from everos.config import LanceDBSettings
from everos.core.errors import VectorStoreBusyError
from everos.core.persistence import (
    BaseLanceTable,
    MemoryRoot,
    Vector,
    open_lancedb_connection,
)
from everos.core.persistence.lancedb import LanceRepoBase
from everos.core.persistence.lancedb import (
    repository as repository_mod,
)


class _Note(BaseLanceTable):
    TABLE_NAME: ClassVar[str] = "_hardening_note"

    id: str
    md_path: str
    text: str
    vector: Vector(4)  # type: ignore[valid-type]


class _SearchNote(BaseLanceTable):
    """Schema declaring one BM25 column — exercises FTS index setup."""

    TABLE_NAME: ClassVar[str] = "_hardening_search_note"
    BM25_FIELDS: ClassVar[list[str]] = ["tokens"]

    id: str
    tokens: str


class _SearchRepo(LanceRepoBase[_SearchNote]):
    schema = _SearchNote


class _NoteRepo(LanceRepoBase[_Note]):
    schema = _Note


@pytest.fixture(autouse=True)
def _reset_write_locks() -> None:
    LanceRepoBase._reset_locks_for_tests()


# ── Stub tables (deadline + rebuild mechanics, no real lancedb IO) ─────────


class _IndexCfg:
    """Minimal stand-in for lancedb's ``IndexConfig``."""

    def __init__(self, name: str, columns: list[str]) -> None:
        self.name = name
        self.columns = columns
        self.index_type = "FTS"


class _HangingTable:
    """Every method hangs forever — deadline tests cancel them."""

    def __init__(self, *, method: str) -> None:
        self._method = method
        self.calls: list[tuple[object, ...]] = []

    async def _hang(self, *args: object) -> None:
        self.calls.append(args)
        await asyncio.sleep(600)

    def __getattr__(self, name: str):  # noqa: ANN401 — dynamic stub
        if name.startswith("_"):
            raise AttributeError(name)

        async def hang(*args: object, **kwargs: object) -> object:
            if name == self._method or self._method == "*":
                await self._hang(name, *args)
            raise AssertionError(f"unexpected call to {name}")

        return hang


class _RecordingTable:
    """Records drop/create/list calls for rebuild-mechanics assertions."""

    def __init__(self, indices: list[_IndexCfg]) -> None:
        self._indices = indices
        self.created: list[dict[str, object]] = []
        self.dropped: list[str] = []
        self.hang_create = False

    async def list_indices(self) -> list[_IndexCfg]:
        return list(self._indices)

    async def drop_index(self, name: str) -> None:
        self.dropped.append(name)

    async def create_index(
        self,
        column: str | None = None,
        *,
        replace: bool | None = None,
        config: object = None,
    ) -> None:
        if self.hang_create:
            await asyncio.sleep(600)
        self.created.append({"column": column, "replace": replace})

    # Reads used incidentally by the chassis.
    async def uri(self) -> str:
        return "file:///nonexistent-table-uri"


# ── Deadlines fire ──────────────────────────────────────────────────────────


async def test_write_deadline_fires_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repository_mod, "_WRITE_TIMEOUT_SECONDS", 0.05)
    repo = _NoteRepo(table=_HangingTable(method="add"))
    start = time.monotonic()
    with pytest.raises(VectorStoreBusyError):
        await repo.add([_Note(id="a", md_path="a.md", text="x", vector=[1.0] * 4)])
    elapsed = time.monotonic() - start
    assert elapsed < 3.0, f"deadline must fire promptly, took {elapsed:.2f}s"

    # The lock must be free again: a fast write succeeds afterwards.
    class _FastTable:
        async def add(self, records: object) -> None:
            return None

    repo._table_override = _FastTable()
    await asyncio.wait_for(repo.add([]), timeout=2.0)


async def test_read_deadline_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repository_mod, "_READ_TIMEOUT_SECONDS", 0.05)
    repo = _NoteRepo(table=_HangingTable(method="*"))
    with pytest.raises(VectorStoreBusyError):
        await asyncio.wait_for(repo.count(), timeout=3.0)


async def test_prune_passes_write_locked_cleanup_and_delete_unverified_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Heavy beat contract: cleanup arg + cross-process-safe flag, under lock."""
    seen: dict[str, object] = {}

    class _PruneTable(_HangingTable):
        async def optimize(self, *, cleanup_older_than=None, delete_unverified=None):  # type: ignore[no-untyped-def]
            seen["cleanup"] = cleanup_older_than
            seen["delete_unverified"] = delete_unverified

        async def uri(self) -> str:
            return "file:///nonexistent-table-uri"

    monkeypatch.setattr(repository_mod, "_PRUNE_TIMEOUT_SECONDS", 0.05)
    repo = _NoteRepo(table=_PruneTable(method="__none__"))
    retention = dt.timedelta(seconds=42)
    await asyncio.wait_for(repo.prune(retention), timeout=3.0)
    assert seen["cleanup"] == retention
    assert seen["delete_unverified"] is False


async def test_prune_deadline_fires_when_cleanup_hangs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repository_mod, "_PRUNE_TIMEOUT_SECONDS", 0.05)
    repo = _NoteRepo(table=_HangingTable(method="optimize"))

    class _HangOptimize(_HangingTable):
        pass

    table = _HangingTable(method="optimize")

    async def optimize(*args: object, **kwargs: object) -> None:
        await asyncio.sleep(600)

    table.optimize = optimize  # type: ignore[method-assign]
    repo._table_override = table
    with pytest.raises(VectorStoreBusyError):
        await asyncio.wait_for(repo.prune(dt.timedelta(seconds=60)), timeout=3.0)


# ── Mutual exclusion (optimize / prune / rebuild / writes) ─────────────────


async def test_rebuild_waits_bounded_behind_held_write_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged lock holder cannot park a rebuild forever."""
    monkeypatch.setattr(repository_mod, "_REBUILD_TIMEOUT_SECONDS", 0.05)
    table = _RecordingTable([])
    repo = _SearchRepo(table=table)
    lock = LanceRepoBase._write_lock(_SearchNote.TABLE_NAME)
    await lock.acquire()
    try:
        with pytest.raises(VectorStoreBusyError):
            await asyncio.wait_for(repo.rebuild_indexes(), timeout=3.0)
    finally:
        lock.release()


async def test_upsert_waits_bounded_behind_in_flight_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optimize-vs-rebuild style mutual exclusion: while a rebuild holds the
    per-table lock, a concurrent write fails fast (bounded) instead of
    racing on the manifest or hanging."""
    monkeypatch.setattr(repository_mod, "_WRITE_TIMEOUT_SECONDS", 0.05)
    table = _RecordingTable([])
    table.hang_create = True  # rebuild will hold the lock until cancelled
    repo = _SearchRepo(table=table)
    lock = LanceRepoBase._write_lock(_SearchNote.TABLE_NAME)

    rebuild_task = asyncio.create_task(repo.rebuild_indexes())
    await asyncio.sleep(0.02)  # let the rebuild acquire the lock
    assert lock.locked()

    with pytest.raises(VectorStoreBusyError):
        await asyncio.wait_for(repo.upsert([]), timeout=3.0)
    rebuild_task.cancel()
    with contextlib_suppress():
        await rebuild_task


def contextlib_suppress():  # noqa: ANN201 — local helper
    import contextlib

    return contextlib.suppress(asyncio.CancelledError)


# ── Rebuild in place (replace=True) ────────────────────────────────────────


async def test_rebuild_replaces_indexes_without_drop_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BM25 columns are rebuilt via ``create_index(replace=True)`` — never
    dropped first (an FTS query in a no-index window raises, not degrades).
    Only indices whose columns left the BM25 set get dropped."""
    monkeypatch.setattr(repository_mod, "_REBUILD_TIMEOUT_SECONDS", 5.0)
    table = _RecordingTable(
        [
            _IndexCfg("tokens_idx", ["tokens"]),  # still wanted
            _IndexCfg("legacy_idx", ["obsolete_col"]),  # no longer indexed
        ]
    )
    repo = _SearchRepo(table=table)
    await asyncio.wait_for(repo.rebuild_indexes(), timeout=10.0)

    assert table.dropped == ["legacy_idx"], "only unqueried leftovers may be dropped"
    assert len(table.created) == 1
    assert table.created[0]["column"] == "tokens"
    assert table.created[0]["replace"] is True, (
        "in-place replace keeps FTS available throughout"
    )


# ── Husk sweep ──────────────────────────────────────────────────────────────


def test_husk_sweep_removes_only_old_empty_dirs(tmp_path: Path) -> None:
    from everos.core.persistence.lancedb.repository import _remove_empty_index_dirs

    table_dir = tmp_path / "table"
    table_dir.mkdir()
    indices = table_dir / "_indices"
    old_empty = indices / "uuid-old-empty"
    fresh_empty = indices / "uuid-fresh-empty"
    old_nonempty = indices / "uuid-old-nonempty"
    for d in (old_empty, fresh_empty, old_nonempty):
        d.mkdir(parents=True)
    (old_nonempty / "data.lance").write_bytes(b"x")
    (indices / "afile.txt").write_text("not a dir")
    long_ago = time.time() - 14 * 24 * 3600
    os.utime(old_empty, (long_ago, long_ago))
    os.utime(old_nonempty, (long_ago, long_ago))

    removed = _remove_empty_index_dirs(str(table_dir), min_age_seconds=7 * 24 * 3600.0)

    assert removed == 1
    assert not old_empty.exists(), "old empty husk reclaimed"
    assert fresh_empty.exists(), "fresh dir may be a build in progress"
    assert old_nonempty.exists(), "rmdir must refuse non-empty dirs"


async def test_prune_husk_sweep_failure_is_swallowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cleanup commit already succeeded; a sweep timeout must not escape
    prune() (it would corrupt the failure ledger of the maintenance beat)."""
    monkeypatch.setattr(repository_mod, "_HUSK_SWEEP_TIMEOUT_SECONDS", 0.05)

    import time as _time

    def slow_walk(*args: object, **kwargs: object) -> int:
        _time.sleep(1.0)  # simulates a wedged filesystem walk
        return 0

    monkeypatch.setattr(repository_mod, "_remove_empty_index_dirs", slow_walk)

    class _CalmTable:
        async def optimize(self, *, cleanup_older_than=None, delete_unverified=None):  # type: ignore[no-untyped-def]
            return None

        async def uri(self) -> str:
            return str(tmp_path)

    repo = _NoteRepo(table=_CalmTable())
    started = time.monotonic()
    await asyncio.wait_for(repo.prune(dt.timedelta(seconds=1)), timeout=10.0)
    assert time.monotonic() - started < 5.0, "sweep timeout must bound the walk"


# ── Real-lancedb behavioural sanity ────────────────────────────────────────


async def test_prune_on_real_table_preserves_rows(tmp_path: Path) -> None:
    """End-to-end on a real table: prune commits cleanly and never loses
    live rows (cleanup only touches superseded versions)."""
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    table = await conn.create_table("_hardening_real", schema=_Note)

    repo = _NoteRepo(table=table)
    rows = [
        _Note(id=f"i{n}", md_path=f"u/{n}.md", text="t", vector=[float(n)] * 4)
        for n in range(5)
    ]
    await repo.add(rows)
    # Retention ~0: everything replaced before "now" is reclaimable; the
    # live rows themselves must survive.
    await asyncio.wait_for(repo.prune(dt.timedelta(seconds=0.001)), timeout=30.0)
    assert await repo.count() == 5
