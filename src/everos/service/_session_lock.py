"""Per-session process + cross-process lock for memorize() calls.

Two concurrent ``POST /add`` (or ``/flush``) calls on the **same**
``session_id`` race on the unprocessed_buffer:

1. Both read ``unprocessed_buffer`` for the session and see the same
   pre-existing rows.
2. Both run boundary detection independently against their own merged
   slice (each task only sees its own newly-arrived messages plus the
   shared pre-existing buffer rows — neither sees the other's messages).
3. Both call ``_replace_buffer(session_id, tail)`` — the later write
   silently overwrites the earlier write's tail and the earlier task's
   tail messages are lost forever (they never made it into any memcell
   either, since each task's boundary call only saw its own slice).

This module serialises memorize() at the ``session_id`` granularity so
the read-merge-boundary-write cycle is atomic per session.

The asyncio lock covers tasks in one process.  A hashed portalocker anchor
under the memory root covers multiple workers and is released by the OS if a
process crashes.  Cross-session calls remain fully parallel.

Wrap acquire + work in ``asyncio.timeout(...)`` (see
``MemorizeSettings.session_lock_timeout_seconds``) so a hung LLM cannot
hold the lock forever — on timeout the task is cancelled and
``async with`` releases the lock automatically.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO

import anyio
import portalocker

from everos.core.persistence import MemoryRoot

# Plain dict (not WeakValueDictionary): a Lock with pending waiters must
# outlive the dict entry, otherwise GC racing with waiters can drop a
# lock mid-flight (CPython bpo-28427). Same rationale as
# ``everos.core.persistence.markdown.writer.MarkdownWriter._path_locks``.
_session_locks: dict[tuple[str, str, str], asyncio.Lock] = {}


def get_session_lock(
    session_id: str,
    *,
    app_id: str = "default",
    project_id: str = "default",
) -> asyncio.Lock:
    """Return the per-scope session lock; create on first use.

    ``dict.setdefault`` is atomic under single-threaded asyncio (no GIL
    release between the get and the insert), so no meta-lock is needed
    around the registry.
    """
    return _session_locks.setdefault((app_id, project_id, session_id), asyncio.Lock())


@asynccontextmanager
async def scoped_session_lock(
    memory_root: MemoryRoot,
    session_id: str,
    *,
    app_id: str = "default",
    project_id: str = "default",
) -> AsyncIterator[None]:
    """Serialise one scoped session across tasks and service workers."""

    local = get_session_lock(session_id, app_id=app_id, project_id=project_id)
    digest = hashlib.sha256(
        f"{app_id}\0{project_id}\0{session_id}".encode()
    ).hexdigest()
    lock_path = (
        memory_root.root / ".index" / "locks" / "memorize-session" / f"{digest}.lock"
    )

    async with local:
        await anyio.Path(lock_path.parent).mkdir(parents=True, exist_ok=True)
        handle: IO[str] = await anyio.to_thread.run_sync(
            lambda: open(Path(lock_path), "a+", encoding="utf-8")  # noqa: SIM115
        )
        acquired = False
        try:
            while not acquired:
                try:
                    await anyio.to_thread.run_sync(
                        portalocker.lock,
                        handle,
                        portalocker.LOCK_EX | portalocker.LOCK_NB,
                    )
                    acquired = True
                except portalocker.LockException:
                    await anyio.sleep(0.05)
            yield
        finally:
            try:
                if acquired:
                    await anyio.to_thread.run_sync(portalocker.unlock, handle)
            finally:
                await anyio.to_thread.run_sync(handle.close)


def _reset_for_tests() -> None:
    """Test-only: drop all registered locks.

    Used by integration test fixtures that rebuild memorize singletons
    against a fresh tmp memory_root; ensures no stale lock state leaks
    across tests.
    """
    _session_locks.clear()
