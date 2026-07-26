"""Filesystem watcher — emits cascade enqueue events on md changes.

watchdog 6's :class:`Observer` runs in its own native thread; the
event handler callback fires there too. We bridge those events back
onto the orchestrator's asyncio loop via
:func:`asyncio.run_coroutine_threadsafe` so every state-table write
goes through the same async repo as the scanner / CLI sync paths.

The handler is intentionally cheap: pattern-match the path against
the kind registry, then enqueue. The watcher does **not** read the
file content — that's the worker's job after :meth:`claim_one`.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from watchdog.events import FileMovedEvent, FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.sqlite import md_change_state_repo

from .registry import KindSpec, match_kind

logger = get_logger(__name__)


class CascadeWatcher:
    """Bridge watchdog → md_change_state for the configured memory root.

    The watchdog observer is started on :meth:`start` and stopped on
    :meth:`stop`. Events outside the registered kind paths are silently
    ignored — DD-7 (single whitelist layer) keeps the watcher free of
    bespoke exclusion rules.
    """

    def __init__(
        self,
        memory_root: MemoryRoot,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._memory_root = memory_root
        self._loop = loop
        self._observer = Observer()
        self._handler = _Handler(memory_root, loop)
        self._root_handler = _RootHandler(
            on_add=self._schedule_app_root,
            on_remove=self._forget_app_root,
        )
        self._watch_lock = threading.Lock()
        self._app_watches: dict[Path, Any] = {}
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        # The memory root is created lazily by other layers; watchdog
        # rejects non-existent paths so we ensure it exists here.
        self._memory_root.ensure()
        # Watch only direct children at the memory root. New app directories
        # get their own recursive watch, while hidden .index/LanceDB trees
        # never consume an inotify watch.
        self._observer.schedule(
            self._root_handler,
            str(self._memory_root.root),
            recursive=False,
        )
        watch_roots = _watch_roots(self._memory_root.root)
        for watch_root in watch_roots:
            self._schedule_app_root(watch_root, catch_up=False)
        self._observer.start()
        self._started = True
        logger.info(
            "cascade_watcher_started",
            root=str(self._memory_root.root),
            watched_app_roots=len(watch_roots),
        )

    def stop(self) -> None:
        if not self._started:
            return
        self._observer.stop()
        self._observer.join(timeout=5)
        self._started = False
        logger.info("cascade_watcher_stopped")

    def _schedule_app_root(
        self,
        raw_path: str | Path,
        *,
        catch_up: bool = True,
    ) -> None:
        """Recursively watch one direct app directory and close the create race."""

        path = Path(raw_path).resolve()
        if not _is_app_root(self._memory_root.root, path) or not path.is_dir():
            return
        with self._watch_lock:
            if path in self._app_watches:
                return
            try:
                watch = self._observer.schedule(
                    self._handler,
                    str(path),
                    recursive=True,
                )
            except OSError:
                # The directory may have been moved/deleted between the root
                # event and schedule(). The periodic scanner remains a
                # correctness fallback.
                return
            self._app_watches[path] = watch
        if catch_up:
            # Files can be created between mkdir(app) and schedule(). The
            # recursive watch is active first; this scan repairs that window.
            self._handler.enqueue_existing_tree(path)

    def _forget_app_root(self, raw_path: str | Path) -> None:
        """Drop bookkeeping when a direct app directory leaves the root."""

        path = Path(raw_path).resolve()
        if not _is_app_root(self._memory_root.root, path):
            return
        with self._watch_lock:
            watch = self._app_watches.pop(path, None)
        if watch is not None:
            with suppress(KeyError):
                self._observer.unschedule(watch)


class _Handler(FileSystemEventHandler):
    """Watchdog callback — fires in the watchdog thread."""

    def __init__(
        self, memory_root: MemoryRoot, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._memory_root = memory_root
        self._loop = loop

    def on_created(self, event: FileSystemEvent) -> None:
        self._enqueue(event.src_path, "added")

    def on_modified(self, event: FileSystemEvent) -> None:
        self._enqueue(event.src_path, "modified")

    def on_deleted(self, event: FileSystemEvent) -> None:
        # macOS FSEvents fires a synthetic deletion for the OLD inode
        # whenever ``os.replace`` overwrites an existing file — the path
        # itself is still present, now pointing at the new inode, and the
        # paired ``on_moved`` has already enqueued the dest as 'added'.
        # Propagating this false-positive 'deleted' drives the worker to
        # call ``delete_by_md_path`` and wipe LanceDB while md is fine.
        # The stat is on the watcher thread but cheap on APFS (~µs);
        # real unlinks still surface because the path is truly gone.
        if Path(event.src_path).exists():
            return
        self._enqueue(event.src_path, "deleted")

    def on_moved(self, event: FileSystemEvent) -> None:
        # A rename emits both a `moved` for the src and effectively a
        # `created` for the dest. We materialise both sides so the
        # state table tracks the source as deleted and the destination
        # as added.
        #
        # Symmetric to ``on_deleted``: stat src first. If the path still
        # exists (e.g. macOS reports a synthetic move for the old inode
        # of an atomic-replace pair, or a hardlink survives the rename
        # so the named path is still bound), the 'deleted' enqueue
        # would wipe LanceDB while the file is intact. Real renames
        # (src genuinely gone, dest the new home) keep both legs.
        if not Path(event.src_path).exists():
            self._enqueue(event.src_path, "deleted")
        if isinstance(event, FileMovedEvent):
            self._enqueue(event.dest_path, "added")

    def enqueue_existing_tree(self, root: Path) -> None:
        """Enqueue markdown already present when a new app watch is attached."""

        try:
            paths = sorted(root.rglob("*.md"))
        except OSError:
            return
        for path in paths:
            if path.is_file():
                self._enqueue(str(path), "added")

    def _enqueue(self, raw_path: str, change_type: str) -> None:
        rel = _relative_to_root(self._memory_root.root, raw_path)
        if rel is None:
            return
        spec = match_kind(rel)
        if spec is None:
            return
        mtime = _safe_mtime(raw_path)
        asyncio.run_coroutine_threadsafe(
            _enqueue_async(spec, rel, change_type, mtime),
            self._loop,
        )


class _RootHandler(FileSystemEventHandler):
    """Maintain recursive watches for direct app directories only."""

    def __init__(
        self,
        *,
        on_add: Callable[[str], None],
        on_remove: Callable[[str], None],
    ) -> None:
        self._on_add = on_add
        self._on_remove = on_remove

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            self._on_add(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            self._on_remove(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            return
        self._on_remove(event.src_path)
        if isinstance(event, FileMovedEvent):
            self._on_add(event.dest_path)


async def _enqueue_async(
    spec: KindSpec, rel: str, change_type: str, mtime: float
) -> None:
    """Coroutine variant — runs on the orchestrator's event loop."""
    try:
        await md_change_state_repo.upsert(
            rel,
            kind=spec.name,
            change_type=change_type,
            mtime=mtime,
        )
    except Exception as exc:  # noqa: BLE001 — defensive: never crash watcher
        logger.warning(
            "cascade_watcher_upsert_failed",
            md_path=rel,
            kind=spec.name,
            error=str(exc),
        )


def _relative_to_root(root: Path, raw: str) -> str | None:
    """Return ``raw`` relative to ``root`` using POSIX separators.

    ``None`` when the path is outside the memory root (defensive — the
    watcher only watches inside ``root``, but external symlinks could
    surface).
    """
    try:
        rel = Path(raw).resolve().relative_to(root)
    except ValueError:
        return None
    return rel.as_posix()


def _safe_mtime(raw: str) -> float:
    """Return mtime in seconds, falling back to 0.0 on stat failure."""
    try:
        return Path(raw).stat().st_mtime
    except OSError:
        return 0.0


def _watch_roots(root: Path) -> list[Path]:
    """Return app roots that may contain memory markdown.

    Never recursively watch the hidden .index storage tree: LanceDB can create
    hundreds of thousands of UUID directories and exhaust Linux inotify before
    the service starts. A shallow root watch adds new app roots dynamically.
    """
    ignored = {"lancedb"}
    try:
        children = root.iterdir()
    except OSError:
        return []
    return sorted(
        (
            child
            for child in children
            if child.is_dir() and _is_app_root(root, child, ignored=ignored)
        ),
        key=lambda path: path.name,
    )


def _is_app_root(
    root: Path,
    candidate: Path,
    *,
    ignored: set[str] | None = None,
) -> bool:
    """Return whether ``candidate`` is an allowed direct app directory."""

    ignored = ignored or {"lancedb"}
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    return (
        candidate_resolved.parent == root_resolved
        and not candidate_resolved.name.startswith(".")
        and candidate_resolved.name not in ignored
    )
