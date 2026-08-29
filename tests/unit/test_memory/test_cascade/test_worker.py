"""Tests for :class:`CascadeWorker` retry classification + maintenance scheduler.

The pure-function pieces (registry / reconciler) get coverage in
their own files. Here we focus on the worker's branch behaviour
without touching the real handler / lancedb stack:

- ``RecoverableError`` retries up to ``max_retry`` and then marks
  ``retryable=TRUE``; store-busy deadline errors retry the same way.
- Any other exception marks ``retryable=FALSE`` immediately.
- Successful handler ⇒ ``mark_done``.
- Unknown kind ⇒ ``mark_failed(retryable=False)``.

A second group covers the per-kind throttle + trailing-edge
maintenance scheduler that fires LanceDB beats outside the drain loop
— coalescing under burst writes, re-running when dirty is re-raised
mid-beat, heavy/light beat splitting, benign commit-conflict handling,
and flushing on drain-until-empty / stop.

A third group covers loop supervision (bounded restart with backoff,
budget exhaustion → process-exit request, stable-run reset) and the
in-memory health signals feeding ``/health``.

The repo singleton is monkey-patched onto a recording fake so the
test stays in-memory.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
import unittest.mock as mock
from dataclasses import dataclass

import pytest

from everos.core.errors import VectorStoreBusyError
from everos.memory.cascade import worker as worker_mod
from everos.memory.cascade.errors import RecoverableError, UnrecoverableError
from everos.memory.cascade.handlers import Handler, HandlerDeps
from everos.memory.cascade.types import HandlerOutcome
from everos.memory.cascade.worker import (
    DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS,
    DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS,
    CascadeWorker,
    _KindOptimizerState,
)

# Retention window is the disk-growth bound now: it must stay short enough
# that superseded full-table copies are reclaimed quickly on a small disk.
# The prune *cadence* is frequency-only and configurable via settings.


def test_default_retention_bounds_bulk_import_disk_growth() -> None:
    """Keep local retention below the measured small-disk safety limit."""
    assert DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS >= 10.0
    assert DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS <= 60.0


@dataclass
class _Row:
    """Minimal MdChangeState shape the worker reads off."""

    md_path: str
    kind: str = "episode"
    change_type: str = "added"
    retry_count: int = 0


class _FakeRepo:
    """Records every state-machine transition the worker drives."""

    def __init__(self, batch: list[_Row]) -> None:
        self.batch = list(batch)
        self.done: list[str] = []
        self.failed: list[tuple[str, bool, str, int]] = []

    async def claim_pending_batch(self, _limit: int) -> list[_Row]:
        items, self.batch = self.batch, []
        return items

    async def mark_done(self, md_path: str) -> None:
        self.done.append(md_path)

    async def mark_failed(
        self,
        md_path: str,
        *,
        retryable: bool,
        error: str,
        new_retry_count: int,
    ) -> None:
        self.failed.append((md_path, retryable, error, new_retry_count))


class _OkHandler(Handler):
    def __init__(self) -> None:  # noqa: D401 — no deps needed
        pass

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=1, deleted=0, skipped=0
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=0, deleted=1, skipped=0
        )


class _RecoverableHandler(_OkHandler):
    """Always raises RecoverableError."""

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        raise RecoverableError("embedding 503")


class _UnrecoverableHandler(_OkHandler):
    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        raise UnrecoverableError("YAML parse error")


class _BareExceptionHandler(_OkHandler):
    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        raise RuntimeError("unexpected boom")


@pytest.fixture
def patched_repo(monkeypatch: pytest.MonkeyPatch) -> _FakeRepo:
    """Drop a fake repo onto the module the worker imports."""
    from everos.memory.cascade import worker as worker_mod

    repo = _FakeRepo(batch=[])
    monkeypatch.setattr(worker_mod, "md_change_state_repo", repo)
    return repo


async def test_ok_handler_marks_done(patched_repo: _FakeRepo) -> None:
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    await w.drain_once()
    assert patched_repo.done == ["a.md"]
    assert patched_repo.failed == []


async def test_recoverable_handler_marks_retryable_after_max_retry(
    patched_repo: _FakeRepo,
) -> None:
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker(
        {"episode": _RecoverableHandler()}, max_retry=2, retry_backoff_seconds=0
    )
    await w.drain_once()
    assert patched_repo.done == []
    assert len(patched_repo.failed) == 1
    path, retryable, _err, retry_count = patched_repo.failed[0]
    assert path == "a.md"
    assert retryable is True
    assert retry_count == 2  # 2 retries after the initial attempt


async def test_unrecoverable_handler_marks_permanent(
    patched_repo: _FakeRepo,
) -> None:
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker({"episode": _UnrecoverableHandler()}, retry_backoff_seconds=0)
    await w.drain_once()
    _path, retryable, err, _retry = patched_repo.failed[0]
    assert retryable is False
    assert "UnrecoverableError" in err or "YAML parse error" in err


async def test_bare_exception_marked_permanent(patched_repo: _FakeRepo) -> None:
    """Anything that isn't RecoverableError counts as unrecoverable."""
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker({"episode": _BareExceptionHandler()}, retry_backoff_seconds=0)
    await w.drain_once()
    _path, retryable, _err, _retry = patched_repo.failed[0]
    assert retryable is False


async def test_unknown_kind_marks_permanent_without_handler(
    patched_repo: _FakeRepo,
) -> None:
    patched_repo.batch = [_Row(md_path="a.md", kind="mystery")]
    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    await w.drain_once()
    assert patched_repo.failed[0][1] is False
    assert "no handler" in patched_repo.failed[0][2]


async def test_drain_until_empty_loops_until_no_batch(
    patched_repo: _FakeRepo,
) -> None:
    """Worker keeps draining until claim returns an empty list."""

    rows = [_Row(md_path=f"a{i}.md") for i in range(3)]

    class _ChunkedRepo(_FakeRepo):
        async def claim_pending_batch(self, _limit: int) -> list[_Row]:
            if not self.batch:
                return []
            head, self.batch = self.batch[:1], self.batch[1:]
            return head

    chunked = _ChunkedRepo(rows)
    from everos.memory.cascade import worker as worker_mod

    with mock.patch.object(worker_mod, "md_change_state_repo", chunked):
        w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
        total = await w.drain_until_empty()
    assert total == 3
    assert len(chunked.done) == 3


def test_worker_handler_deps_construct_with_real_classes() -> None:
    """Sanity: HandlerDeps accepts the real provider Protocols."""
    # No instantiation needed — just verifies the dataclass shape.
    assert {"memory_root", "embedder", "tokenizer"} == {
        f.name for f in HandlerDeps.__dataclass_fields__.values()
    }


# ── Optimize scheduler tests ───────────────────────────────────────────────


class _FakeLanceRepo:
    """Records every maintenance call the worker drives.

    ``optimize_delay`` / ``prune_delay`` / ``rebuild_delay`` simulate slow
    operations. ``optimize_raises`` / ``rebuild_raises`` make the
    corresponding call raise (for failure-classification tests). Prune
    calls record their ``cleanup_older_than`` argument so beat-splitting
    tests can assert the heavy/light paths.
    """

    def __init__(
        self,
        *,
        optimize_delay: float = 0.0,
        prune_delay: float = 0.0,
        rebuild_delay: float = 0.0,
        optimize_raises: Exception | None = None,
        rebuild_raises: bool = False,
    ) -> None:
        self.optimize_calls: list[float] = []
        self.prune_calls: list[dt.timedelta] = []
        self.rebuild_calls: list[float] = []
        self.optimize_delay = optimize_delay
        self.prune_delay = prune_delay
        self.rebuild_delay = rebuild_delay
        self.optimize_raises = optimize_raises
        self.rebuild_raises = rebuild_raises

    async def optimize(self) -> None:
        if self.optimize_delay > 0:
            await asyncio.sleep(self.optimize_delay)
        if self.optimize_raises is not None:
            raise self.optimize_raises
        self.optimize_calls.append(time.monotonic())

    async def prune(self, older_than: dt.timedelta) -> None:
        # Recorded before the failure channel so callers can observe that a
        # heavy beat *ran* even when it then raised.
        self.prune_calls.append(older_than)
        if self.prune_delay > 0:
            await asyncio.sleep(self.prune_delay)
        # Heavy beats share the raise channel with light beats so a failing
        # repo surfaces regardless of which cadence fired.
        if self.optimize_raises is not None:
            raise self.optimize_raises

    async def rebuild_indexes(self) -> None:
        if self.rebuild_delay > 0:
            await asyncio.sleep(self.rebuild_delay)
        if self.rebuild_raises:
            raise RuntimeError("rebuild boom")
        self.rebuild_calls.append(time.monotonic())


class _OkHandlerWithRepo(_OkHandler):
    """OK handler exposing a fake ``lance_repo`` for scheduler tests."""

    def __init__(self, repo: _FakeLanceRepo) -> None:
        super().__init__()
        self.lance_repo = repo


async def test_schedule_optimize_noop_when_handler_has_no_lance_repo(
    patched_repo: _FakeRepo,
) -> None:
    """Test stubs without ``lance_repo`` should not even register state."""
    w = CascadeWorker(
        {"episode": _OkHandler()},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.05,
    )
    w._schedule_optimize("episode")
    assert "episode" not in w._optimizer_states


def _beat_count(fake: _FakeLanceRepo) -> int:
    """Total maintenance beats recorded on a fake repo (heavy + light)."""
    return len(fake.prune_calls) + len(fake.optimize_calls)


async def test_schedule_optimize_collapses_burst_within_throttle_window(
    patched_repo: _FakeRepo,
) -> None:
    """A burst of synchronous schedules creates at most one in-flight task.

    The first call starts the beat; subsequent calls during the
    same window only flip ``dirty``. With no time advance between
    schedules, the runner sees ``dirty=False`` after the first run
    and exits — total maintenance beats collapse to one.
    """
    fake = _FakeLanceRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.05,
    )
    for _ in range(10):
        w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert _beat_count(fake) >= 1, "expected at least one maintenance beat"
    assert _beat_count(fake) == 1, (
        f"burst should collapse, got {_beat_count(fake)} beats"
    )


async def test_schedule_optimize_reruns_when_dirty_set_during_beat(
    patched_repo: _FakeRepo,
) -> None:
    """A write that lands mid-beat re-raises ``dirty`` and triggers a re-run.

    Uses an artificially slow first beat so the second schedule fires
    while the first run is still in flight. Trailing-edge semantics
    guarantee the second run happens after the throttle interval.
    """
    fake = _FakeLanceRepo(prune_delay=0.05)
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.02,
    )
    w._schedule_optimize("episode")
    await asyncio.sleep(0.01)  # ensure first task is mid-beat
    w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert _beat_count(fake) == 2


async def test_concurrent_schedules_keep_one_task_per_kind(
    patched_repo: _FakeRepo,
) -> None:
    """LanceDB manifest contention guard: per-kind in-flight task is unique."""
    fake = _FakeLanceRepo(optimize_delay=0.05)
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.02,
    )
    w._schedule_optimize("episode")
    first_task = w._optimizer_states["episode"].task
    # Re-schedule while first task is still in flight; slot must not
    # be replaced.
    for _ in range(5):
        w._schedule_optimize("episode")
        assert w._optimizer_states["episode"].task is first_task
    await w._flush_optimizers()


async def test_flush_optimizers_awaits_pending_task(
    patched_repo: _FakeRepo,
) -> None:
    """flush_optimizers blocks until the in-flight beat commits and clears slot."""
    fake = _FakeLanceRepo(prune_delay=0.05)
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.02,
    )
    w._schedule_optimize("episode")
    assert w._optimizer_states["episode"].task is not None
    await w._flush_optimizers()
    assert _beat_count(fake), "flush should not return before the beat ran"
    assert w._optimizer_states["episode"].task is None


async def test_drain_until_empty_flushes_optimizers_before_returning(
    patched_repo: _FakeRepo,
) -> None:
    """CLI ``cascade sync`` expects FTS to be current when the call returns."""
    fake = _FakeLanceRepo(prune_delay=0.03)
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.02,
    )
    await w.drain_until_empty()
    assert patched_repo.done == ["a.md"]
    assert _beat_count(fake) == 1
    assert w._optimizer_states["episode"].task is None


async def test_drain_once_does_not_block_on_maintenance(
    patched_repo: _FakeRepo,
) -> None:
    """drain_once is fire-and-forget — it must return before a beat commits."""
    fake = _FakeLanceRepo(prune_delay=0.2)
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
    )
    started = time.monotonic()
    await w.drain_once()
    drain_elapsed = time.monotonic() - started
    # drain returned long before the 0.2s prune would finish
    assert drain_elapsed < 0.1, f"drain blocked on maintenance: {drain_elapsed:.3f}s"
    assert not fake.prune_calls, "prune should still be in flight"
    await w._flush_optimizers()
    assert len(fake.prune_calls) == 1


async def test_stop_waits_for_in_flight_maintenance(
    patched_repo: _FakeRepo,
) -> None:
    """stop() must give an in-flight maintenance beat a chance to commit."""
    fake = _FakeLanceRepo(prune_delay=0.05)
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.02,
        optimize_heartbeat_seconds=10.0,
        # Park rebuild interval — startup sweep still fires but we wait
        # for it before testing beat semantics.
        optimize_rebuild_interval_seconds=10.0,
    )
    await w.start()
    # Let the startup rebuild sweep complete (instant for the fake repo)
    # before scheduling a beat — otherwise it would queue behind the rebuild.
    await asyncio.sleep(0.02)
    assert fake.rebuild_calls, "startup rebuild should have fired by now"
    w._schedule_optimize("episode")
    await asyncio.sleep(0.01)  # let the beat start
    await w.stop()
    assert len(fake.prune_calls) == 1


async def test_optimize_failure_does_not_crash_drain_loop(
    patched_repo: _FakeRepo,
) -> None:
    """Repo.optimize() raising should be logged but never propagate."""

    class _FailingRepo:
        async def optimize(self) -> None:
            raise RuntimeError("simulated lancedb manifest conflict")

        async def prune(self, older_than: object) -> None:
            raise RuntimeError("simulated lancedb manifest conflict")

    class _HandlerWithFailingRepo(_OkHandler):
        def __init__(self) -> None:
            super().__init__()
            self.lance_repo = _FailingRepo()

    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker(
        {"episode": _HandlerWithFailingRepo()},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.02,
    )
    # If the failure propagated, drain_until_empty would raise.
    await w.drain_until_empty()
    assert patched_repo.done == ["a.md"]
    assert patched_repo.failed == []


async def test_idle_heartbeat_does_not_schedule_optimize(
    patched_repo: _FakeRepo,
) -> None:
    """An idle daemon must not manufacture index versions every minute."""
    fake_a = _FakeLanceRepo()
    fake_b = _FakeLanceRepo()
    w = CascadeWorker(
        {
            "episode": _OkHandlerWithRepo(fake_a),
            "atomic_fact": _OkHandlerWithRepo(fake_b),
        },
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_heartbeat_seconds=0.05,
    )
    await w.start()
    # Let at least one heartbeat tick happen.
    await asyncio.sleep(0.12)
    await w.stop()
    assert fake_a.optimize_calls == []
    assert fake_b.optimize_calls == []


async def test_heartbeat_recovers_dirty_state_with_lost_task(
    patched_repo: _FakeRepo,
) -> None:
    """Heartbeat restarts real pending optimizer work after task loss."""
    fake = _FakeLanceRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_heartbeat_seconds=0.05,
        optimize_rebuild_interval_seconds=10.0,
    )
    w._optimizer_states["episode"] = _KindOptimizerState(dirty=True)
    await w.start()
    await asyncio.sleep(0.12)
    await w.stop()
    assert _beat_count(fake), "dirty state should be recovered"


async def test_first_beat_prunes_then_throttles_to_light_compaction(
    patched_repo: _FakeRepo,
) -> None:
    """First maintenance beat per kind takes the heavy prune path; subsequent
    beats within ``optimize_prune_interval_seconds`` take the light
    lock-free compaction path.

    Rationale lives in ``LanceRepoBase.prune``: a lock-free bundled
    ``cleanup_older_than`` loses the manifest commit race under churn and
    never reclaims; the heavy beat therefore runs under the per-table
    write lock on a slow cadence, while every other tick stays cheap and
    lock-free.
    """
    fake = _FakeLanceRepo()
    retention = 25.0
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_prune_interval_seconds=10.0,  # long — second beat must be light
        optimize_prune_retention_seconds=retention,
    )
    # First beat: state has never attempted a prune — must go heavy.
    w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert len(fake.prune_calls) == 1
    assert fake.optimize_calls == []
    assert fake.prune_calls[0] == dt.timedelta(seconds=retention)

    # Second beat within the prune cadence: light path (compaction only).
    await asyncio.sleep(0.02)  # exceed optimize throttle (0.01), not prune (10)
    w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert len(fake.prune_calls) == 1, "prune cadence must gate the heavy beat"
    assert len(fake.optimize_calls) == 1


# ── Rebuild scheduler tests ────────────────────────────────────────────────


async def test_rebuild_runs_on_startup_for_every_kind(
    patched_repo: _FakeRepo,
) -> None:
    """The first rebuild sweep fires on worker start, before any interval.

    Otherwise a daemon that restarts more often than the rebuild
    interval would never bound accumulated UUIDs.
    """
    fake_a = _FakeLanceRepo()
    fake_b = _FakeLanceRepo()
    w = CascadeWorker(
        {
            "episode": _OkHandlerWithRepo(fake_a),
            "atomic_fact": _OkHandlerWithRepo(fake_b),
        },
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_heartbeat_seconds=10.0,  # park heartbeat
        optimize_rebuild_interval_seconds=10.0,  # only the startup sweep should fire
    )
    await w.start()
    # Allow the startup sweep to complete; the next tick is 10s away.
    await asyncio.sleep(0.1)
    await w.stop()
    # Exactly one rebuild per kind: the startup sweep. Next interval is 10s.
    assert len(fake_a.rebuild_calls) == 1
    assert len(fake_b.rebuild_calls) == 1


async def test_rebuild_runs_periodically(
    patched_repo: _FakeRepo,
) -> None:
    """After the startup sweep, rebuild repeats every interval."""
    fake = _FakeLanceRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_heartbeat_seconds=10.0,
        optimize_rebuild_interval_seconds=0.05,  # ~tick every 50ms in this test
    )
    await w.start()
    await asyncio.sleep(0.2)  # ~4 ticks plus startup sweep
    await w.stop()
    # Startup sweep + at least 2 interval-driven sweeps.
    assert len(fake.rebuild_calls) >= 3, (
        f"expected ≥3 rebuilds (1 startup + ≥2 periodic), got {len(fake.rebuild_calls)}"
    )


async def test_rebuild_failure_does_not_crash_daemon(
    patched_repo: _FakeRepo,
) -> None:
    """A throwing rebuild is logged and absorbed; the worker keeps running."""
    fake = _FakeLanceRepo(rebuild_raises=True)
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_heartbeat_seconds=0.05,
        optimize_rebuild_interval_seconds=10.0,
    )
    await w.start()
    # Give startup rebuild and an idle heartbeat a chance to run.
    await asyncio.sleep(0.12)
    assert w._task is not None and not w._task.done()
    assert fake.optimize_calls == [], "idle heartbeat must not force optimize"
    await w.stop()
    # Worker is still alive (stop() returned cleanly).
    assert w._task is None


# ── Loop supervision (failure injection) ───────────────────────────────────


class _CrashThenBlockBody:
    """Callable loop body that raises ``crashes`` times, then parks on stop."""

    def __init__(self, worker: CascadeWorker, crashes: int = 1) -> None:
        self.worker = worker
        self.crashes = crashes
        self.runs = 0
        self.restarted = asyncio.Event()

    async def __call__(self) -> None:
        self.runs += 1
        if self.runs <= self.crashes:
            raise RuntimeError("injected loop crash")
        self.restarted.set()
        await self.worker._stop.wait()


async def test_supervise_restarts_crashed_loop_with_backoff(
    patched_repo: _FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loop body that raises is restarted after the (shortened) backoff."""
    monkeypatch.setattr(worker_mod, "_LOOP_RESTART_BACKOFF_SECONDS", (0.01,))
    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    body = _CrashThenBlockBody(w, crashes=1)
    task = asyncio.create_task(w._supervise("drain", body))  # type: ignore[arg-type]
    # Deterministic: wait for the *restart itself* instead of guessing when
    # the backoff has elapsed.
    await asyncio.wait_for(body.restarted.wait(), timeout=2.0)
    w._stop.set()  # let the parked body observe stop; supervisor returns
    await asyncio.wait_for(task, timeout=2.0)
    assert body.runs == 2, "crashing body must be restarted exactly once"


class _AlwaysCrashBody:
    def __init__(self) -> None:
        self.runs = 0

    async def __call__(self) -> None:
        self.runs += 1
        raise RuntimeError("deterministic crash")


async def test_supervise_exhausts_budget_and_requests_process_exit(
    patched_repo: _FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consecutive quick crashes past the budget trigger process-exit escalation."""
    monkeypatch.setattr(worker_mod, "_LOOP_RESTART_BACKOFF_SECONDS", (0.01, 0.01))
    exits: list[str] = []
    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    monkeypatch.setattr(
        w, "_request_process_exit", lambda loop_name: exits.append(loop_name)
    )
    body = _AlwaysCrashBody()
    await asyncio.wait_for(w._supervise("drain", body), timeout=5.0)  # type: ignore[arg-type]
    # budget=2 → the 3rd strike (strikes > budget) escalates.
    assert body.runs == 3
    assert exits == ["drain"], "budget exhaustion must request process exit"


class _LongRunThenCrashBody:
    """Runs >= stable threshold, then crashes; parks on stop afterwards."""

    def __init__(self, worker: CascadeWorker, crashes: int = 4) -> None:
        self.worker = worker
        self.crashes = crashes
        self.runs = 0
        self.phase_done = asyncio.Event()

    async def __call__(self) -> None:
        self.runs += 1
        if self.runs <= self.crashes:
            await asyncio.sleep(0.06)  # >= stable threshold → strike reset
            raise RuntimeError("transient after honest work")
        self.phase_done.set()
        await self.worker._stop.wait()


async def test_stable_run_resets_quick_crash_budget(
    patched_repo: _FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A body that ran >= stable-run seconds before crashing starts a fresh
    incident — independent transients never accumulate into an escalation."""
    monkeypatch.setattr(worker_mod, "_LOOP_RESTART_BACKOFF_SECONDS", (0.005,))
    monkeypatch.setattr(worker_mod, "_LOOP_STABLE_RUN_SECONDS", 0.05)

    exits: list[str] = []
    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    monkeypatch.setattr(
        w, "_request_process_exit", lambda loop_name: exits.append(loop_name)
    )
    body = _LongRunThenCrashBody(w, crashes=4)
    task = asyncio.create_task(w._supervise("drain", body))  # type: ignore[arg-type]
    # Deterministic: wait until several post-stable crashes have happened
    # (with budget=1, two *consecutive* quick crashes would escalate).
    # Generous ceiling: only event-loop starvation can stretch this.
    await asyncio.wait_for(body.phase_done.wait(), timeout=30.0)
    w._stop.set()
    await asyncio.wait_for(task, timeout=10.0)
    assert body.runs == 5, "every post-stable crash must get a fresh budget"
    assert exits == [], "stable runs reset strikes; escalation must not fire"


def test_done_callback_logs_unexpected_supervisor_death(
    patched_repo: _FakeRepo,
) -> None:
    """A supervised task that ends from a BaseException (not cancellation,
    not stop()) is observed by its done-callback and logged.

    Simulated with a stand-in task object: raising a real ``BaseException``
    inside a coroutine would tear down the event loop under the test runner
    — the very behaviour the callback exists to *observe* in production."""
    import structlog

    class _FakeTask:
        def __init__(self, *, cancelled: bool, exc: BaseException | None) -> None:
            self._cancelled = cancelled
            self._exc = exc

        def cancelled(self) -> bool:
            return self._cancelled

        def exception(self) -> BaseException | None:
            return self._exc

    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    with structlog.testing.capture_logs() as logs:
        w._on_loop_task_done(
            "drain",
            _FakeTask(cancelled=False, exc=KeyboardInterrupt("escape")),  # type: ignore[arg-type]
        )
    death_events = [
        e for e in logs if e.get("event") == "cascade_loop_task_ended_unexpectedly"
    ]
    assert len(death_events) == 1
    assert "KeyboardInterrupt" in str(death_events[0].get("error"))

    # Cancelled tasks (stop() path) must stay silent.
    with structlog.testing.capture_logs() as logs2:
        w._on_loop_task_done(
            "drain",
            _FakeTask(cancelled=True, exc=None),  # type: ignore[arg-type]
        )
    assert logs2 == []


# ── Health signals ─────────────────────────────────────────────────────────


async def test_health_defaults_are_clean_before_start(patched_repo: _FakeRepo) -> None:
    w = CascadeWorker({"episode": _OkHandler()})
    h = w.health()
    assert h.drain_consecutive_failures == 0
    assert h.unrecoverable_total == 0
    assert h.optimize_failure_streak == 0
    assert h.prune_stale_seconds == 0.0
    assert h.prune_stale_kind is None
    assert h.reasons() == []


class _FlakyClaimRepo(_FakeRepo):
    """claim_pending_batch raises for the first N calls, then drains clean."""

    def __init__(self, fail_times: int) -> None:
        super().__init__(batch=[])
        self.fail_times = fail_times
        self.calls = 0

    async def claim_pending_batch(self, _limit: int) -> list[_Row]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("sqlite busy")
        return []


async def test_drain_failure_counter_counts_then_resets(
    patched_repo: _FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consecutive drain exceptions are counted; a clean drain resets them."""
    flaky = _FlakyClaimRepo(fail_times=3)
    monkeypatch.setattr(worker_mod, "md_change_state_repo", flaky)
    w = CascadeWorker(
        {"episode": _OkHandler()},
        retry_backoff_seconds=0,
        poll_interval_seconds=0.01,
    )
    await w.start()
    deadline = time.monotonic() + 2.0
    saw_degraded = False
    while time.monotonic() < deadline:
        if w.health().drain_consecutive_failures >= 1:
            saw_degraded = True
            break
        await asyncio.sleep(0.005)
    assert saw_degraded, "failing drains must surface in health"
    # The flaky repo recovers after 3 failures; the counter must reset to 0.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if w.health().drain_consecutive_failures == 0:
            break
        await asyncio.sleep(0.005)
    await w.stop()
    assert w.health().drain_consecutive_failures == 0, (
        "a clean drain resets the consecutive-failure counter"
    )


async def test_unrecoverable_handler_increments_health_counter(
    patched_repo: _FakeRepo,
) -> None:
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker({"episode": _BareExceptionHandler()}, retry_backoff_seconds=0)
    await w.drain_once()
    assert w.health().unrecoverable_total == 1


async def test_prune_staleness_reports_worst_kind(patched_repo: _FakeRepo) -> None:
    """One stalled kind must not be masked by siblings pruning on schedule."""
    now = time.monotonic()
    w = CascadeWorker({"a": _OkHandler(), "b": _OkHandler()})
    w._started_at = now - 100.0
    # Both kinds have unresolved heavy cleanup incidents; b has been pending
    # longer, so it is the worst.
    w._optimizer_states["a"] = _KindOptimizerState(
        first_scheduled_at=now - 100.0, prune_pending_since=now - 50.0
    )
    w._optimizer_states["b"] = _KindOptimizerState(
        first_scheduled_at=now - 100.0, prune_pending_since=now - 90.0
    )
    stale, kind = w._prune_staleness()
    assert kind == "b"
    assert 89.0 < stale <= 91.0

    # Before start (no baseline clock) nothing is reported stale.
    fresh = CascadeWorker({"a": _OkHandler()})
    fresh._optimizer_states["a"] = _KindOptimizerState(
        first_scheduled_at=now, last_prune_at=now
    )
    assert fresh._prune_staleness() == (0.0, None)


async def test_idle_worker_past_alert_threshold_stays_healthy(
    patched_repo: _FakeRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (review P2-1): an idle deployment must not report version
    cleanup as stalled.

    Startup registers optimizer states via the rebuild sweep, but prune beats
    are write-driven — zero writes means zero prune opportunities. Staleness
    baselines at the *first scheduled opportunity*, so a kind that was never
    scheduled cannot count as stalled no matter how long the worker runs."""
    monkeypatch.setattr(worker_mod, "_PRUNE_STALE_SECONDS_ALERT", 1.0)
    fake = _FakeLanceRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_heartbeat_seconds=10.0,
        optimize_rebuild_interval_seconds=10.0,
    )
    await w.start()  # startup rebuild sweep registers the optimizer state
    await asyncio.sleep(0.05)
    assert "episode" in w._optimizer_states, (
        "precondition: state registered by startup rebuild"
    )
    assert fake.prune_calls == [], "precondition: no writes → no prune beat"
    await w.stop()

    # Simulate a long silent night: worker started ~2h ago, still zero writes
    # and therefore zero prune opportunities since boot.
    w._started_at -= 2 * 3600.0

    h = w.health()
    assert h.prune_stale_seconds == 0.0
    assert h.prune_stale_kind is None
    assert h.reasons() == [], (
        f"idle deployment must not flip readiness red (got reasons={h.reasons()})"
    )


async def test_scheduled_but_failing_prune_flips_unhealthy(
    patched_repo: _FakeRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure path preserved (review P2-1): once a kind HAS a maintenance
    opportunity and its prune keeps not succeeding, staleness grows from that
    opportunity and the stall reason fires with the kind named."""
    monkeypatch.setattr(worker_mod, "_PRUNE_STALE_SECONDS_ALERT", 1.0)
    now = time.monotonic()
    w = CascadeWorker({"agent_skill": _OkHandler()})
    w._started_at = now - 300.0
    # Scheduled once 200s ago; the heavy cleanup incident has remained
    # unresolved since then even if individual retries happened later.
    w._optimizer_states["agent_skill"] = _KindOptimizerState(
        first_scheduled_at=now - 200.0,
        last_prune_attempt_at=now - 1.0,
        prune_pending_since=now - 200.0,
    )
    # An idle sibling registered by the rebuild sweep must not mask it — and
    # must not be counted itself.
    w._optimizer_states["episode"] = _KindOptimizerState()

    h = w.health()
    assert h.prune_stale_kind == "agent_skill"
    assert 199.0 <= h.prune_stale_seconds <= 201.0, (
        "staleness measured from the opportunity, not worker start"
    )
    reasons = h.reasons()
    assert any("version cleanup stalled" in r for r in reasons), reasons


async def test_successful_prune_then_idle_does_not_age_into_unhealthy(
    patched_repo: _FakeRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a completed heavy beat must stay healthy while idle."""
    monkeypatch.setattr(worker_mod, "_PRUNE_STALE_SECONDS_ALERT", 1.0)
    now = time.monotonic()
    fake = _FakeLanceRepo()
    w = CascadeWorker({"a": _OkHandlerWithRepo(fake)})
    w._started_at = now - 500.0  # daemon up for ages, silent
    # 'a' gets its very first write-driven schedule right now.
    w._schedule_optimize("a")
    await w._flush_optimizers()
    state = w._optimizer_states["a"]
    assert state.first_scheduled_at > 0.0
    assert state.last_prune_at > 0.0
    assert state.prune_pending_since == 0.0

    # Simulate a long idle period after the successful prune.  There is no
    # unresolved cleanup, so wall-clock age alone must not degrade readiness.
    state.first_scheduled_at -= 2 * 3600.0
    state.last_prune_at -= 2 * 3600.0
    state.last_prune_attempt_at -= 2 * 3600.0
    assert w._prune_staleness() == (0.0, None)

    # Reviewer regression: a *new* write after the long idle period starts a
    # fresh dirty episode.  Before its task gets CPU time, health must use the
    # new opportunity timestamp rather than the hours-old prior episode.
    stale_timestamp = state.first_scheduled_at
    w._schedule_optimize("a")
    assert state.dirty is True
    assert state.first_scheduled_at > stale_timestamp
    h = w.health()
    assert h.prune_stale_seconds < 1.0
    assert h.reasons() == []
    await w._flush_optimizers()


async def test_write_burst_preserves_earliest_dirty_opportunity(
    patched_repo: _FakeRepo,
) -> None:
    """Repeated schedules in one dirty episode must not keep resetting age."""
    w = CascadeWorker({"a": _OkHandlerWithRepo(_FakeLanceRepo())})
    w._schedule_optimize("a")
    state = w._optimizer_states["a"]
    first = state.first_scheduled_at
    w._schedule_optimize("a")
    assert state.first_scheduled_at == first
    await w._flush_optimizers()


async def test_failed_prune_pending_clock_survives_retries(
    patched_repo: _FakeRepo,
) -> None:
    """A later retry must not hide how long cleanup has been unresolved."""
    now = time.monotonic()
    w = CascadeWorker({"a": _OkHandler()})
    w._started_at = now - 500.0
    w._optimizer_states["a"] = _KindOptimizerState(
        first_scheduled_at=now - 400.0,
        last_prune_attempt_at=now - 5.0,
        prune_pending_since=now - 300.0,
    )
    stale, kind = w._prune_staleness()
    assert kind == "a"
    assert 299.0 <= stale <= 301.0


def test_worker_health_reasons_thresholds() -> None:
    from everos.memory.cascade.worker import CascadeWorkerHealth as H

    clean = H(0, 0, 0, 0.0)
    assert clean.reasons() == []

    degraded_drain = H(3, 0, 0, 0.0)
    assert any("drain loop failing" in r for r in degraded_drain.reasons())

    stuck_optimize = H(0, 0, 5, 0.0)
    assert any("optimize stuck" in r for r in stuck_optimize.reasons())

    stalled_prune = H(0, 0, 0, 901.0, prune_stale_kind="agent_skill")
    reasons = stalled_prune.reasons()
    assert any("version cleanup stalled" in r and "agent_skill" in r for r in reasons)


def test_failed_permanent_is_informational_not_operational() -> None:
    """The data-quality backlog must not appear in operational ``reasons``.

    ``failed_permanent`` lives in the SQLite summary (orchestrator level),
    never in :meth:`CascadeWorkerHealth.reasons` — otherwise /health sits
    red forever on a normal backlog."""
    from everos.memory.cascade.worker import CascadeWorkerHealth as H

    backlog = H(
        drain_consecutive_failures=0,
        unrecoverable_total=7,
        optimize_failure_streak=0,
        prune_stale_seconds=0.0,
    )
    assert backlog.reasons() == []


# ── Maintenance failure classification (failure injection) ────────────────


async def test_benign_commit_conflict_not_counted_and_no_fallback_rebuild(
    patched_repo: _FakeRepo,
) -> None:
    """A lost manifest race is benign: debug-logged, streak stays 0, no rebuild."""
    fake = _FakeLanceRepo(
        optimize_raises=RuntimeError(
            "Retryable commit conflict: This Rewrite transaction was preempted"
        )
    )
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.001,
    )
    for _ in range(6):  # far beyond the alert threshold (5)
        w._schedule_optimize("episode")
        await w._flush_optimizers()
        await asyncio.sleep(0.002)
    assert w.health().optimize_failure_streak == 0
    assert fake.rebuild_calls == [], "benign conflicts must not trigger rebuilds"


async def test_real_failures_trigger_one_fallback_per_threshold(
    patched_repo: _FakeRepo,
) -> None:
    """Non-benign beats escalate: fallback rebuild at threshold, once per
    threshold; the health streak survives the rebuild it triggered."""
    fake = _FakeLanceRepo(optimize_raises=RuntimeError("disk exploded"))
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.001,
    )
    for _ in range(5):  # reach the alert threshold
        w._schedule_optimize("episode")
        await w._flush_optimizers()
        await asyncio.sleep(0.002)
    assert len(fake.rebuild_calls) == 1, "threshold must fire one fallback rebuild"
    h = w.health()
    assert h.optimize_failure_streak == 5, (
        "fallback rebuild must NOT reset the health signal"
    )
    state = w._optimizer_states["episode"]
    assert state.failures_since_fallback == 0, "rate limiter resets after firing"

    # Five more failures → exactly one more fallback (rate-limited).
    for _ in range(5):
        w._schedule_optimize("episode")
        await w._flush_optimizers()
        await asyncio.sleep(0.002)
    assert len(fake.rebuild_calls) == 2


async def test_failed_prune_advances_attempt_clock_next_beat_is_light(
    patched_repo: _FakeRepo,
) -> None:
    """A hung/failed heavy beat still backs off a full cadence before the next
    attempt (attempt clock advanced pre-call), instead of re-pinning the lock
    every throttle tick."""

    class _HungPruneRepo(_FakeLanceRepo):
        async def prune(self, older_than: dt.timedelta) -> None:
            self.prune_calls.append(older_than)
            raise TimeoutError("hung cleanup")

    fake = _HungPruneRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.001,
        optimize_prune_interval_seconds=10.0,  # long cadence
    )
    w._schedule_optimize("episode")
    await w._flush_optimizers()  # beat 1: heavy prune → fails
    assert len(fake.prune_calls) == 1
    state = w._optimizer_states["episode"]
    assert state.last_prune_attempt_at > 0.0, "attempt clock advances on failure"
    assert state.last_prune_at == 0.0, "success clock only moves on success"
    pending_since = state.prune_pending_since
    assert pending_since > 0.0, "failed heavy beat starts unresolved-cleanup clock"

    # Beat 2 within the cadence: light compaction (not another lock-holding
    # prune attempt).
    await asyncio.sleep(0.002)
    w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert len(fake.prune_calls) == 1, "failed prune backs off the full cadence"
    assert len(fake.optimize_calls) == 1
    assert state.prune_pending_since == pending_since, (
        "light retry must not hide how long heavy cleanup has been unresolved"
    )


async def test_store_busy_deadline_error_marks_row_retryable(
    patched_repo: _FakeRepo,
) -> None:
    """``VectorStoreBusyError`` is transient by contract: retried inline, then
    marked ``retryable=True`` — never permanently failed."""
    patched_repo.batch = [_Row(md_path="a.md")]

    class _BusyHandler(_OkHandler):
        async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
            raise VectorStoreBusyError("upsert on 'episode' exceeded its 15s deadline")

    w = CascadeWorker({"episode": _BusyHandler()}, max_retry=1, retry_backoff_seconds=0)
    await w.drain_once()
    assert patched_repo.done == []
    assert len(patched_repo.failed) == 1
    _path, retryable, err, _count = patched_repo.failed[0]
    assert retryable is True
    assert "VectorStoreBusyError" in err


# ── Manifest commit-race handling (deterministic, both orderings) ──────────


async def test_rebuild_commit_conflict_schedules_bounded_retry(
    patched_repo: _FakeRepo,
) -> None:
    """A rebuild that loses the manifest race records a backoff *deadline*
    instead of failing the kind or sleeping the loop."""
    monkeypatch_backoffs = (600.0, 1800.0)
    original = worker_mod._REBUILD_CONFLICT_BACKOFFS_SECONDS
    worker_mod._REBUILD_CONFLICT_BACKOFFS_SECONDS = monkeypatch_backoffs
    try:
        conflict = RuntimeError("Retryable commit conflict: preempted")

        class _ConflictOnceRepo(_FakeLanceRepo):
            def __init__(self) -> None:
                super().__init__()
                self.rebuild_attempts = 0

            async def rebuild_indexes(self) -> None:
                self.rebuild_attempts += 1
                if self.rebuild_attempts == 1:
                    raise conflict
                self.rebuild_calls.append(time.monotonic())

        fake = _ConflictOnceRepo()
        w = CascadeWorker(
            {"episode": _OkHandlerWithRepo(fake)},
            retry_backoff_seconds=0,
            optimize_min_interval_seconds=0.01,
        )
        await w._run_rebuild_once("episode")  # loses the race
        state = w._optimizer_states["episode"]
        assert state.rebuild_attempt == 1
        assert state.rebuild_retry_at > time.monotonic(), (
            "retry recorded as a future deadline"
        )
        assert len(fake.rebuild_calls) == 0

        # Due deadline → next loop pass retries and success resets the ledger.
        state.rebuild_retry_at = time.monotonic() - 0.001
        await w._run_rebuild_once("episode")
        assert len(fake.rebuild_calls) == 1
        assert state.rebuild_attempt == 0
        assert state.rebuild_retry_at == 0.0
    finally:
        worker_mod._REBUILD_CONFLICT_BACKOFFS_SECONDS = original


async def test_rebuild_real_failure_defers_to_next_sweep(
    patched_repo: _FakeRepo,
) -> None:
    """A non-conflict rebuild error must NOT be retried via backoff — retrying
    a real error just burns the write lock."""
    fake = _FakeLanceRepo(rebuild_raises=True)
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
    )
    await w._run_rebuild_once("episode")
    state = w._optimizer_states["episode"]
    assert state.rebuild_attempt == 0
    assert state.rebuild_retry_at == 0.0, "no retry scheduled for real errors"


async def test_light_beat_conflict_then_success_recovers_streak(
    patched_repo: _FakeRepo,
) -> None:
    """Both race orderings are benign and self-heal: heavy-beat conflict then
    light-beat conflict, then a clean beat resets every counter."""
    calls = {"n": 0}
    outcomes = ["conflict", "conflict", "ok"]

    class _ScriptedRepo(_FakeLanceRepo):
        async def prune(self, older_than: dt.timedelta) -> None:
            calls["n"] += 1
            if outcomes[min(calls["n"], 3) - 1] == "conflict":
                raise RuntimeError(
                    "This Rewrite transaction was preempted by concurrent "
                    "transaction — Retryable commit conflict"
                )
            self.prune_calls.append(older_than)

        async def optimize(self) -> None:
            calls["n"] += 1
            if outcomes[min(calls["n"], 3) - 1] == "conflict":
                raise RuntimeError("Retryable commit conflict: preempted")
            self.optimize_calls.append(time.monotonic())

    fake = _ScriptedRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.005,
        optimize_prune_interval_seconds=10.0,  # force distinct beats
    )
    for _ in range(3):
        w._schedule_optimize("episode")
        await w._flush_optimizers()
        await asyncio.sleep(0.006)
    assert w.health().optimize_failure_streak == 0
    assert fake.rebuild_calls == []
