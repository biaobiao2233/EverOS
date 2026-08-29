"""Cascade worker — consumes pending rows and runs the matching handler.

The worker is the only piece that crosses the md → LanceDB boundary.
Each cycle:

1. ``claim_pending_batch(BATCH_SIZE)`` atomically flips pending rows to
   ``processing`` and returns them in LSN order.
2. For each row, look up the kind's :class:`Handler` and call either
   :meth:`handle_added_or_modified` or :meth:`handle_deleted` based on
   the row's ``change_type``.
3. On success: ``mark_done``.
4. On :class:`RecoverableError` (or a store-busy deadline error):
   retry inline up to ``MAX_RETRY``; if all attempts fail,
   ``mark_failed(retryable=True)``.
5. On any other exception: ``mark_failed(retryable=False)`` (treated
   as unrecoverable, surfaces in ``cascade fix`` for the user to
   triage by editing the md).

Batch processing is concurrent inside a batch (``asyncio.gather``);
ordering across rows is best-effort — the LSN gives a deterministic
prefix but the handlers themselves are independent.

After a batch completes, each kind that mutated its LanceDB table is
passed to the per-kind optimize scheduler, which splits maintenance
into two beats:

- **light** — lock-free :meth:`LanceRepoBase.optimize` compaction on
  every throttle tick;
- **heavy** — :meth:`LanceRepoBase.prune` under the per-table write
  lock on a slow cadence, which physically reclaims files of
  superseded dataset versions (a lock-free bundled cleanup loses the
  manifest commit race under churn and never runs — see
  :meth:`LanceRepoBase.prune` for the provenance).

A 60-second heartbeat only restarts optimizer work that is already
marked dirty but lost its task. It never manufactures idle optimize
work; startup rebuild and write-driven scheduling cover normal
recovery.

A separate 12-hour loop (:meth:`_rebuild_loop`) rebuilds every index
in place per kind to bound the **active** index UUID / FTS segment
count growth — a workaround for an upstream gap in the lancedb Python
async API; see :meth:`everos.core.persistence.lancedb.LanceRepoBase.rebuild_indexes`
for the full provenance and the conditions under which this scheduler
can be removed.

All three long-lived loops (drain / heartbeat / rebuild) run under
:meth:`CascadeWorker._supervise`: one uncaught exception used to end a
loop permanently and silently while the HTTP process stayed alive,
making ``/health`` lie. The supervisor logs, waits with bounded
backoff, restarts the loop, counts consecutive quick crashes against a
budget, resets the budget after a stable run, and — when the budget is
exhausted — asks the process to exit so the service supervisor can
restart it (see :meth:`CascadeWorker._request_process_exit`).
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import functools
import os
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from everos.core.errors import VectorStoreBusyError
from everos.core.observability.logging import get_logger
from everos.infra.persistence.sqlite import MdChangeState, md_change_state_repo

from .errors import RecoverableError
from .handlers import Handler

logger = get_logger(__name__)

# Conservative defaults — surface in settings if tuning is needed.
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRY = 3
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS = 10.0
DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS = 60.0
_OPTIMIZE_FAILURE_ALERT_THRESHOLD = 5
"""Consecutive **non-benign** ``optimize()``/prune failures (per kind) before
the log escalates ``warning``→``error``, a fallback rebuild is triggered
(:meth:`_run_rebuild_once`), and :meth:`CascadeWorker.health` reports the
kind degraded. A benign commit conflict (lost concurrency race, either
beat) is expected under churn, logged at ``debug``, and does **not**
count — otherwise the streak pins high on a busy table and drowns the
real signal (the disk-bloat failure mode behind lance-format/lance#7653).

Prune that stops succeeding is **not** detected here: a lost race is
excluded by design, and an intervening light-beat success resets the
streak anyway. That failure mode belongs to the per-kind prune-staleness
signal (:data:`_PRUNE_STALE_FACTOR`), which fires on the symptom (nothing
reclaimed for several cadences) rather than on a particular exception."""
_DRAIN_FAILURE_ALERT_THRESHOLD = 3
"""Consecutive :meth:`drain_once` exceptions at or above which
:meth:`CascadeWorker.health` reports degraded — the md → LanceDB
projection is not making progress, so accepted writes are not indexed."""
_PRUNE_STALE_FACTOR = 3.0
"""Multiple of the prune interval without any successful prune before
:meth:`CascadeWorker.health` reports degraded. The primary disk-bloat
signal — fires whether the cause is lost commit-conflict races, a stuck
heavy beat, or a wedged worker, without depending on a particular
exception showing up."""
_MAINTENANCE_TASK_TIMEOUT_SECONDS = 180.0
"""Last-resort deadline on one whole maintenance call (compact or prune).

The repo bounds its own critical sections, but this scheduler is the thing that
breaks if a call never returns at all: it runs one task per kind and skips a
kind whose task is still in flight, so a single non-returning await parks that
kind's maintenance forever — silently, since nothing failed. Generous enough
never to fire on a healthy beat (prune's own budget is 60s), tight enough that
a hang costs one cadence, not forever.
"""

_REBUILD_CONFLICT_BACKOFFS_SECONDS = (600.0, 1800.0, 10800.0)
"""Delay before each re-attempt at a kind's index rebuild: 10min, 30min, 3h.

Only a lost commit race is retried — lance marks it ``Retryable`` and the table
is fine; a concurrent writer in another process simply won the manifest. Any
other failure defers to the next sweep, because retrying a real error just
burns the write lock.

Scaled to the 12h rebuild cadence, not to the conflict. A seconds-long backoff
would spend the whole budget inside one contention window and then leave the
kind unindexed for a full cadence; spreading the attempts over hours means the
retries land in genuinely different load conditions. The schedule is a
*deadline recorded on the kind*, never a sleep: :meth:`_rebuild_loop` walks
kinds sequentially, so sleeping here would park every later kind behind this
one."""

_REBUILD_LOOP_TICK_SECONDS = 60.0
"""How often the rebuild loop wakes to pick up due retries between sweeps.
Cheap: a dict scan per tick, and only kinds with a due deadline do work."""

_LOOP_RESTART_BACKOFF_SECONDS = (5.0, 15.0, 45.0)
"""Backoff before each restart of a background loop that raised.

The three long-lived loops (drain / heartbeat / rebuild) are plain
``create_task`` coroutines. Without supervision, one uncaught exception ends
that loop **permanently and silently**: nothing restarts it, and because
``self._*_task`` keeps a strong reference the interpreter never prints the
"Task exception was never retrieved" warning either (that fires on GC). The
loop's job simply stops happening. So each loop runs under
:meth:`CascadeWorker._supervise`, which logs, waits, and restarts.

Escalating rather than fixed: a transient cause (a momentarily unavailable
table) clears within seconds, while a deterministic one would otherwise spin.
After the last entry is used the worker asks the process to exit (see
:meth:`CascadeWorker._request_process_exit`) — a server whose projection
pipeline is permanently dead should not keep serving as if healthy.

The budget is **per incident, not per process lifetime**: a body that ran at
least :data:`_LOOP_STABLE_RUN_SECONDS` before crashing gets a full budget
again. Without that reset, rare *independent* transients — one every few
days, each recovered by a single restart — would still spend the budget one
by one and kill the server weeks later on the 4th, which punishes exactly
the case supervision exists to absorb. This mirrors how process supervisors
count restarts within a window (systemd ``StartLimitIntervalSec``, Erlang
``max_restarts`` per ``max_seconds``) rather than forever.
"""

_LOOP_STABLE_RUN_SECONDS = 60.0
"""A supervised loop body that ran at least this long before raising is
treated as a fresh incident (restart budget resets): a deterministic
crash-on-startup fails in milliseconds and cannot reach it, while a loop that
did an hour of honest work before hitting a transient obviously should not
inherit stale strikes."""

DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS = 12 * 60 * 60.0
"""How often (per kind) to do a full index rebuild.

This is the **only** application-level mechanism we have to bound the
active index UUID / segment count growth — see
:meth:`LanceRepoBase.rebuild_indexes` for the full provenance: Rust
``OptimizeOptions.num_indices_to_merge`` is the right knob but
``lancedb.AsyncTable.optimize()`` does not expose it, and on the embedded
``lance crate 4.0`` the merge behaviour itself is broken so even calling
Lance directly wouldn't help.

12 hours is a conservative pick: rebuild cost is ~0.3s per 50k rows
× indexed columns (measured locally), so even a small EverOS
deployment can absorb it without scheduling around peaks.

**Remove this scheduler** once lancedb exposes ``num_indices_to_merge``
on the async Python API and the embedded lance crate ships the
working merge implementation; ``optimize(num_indices_to_merge=1)``
in the regular hot path will do the same job for ~free.
"""

DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS = 300.0
"""**Cadence** — how often (per kind) the heavy write-locked
:meth:`LanceRepoBase.prune` beat runs, vs the light lock-free
:meth:`~LanceRepoBase.optimize` compaction on every other tick.

prune holds the per-table write lock (brief same-table write stall,
~seconds on a churned table), so it runs on this slow beat rather than
every throttle tick. This is the *frequency* of reclamation; how much
history each prune keeps is a separate knob — see
:data:`DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS`.

Does **not** shrink active index internals (FTS ``part_N`` count or
vector index UUID count): those only collapse via a full index rebuild,
which is intentionally out of scope here.
"""

DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS = 60.0
"""**Retention window** — ``cleanup_older_than`` passed to prune: files
belonging to dataset versions replaced more than this long ago become
eligible for physical deletion.

Decoupled from (and much shorter than) the prune *cadence*: prune runs
under the per-table write lock, so no concurrent writer can be
referencing an old version, and the only thing the window must outlive
is an **in-flight read** (a ``/search`` holding a version reference) —
sub-second to a few seconds. 60s is comfortably safe.

Why short matters: under sustained churn each compaction leaves a
full-table-sized *superseded* fragment behind; the retention window is
how long those pile up before reclamation. A long window × the churn
rate retains many full-table copies of transient footprint. Tune via
the constructor."""

_PRUNE_STALE_SECONDS_ALERT = (
    _PRUNE_STALE_FACTOR * DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS
)
"""Absolute prune-staleness alert threshold (seconds) under the shipped
prune cadence — three missed heavy beats."""


@dataclass
class _KindOptimizerState:
    """Per-kind throttle state for LanceDB maintenance.

    ``dirty`` is the trailing-edge signal: every write sets it, the
    runner consumes it before each beat. If a write arrives mid-beat,
    ``dirty`` is re-raised and the runner loops once more after
    honouring the throttle interval.

    ``task`` holds the in-flight runner; at most one runner exists
    per kind so concurrent LanceDB writes never collide on the same
    table's manifest.

    Two prune clocks, deliberately split:

    - ``last_prune_attempt_at`` gates **scheduling** — the monotonic time
      of the last heavy (prune) beat *attempt*, advanced whether it
      succeeds or times out. So a prune that hangs and is killed by the
      write-lock timeout backs off a full cadence before the next attempt
      instead of retrying every ~10s and holding the lock most of the
      time.
    - ``last_prune_at`` records the last *successful* prune, advanced only
      after the call returns. It drives the prune-staleness **health**
      signal (:meth:`CascadeWorker._prune_staleness`), so a persistently
      failing/hanging prune still surfaces as degraded instead of being
      masked by advancing the schedule clock.

    Both default ``0`` ("never") — the first run after worker startup
    always prunes, catching up from a prior session. ``monotonic()`` is
    boot-relative, so the sentinel matters: comparing against ``0`` with
    ``>=`` would misbehave for the first cadence of machine uptime.
    """

    last_run_at: float = 0.0
    last_prune_attempt_at: float = 0.0
    last_prune_at: float = 0.0
    prune_pending_since: float = 0.0
    """Monotonic start of an unresolved heavy prune beat.

    Set immediately before the first heavy prune attempt in an unresolved
    cleanup incident and cleared only after a prune completes successfully.
    Repeated failed/conflicted attempts deliberately do not move this clock:
    health needs to measure how long cleanup has failed to reach a success,
    not how recently the last retry happened.

    A completed prune followed by an idle period leaves this at ``0``.  That
    distinction is important: time since the last successful cleanup is not
    itself a stalled cleanup when there has been no unresolved heavy beat.
    """
    first_scheduled_at: float = 0.0
    """Monotonic start of the current dirty maintenance opportunity.

    ``_schedule_optimize`` refreshes this only on a clean -> dirty transition,
    so a burst of writes preserves the earliest unresolved opportunity while a
    new write after a long idle period gets a fresh clock. ``0`` means no
    maintenance opportunity has ever been scheduled in this process.

    Health uses this only while an optimizer is still ``dirty`` before a heavy
    beat has begun. Once a heavy prune attempt starts, ``prune_pending_since``
    becomes the incident clock instead. A state registered by the startup
    rebuild sweep on an idle deployment (no writes → no beats) is therefore
    not a stalled cleanup.
    """
    dirty: bool = False
    optimize_failures: int = 0
    """Consecutive non-benign optimize/prune failures **since the last
    success**. Drives the health verdict and escalation to ``error`` at
    :data:`_OPTIMIZE_FAILURE_ALERT_THRESHOLD`.

    Reset **only** by a successful beat — never by the fallback rebuild it
    triggers. Zeroing the alert counter inside the branch that fires at the
    threshold made the alert effectively unobservable (a table failing 100%
    cycled 1..threshold -> 0 -> 1.., readable only during the sub-second
    rebuild). Same shape as the cross-kind max()-masking bug: a remediation
    path refreshing the very signal that is supposed to report it.
    """
    failures_since_fallback: int = 0
    """Failures since the last fallback rebuild — the rate limiter.

    Reset by the fallback rebuild so it fires at most once per
    :data:`_OPTIMIZE_FAILURE_ALERT_THRESHOLD` failures instead of on every
    failure."""
    rebuild_retry_at: float = 0.0
    """Monotonic deadline for re-attempting a rebuild that lost a commit race.
    ``0`` means nothing pending. Set instead of sleeping so the rebuild loop
    stays free to serve the other kinds — see
    :data:`_REBUILD_CONFLICT_BACKOFFS_SECONDS`."""
    rebuild_attempt: int = 0
    """Consecutive lost commit races for this kind; indexes into the backoff
    schedule and resets on any completed rebuild."""
    task: asyncio.Task[None] | None = None
    rebuild_task: asyncio.Task[None] | None = None
    """In-flight rebuild task slot, separate from ``task`` so ordinary
    ``_schedule_optimize`` calls during a rebuild can still register
    ``dirty`` + spawn an optimize runner. The runner itself waits for
    ``rebuild_task`` before touching the LanceDB manifest, so the two
    operations never race on commits — only the dispatch slot is split.
    """


@dataclass(frozen=True)
class CascadeWorkerHealth:
    """Snapshot of the worker's in-memory health signals.

    Cheap to produce (no IO — pure in-process counters). The orchestrator
    combines it with the SQLite queue summary for the full verdict.
    """

    drain_consecutive_failures: int
    """Consecutive :meth:`drain_once` exceptions; 0 when the last drain
    completed cleanly."""

    unrecoverable_total: int
    """Cumulative unrecoverable handler failures since worker start —
    each is a md file whose projection to LanceDB permanently failed
    and now needs a user edit (surfaced by ``cascade fix``)."""

    optimize_failure_streak: int
    """Max consecutive non-benign optimize/prune failure count across
    kinds; benign light-beat commit conflicts do not count."""

    prune_stale_seconds: float
    """Age of the **worst unresolved** version-cleanup incident.

    ``0`` means no kind currently has a dirty pre-prune optimizer or a heavy
    prune attempt that has failed to reach a successful completion. Time since
    a successful prune does not accumulate here while the kind is idle.

    Deliberately the worst kind, not the newest prune across kinds: a
    per-kind cleanup that dies grows *that table's* index dir unbounded,
    and every other kind pruning normally must not mask it."""

    prune_stale_kind: str | None = None
    """The kind :attr:`prune_stale_seconds` belongs to; ``None`` when no
    kind has state yet. Named in :meth:`reasons` so an operator knows
    which table to look at."""

    def reasons(self) -> list[str]:
        """Operational degradation reasons; empty when healthy.

        Combined with the SQLite ``failed_permanent`` count in
        ``CascadeOrchestrator.health`` to produce the ``/health``
        verdict. These in-memory signals do not include the permanent-
        failure backlog (that needs a query)."""
        out: list[str] = []
        if self.drain_consecutive_failures >= _DRAIN_FAILURE_ALERT_THRESHOLD:
            out.append(
                f"drain loop failing ({self.drain_consecutive_failures} in a row)"
            )
        if self.optimize_failure_streak >= _OPTIMIZE_FAILURE_ALERT_THRESHOLD:
            out.append(f"optimize stuck ({self.optimize_failure_streak} in a row)")
        if self.prune_stale_seconds >= _PRUNE_STALE_SECONDS_ALERT:
            kind = self.prune_stale_kind or "unknown"
            out.append(
                f"version cleanup stalled for kind '{kind}' "
                f"({int(self.prune_stale_seconds)}s since its last prune "
                "— that table's index dir may grow)"
            )
        return out


def _is_benign_commit_conflict(exc: BaseException) -> bool:
    """True for a LanceDB optimistic-concurrency retry error.

    LanceDB surfaces the Rust ``Retryable commit conflict`` as a plain
    exception whose message carries the phrase — there is no dedicated
    class to catch. Either beat can lose the race: the light beat is
    lock-free, and the heavy beat's write lock is in-process only, so a
    second process (CLI ``cascade sync`` / ``backfill``) can preempt it.
    Both are expected under churn and benign (the next beat retries), so
    they log at ``debug`` and do not count toward the failure streak.

    Match only the specific ``commit conflict`` phrase (a substring of the
    Rust message), not a bare ``retryable`` — the latter appears in the
    repr of unrelated recoverable errors and would silently swallow real
    failures.
    """
    return "commit conflict" in str(exc).lower()


class CascadeWorker:
    """Owns the claim → dispatch → mark cycle.

    Created with the ``{kind: Handler}`` map produced by
    :func:`memory.cascade.registry.build_handlers`. Holds no other
    state — every per-row decision goes through the repo.
    """

    def __init__(
        self,
        handlers: dict[str, Handler],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retry: int = DEFAULT_MAX_RETRY,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        optimize_min_interval_seconds: float = DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS,
        optimize_heartbeat_seconds: float = DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS,
        optimize_prune_interval_seconds: float = (
            DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS
        ),
        optimize_prune_retention_seconds: float = (
            DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS
        ),
        optimize_rebuild_interval_seconds: float = (
            DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS
        ),
    ) -> None:
        self._handlers = handlers
        self._batch_size = batch_size
        self._max_retry = max_retry
        self._poll_interval = poll_interval_seconds
        self._retry_backoff = retry_backoff_seconds
        self._optimize_min_interval = optimize_min_interval_seconds
        self._optimize_heartbeat = optimize_heartbeat_seconds
        self._optimize_prune_interval = optimize_prune_interval_seconds
        self._optimize_prune_retention = optimize_prune_retention_seconds
        self._optimize_rebuild_interval = optimize_rebuild_interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._rebuild_task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._optimizer_states: dict[str, _KindOptimizerState] = {}
        # ── in-memory health signals (see :meth:`health`) ──────────────
        self._started_at: float = 0.0
        self._drain_consecutive_failures: int = 0
        self._unrecoverable_total: int = 0

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._started_at = time.monotonic()
        self._task = self._spawn_loop("drain", self._run_loop, "cascade-worker")
        self._heartbeat_task = self._spawn_loop(
            "heartbeat", self._heartbeat_loop, "cascade-worker-heartbeat"
        )
        self._rebuild_task = self._spawn_loop(
            "rebuild", self._rebuild_loop, "cascade-worker-rebuild"
        )
        logger.info("cascade_worker_started", batch_size=self._batch_size)

    def _spawn_loop(
        self,
        loop_name: str,
        body: Callable[[], Coroutine[Any, Any, None]],
        task_name: str,
    ) -> asyncio.Task[None]:
        """Start one supervised background loop.

        Two layers, because they fail differently: :meth:`_supervise` restarts
        the loop body when it raises, and the done-callback is the last-resort
        observer for the case the supervisor itself ends unexpectedly (a
        ``BaseException`` it deliberately does not catch). Without the callback
        that ending is invisible — see :data:`_LOOP_RESTART_BACKOFF_SECONDS`.
        """
        task = asyncio.create_task(self._supervise(loop_name, body), name=task_name)
        task.add_done_callback(functools.partial(self._on_loop_task_done, loop_name))
        return task

    async def _supervise(
        self,
        loop_name: str,
        body: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """Run ``body`` and restart it on failure, bounded, then give up.

        A clean return means the loop observed ``self._stop`` — nothing to do.
        ``CancelledError`` is re-raised so :meth:`stop` still works. Everything
        else is logged with the loop name and retried per
        :data:`_LOOP_RESTART_BACKOFF_SECONDS`; when those are exhausted the
        process is asked to exit rather than run on with a dead loop.

        The budget counts **consecutive quick crashes**, not crashes over the
        process lifetime: a body that ran at least
        :data:`_LOOP_STABLE_RUN_SECONDS` before raising starts a fresh
        incident. A deterministic crash-on-entry still exhausts the budget in
        ~65s; independent transients days apart each get the full ladder.
        """
        budget = len(_LOOP_RESTART_BACKOFF_SECONDS)
        strikes = 0
        while True:
            if strikes and await self._wait_or_stop(
                _LOOP_RESTART_BACKOFF_SECONDS[strikes - 1]
            ):
                return
            if self._stop.is_set():
                return
            started = time.monotonic()
            try:
                await body()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                ran = time.monotonic() - started
                if ran >= _LOOP_STABLE_RUN_SECONDS:
                    strikes = 0
                strikes += 1
                logger.exception(
                    "cascade_loop_crashed",
                    loop=loop_name,
                    strike=strikes,
                    restarts_left=max(0, budget - strikes + 1),
                    ran_seconds=round(ran, 1),
                    error=f"{type(exc).__name__}: {exc}",
                )
                if strikes > budget:
                    break
        logger.error("cascade_loop_unrecoverable", loop=loop_name, restarts=budget)
        self._request_process_exit(loop_name)

    def _on_loop_task_done(self, loop_name: str, task: asyncio.Task[None]) -> None:
        """Log a supervised loop task that ended without ``stop()`` asking it to."""
        if task.cancelled() or self._stop.is_set():
            return
        exc = task.exception()
        logger.error(
            "cascade_loop_task_ended_unexpectedly",
            loop=loop_name,
            error=f"{type(exc).__name__}: {exc}" if exc is not None else None,
        )

    def _request_process_exit(self, loop_name: str) -> None:
        """Terminate this process so the service supervisor restarts it.

        **Deliberate divergence from upstream, driven by verified production
        configuration.** Upstream raises ``SIGTERM`` to its own pid so the
        ASGI server shuts down gracefully, assuming a supervisor with
        ``Restart=always`` (or Docker / k8s). Production EverOS-v2 runs under
        systemd with ``Restart=on-failure`` (verified 2026-08-25 via
        ``systemctl show everos-v2.service``), and the server entrypoint is
        uvicorn — which installs its own SIGTERM handler and exits with code
        0 on graceful shutdown. A self-SIGTERM there is therefore classified
        as a *clean* exit and systemd will **not** restart us: the dead
        projection pipeline would stay down until manual intervention.

        Exiting with a **non-zero status** is what ``Restart=on-failure``
        acts on. Data safety of the hard exit: LanceDB commits are atomic at
        the manifest level (an interrupted commit leaves the previous
        version valid), SQLite runs in WAL mode, and
        ``CascadeOrchestrator.start`` already recovers orphaned
        ``processing`` rows on boot — exactly the crash-recovery path this
        exercises. ``RestartUSec=10s`` spaces the restart; the unit's start
        limit (5 per 10s window) cannot trip on a single exit.

        Overridable seam for tests — they replace this rather than kill the
        pytest process.
        """
        logger.error("cascade_worker_requesting_process_exit", loop=loop_name)
        os._exit(1)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._heartbeat_task
        if self._rebuild_task is not None:
            self._rebuild_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._rebuild_task
        # Optimize tasks coalesce on the stop signal (their inter-run
        # cooldowns observe ``self._stop``), so flushing them just
        # waits out the currently in-flight commit rather than
        # blocking on a fresh throttle window.
        await self._flush_optimizers()
        self._task = None
        self._heartbeat_task = None
        self._rebuild_task = None
        logger.info("cascade_worker_stopped")

    async def drain_once(self) -> int:
        """Process one batch, return the number of rows handled.

        Used by CLI ``cascade sync`` and ``fix --apply`` to flush the
        queue without spinning the background task. Returns ``0`` when
        the queue is empty.

        For each kind that mutated its LanceDB table this batch,
        :meth:`_schedule_optimize` records a throttled optimize
        intent. The actual maintenance runs as separate tasks so
        drain throughput is decoupled from index maintenance; callers
        that need a "fully indexed" barrier (CLI ``cascade sync``)
        must call :meth:`_flush_optimizers` — :meth:`drain_until_empty`
        does this on their behalf.
        """
        batch = await md_change_state_repo.claim_pending_batch(self._batch_size)
        if not batch:
            return 0
        results = await asyncio.gather(*(self._process_one(row) for row in batch))
        touched_kinds = {kind for kind in results if kind is not None}
        for kind in touched_kinds:
            self._schedule_optimize(kind)
        return len(batch)

    async def drain_until_empty(self, *, max_passes: int = 100) -> int:
        """Drain repeatedly until the queue is empty (or ``max_passes``).

        Returns the total number of rows processed. Bounded passes
        prevent a livelock if a stuck row keeps re-failing back to
        pending (which can't happen in the current design but is a
        cheap safety net).

        Awaits :meth:`_flush_optimizers` before returning so callers
        (CLI ``cascade sync``) can rely on FTS being up-to-date when
        the call completes.
        """
        total = 0
        for _ in range(max_passes):
            processed = await self.drain_once()
            if processed == 0:
                break
            total += processed
        await self._flush_optimizers()
        return total

    def health(self) -> CascadeWorkerHealth:
        """Snapshot the worker's in-memory health signals (no IO).

        Combines the drain-loop and unrecoverable counters with the worst
        per-kind optimize streak and the prune-staleness clock.
        """
        states = self._optimizer_states.values()
        optimize_failure_streak = max((s.optimize_failures for s in states), default=0)
        stale_seconds, stale_kind = self._prune_staleness()
        return CascadeWorkerHealth(
            drain_consecutive_failures=self._drain_consecutive_failures,
            unrecoverable_total=self._unrecoverable_total,
            optimize_failure_streak=optimize_failure_streak,
            prune_stale_seconds=stale_seconds,
            prune_stale_kind=stale_kind,
        )

    def _prune_staleness(self) -> tuple[float, str | None]:
        """Staleness of the **worst** kind: ``(seconds, kind)``.

        Per kind, staleness is measured only while there is unresolved cleanup:

        - an in-flight/lost optimizer that is still ``dirty`` and has not yet
          reached a heavy prune attempt, measured from its first scheduling
          opportunity; or
        - a heavy prune incident that started but has not completed
          successfully, measured from ``prune_pending_since``.

        A successful prune followed by an idle period is *not* stale.  The
        previous implementation measured wall-clock time since the last
        successful prune forever, so a healthy deployment inevitably flipped
        readiness red after the alert threshold even with an empty queue and
        no failed maintenance work.

        The worst (largest) unresolved incident is reported. Taking the worst
        across kinds prevents one genuinely stalled cleanup from being masked
        by healthy siblings.

        A kind that has **never been scheduled** (no write batch touched it,
        heartbeat never revived it) is skipped entirely: its optimizer state
        may exist purely because the startup rebuild sweep registered it, and
        "idle since boot" is not a stalled cleanup. Without this gate an idle
        single-user deployment would drift past the alert threshold every
        silent night and flip ``/health`` red while nothing is stuck.

        Returns ``(0.0, None)`` before the worker has started
        (``_started_at == 0``), before any kind has registered a state, or
        when no state carries a maintenance opportunity yet.
        """
        states = list(self._optimizer_states.items())
        if not states or self._started_at == 0.0:
            return 0.0, None
        now = time.monotonic()
        worst_seconds = -1.0
        worst_kind: str | None = None
        for kind, state in states:
            if state.first_scheduled_at == 0.0:
                continue
            if state.prune_pending_since > 0.0:
                baseline = state.prune_pending_since
            elif state.dirty:
                baseline = state.first_scheduled_at
            else:
                # No unresolved heavy cleanup.  In particular, do not turn a
                # successful prune into a future readiness failure merely
                # because the table stayed idle afterwards.
                continue
            stale = max(0.0, now - baseline)
            if stale > worst_seconds:
                worst_seconds, worst_kind = stale, kind
        if worst_kind is None:
            return 0.0, None
        return worst_seconds, worst_kind

    # ── internals ──────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                processed = await self.drain_once()
                self._drain_consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 — counted, never crash the loop
                self._drain_consecutive_failures += 1
                logger.exception(
                    "cascade_worker_drain_failed",
                    error=str(exc),
                    consecutive_failures=self._drain_consecutive_failures,
                )
                processed = 0
            if processed == 0:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self._poll_interval
                    )
                except TimeoutError:
                    continue

    async def _process_one(self, row: MdChangeState) -> str | None:
        """Process one ``md_change_state`` row.

        Returns the ``row.kind`` when the handler actually mutated the
        kind's LanceDB table (``upserted`` or ``deleted`` > 0) so the
        caller can collect a set of "touched kinds" and maintain them
        after the batch. Returns ``None`` for skipped-only rows, failed
        rows, and rows where no handler is registered — the maintenance
        step is gated on actual writes happening this batch.
        """
        handler = self._handlers.get(row.kind)
        if handler is None:
            await md_change_state_repo.mark_failed(
                row.md_path,
                retryable=False,
                error=f"no handler registered for kind {row.kind!r}",
                new_retry_count=row.retry_count,
            )
            return None

        retry_count = row.retry_count
        last_error: str = ""
        for attempt in range(self._max_retry + 1):
            try:
                if row.change_type == "deleted":
                    outcome = await handler.handle_deleted(row.md_path)
                else:
                    outcome = await handler.handle_added_or_modified(row.md_path)
            except (RecoverableError, VectorStoreBusyError) as exc:
                # VectorStoreBusyError (repo deadline exceeded) is transient by
                # contract — the table did not answer in time; treat it exactly
                # like a RecoverableError so the row is retried instead of
                # being marked permanently failed.
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "cascade_worker_recoverable",
                    md_path=row.md_path,
                    attempt=attempt,
                    error=last_error,
                )
                if attempt < self._max_retry:
                    retry_count += 1
                    await asyncio.sleep(self._retry_backoff * (attempt + 1))
                    continue
                await md_change_state_repo.mark_failed(
                    row.md_path,
                    retryable=True,
                    error=last_error,
                    new_retry_count=retry_count,
                )
                return None
            except Exception as exc:  # noqa: BLE001 — surface as unrecoverable
                last_error = f"{type(exc).__name__}: {exc}"
                self._unrecoverable_total += 1
                logger.exception(
                    "cascade_worker_unrecoverable",
                    md_path=row.md_path,
                    kind=row.kind,
                    unrecoverable_total=self._unrecoverable_total,
                )
                await md_change_state_repo.mark_failed(
                    row.md_path,
                    retryable=False,
                    error=last_error,
                    new_retry_count=retry_count,
                )
                return None

            logger.info(
                "cascade_worker_processed",
                md_path=row.md_path,
                kind=row.kind,
                change_type=row.change_type,
                upserted=outcome.upserted,
                deleted=outcome.deleted,
                skipped=outcome.skipped,
            )
            await md_change_state_repo.mark_done(row.md_path)
            # Only flag the kind as "touched" when we actually wrote
            # something — skipped rows leave the table untouched, so
            # maintaining it would be pure overhead.
            return row.kind if (outcome.upserted or outcome.deleted) else None
        return None

    # ── optimizer scheduling ───────────────────────────────────────────────

    def _schedule_optimize(self, kind: str) -> None:
        """Throttle + trailing-edge schedule for a kind's maintenance.

        Per-kind semantics — for any one ``kind``:

        - The first call after the throttle window starts a beat
          immediately (initial_delay=0).
        - Subsequent calls within the window only set ``dirty=True``;
          the in-flight runner picks the flag up and re-runs after the
          throttle interval has elapsed.
        - A call while a task is in flight returns without starting a
          new task — only the dirty flag matters. This guarantees at
          most one concurrent maintenance task per kind, which is what
          LanceDB's per-table manifest version expects.

        No-op when the handler for ``kind`` doesn't expose a
        ``lance_repo`` (test stubs, handlers that intentionally skip
        LanceDB).

        Idempotent and cheap (a single dict lookup + flag write in
        the hot path) — safe to call on every batch and from the
        heartbeat sweep.
        """
        handler = self._handlers.get(kind)
        repo = getattr(handler, "lance_repo", None) if handler else None
        if repo is None:
            return
        state = self._optimizer_states.setdefault(kind, _KindOptimizerState())
        if not state.dirty:
            # Start (or refresh) the clock for this dirty episode.  Repeated
            # calls while already dirty intentionally preserve the earliest
            # unresolved opportunity.  After a completed/idle episode, a new
            # write must not inherit an hours-old timestamp and instantly turn
            # readiness red before its optimizer has had a chance to run.
            state.first_scheduled_at = time.monotonic()
        state.dirty = True
        if state.task is not None and not state.task.done():
            return
        elapsed = time.monotonic() - state.last_run_at
        delay = max(0.0, self._optimize_min_interval - elapsed)
        state.task = asyncio.create_task(
            self._optimize_runner(kind, initial_delay=delay),
            name=f"cascade-optimize-{kind}",
        )

    async def _optimize_runner(self, kind: str, *, initial_delay: float) -> None:
        """Run maintenance for ``kind`` until ``dirty`` clears.

        Honours the throttle interval on entry (when scheduled
        mid-cooldown) and between consecutive runs (when a write
        re-raised ``dirty`` during the previous beat). The cooldown
        waits respect the worker's stop signal so shutdown doesn't have
        to outlast the throttle window.

        If a rebuild is in flight for this kind we wait for it (bounded)
        before touching the manifest — concurrent maintenance + rebuild
        on the same LanceDB table would race on the version commit.
        """
        state = self._optimizer_states[kind]
        try:
            if initial_delay > 0 and await self._wait_or_stop(initial_delay):
                return
            # Serialise behind any in-flight rebuild (rare; only during
            # the 12h sweep). Failures are absorbed in _run_rebuild_once.
            #
            # Bounded, and symmetric with the wait on the other side: whichever
            # of the two maintenance jobs arrives second parks on the first, so
            # an unbounded wait here is the same defect as an unbounded wait
            # there — this kind's task slot never frees, _schedule_optimize
            # keeps short-circuiting on it, and that table silently stops being
            # maintained.
            if state.rebuild_task is not None and not state.rebuild_task.done():
                try:
                    async with asyncio.timeout(_MAINTENANCE_TASK_TIMEOUT_SECONDS):
                        await state.rebuild_task
                except TimeoutError:
                    # Give up this beat rather than compact under a live
                    # rebuild — the two commit on the same manifest, which is
                    # what the wait exists to prevent. Writes keep the dirty
                    # flag set, so the next beat retries.
                    logger.warning(
                        "cascade_lancedb_optimize_skipped_rebuild_unfinished",
                        kind=kind,
                        waited_seconds=_MAINTENANCE_TASK_TIMEOUT_SECONDS,
                    )
                    return
                except Exception:  # noqa: BLE001 — already logged+counted there
                    pass
            while state.dirty and not self._stop.is_set():
                state.dirty = False
                state.last_run_at = time.monotonic()
                await self._run_optimize_once(kind)
                if (
                    state.dirty
                    and not self._stop.is_set()
                    and await self._wait_or_stop(self._optimize_min_interval)
                ):
                    return
        finally:
            if state.task is asyncio.current_task():
                state.task = None

    async def _wait_or_stop(self, seconds: float) -> bool:
        """Sleep up to ``seconds``; return True if stop was set."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            return False
        return True

    async def _run_optimize_once(self, kind: str) -> None:
        """Run one maintenance beat for ``kind``.

        Most calls take the **light** path — lock-free ``optimize()``,
        pure compaction + index merge, fast. Every
        ``_optimize_prune_interval`` seconds the next call takes the
        **heavy** path — ``prune()`` under the per-table write lock,
        which compacts *and* physically deletes files belonging to
        versions older than the retention window (short + decoupled
        from the cadence; see
        :data:`DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS`).

        Pruning is opt-in per call rather than a separate task so the
        existing per-kind serialisation (one in-flight runner per kind)
        keeps holding — LanceDB serialises writes on the table's
        manifest, and prune is a write.
        """
        handler = self._handlers.get(kind)
        repo = getattr(handler, "lance_repo", None) if handler else None
        if repo is None:
            return
        state = self._optimizer_states.get(kind)
        now = time.monotonic()
        should_prune = (
            state is None
            # 0.0 means "never attempted" — always prune, don't compare clocks.
            # ``monotonic()`` is boot-relative, so ``now - 0 >= interval`` is
            # false for the first ~cadence of machine uptime and the
            # catch-up prune would be skipped exactly when a fresh process
            # most needs it.
            or state.last_prune_attempt_at == 0.0
            or (now - state.last_prune_attempt_at) >= self._optimize_prune_interval
        )
        try:
            if should_prune:
                # Heavy beat: physically reclaim old versions under the
                # per-table write lock (inside ``repo.prune``) so churn
                # can't preempt the cleanup commit. Advance the *attempt*
                # clock before the call: if prune hangs and the timeout kills
                # it, the next beat still waits a full cadence instead of
                # retrying immediately and pinning the write lock. The
                # *success* clock (last_prune_at, for the staleness health
                # signal) advances only after the call returns.
                if state is not None:
                    state.last_prune_attempt_at = now
                    if state.prune_pending_since == 0.0:
                        state.prune_pending_since = now
                async with asyncio.timeout(_MAINTENANCE_TASK_TIMEOUT_SECONDS):
                    await repo.prune(
                        dt.timedelta(seconds=self._optimize_prune_retention)
                    )
                if state is not None:
                    state.last_prune_at = now
                    state.prune_pending_since = 0.0
            else:
                # Light beat: lock-free compaction. A commit conflict here
                # is benign — handled below.
                async with asyncio.timeout(_MAINTENANCE_TASK_TIMEOUT_SECONDS):
                    await repo.optimize()
            if state is not None:
                state.optimize_failures = 0
                state.failures_since_fallback = 0
            logger.debug(
                "cascade_lancedb_optimized",
                kind=kind,
                pruned=should_prune,
            )
        except Exception as exc:
            # Benign light-beat commit conflict: the lock-free compaction
            # lost the optimistic-concurrency race against a live writer.
            # Expected under churn, self-heals next beat — log at debug and
            # do NOT count it toward the streak (which would otherwise pin
            # high on a busy table) or trigger a fallback rebuild.
            #
            # This applies to the HEAVY beat too: the per-table write lock is
            # in-process only (see LanceRepoBase.prune), so a second process
            # — a long ``cascade backfill``, a ``cascade sync`` — can still
            # preempt prune's Rewrite commit. Counting those as real failures
            # could fire a spurious fallback rebuild, which replaces every
            # index; if the rebuild lost the race too, the table could sit
            # without an FTS index (all ``/search`` on that kind failing)
            # until the next 12h sweep. A prune that genuinely stops
            # succeeding is caught by the prune-staleness health signal
            # instead (per-kind, see _prune_staleness) — that is the right
            # detector for it, and unlike a rebuild it does not destroy
            # indexes to "fix" a lost race.
            if _is_benign_commit_conflict(exc):
                # ``pruned`` is the whole diagnostic value of this line. Lance
                # labels both beats' commit the same way ("This Rewrite
                # transaction was preempted by ..."), so the message alone
                # cannot tell them apart — but the consequences differ:
                # a lost LIGHT beat is free (compaction retries ~10s later),
                # while a lost HEAVY beat means that table skipped a whole
                # prune cadence and its index dir keeps the superseded files.
                logger.debug(
                    "cascade_lancedb_optimize_conflict",
                    kind=kind,
                    pruned=should_prune,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return
            failures = 0
            since_fallback = 0
            if state is not None:
                state.optimize_failures += 1
                state.failures_since_fallback += 1
                failures = state.optimize_failures
                since_fallback = state.failures_since_fallback
            log = (
                logger.error
                if failures >= _OPTIMIZE_FAILURE_ALERT_THRESHOLD
                else logger.warning
            )
            log(
                "cascade_lancedb_optimize_failed",
                kind=kind,
                pruned=should_prune,
                consecutive_failures=failures,
                error=f"{type(exc).__name__}: {exc}",
            )
            if since_fallback >= _OPTIMIZE_FAILURE_ALERT_THRESHOLD:
                logger.info(
                    "cascade_lancedb_optimize_fallback_rebuild",
                    kind=kind,
                    consecutive_failures=failures,
                )
                await self._run_rebuild_once(kind)
                # Reset the *rate limiter* even when the rebuild fails, so the
                # fallback fires at most once per threshold failures. A failed
                # rebuild defers cleanup to the 12h periodic sweep — harmless
                # for correctness (see _run_rebuild_once docstring).
                #
                # ``optimize_failures`` is deliberately NOT reset here: it is
                # the health signal, and zeroing it in the branch that fires at
                # the threshold is what made the alert unreachable.
                if state is not None:
                    state.failures_since_fallback = 0

    async def _heartbeat_loop(self) -> None:
        """Periodic safety net for the optimizer.

        Recover only a kind whose optimizer state is already dirty and whose
        task is missing or done. Calling the scheduler for every idle handler
        is unsafe because that method deliberately sets dirty=True; an
        unconditional sweep manufactured index UUID directories forever on
        this stack (observed locally) — the same reason the restriction was
        introduced here. Write-driven scheduling plus the startup rebuild
        cover the remaining recovery cases.
        """
        while not self._stop.is_set():
            if await self._wait_or_stop(self._optimize_heartbeat):
                return
            for kind, state in tuple(self._optimizer_states.items()):
                if state.dirty and (state.task is None or state.task.done()):
                    self._schedule_optimize(kind)

    async def _rebuild_loop(self) -> None:
        """Slow per-kind index rebuild loop.

        Workaround for the upstream lancedb / lance gap documented on
        :meth:`LanceRepoBase.rebuild_indexes`. Every
        ``_optimize_rebuild_interval`` seconds we sweep each kind and
        do a full rebuild — this is the **only** lever we have on the
        current stack (lancedb 0.30.2 / lance 4.0) to bound active
        index UUID / FTS ``part_N`` accumulation.

        First sweep fires immediately on worker start to bound any
        accumulation from a previous session. Subsequent sweeps honour
        the interval. Both sweep and each per-kind step respect
        ``self._stop`` so shutdown is prompt.

        Between sweeps the loop wakes on a coarse tick solely to serve
        kinds whose conflict backoff became due (see
        :data:`_REBUILD_CONFLICT_BACKOFFS_SECONDS`). The tick is never
        coarser than the configured cadence so a short configured
        interval keeps its granularity.
        """
        # First sweep: catch up from any prior session before honouring the
        # interval. Rebuild is cheap (~0.3s per 50k rows × indexed columns
        # in local benchmarks); deferring it 12h after startup risks long
        # accumulation if the daemon restarts often.
        for kind in self._handlers:
            if self._stop.is_set():
                return
            await self._run_rebuild_once(kind)
        last_sweep = time.monotonic()
        tick = min(_REBUILD_LOOP_TICK_SECONDS, self._optimize_rebuild_interval)
        while not self._stop.is_set():
            if await self._wait_or_stop(tick):
                return
            now = time.monotonic()
            if now - last_sweep >= self._optimize_rebuild_interval:
                last_sweep = now
                for kind in self._handlers:
                    if self._stop.is_set():
                        return
                    await self._run_rebuild_once(kind)
                continue
            # Between sweeps, serve only kinds whose conflict backoff is due.
            for kind in self._handlers:
                state = self._optimizer_states.get(kind)
                if state is None or not state.rebuild_retry_at:
                    continue
                if now < state.rebuild_retry_at:
                    continue
                state.rebuild_retry_at = 0.0
                if self._stop.is_set():
                    return
                await self._run_rebuild_once(kind)

    async def _run_rebuild_once(self, kind: str) -> None:
        """Rebuild every index on ``kind``'s LanceDB table (in place).

        Waits (bounded) for any in-flight maintenance task to settle,
        then claims the per-kind rebuild slot so ``schedule_optimize``
        calls during the rebuild coalesce instead of racing on the
        manifest.
        """
        handler = self._handlers.get(kind)
        repo = getattr(handler, "lance_repo", None) if handler else None
        if repo is None:
            return
        state = self._optimizer_states.setdefault(kind, _KindOptimizerState())
        # Drain any in-flight optimize before taking the rebuild slot —
        # both would commit on the same manifest version. The optimize
        # runner reciprocates (it awaits ``state.rebuild_task`` on entry).
        # Skip when ``state.task`` is the current task: the fallback-rebuild
        # path in ``_run_optimize_once`` reaches here from *inside* the
        # optimize runner itself, so awaiting ``state.task`` would be
        # self-await (asyncio raises RuntimeError).
        if (
            state.task is not None
            and not state.task.done()
            and state.task is not asyncio.current_task()
        ):
            try:
                # Bounded: an optimize task that hangs must not park the
                # rebuild loop behind it.
                async with asyncio.timeout(_MAINTENANCE_TASK_TIMEOUT_SECONDS):
                    await state.task
            except TimeoutError:
                # Skip this sweep rather than rebuild concurrently with an
                # optimize that is still running: replacing indices under it
                # is exactly the interleaving this wait exists to prevent.
                # The 12h loop retries, and the prune-staleness signal covers
                # the stall.
                logger.warning(
                    "cascade_lancedb_rebuild_skipped_optimize_unfinished",
                    kind=kind,
                    waited_seconds=_MAINTENANCE_TASK_TIMEOUT_SECONDS,
                )
                return
            except Exception:  # noqa: BLE001 — the runner already logged it
                pass
        # Retry a lost commit race in place. Lance labels it "Retryable" and
        # means it: the rebuild transaction was preempted by a concurrent
        # writer (another process, since the write lock is in-process only)
        # and nothing about the table is wrong. Without a retry, a conflict
        # costs a whole rebuild cadence — 12h in production — for what a
        # backoff resolves.
        rebuild_task = asyncio.create_task(
            repo.rebuild_indexes(), name=f"cascade-rebuild-{kind}-inner"
        )
        state.rebuild_task = rebuild_task
        try:
            await rebuild_task
            logger.info("cascade_lancedb_rebuilt", kind=kind)
            state.rebuild_attempt = 0
            state.rebuild_retry_at = 0.0
        except Exception as exc:  # noqa: BLE001 — classified below
            attempt = state.rebuild_attempt
            if _is_benign_commit_conflict(exc) and attempt < len(
                _REBUILD_CONFLICT_BACKOFFS_SECONDS
            ):
                # Lost the manifest race to a concurrent writer; the table is
                # fine. Record a deadline instead of sleeping so the other
                # kinds in this sweep are not parked behind the backoff.
                delay = _REBUILD_CONFLICT_BACKOFFS_SECONDS[attempt]
                state.rebuild_attempt = attempt + 1
                state.rebuild_retry_at = time.monotonic() + delay
                logger.info(
                    "cascade_lancedb_rebuild_conflict_retry_scheduled",
                    kind=kind,
                    attempt=attempt,
                    retry_in_seconds=delay,
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                state.rebuild_attempt = 0
                state.rebuild_retry_at = 0.0
                logger.warning(
                    "cascade_lancedb_rebuild_failed",
                    kind=kind,
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
        finally:
            if state.rebuild_task is rebuild_task:
                state.rebuild_task = None

    async def _flush_optimizers(self) -> None:
        """Wait for every in-flight maintenance task to settle.

        Drain-loop path is fire-and-forget for throughput; this is the
        explicit barrier used by CLI ``cascade sync`` and worker
        shutdown to ensure FTS is consistent with the data on disk
        before the call returns.

        Exceptions from maintenance tasks are already logged in
        :meth:`_run_optimize_once`; ``return_exceptions=True`` here
        keeps the flush itself from raising.
        """
        pending: list[asyncio.Task[None]] = []
        for state in self._optimizer_states.values():
            if state.task is not None and not state.task.done():
                pending.append(state.task)
            if state.rebuild_task is not None and not state.rebuild_task.done():
                pending.append(state.rebuild_task)
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
