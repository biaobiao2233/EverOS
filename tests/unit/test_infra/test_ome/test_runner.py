"""Runner contract tests: attempt-level RunRecord state machine, retry
chain terminal states, DLQ callback, and the per-attempt semaphore +
exponential-backoff-with-jitter behaviour between attempts.

The concurrency regression core: ``engine_sem`` must be released while a
failed run waits out its backoff sleep, so other strategies can use the
slot — a partial outage must not park every slot in ``asyncio.sleep``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from everos.infra.ome._dispatch.runner import Runner
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import StrategyMeta, offline_strategy
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.records import RunStatus
from everos.infra.ome.triggers import Immediate


class _E(BaseEvent):
    user_id: str = "u1"


@pytest.fixture
async def setup(tmp_path: Path):
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    rec_store = RunRecordStore(storage=storage, max_records_per_strategy=1000)
    sem = asyncio.Semaphore(20)
    # Base 0.0 keeps the retry chain instant for tests that exercise
    # retry *counting* rather than backoff timing.
    config = OMEConfig(
        jobstore_path=tmp_path / "ome.db", retry_backoff_base_seconds=0.0
    )
    return rec_store, sem, config


def _runner(
    rec_store: RunRecordStore,
    sem: asyncio.Semaphore,
    config: OMEConfig,
    **kwargs: Any,
) -> Runner:
    return Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        config=config,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_runner_success_marks_record(setup) -> None:
    rec_store, sem, config = setup

    @offline_strategy(name="ok", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    runner = _runner(rec_store, sem, config)
    await runner.run(
        s._ome_strategy_meta,
        _E(),
        run_id="r1",
        max_retries_snapshot=1,
    )

    rec = await rec_store.get("r1")
    assert rec.status == RunStatus.SUCCESS


@pytest.mark.asyncio
async def test_runner_retries_on_failure(setup) -> None:
    rec_store, sem, config = setup
    calls = {"n": 0}

    @offline_strategy(
        name="flaky",
        trigger=Immediate(on=[_E]),
        emits=[],
        max_retries=2,
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")

    runner = _runner(rec_store, sem, config)
    await runner.run(
        s._ome_strategy_meta,
        _E(),
        run_id="r1",
        max_retries_snapshot=2,
    )
    assert calls["n"] == 3
    # Final successful attempt 2 has a new run_id (not "r1");
    # find by status=SUCCESS, strategy_name=flaky
    success_runs = await rec_store.list_runs(
        strategy_name="flaky",
        status=RunStatus.SUCCESS,
    )
    assert len(success_runs) == 1
    assert success_runs[0].attempt == 2


@pytest.mark.asyncio
async def test_runner_dead_letter_after_exhaust(setup) -> None:
    rec_store, sem, config = setup

    @offline_strategy(
        name="bad",
        trigger=Immediate(on=[_E]),
        emits=[],
        max_retries=1,
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        raise RuntimeError("always-fail")

    dl_calls: list = []

    runner = _runner(
        rec_store, sem, config, on_dead_letter=lambda r: dl_calls.append(r)
    )
    await runner.run(
        s._ome_strategy_meta,
        _E(),
        run_id="r1",
        max_retries_snapshot=1,
    )
    dead_runs = await rec_store.list_runs(
        strategy_name="bad",
        status=RunStatus.DEAD_LETTER,
    )
    assert len(dead_runs) == 1
    assert len(dl_calls) == 1


@pytest.mark.asyncio
async def test_runner_emit_must_be_declared(setup) -> None:
    rec_store, sem, config = setup

    class _Other(BaseEvent):
        pass

    @offline_strategy(
        name="emit_undeclared",
        trigger=Immediate(on=[_E]),
        emits=[],
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        await ctx.emit(_Other())  # not declared

    runner = _runner(rec_store, sem, config)
    await runner.run(
        s._ome_strategy_meta,
        _E(),
        run_id="r1",
        max_retries_snapshot=0,
    )
    rec = await rec_store.get("r1")
    assert rec.status == RunStatus.DEAD_LETTER
    assert "EmitNotDeclaredError" in (rec.error or "")


@pytest.mark.asyncio
async def test_runner_negative_max_retries_raises(setup) -> None:
    """``max_retries_snapshot < 0`` is an internal-bug condition (Pydantic
    constrains the user-supplied source to ``>= 0``), so the framework
    fails fast rather than silently no-op the run.
    """
    rec_store, sem, config = setup

    @offline_strategy(name="ok", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    runner = _runner(rec_store, sem, config)
    with pytest.raises(ValueError, match=r"max_retries_snapshot must be >= 0"):
        await runner.run(
            s._ome_strategy_meta,
            _E(),
            run_id="r1",
            max_retries_snapshot=-1,
        )


@pytest.mark.asyncio
async def test_runner_aborts_silently_when_mark_running_fails(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When persistence itself fails before the strategy is invoked,
    the run must exit cleanly (no exception escaping the framework) and
    the strategy body must NOT execute — no RUNNING row exists for
    crash recovery to pick up, so re-execution via recovery is
    impossible. The emergency log is the only audit trail.
    """
    rec_store, sem, config = setup
    called = {"n": 0}

    @offline_strategy(name="ok", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        called["n"] += 1

    async def _boom(**_: object) -> None:
        raise RuntimeError("disk_full")

    monkeypatch.setattr(rec_store, "mark_running", _boom)

    runner = _runner(rec_store, sem, config)
    # Must NOT raise; the framework swallows + logs.
    await runner.run(
        s._ome_strategy_meta,
        _E(),
        run_id="r1",
        max_retries_snapshot=1,
    )
    assert called["n"] == 0


# ── Backoff / per-attempt semaphore ──────────────────────────────────────


async def _make_transient_failing_runner(
    tmp_path: Path,
    *,
    max_retries: int,
    backoff_base: float = 1.0,
    backoff_cap: float = 10.0,
    jitter: float = 0.5,
) -> tuple[Runner, StrategyMeta]:
    """Build a ``Runner`` wired to a strategy that raises on every attempt.

    Drives the retry loop to exhaustion so the backoff sleep fires between
    each of the ``max_retries`` retries, for tests asserting on
    ``asyncio.sleep`` call arguments.
    """
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    rec_store = RunRecordStore(storage=storage, max_records_per_strategy=1000)
    sem = asyncio.Semaphore(20)
    config = OMEConfig(
        jobstore_path=tmp_path / "ome.db",
        retry_backoff_base_seconds=backoff_base,
        retry_backoff_cap_seconds=backoff_cap,
        retry_jitter_seconds=jitter,
    )

    @offline_strategy(
        name="transient_failing",
        trigger=Immediate(on=[_E]),
        emits=[],
        max_retries=max_retries,
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        raise RuntimeError("transient")

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        config=config,
    )
    return runner, s._ome_strategy_meta


@pytest.mark.asyncio
async def test_runner_applies_exponential_backoff_between_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each attempt after the first sleeps ~ base * 2**(attempt-1), capped at
    cap_seconds, with up to jitter_seconds added. Verifies that a strategy
    raising a transient exception gets real wall-clock breathing room."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("everos.infra.ome._dispatch.runner.asyncio.sleep", fake_sleep)

    runner, meta = await _make_transient_failing_runner(tmp_path, max_retries=3)
    await runner.run(meta, _E(), run_id="r1", max_retries_snapshot=3)

    # attempts 1, 2, 3 (0 has no preceding sleep); base=1, cap=10, jitter=0.5
    assert len(sleeps) == 3
    assert 1.0 <= sleeps[0] <= 1.5  # ~1s + jitter
    assert 2.0 <= sleeps[1] <= 2.5  # ~2s + jitter
    assert 4.0 <= sleeps[2] <= 4.5  # ~4s + jitter


@pytest.mark.asyncio
async def test_runner_backoff_caps_at_configured_maximum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backoff cap prevents unbounded growth on high max_retries."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("everos.infra.ome._dispatch.runner.asyncio.sleep", fake_sleep)

    runner, meta = await _make_transient_failing_runner(
        tmp_path,
        max_retries=6,
        backoff_base=1.0,
        backoff_cap=3.0,
        jitter=0.0,
    )
    await runner.run(meta, _E(), run_id="r1", max_retries_snapshot=6)

    # 1, 2, 3, 3, 3, 3 -- all capped after attempt 3
    assert sleeps == [1.0, 2.0, 3.0, 3.0, 3.0, 3.0]


@pytest.mark.asyncio
async def test_runner_zero_base_disables_backoff_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``retry_backoff_base_seconds == 0.0`` must skip the sleep entirely
    (the escape hatch tests and latency-insensitive deployments rely on);
    jitter alone must not reintroduce a sleep when base is 0."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("everos.infra.ome._dispatch.runner.asyncio.sleep", fake_sleep)

    runner, meta = await _make_transient_failing_runner(
        tmp_path,
        max_retries=2,
        backoff_base=0.0,
        jitter=0.5,
    )
    await runner.run(meta, _E(), run_id="r1", max_retries_snapshot=2)

    assert sleeps == []


@pytest.mark.asyncio
async def test_runner_releases_engine_sem_across_backoff_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The concurrency slot must be free while a run waits to retry.

    Uses a single-permit semaphore so ``locked()`` is unambiguous, and
    asserts a second waiter actually acquires during the sleep —
    ``locked()`` alone would pass on an implementation that released the
    slot but left a waiter unable to take it.
    """
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    rec_store = RunRecordStore(storage=storage, max_records_per_strategy=1000)
    sem = asyncio.Semaphore(1)
    config = OMEConfig(
        jobstore_path=tmp_path / "ome.db",
        retry_backoff_base_seconds=1.0,
        retry_backoff_cap_seconds=10.0,
        retry_jitter_seconds=0.0,
    )

    held_during_sleep: list[bool] = []
    acquired_during_sleep: list[bool] = []

    async def fake_sleep(seconds: float) -> None:
        held_during_sleep.append(sem.locked())
        try:
            async with asyncio.timeout(0.5):
                await sem.acquire()
        except TimeoutError:
            acquired_during_sleep.append(False)
        else:
            acquired_during_sleep.append(True)
            sem.release()

    monkeypatch.setattr("everos.infra.ome._dispatch.runner.asyncio.sleep", fake_sleep)

    @offline_strategy(
        name="sem_probe_failing",
        trigger=Immediate(on=[_E]),
        emits=[],
        max_retries=2,
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        raise RuntimeError("transient")

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        config=config,
    )
    await runner.run(s._ome_strategy_meta, _E(), run_id="r1", max_retries_snapshot=2)

    assert held_during_sleep == [False, False]
    assert acquired_during_sleep == [True, True]


@pytest.mark.asyncio
async def test_runner_slot_serves_other_strategy_while_one_is_backing_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance core for per-attempt fairness: with a single concurrency
    slot, Strategy A's failed attempt enters backoff sleep; DURING that
    sleep Strategy B must acquire the semaphore and complete successfully;
    afterwards A's retry completes on the reclaimed slot. Audit rows stay
    exact: every real attempt writes its row, sleeps write none.
    """
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    rec_store = RunRecordStore(storage=storage, max_records_per_strategy=1000)
    sem = asyncio.Semaphore(1)  # max_concurrent_runs=1
    config = OMEConfig(
        jobstore_path=tmp_path / "ome.db",
        retry_backoff_base_seconds=1.0,
        retry_backoff_cap_seconds=10.0,
        retry_jitter_seconds=0.0,
    )

    a_attempts = {"n": 0}
    entered_sleep = asyncio.Event()
    release_sleep = asyncio.Event()

    async def gated_sleep(seconds: float) -> None:
        entered_sleep.set()
        await asyncio.wait_for(release_sleep.wait(), timeout=5.0)

    monkeypatch.setattr("everos.infra.ome._dispatch.runner.asyncio.sleep", gated_sleep)

    @offline_strategy(name="backoff_a", trigger=Immediate(on=[_E]), emits=[])
    async def strat_a(event: _E, ctx: StrategyContext) -> None:
        a_attempts["n"] += 1
        if a_attempts["n"] == 1:
            raise RuntimeError("transient A")

    @offline_strategy(name="steady_b", trigger=Immediate(on=[_E]), emits=[])
    async def strat_b(event: _E, ctx: StrategyContext) -> None:
        return None

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        config=config,
    )

    task_a = asyncio.create_task(
        runner.run(
            strat_a._ome_strategy_meta,
            _E(),
            run_id="a-run",
            max_retries_snapshot=1,
        )
    )

    # Wait until A has failed attempt 0 and parked in its backoff sleep.
    await asyncio.wait_for(entered_sleep.wait(), timeout=5.0)

    # The slot must be free while A sleeps...
    assert not sem.locked()

    # ...and B must complete on it while A is mid-retry-chain. If the
    # semaphore were still held across the sleep this would time out.
    await asyncio.wait_for(
        runner.run(
            strat_b._ome_strategy_meta,
            _E(),
            run_id="b-run",
            max_retries_snapshot=0,
        ),
        timeout=5.0,
    )

    b_rec = await rec_store.get("b-run")
    assert b_rec.status == RunStatus.SUCCESS
    # A is mid-chain: exactly its failed attempt-0 row, nothing terminal.
    a_mid = await rec_store.list_runs(strategy_name="backoff_a")
    assert [(r.attempt, r.status) for r in a_mid] == [(0, RunStatus.FAILED)]

    # Release A's sleep; its retry re-acquires the slot and succeeds.
    release_sleep.set()
    await asyncio.wait_for(task_a, timeout=5.0)

    a_final = await rec_store.list_runs(strategy_name="backoff_a")
    by_attempt = {r.attempt: r.status for r in a_final}
    assert by_attempt == {
        0: RunStatus.FAILED,
        1: RunStatus.SUCCESS,
    }
    assert a_attempts["n"] == 2
    # B's record untouched by A's retry chain.
    assert (await rec_store.get("b-run")).status == RunStatus.SUCCESS


async def _no_emit(event: BaseEvent) -> None:
    return None
