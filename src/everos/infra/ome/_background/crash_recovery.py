"""Startup crash recovery — orphaned RUNNING rows → re-enqueue + CRASHED.

The engine calls this while holding its single-instance lock and while
APScheduler is paused.  Therefore every pre-existing RUNNING row belongs to
the dead process, regardless of age.  The recovery job is persisted first;
only then is the old row marked CRASHED, so an enqueue failure remains
retryable on the next service start.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from uuid import NAMESPACE_URL, uuid5

from everos.component.utils.datetime import get_utc_now
from everos.core.observability.logging import get_logger
from everos.infra.ome._stores.run_record import RunRecordStore

logger = get_logger(__name__)


async def scan_and_resume(
    *,
    run_record_store: RunRecordStore,
    timeout_seconds: int,
    add_job: Callable[[str, str, str, str, int], Awaitable[None]],
    recover_all: bool = False,
) -> None:
    """Scan RUNNING rows and durably schedule their replacements.

    ``add_job`` is called with positional args
    ``(strategy_name, run_id, event_topic, event_payload, max_retries)``.

    Raises:
        ValueError: If ``timeout_seconds`` is not positive.
    """
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
    now = get_utc_now()
    cutoff = now - timedelta(seconds=timeout_seconds)
    running = await run_record_store.find_running()
    failures: list[tuple[str, Exception]] = []
    for rec in running:
        if not recover_all and rec.started_at >= cutoff:
            continue
        new_run_id = uuid5(NAMESPACE_URL, f"everos:ome:recovery:{rec.run_id}").hex
        remaining_retries = max(0, rec.max_retries_snapshot - rec.attempt)
        try:
            await add_job(
                rec.strategy_name,
                new_run_id,
                rec.event_topic,
                rec.event_payload,
                remaining_retries,
            )
            await run_record_store.mark_crashed(
                run_id=rec.run_id,
                finished_at=now,
                error="crash recovery: replacement job persisted",
            )
            logger.info(
                "crash_recovery_resumed",
                strategy_name=rec.strategy_name,
                event_topic=rec.event_topic,
                old_run_id=rec.run_id,
                new_run_id=new_run_id,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((rec.run_id, exc))
            logger.exception(
                "crash_recovery_resume_failed",
                strategy_name=rec.strategy_name,
                event_topic=rec.event_topic,
                old_run_id=rec.run_id,
            )
    if failures:
        run_ids = ", ".join(run_id for run_id, _ in failures)
        raise RuntimeError(
            f"crash recovery could not persist replacements for: {run_ids}"
        ) from failures[0][1]
