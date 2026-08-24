"""``GET /health`` — liveness vs readiness contract.

Verifies:

1. ``status`` stays ``"ok"`` with HTTP 200 whenever the process is up
   (liveness compat for existing pollers) — including when the cascade
   subsystem is degraded.
2. The ``cascade`` block maps healthy / degraded states correctly and is
   absent (``null``) on an app built without the cascade lifespan.
3. ``failed_permanent`` is informational: it never flips
   ``cascade.healthy``.
4. A failing health probe (locked SQLite etc.) degrades *readiness* but
   keeps HTTP 200 + ``status "ok"``.
"""

from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from everos.entrypoints.api.routes import health as health_module
from everos.memory.cascade import CascadeHealth, CascadeOrchestrator


def _verdict(
    *,
    healthy: bool = True,
    reasons: list[str] | None = None,
    failed_permanent: int = 0,
    pending: int = 0,
) -> CascadeHealth:
    return CascadeHealth(
        healthy=healthy,
        reasons=reasons or [],
        pending=pending,
        failed_permanent=failed_permanent,
        failed_retryable=0,
        drain_consecutive_failures=0,
        unrecoverable_total=0,
        optimize_failure_streak=0,
        prune_stale_seconds=0.0,
    )


class _StubOrchestrator(CascadeOrchestrator):
    """Bypasses real construction; only :meth:`health` is consulted."""

    def __init__(self, verdict_or_exc: object) -> None:
        self._verdict_or_exc = verdict_or_exc

    async def health(self) -> CascadeHealth:
        if isinstance(self._verdict_or_exc, Exception):
            raise self._verdict_or_exc
        assert isinstance(self._verdict_or_exc, CascadeHealth)
        return self._verdict_or_exc


def _app(orch: CascadeOrchestrator | None) -> FastAPI:
    app = FastAPI()
    app.include_router(health_module.router)
    if orch is not None:
        app.state.lifespan_data = {"cascade": orch}
    else:
        app.state.lifespan_data = {}
    return app


async def _get(app: FastAPI) -> tuple[int, dict[str, object]]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")
        return resp.status_code, resp.json()


async def test_no_cascade_lifespan_keeps_legacy_shape() -> None:
    """Minimal app: legacy consumers see ``{"status": "ok", ...}``, cascade null."""
    code, body = await _get(_app(None))
    assert code == 200
    assert body["status"] == "ok"
    assert body["cascade"] is None
    assert isinstance(body["version"], str)


async def test_healthy_cascade_maps_all_fields() -> None:
    orch = _StubOrchestrator(_verdict(pending=2, failed_permanent=4))
    code, body = await _get(_app(orch))
    assert code == 200
    assert body["status"] == "ok"
    block = body["cascade"]
    assert block is not None
    assert block["healthy"] is True
    assert block["reasons"] == []
    assert block["pending"] == 2
    assert block["failed_permanent"] == 4
    # Every contract field present for alerting consumers.
    expected_keys = {
        "healthy",
        "reasons",
        "pending",
        "failed_permanent",
        "failed_retryable",
        "drain_consecutive_failures",
        "unrecoverable_total",
        "optimize_failure_streak",
        "prune_stale_seconds",
    }
    assert expected_keys <= set(block.keys())


async def test_degraded_cascade_does_not_flip_liveness() -> None:
    """Operational degradation flips ``cascade.healthy`` only."""
    orch = _StubOrchestrator(
        _verdict(
            healthy=False,
            reasons=["drain loop failing (3 in a row)"],
            pending=7,
        )
    )
    code, body = await _get(_app(orch))
    assert code == 200, "readiness failure must not become an HTTP error"
    assert body["status"] == "ok", "liveness key unchanged for existing pollers"
    block = body["cascade"]
    assert block["healthy"] is False
    assert block["reasons"] == ["drain loop failing (3 in a row)"]
    assert block["pending"] == 7


async def test_failed_permanent_backlog_is_informational_only() -> None:
    """A large permanent-failure backlog keeps ``healthy=True``.

    It is normal steady state (files awaiting ``cascade fix``); folding it
    into readiness would pin the signal red forever."""
    orch = _StubOrchestrator(_verdict(failed_permanent=1234))
    _code, body = await _get(_app(orch))
    block = body["cascade"]
    assert block["healthy"] is True
    assert block["failed_permanent"] == 1234


async def test_probe_failure_degrades_readiness_keeps_http_200() -> None:
    """SQLite locked/mid-migration → unhealthy readiness, HTTP still 200."""
    orch = _StubOrchestrator(RuntimeError("database is locked"))
    code, body = await _get(_app(orch))
    assert code == 200
    assert body["status"] == "ok"
    block = body["cascade"]
    assert block["healthy"] is False
    assert any("cascade health probe failed" in r for r in block["reasons"])
    assert block["pending"] == 0
