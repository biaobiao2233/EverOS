"""Health check route — liveness + cascade readiness probe.

``status`` stays ``"ok"`` with HTTP 200 whenever the process is up: the
endpoint's top-level key is a **liveness** signal and existing consumers
(the production Control Center poller) key off exactly this shape. A
degraded cascade must not flip it — restarting a server neither fixes a
bad md file nor reclaims disk.

The optional ``cascade`` block is the **readiness** signal for the
md → LanceDB projection: ``healthy=false`` with human-readable
``reasons`` only when the pipeline itself is stuck (drain failing,
optimize stuck, version cleanup stalled). ``failed_permanent`` — files
awaiting ``cascade fix`` — is a data-quality backlog reported as an
informational count that does **not** flip ``healthy``. Alert on
``cascade.healthy``, not on ``status``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from everos import __version__
from everos.core.observability.logging import get_logger
from everos.entrypoints.api.utils import cascade_orchestrator

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


class CascadeHealthBlock(BaseModel):
    """Readiness of the md → LanceDB projection (cascade) subsystem.

    ``healthy`` reflects **operational** health only — drain loop alive,
    optimize not stuck, version cleanup (prune) not stalled — and is what
    alerting should watch. ``failed_permanent`` (md files awaiting
    ``cascade fix``) is a normal data-quality backlog reported as an
    informational count; it does **not** flip ``healthy``, otherwise the
    signal would sit red forever.
    """

    healthy: bool
    reasons: list[str]
    pending: int
    failed_permanent: int
    failed_retryable: int
    drain_consecutive_failures: int
    unrecoverable_total: int
    optimize_failure_streak: int
    prune_stale_seconds: float


class HealthResponse(BaseModel):
    """Response schema for ``GET /health``.

    Declared as a Pydantic model (not ``dict``) so the generated
    OpenAPI schema carries the full field shape — ``cascade`` is typed.
    A bare ``-> dict`` return type degrades the OpenAPI response to
    ``additionalProperties: true``, which robs clients (and codegen) of
    any structure to lean on.
    """

    status: str
    version: str
    cascade: CascadeHealthBlock | None = None
    """Present when the cascade lifespan is running; ``None`` for a
    minimal app built without it."""


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness + cascade readiness probe."""
    cascade: CascadeHealthBlock | None = None
    orch = cascade_orchestrator(request)
    if orch is not None:
        try:
            ch = await orch.health()
            cascade = CascadeHealthBlock(
                healthy=ch.healthy,
                reasons=ch.reasons,
                pending=ch.pending,
                failed_permanent=ch.failed_permanent,
                failed_retryable=ch.failed_retryable,
                drain_consecutive_failures=ch.drain_consecutive_failures,
                unrecoverable_total=ch.unrecoverable_total,
                optimize_failure_streak=ch.optimize_failure_streak,
                prune_stale_seconds=round(ch.prune_stale_seconds, 1),
            )
        except Exception as exc:
            # The probe reads SQLite (queue_summary runs aggregate counts).
            # A locked / full / mid-migration DB must NOT turn /health into a
            # 500 — that flips the liveness signal and makes a supervisor
            # restart the service, which fixes neither a stuck DB nor disk
            # bloat. Surface it as unhealthy *readiness* with a reason and
            # keep HTTP 200 + status "ok".
            logger.warning("cascade_health_probe_failed", error=repr(exc))
            cascade = CascadeHealthBlock(
                healthy=False,
                reasons=[f"cascade health probe failed: {exc!r}"],
                pending=0,
                failed_permanent=0,
                failed_retryable=0,
                drain_consecutive_failures=0,
                unrecoverable_total=0,
                optimize_failure_streak=0,
                prune_stale_seconds=0.0,
            )
    return HealthResponse(status="ok", version=__version__, cascade=cascade)
