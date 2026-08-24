"""Shared helpers for the API layer (routes)."""

from __future__ import annotations

from fastapi import Request

from everos.memory.cascade import CascadeOrchestrator


def cascade_orchestrator(request: Request) -> CascadeOrchestrator | None:
    """Return the running cascade orchestrator, or ``None``.

    The cascade lifespan stashes the orchestrator at
    ``app.state.lifespan_data["cascade"]``. An app built without that
    lifespan (e.g. a minimal test app) has no entry, so callers get
    ``None`` and degrade gracefully instead of erroring.
    """
    data = getattr(request.app.state, "lifespan_data", None) or {}
    orch = data.get("cascade")
    return orch if isinstance(orch, CascadeOrchestrator) else None
