"""Read-side service for materialized Memory Wiki snapshots."""

from __future__ import annotations

from typing import Any

from everos.memory.knowledge.materializer import (
    load_current_manifest,
    load_current_page,
)


def index() -> dict[str, object]:
    manifest = load_current_manifest()
    if manifest is None:
        return {
            "available": False,
            "snapshot_id": None,
            "compiler_version": None,
            "materialized_at": None,
            "pages": [],
        }
    return {"available": True, **manifest.as_dict()}


def page(slug: str) -> dict[str, Any] | None:
    return load_current_page(slug)
