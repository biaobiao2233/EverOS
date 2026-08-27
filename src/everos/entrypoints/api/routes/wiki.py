"""Read-only Memory Wiki routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from everos.memory.knowledge.dto import WikiIndexResponse, WikiPageResponse
from everos.service import wiki as wiki_service

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


@router.get(
    "/wiki",
    response_model=WikiIndexResponse,
    response_model_exclude_unset=True,
)
def wiki_index() -> WikiIndexResponse:
    return WikiIndexResponse(data=wiki_service.index())


@router.get("/wiki/{slug}", response_model=WikiPageResponse)
def wiki_page(slug: str) -> WikiPageResponse:
    if len(slug) > 160 or "/" in slug or "\\" in slug or slug in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid wiki page slug")
    page = wiki_service.page(slug)
    if page is None:
        raise HTTPException(status_code=404, detail="Wiki page not found")
    return WikiPageResponse(data=page)
