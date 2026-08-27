"""Public read-only DTOs for materialized Memory Wiki endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WikiPageSummaryDto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1)
    project_key: str = Field(min_length=1)
    artifact_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    rendered_claim_count: int = Field(ge=0)
    backed_claim_count: int = Field(ge=0)
    unsupported_claim_count: int = Field(ge=0)
    stale_claim_count: int = Field(ge=0)
    verification_status: Literal["VERIFIED", "FAILED"]
    claim_set_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    agent_view_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    markdown_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class WikiIndexData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    schema_version: int | None = Field(default=None, ge=1)
    snapshot_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    source_truth_view_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    compiler_version: str | None = None
    materialized_at: str | None = None
    pages: list[WikiPageSummaryDto] = Field(default_factory=list)


class WikiIndexResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: WikiIndexData


class WikiPageData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    snapshot_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    page: WikiPageSummaryDto
    agent_view: dict[str, Any]
    markdown: str


class WikiPageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: WikiPageData
