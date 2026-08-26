"""Deterministic Memory Wiki compiler.

The compiler consumes an explicit accepted Truth snapshot.  It does not
search, summarize, infer, or write back to Truth.  Human Markdown and agent
JSON are two renderings of one immutable compiled claim set.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

from .truth import AuthorityState, TruthClaim, TruthClass

COMPILER_VERSION = "wiki-compiler-v1"
_SECTION_NAMES = {
    "PROJECT_STATE": "Current State",
    "GOAL": "Goals",
    "DECISION": "Active Decisions",
    "CONSTRAINT": "Constraints",
    "TODO": "Open Work",
    "HISTORY": "History",
}


@dataclass(frozen=True, slots=True)
class WikiPage:
    snapshot_id: str
    compiler_version: str
    slug: str
    title: str
    claims: tuple[TruthClaim, ...]
    rendered_claim_count: int
    backed_claim_count: int
    unsupported_claim_count: int
    stale_claim_count: int
    verification_status: str
    claim_set_sha256: str

    def agent_view(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "compiler_version": self.compiler_version,
            "slug": self.slug,
            "title": self.title,
            "claims": [claim.canonical_dict() for claim in self.claims],
            "rendered_claim_count": self.rendered_claim_count,
            "backed_claim_count": self.backed_claim_count,
            "unsupported_claim_count": self.unsupported_claim_count,
            "stale_claim_count": self.stale_claim_count,
            "verification_status": self.verification_status,
            "claim_set_sha256": self.claim_set_sha256,
        }

    def human_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"- Truth Snapshot ID: `{self.snapshot_id}`",
            f"- Compiler Version: `{self.compiler_version}`",
            f"- Rendered Claim Count: {self.rendered_claim_count}",
            f"- Backed Claim Count: {self.backed_claim_count}",
            f"- Unsupported Claim Count: {self.unsupported_claim_count}",
            f"- Stale Claim Count: {self.stale_claim_count}",
            f"- Verification Status: `{self.verification_status}`",
            "",
        ]
        for kind, heading in _SECTION_NAMES.items():
            section = [claim for claim in self.claims if _section_kind(claim) == kind]
            if not section:
                continue
            lines.extend([f"## {heading}", ""])
            for claim in section:
                refs = ", ".join(f"`{ref}`" for ref in claim.source_refs)
                lines.append(f"- **{claim.claim_id}** — {claim.text}")
                if refs:
                    lines.append(f"  - Provenance: {refs}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


class WikiCompiler:
    """Compile a stable page from a supplied Truth snapshot."""

    def __init__(self, *, version: str = COMPILER_VERSION) -> None:
        self.version = version

    def compile(
        self,
        *,
        snapshot_id: str,
        title: str,
        claims: Iterable[TruthClaim],
        scope: dict[str, str] | None = None,
    ) -> WikiPage:
        selected = [
            claim
            for claim in claims
            if claim.authority == AuthorityState.ACCEPTED
            and claim.truth_class != TruthClass.EXPIRED
            and claim.truth_class in {TruthClass.CURRENT, TruthClass.HISTORY}
            and claim.superseded_by is None
            and bool(claim.source_refs)
            and (
                scope is None or all(claim.scope.get(k) == v for k, v in scope.items())
            )
            and claim.kind in _SECTION_NAMES
        ]
        ordered = tuple(sorted(selected, key=lambda c: (c.kind, c.claim_id)))
        backed = sum(bool(claim.source_refs) for claim in ordered)
        stale = sum(claim.truth_class == TruthClass.EXPIRED for claim in ordered)
        unsupported = len(ordered) - backed
        payload = json.dumps(
            [claim.canonical_dict() for claim in ordered],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return WikiPage(
            snapshot_id=snapshot_id,
            compiler_version=self.version,
            slug=slugify(title),
            title=title,
            claims=ordered,
            rendered_claim_count=len(ordered),
            backed_claim_count=backed,
            unsupported_claim_count=unsupported,
            stale_claim_count=stale,
            verification_status=(
                "VERIFIED" if unsupported == 0 and stale == 0 else "FAILED"
            ),
            claim_set_sha256=hashlib.sha256(payload).hexdigest(),
        )


def slugify(value: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", value.strip().casefold(), flags=re.UNICODE)
    return slug.strip("-") or "memory"


def _section_kind(claim: TruthClaim) -> str:
    """Keep historical truth in the History section regardless of source kind."""

    if claim.truth_class == TruthClass.HISTORY or claim.kind == "HISTORY":
        return "HISTORY"
    return claim.kind
