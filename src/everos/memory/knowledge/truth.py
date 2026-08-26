"""Small, explicit truth envelope used by read-side knowledge services.

Existing EverOS rows predate these fields.  Adapters therefore default rows
without an envelope to ``accepted/current`` because extracted Markdown and
its LanceDB projection are already the accepted semantic-memory surface.  A
row that does carry an envelope is never allowed to bypass its authority.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from everos.component.utils.datetime import ensure_utc


class AuthorityState(StrEnum):
    ACCEPTED = "accepted"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


class TruthClass(StrEnum):
    CURRENT = "current"
    HISTORY = "history"
    CONFLICT = "conflict"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class TruthClaim:
    """One claim with enough metadata for a safe read-side decision."""

    claim_id: str
    text: str
    kind: str
    scope: Mapping[str, str] = field(default_factory=dict)
    authority: AuthorityState = AuthorityState.ACCEPTED
    truth_class: TruthClass = TruthClass.CURRENT
    superseded_by: str | None = None
    valid_from: dt.datetime | None = None
    valid_until: dt.datetime | None = None
    conflict_group: str | None = None
    source_refs: tuple[str, ...] = ()
    created_at: dt.datetime | None = None
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def is_temporally_valid(self, when: dt.datetime | None) -> bool:
        when_utc = ensure_utc(when)
        if when_utc is None:
            return True
        valid_from = ensure_utc(self.valid_from)
        valid_until = ensure_utc(self.valid_until)
        if valid_from is not None and valid_from > when_utc:
            return False
        return valid_until is None or valid_until > when_utc

    def canonical_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, deterministic representation."""

        def iso(value: dt.datetime | None) -> str | None:
            return value.isoformat() if value is not None else None

        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "kind": self.kind,
            "scope": dict(sorted(self.scope.items())),
            "authority": self.authority.value,
            "truth_class": self.truth_class.value,
            "superseded_by": self.superseded_by,
            "valid_from": iso(self.valid_from),
            "valid_until": iso(self.valid_until),
            "conflict_group": self.conflict_group,
            "source_refs": list(self.source_refs),
            "created_at": iso(self.created_at),
            "semantic_score": self.semantic_score,
            "bm25_score": self.bm25_score,
            "metadata": dict(self.metadata),
        }
