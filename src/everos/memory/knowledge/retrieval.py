"""Truth-aware hybrid retrieval.

The algorithm is intentionally independent of a particular vector engine.
LanceDB/everalgo candidates are adapted at the boundary and then pass through
the same hard truth gates.  This keeps semantic relevance from becoming an
authority decision.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from everalgo.types import Candidate

from everos.component.utils.datetime import ensure_utc, get_utc_now

from .truth import AuthorityState, TruthClaim, TruthClass


class TruthView(StrEnum):
    CURRENT = "current"
    HISTORY = "history"
    ALL_ACCEPTED = "all_accepted"


_HISTORY_WORDS = re.compile(
    r"\b(previous|formerly|historical|history|before|old|past)\b|以前|曾经|过去|历史|原来",
    re.I,
)


@dataclass(frozen=True, slots=True)
class TruthAwareQuery:
    query: str
    scope: dict[str, str]
    view: TruthView = TruthView.CURRENT
    as_of: dt.datetime | None = None
    include_candidates: bool = False
    top_k: int = 10
    mmr_lambda: float = 0.85

    @property
    def asks_history(self) -> bool:
        return self.view == TruthView.HISTORY or bool(_HISTORY_WORDS.search(self.query))


@dataclass(frozen=True, slots=True)
class RankedTruthClaim:
    claim: TruthClaim
    score: float
    truth_priority: int
    reasons: tuple[str, ...]


class TruthAwareRetriever:
    """Deterministic scope → truth → hybrid → recency/MMR pipeline."""

    def retrieve(
        self,
        claims: Iterable[TruthClaim],
        request: TruthAwareQuery,
    ) -> list[RankedTruthClaim]:
        scoped = [c for c in claims if _scope_matches(c, request.scope)]
        gated = self._truth_gate(scoped, request)
        ranked = [self._rank(c, request) for c in gated]
        ranked.sort(key=_sort_key)
        if request.mmr_lambda < 1.0:
            ranked = _mmr(
                ranked,
                len(ranked) if request.top_k < 0 else request.top_k,
                request.mmr_lambda,
            )
        return ranked if request.top_k < 0 else ranked[: request.top_k]

    def _truth_gate(
        self,
        claims: list[TruthClaim],
        request: TruthAwareQuery,
    ) -> list[TruthClaim]:
        accepted = [
            claim
            for claim in claims
            if claim.authority == AuthorityState.ACCEPTED
            or (
                request.include_candidates
                and claim.authority == AuthorityState.CANDIDATE
            )
        ]
        validity_time = _aware(request.as_of) if request.as_of else get_utc_now()
        active = [
            claim for claim in accepted if claim.is_temporally_valid(validity_time)
        ]
        if request.view == TruthView.CURRENT and not request.asks_history:
            active = [
                claim
                for claim in active
                if (
                    claim.truth_class == TruthClass.CURRENT
                    and claim.superseded_by is None
                )
            ]
            # A candidate can never displace accepted current truth.  If two
            # accepted active claims are unresolved conflicts, return neither
            # instead of choosing by vector score.
            conflict_ids = _unresolved_conflict_ids(active)
            active = [claim for claim in active if claim.claim_id not in conflict_ids]
        elif request.view == TruthView.HISTORY:
            active = [
                claim for claim in active if claim.truth_class == TruthClass.HISTORY
            ]
        elif request.asks_history:
            # Natural-language historical intent is a read of the historical
            # view, not permission to mix current truth into the answer.
            active = [
                claim for claim in active if claim.truth_class == TruthClass.HISTORY
            ]
        else:
            active = [
                claim for claim in active if claim.truth_class != TruthClass.EXPIRED
            ]
        return active

    @staticmethod
    def _rank(claim: TruthClaim, request: TruthAwareQuery) -> RankedTruthClaim:
        lexical = _lexical_score(request.query, claim.text)
        semantic = max(0.0, min(1.0, float(claim.semantic_score)))
        bm25 = max(0.0, min(1.0, float(claim.bm25_score)))
        # Sparse and dense scores are fused before recency.  A row that has
        # only one score still remains searchable, but the truth gate always
        # ran first.
        score = 0.45 * bm25 + 0.35 * semantic + 0.20 * lexical
        if claim.created_at is not None:
            reference_time = _aware(request.as_of) if request.as_of else get_utc_now()
            age_days = max(
                0.0,
                (reference_time - _aware(claim.created_at)).total_seconds() / 86400.0,
            )
            score += 0.05 / (1.0 + age_days / 30.0)
        priority = 0 if claim.truth_class == TruthClass.CURRENT else 1
        reasons = ("scope_pass", "accepted_authority", "hybrid_fused")
        return RankedTruthClaim(claim, score, priority, reasons)


def retrieve_candidates(
    candidates: Sequence[Candidate],
    *,
    query: str,
    scope: dict[str, str],
    view: TruthView = TruthView.CURRENT,
    as_of: dt.datetime | None = None,
    top_k: int = 10,
    include_candidates: bool = False,
) -> list[Candidate]:
    """Apply the same policy to existing EverOS Candidate objects.

    Missing envelope fields are treated as accepted/current for backward
    compatibility with the pre-EV-MEMORY-01 LanceDB schema. New writers can
    add the fields without changing the ranking API.
    """

    claims: list[TruthClaim] = []
    by_id: dict[str, Candidate] = {}
    envelope_keys = {
        "authority",
        "authority_state",
        "truth_class",
        "superseded_by",
        "valid_from",
        "valid_until",
        "conflict_group",
    }
    if not any(envelope_keys & candidate.metadata.keys() for candidate in candidates):
        # Existing 0.30.2 rows are already the accepted semantic-memory
        # projection and have no envelope columns. Preserve their exact
        # ranking/score until a writer adds explicit truth metadata.
        return list(candidates[:top_k])
    for candidate in candidates:
        by_id[candidate.id] = candidate
        metadata = candidate.metadata
        has_envelope = bool(envelope_keys & metadata.keys())
        claims.append(
            TruthClaim(
                claim_id=candidate.id,
                text=_candidate_text(metadata),
                kind=str(metadata.get("kind", "memory")),
                # Rows from the pre-envelope schema have no tenant fields.
                # Treat them as already accepted/current and inherit the
                # caller's scope so a mixed old/new recall pool does not
                # silently lose otherwise compatible legacy rows.
                scope=(_candidate_scope(metadata) if has_envelope else dict(scope)),
                authority=_authority(metadata),
                truth_class=_truth_class(metadata),
                superseded_by=_optional_str(metadata.get("superseded_by")),
                valid_from=_optional_datetime(metadata.get("valid_from")),
                valid_until=_optional_datetime(metadata.get("valid_until")),
                conflict_group=_optional_str(metadata.get("conflict_group")),
                source_refs=tuple(_string_list(metadata.get("source_refs"))),
                created_at=_optional_datetime(metadata.get("created_at")),
                semantic_score=_score_for_source(candidate, "vector"),
                bm25_score=_score_for_source(candidate, "keyword"),
                metadata=metadata,
            )
        )
    ranked = TruthAwareRetriever().retrieve(
        claims,
        TruthAwareQuery(
            query=query,
            scope=scope,
            view=view,
            as_of=as_of,
            include_candidates=include_candidates,
            top_k=top_k,
        ),
    )
    result: list[Candidate] = []
    for hit in ranked:
        source = by_id[hit.claim.claim_id]
        result.append(
            Candidate(
                id=source.id,
                score=hit.score,
                source=source.source,
                metadata={**source.metadata, "truth_rank_reasons": list(hit.reasons)},
            )
        )
    return result


def _scope_matches(claim: TruthClaim, requested: dict[str, str]) -> bool:
    return all(claim.scope.get(key) == value for key, value in requested.items())


def _unresolved_conflict_ids(claims: list[TruthClaim]) -> set[str]:
    groups: dict[str, list[TruthClaim]] = {}
    for claim in claims:
        if claim.conflict_group:
            groups.setdefault(claim.conflict_group, []).append(claim)
    return {
        claim.claim_id for group in groups.values() if len(group) > 1 for claim in group
    }


def _sort_key(hit: RankedTruthClaim) -> tuple[int, float, str]:
    # Truth class is intentionally first: recency/semantic score cannot
    # promote HISTORY over CURRENT in a current query.
    return (hit.truth_priority, -hit.score, hit.claim.claim_id)


def _mmr(
    ranked: list[RankedTruthClaim], top_k: int, lambda_: float
) -> list[RankedTruthClaim]:
    if not ranked or lambda_ >= 1.0:
        return ranked[:top_k]
    selected: list[RankedTruthClaim] = []
    remaining = list(ranked)
    while remaining and len(selected) < top_k:
        if not selected:
            selected.append(remaining.pop(0))
            continue
        best = max(
            remaining,
            key=lambda hit: (
                lambda_ * hit.score
                - (1.0 - lambda_)
                * max(
                    _text_similarity(hit.claim.text, prior.claim.text)
                    for prior in selected
                )
            ),
        )
        remaining.remove(best)
        selected.append(best)
    return selected


def _lexical_score(query: str, text: str) -> float:
    q = set(_tokens(query))
    t = set(_tokens(text))
    return len(q & t) / max(1, len(q))


def _text_similarity(left: str, right: str) -> float:
    a, b = set(_tokens(left)), set(_tokens(right))
    return len(a & b) / max(1, len(a | b))


def _tokens(value: str) -> list[str]:
    return [item.casefold() for item in re.findall(r"[\w\u4e00-\u9fff]+", value)]


def _candidate_text(metadata: dict[str, Any]) -> str:
    for key in ("episode", "fact", "content", "approach", "description", "task_intent"):
        value = metadata.get(key)
        if isinstance(value, str):
            return value
    return ""


def _candidate_scope(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        key: value
        for key in ("app_id", "project_id", "owner_id", "session_id")
        if isinstance((value := metadata.get(key)), str)
    }


def _authority(metadata: dict[str, Any]) -> AuthorityState:
    value = metadata.get("authority", metadata.get("authority_state", "accepted"))
    try:
        return AuthorityState(str(value))
    except ValueError:
        return AuthorityState.REJECTED


def _truth_class(metadata: dict[str, Any]) -> TruthClass:
    value = metadata.get("truth_class", "current")
    try:
        return TruthClass(str(value))
    except ValueError:
        return TruthClass.HISTORY


def _score_for_source(candidate: Candidate, source: str) -> float:
    # ``arank`` returns ``source='other'`` after fusion/rerank. Its score is
    # already the fused relevance, so do not erase it when the post-rank truth
    # gate re-adapts the candidate.
    return candidate.score if candidate.source in {source, "other"} else 0.0


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_list(value: Any) -> list[str]:
    return (
        [item for item in value if isinstance(item, str)]
        if isinstance(value, list)
        else []
    )


def _optional_datetime(value: Any) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _aware(value: dt.datetime) -> dt.datetime:
    normalized = ensure_utc(value)
    assert normalized is not None
    return normalized
