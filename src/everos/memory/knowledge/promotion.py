"""Dreaming-inspired consolidation recommendation, without authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .observability import observe_promotion


class PromotionLevel(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True, slots=True)
class PromotionObservation:
    candidate_id: str
    occurrence: int = 0
    recall_count: int = 0
    query_diversity: int = 0
    stability_days: float = 0.0
    cross_session_repetition: int = 0
    project_importance: float = 0.0
    explicit_user_emphasis: float = 0.0
    later_correction: int = 0
    contradiction_count: int = 0
    provenance_quality: float = 0.0
    scope_persistence: float = 0.0
    evidence_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PromotionRecommendation:
    candidate_id: str
    level: PromotionLevel
    score: float
    reasons: tuple[str, ...]
    contributing_signals: dict[str, float]
    evidence_refs: tuple[str, ...]
    scoring_version: str = "promotion-v1"
    authority: str = "none"


def score_promotion(observation: PromotionObservation) -> PromotionRecommendation:
    """Score a candidate for review/retention queues only.

    The return object has ``authority='none'`` by construction and the
    function never accepts or mutates a Truth claim.
    """

    signals = {
        "occurrence": min(1.0, observation.occurrence / 5.0),
        "recall_count": min(1.0, observation.recall_count / 10.0),
        "query_diversity": min(1.0, observation.query_diversity / 5.0),
        "stability_days": min(1.0, observation.stability_days / 30.0),
        "cross_session_repetition": min(
            1.0, observation.cross_session_repetition / 3.0
        ),
        "project_importance": max(0.0, min(1.0, observation.project_importance)),
        "explicit_user_emphasis": max(
            0.0, min(1.0, observation.explicit_user_emphasis)
        ),
        "provenance_quality": max(0.0, min(1.0, observation.provenance_quality)),
        "scope_persistence": max(0.0, min(1.0, observation.scope_persistence)),
    }
    weights = {
        "occurrence": 0.16,
        "recall_count": 0.10,
        "query_diversity": 0.10,
        "stability_days": 0.10,
        "cross_session_repetition": 0.12,
        "project_importance": 0.10,
        "explicit_user_emphasis": 0.12,
        "provenance_quality": 0.06,
        "scope_persistence": 0.04,
    }
    score = sum(signals[key] * weight for key, weight in weights.items())
    penalty = min(
        0.55,
        observation.later_correction * 0.12 + observation.contradiction_count * 0.14,
    )
    score = max(0.0, min(1.0, score - penalty))
    if score >= 0.70 and penalty < 0.30:
        level = PromotionLevel.HIGH
    elif score >= 0.40:
        level = PromotionLevel.MEDIUM
    else:
        level = PromotionLevel.LOW
    reasons = [key for key, value in signals.items() if value >= 0.5]
    if observation.later_correction:
        reasons.append("later_correction_penalty")
    if observation.contradiction_count:
        reasons.append("contradiction_penalty")
    recommendation = PromotionRecommendation(
        candidate_id=observation.candidate_id,
        level=level,
        score=score,
        reasons=tuple(reasons),
        contributing_signals={**signals, "penalty": penalty},
        evidence_refs=observation.evidence_refs,
    )
    observe_promotion(recommendation.level.value)
    return recommendation
