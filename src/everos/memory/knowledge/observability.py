"""Low-cardinality metrics for the EV-MEMORY-01 knowledge surfaces.

The helpers deliberately record operational facts only.  Query text, claim
ids, paths, user ids, and other content-bearing values never become metric
labels.  Metrics are created lazily so tests and embedded callers can replace
the process registry before the first observation.
"""

from __future__ import annotations

from dataclasses import dataclass

from everos.core.observability.metrics import Counter, Histogram, HistogramBuckets
from everos.core.observability.metrics.registry import get_metrics_registry


@dataclass(slots=True)
class _KnowledgeMetrics:
    retrieval_seconds: Histogram
    retrieval_results: Counter
    retrieval_filtered: Counter
    wiki_seconds: Histogram
    wiki_claims: Counter
    promotion_recommendations: Counter
    skill_transitions: Counter
    lifecycle_decisions: Counter
    lifecycle_wait_seconds: Histogram


_METRICS_BY_REGISTRY: dict[int, _KnowledgeMetrics] = {}


def _metrics() -> _KnowledgeMetrics:
    registry = get_metrics_registry()
    key = id(registry)
    current = _METRICS_BY_REGISTRY.get(key)
    if current is not None:
        return current
    current = _KnowledgeMetrics(
        retrieval_seconds=Histogram(
            "everos_knowledge_retrieval_duration_seconds",
            "Truth-aware retrieval duration.",
            labelnames=("view",),
            buckets=HistogramBuckets.FAST,
        ),
        retrieval_results=Counter(
            "everos_knowledge_retrieval_results_total",
            "Truth-aware retrieval results returned.",
            labelnames=("view",),
        ),
        retrieval_filtered=Counter(
            "everos_knowledge_retrieval_filtered_total",
            "Claims removed by scope and truth gates.",
            labelnames=("view",),
        ),
        wiki_seconds=Histogram(
            "everos_knowledge_wiki_build_duration_seconds",
            "Deterministic Memory Wiki build duration.",
            buckets=HistogramBuckets.FAST,
        ),
        wiki_claims=Counter(
            "everos_knowledge_wiki_claims_total",
            "Claims rendered by the deterministic Wiki compiler.",
        ),
        promotion_recommendations=Counter(
            "everos_knowledge_promotion_recommendations_total",
            "Promotion recommendations by level.",
            labelnames=("level",),
        ),
        skill_transitions=Counter(
            "everos_knowledge_skill_transitions_total",
            "Procedural-memory lifecycle transitions.",
            labelnames=("from_state", "to_state"),
        ),
        lifecycle_decisions=Counter(
            "everos_boundary_lifecycle_decisions_total",
            "Boundary lifecycle decisions after authority gating.",
            labelnames=("action", "reason"),
        ),
        lifecycle_wait_seconds=Histogram(
            "everos_boundary_lifecycle_wait_seconds",
            "Observed elapsed time for a boundary lifecycle record.",
            buckets=HistogramBuckets.DEFAULT,
        ),
    )
    _METRICS_BY_REGISTRY[key] = current
    return current


def observe_retrieval(
    *,
    view: str,
    elapsed_seconds: float,
    scoped_count: int,
    gated_count: int,
    returned_count: int,
) -> None:
    metrics = _metrics()
    metrics.retrieval_seconds.labels(view=view).observe(max(0.0, elapsed_seconds))
    metrics.retrieval_results.labels(view=view).inc(max(0, returned_count))
    metrics.retrieval_filtered.labels(view=view).inc(max(0, scoped_count - gated_count))


def observe_wiki(*, elapsed_seconds: float, rendered_count: int) -> None:
    metrics = _metrics()
    metrics.wiki_seconds.observe(max(0.0, elapsed_seconds))
    metrics.wiki_claims.inc(max(0, rendered_count))


def observe_promotion(level: str) -> None:
    _metrics().promotion_recommendations.labels(level=level).inc()


def observe_skill_transition(from_state: str, to_state: str) -> None:
    _metrics().skill_transitions.labels(from_state=from_state, to_state=to_state).inc()


def observe_lifecycle_decision(
    *, action: str, reason: str, elapsed_seconds: float
) -> None:
    metrics = _metrics()
    metrics.lifecycle_decisions.labels(action=action, reason=reason).inc()
    metrics.lifecycle_wait_seconds.observe(max(0.0, elapsed_seconds))
