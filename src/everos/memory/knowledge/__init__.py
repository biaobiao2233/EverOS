"""Knowledge-access primitives built on top of accepted EverOS truth.

The package deliberately contains no write path that can create Truth.  It
provides deterministic lifecycle decisions, truth-aware read ranking, wiki
materialisation, promotion recommendations, and procedural-memory review
state.  Callers must supply the accepted claims/cases explicitly.
"""

from .lifecycle import (
    BoundaryLifecycleDecision,
    BoundaryLifecycleRecord,
    BoundaryLifecycleState,
    BoundaryLifecycleStore,
    LifecyclePolicy,
    build_lifecycle_record,
    evaluate_boundary_lifecycle,
)
from .promotion import (
    PromotionLevel,
    PromotionObservation,
    PromotionRecommendation,
    score_promotion,
)
from .retrieval import (
    TruthAwareQuery,
    TruthAwareRetriever,
    TruthView,
    retrieve_candidates,
)
from .skills import (
    SkillCandidate,
    SkillCandidateState,
    SkillLifecycleFileStore,
    SkillLifecycleManager,
    sanitize_skill_name,
)
from .truth import AuthorityState, TruthClaim, TruthClass
from .wiki import WikiCompiler, WikiPage

__all__ = [
    "AuthorityState",
    "BoundaryLifecycleDecision",
    "BoundaryLifecycleRecord",
    "BoundaryLifecycleState",
    "BoundaryLifecycleStore",
    "LifecyclePolicy",
    "PromotionLevel",
    "PromotionObservation",
    "PromotionRecommendation",
    "SkillCandidate",
    "SkillCandidateState",
    "SkillLifecycleFileStore",
    "SkillLifecycleManager",
    "TruthAwareQuery",
    "TruthAwareRetriever",
    "TruthClaim",
    "TruthClass",
    "TruthView",
    "WikiCompiler",
    "WikiPage",
    "build_lifecycle_record",
    "evaluate_boundary_lifecycle",
    "retrieve_candidates",
    "sanitize_skill_name",
    "score_promotion",
]
