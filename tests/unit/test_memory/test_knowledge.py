"""Adversarial tests for EV-MEMORY-01 knowledge-access primitives."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from everalgo.types import Candidate
from prometheus_client import CollectorRegistry

from everos.memory.knowledge import (
    AuthorityState,
    BoundaryLifecycleState,
    LifecyclePolicy,
    PromotionLevel,
    PromotionObservation,
    SkillCandidateState,
    SkillLifecycleFileStore,
    SkillLifecycleManager,
    TruthAwareQuery,
    TruthAwareRetriever,
    TruthClaim,
    TruthClass,
    TruthView,
    WikiCompiler,
    build_lifecycle_record,
    evaluate_boundary_lifecycle,
    retrieve_candidates,
    sanitize_skill_name,
    score_promotion,
)

UTC = dt.UTC
NOW = dt.datetime(2026, 8, 26, 0, 0, tzinfo=UTC)
SCOPE = {"app_id": "app", "project_id": "p", "owner_id": "u"}


def _claim(
    claim_id: str,
    text: str,
    *,
    truth_class: TruthClass = TruthClass.CURRENT,
    authority: AuthorityState = AuthorityState.ACCEPTED,
    score: float = 0.5,
    scope: dict[str, str] | None = None,
    **kwargs: object,
) -> TruthClaim:
    source_refs = kwargs.pop("source_refs", (f"evidence:{claim_id}",))
    return TruthClaim(
        claim_id=claim_id,
        text=text,
        kind="PROJECT_STATE",
        scope=scope or SCOPE,
        truth_class=truth_class,
        authority=authority,
        semantic_score=score,
        bm25_score=score,
        source_refs=source_refs,
        created_at=NOW,
        **kwargs,
    )


def test_lifecycle_authority_precedes_force_and_timers() -> None:
    record = build_lifecycle_record(
        app_id="a",
        project_id="p",
        session_id="s",
        track="memorize",
        message_ids=["ms-1"],
        revision=1,
        should_wait=True,
        authority_state="pending_publish",
        observed_at=NOW,
        policy=LifecyclePolicy(idle_timeout_seconds=1, max_delay_seconds=5),
    )
    decision = evaluate_boundary_lifecycle(
        record, now=NOW + dt.timedelta(days=1), explicit_final=True, force_flush=True
    )
    assert decision.action == "block"
    assert decision.reason == "pending_publish"
    assert decision.record.state == BoundaryLifecycleState.BLOCKED_PENDING_PUBLISH


@pytest.mark.parametrize(
    ("should_wait", "elapsed", "action"),
    [
        (True, 0.1, "wait"),
        (True, 1.1, "process"),
        (None, 1.1, "process"),
        (False, 0.1, "process"),
    ],
)
def test_lifecycle_should_wait_idle_and_max_rules(
    should_wait: bool | None, elapsed: float, action: str
) -> None:
    record = build_lifecycle_record(
        app_id="a",
        project_id="p",
        session_id="s",
        track="memorize",
        message_ids=["m-1"],
        revision=1,
        should_wait=should_wait,
        authority_state="published",
        observed_at=NOW,
        policy=LifecyclePolicy(idle_timeout_seconds=1, max_delay_seconds=5),
    )
    decision = evaluate_boundary_lifecycle(
        record, now=NOW + dt.timedelta(seconds=elapsed)
    )
    assert decision.action == action


def test_lifecycle_final_consumed_and_revision_supersede_are_durable_shapes() -> None:
    old = build_lifecycle_record(
        app_id="a",
        project_id="p",
        session_id="s",
        track="memorize",
        message_ids=["m-1"],
        revision=1,
        should_wait=True,
        authority_state="published",
        observed_at=NOW,
    )
    new = build_lifecycle_record(
        app_id="a",
        project_id="p",
        session_id="s",
        track="memorize",
        message_ids=["m-1"],
        revision=2,
        should_wait=False,
        authority_state="published",
        observed_at=NOW,
    )
    assert old.message_digest == new.message_digest
    assert new.revision > old.revision
    assert evaluate_boundary_lifecycle(old, explicit_final=True).action == "process"
    consumed = old.__class__(
        **{
            **{name: getattr(old, name) for name in old.__dataclass_fields__},
            "state": BoundaryLifecycleState.CONSUMED,
            "consumed": True,
        }
    )
    assert evaluate_boundary_lifecycle(consumed).action == "skip"


def test_truth_gate_rejects_wrong_scope_candidate_superseded_and_expired() -> None:
    claims = [
        _claim("current", "accepted current deployment", score=0.1),
        _claim(
            "history",
            "old accepted deployment",
            truth_class=TruthClass.HISTORY,
            score=1.0,
        ),
        _claim(
            "candidate",
            "new candidate deployment",
            authority=AuthorityState.CANDIDATE,
            score=1.0,
        ),
        _claim("superseded", "old deployment", score=1.0, superseded_by="current"),
        _claim("expired", "expired deployment", score=1.0, valid_until=NOW),
        _claim(
            "wrong-scope",
            "same words",
            score=1.0,
            scope={"app_id": "other", "project_id": "p", "owner_id": "u"},
        ),
    ]
    current = TruthAwareRetriever().retrieve(
        claims, TruthAwareQuery(query="deployment", scope=SCOPE, top_k=10)
    )
    assert [hit.claim.claim_id for hit in current] == ["current"]
    history = TruthAwareRetriever().retrieve(
        claims,
        TruthAwareQuery(
            query="以前的 deployment", scope=SCOPE, view=TruthView.HISTORY, top_k=10
        ),
    )
    assert [hit.claim.claim_id for hit in history] == ["history"]


def test_explicit_candidates_cannot_displace_accepted_current_truth() -> None:
    claims = [
        _claim("accepted", "accepted deployment", score=0.1),
        _claim(
            "candidate",
            "new candidate deployment",
            authority=AuthorityState.CANDIDATE,
            score=1.0,
        ),
    ]
    hits = TruthAwareRetriever().retrieve(
        claims,
        TruthAwareQuery(
            query="deployment",
            scope=SCOPE,
            include_candidates=True,
            top_k=10,
        ),
    )
    assert [hit.claim.claim_id for hit in hits] == ["accepted", "candidate"]


def test_candidate_conflict_does_not_hide_the_only_accepted_claim() -> None:
    claims = [
        _claim("accepted", "accepted deployment", score=0.1, conflict_group="g"),
        _claim(
            "candidate",
            "candidate deployment",
            authority=AuthorityState.CANDIDATE,
            score=1.0,
            conflict_group="g",
        ),
    ]
    hits = TruthAwareRetriever().retrieve(
        claims,
        TruthAwareQuery(
            query="deployment",
            scope=SCOPE,
            include_candidates=True,
            top_k=10,
        ),
    )
    assert [hit.claim.claim_id for hit in hits] == ["accepted", "candidate"]


def test_truth_conflict_does_not_choose_by_semantic_score() -> None:
    claims = [
        _claim("a", "claim A", conflict_group="c", score=0.1),
        _claim("b", "claim B", conflict_group="c", score=1.0),
    ]
    assert (
        TruthAwareRetriever().retrieve(
            claims, TruthAwareQuery(query="claim", scope=SCOPE)
        )
        == []
    )


def test_candidate_adapter_keeps_legacy_rows_compatible_and_gates_enveloped_rows() -> (
    None
):
    legacy = Candidate(
        id="legacy", score=0.2, source="vector", metadata={"app_id": "a"}
    )
    candidate = Candidate(
        id="candidate",
        score=1.0,
        source="vector",
        metadata={
            "app_id": "a",
            "project_id": "p",
            "owner_id": "u",
            "authority": "candidate",
        },
    )
    assert (
        retrieve_candidates([legacy], query="x", scope={}, top_k=10)[0].id == "legacy"
    )
    assert retrieve_candidates([candidate], query="x", scope=SCOPE, top_k=10) == []

    accepted = Candidate(
        id="accepted",
        score=0.8,
        source="other",
        metadata={
            "app_id": "app",
            "project_id": "p",
            "owner_id": "u",
            "authority": "accepted",
            "truth_class": "current",
            "content": "accepted deployment",
        },
    )
    mixed = retrieve_candidates(
        [legacy, accepted], query="deployment", scope=SCOPE, top_k=10
    )
    assert {item.id for item in mixed} == {"legacy", "accepted"}
    assert mixed[-1].score > 0
    assert [
        item.id
        for item in retrieve_candidates(
            [legacy, Candidate(id="legacy-2", score=0.1, source="vector", metadata={})],
            query="x",
            scope={},
            top_k=-1,
        )
    ] == ["legacy", "legacy-2"]


def test_current_view_excludes_unresolved_conflict_class_without_group() -> None:
    conflict = _claim(
        "conflict-class",
        "conflicting deployment",
        truth_class=TruthClass.CONFLICT,
    )
    assert (
        TruthAwareRetriever().retrieve(
            [conflict], TruthAwareQuery(query="deployment", scope=SCOPE)
        )
        == []
    )


def test_wiki_is_deterministic_and_human_agent_views_share_claim_set() -> None:
    claims = [
        _claim("z", "state Z"),
        _claim("a", "state A"),
        _claim("unsupported", "must not render", source_refs=()),
        _claim("stale", "must not render", truth_class=TruthClass.EXPIRED),
    ]
    compiler = WikiCompiler()
    first = compiler.compile(snapshot_id="snap-1", title="Project A", claims=claims)
    second = compiler.compile(
        snapshot_id="snap-1", title="Project A", claims=reversed(claims)
    )
    assert first.claim_set_sha256 == second.claim_set_sha256
    assert first.claims == second.claims
    assert [c.claim_id for c in first.claims] == ["a", "z"]
    assert first.unsupported_claim_count == first.stale_claim_count == 0
    assert first.verification_status == "VERIFIED"
    agent = first.agent_view()
    assert {item["claim_id"] for item in agent["claims"]} == {
        c.claim_id for c in first.claims
    }
    assert "state A" in first.human_markdown()
    assert "must not render" not in first.human_markdown()

    historical_state = _claim(
        "historical-state",
        "state from before",
        truth_class=TruthClass.HISTORY,
    )
    history_page = compiler.compile(
        snapshot_id="snap-2", title="Project A", claims=[historical_state]
    )
    rendered = history_page.human_markdown()
    assert "## History" in rendered
    assert "## Current State" not in rendered


def test_promotion_is_recommendation_only_and_correction_lowers_level() -> None:
    strong = score_promotion(
        PromotionObservation(
            candidate_id="c",
            occurrence=10,
            recall_count=20,
            query_diversity=10,
            stability_days=60,
            cross_session_repetition=5,
            project_importance=1,
            explicit_user_emphasis=1,
            provenance_quality=1,
            scope_persistence=1,
            evidence_refs=("e1",),
        )
    )
    corrected = score_promotion(
        PromotionObservation(
            candidate_id="c",
            occurrence=10,
            recall_count=20,
            query_diversity=10,
            stability_days=60,
            cross_session_repetition=5,
            project_importance=1,
            explicit_user_emphasis=1,
            provenance_quality=1,
            scope_persistence=1,
            later_correction=3,
            contradiction_count=2,
        )
    )
    assert strong.level == PromotionLevel.HIGH
    assert corrected.score < strong.score
    assert corrected.level != PromotionLevel.HIGH
    assert strong.authority == corrected.authority == "none"


def test_skill_requires_repeated_success_and_explicit_review() -> None:
    manager = SkillLifecycleManager(repeated_success_threshold=3)
    for index in range(2):
        item = manager.observe_success(
            candidate_id="c",
            pattern_key="debug",
            name="Debug / Safe",
            description="debug safely",
            procedure="inspect, test, rollback",
            case_id=f"case-{index}",
            episode_id=f"episode-{index}",
        )
    assert item.state == SkillCandidateState.OBSERVED
    with pytest.raises(ValueError):
        manager.submit_for_review("c")
    item = manager.observe_success(
        candidate_id="c",
        pattern_key="debug",
        name="Debug / Safe",
        description="debug safely",
        procedure="inspect, test, rollback",
        case_id="case-2",
    )
    assert item.state == SkillCandidateState.CANDIDATE
    assert manager.submit_for_review("c").state == SkillCandidateState.REVIEW
    assert manager.accept("c").state == SkillCandidateState.ACCEPTED
    assert manager.retire("c").state == SkillCandidateState.RETIRED


def test_skill_supersession_marks_predecessor_and_versions_replacement() -> None:
    manager = SkillLifecycleManager(repeated_success_threshold=2)
    for candidate_id in ("old", "new"):
        for case_id in (f"{candidate_id}-1", f"{candidate_id}-2"):
            manager.observe_success(
                candidate_id=candidate_id,
                pattern_key="debug",
                name=candidate_id,
                description="debug safely",
                procedure="inspect, test, rollback",
                case_id=case_id,
            )
        manager.submit_for_review(candidate_id)
    replacement = manager.supersede("old", "new")
    assert replacement.supersedes == "old"
    assert replacement.version == 2
    assert manager.get("old").state == SkillCandidateState.SUPERSEDED  # type: ignore[union-attr]


def test_skill_counterexample_rejects_candidate_and_cannot_revive() -> None:
    manager = SkillLifecycleManager(repeated_success_threshold=2)
    for case_id in ("success-1", "success-2"):
        manager.observe_success(
            candidate_id="fragile",
            pattern_key="fragile",
            name="fragile",
            description="fragile procedure",
            procedure="inspect, test",
            case_id=case_id,
        )
    rejected = manager.observe_failure("fragile", "counterexample")
    assert rejected is not None
    assert rejected.state == SkillCandidateState.REJECTED
    with pytest.raises(ValueError, match="new candidate id"):
        manager.observe_success(
            candidate_id="fragile",
            pattern_key="fragile",
            name="fragile",
            description="fragile procedure",
            procedure="inspect, test",
            case_id="success-3",
        )


def test_skill_collision_failure_and_secret_path_guards() -> None:
    assert sanitize_skill_name("../Safe Skill") == "safe_skill"
    manager = SkillLifecycleManager(repeated_success_threshold=2)
    manager.observe_success(
        candidate_id="c",
        pattern_key="one",
        name="one",
        description="d",
        procedure="p",
        case_id="a",
    )
    with pytest.raises(ValueError, match="another pattern"):
        manager.observe_success(
            candidate_id="c",
            pattern_key="two",
            name="two",
            description="d",
            procedure="p",
            case_id="b",
        )
    for case_id in ("a", "b"):
        manager.observe_success(
            candidate_id="unsafe",
            pattern_key="unsafe",
            name="unsafe",
            description="d",
            procedure="use password: secret",
            case_id=case_id,
        )
    manager.submit_for_review("unsafe")
    with pytest.raises(ValueError, match="secret"):
        manager.accept("unsafe", reusable=False)


def test_skill_lifecycle_checkpoint_round_trips_atomically(tmp_path: Path) -> None:
    source = SkillLifecycleManager(repeated_success_threshold=2)
    source.observe_success(
        candidate_id="checkpointed",
        pattern_key="safe-pattern",
        name="Safe / Pattern",
        description="repeatable procedure",
        procedure="inspect, test, rollback",
        case_id="case-a",
        episode_id="episode-a",
    )
    path = tmp_path / "skill-candidates.json"
    SkillLifecycleFileStore(path).save(source)

    restored = SkillLifecycleManager(repeated_success_threshold=2)
    assert SkillLifecycleFileStore(path).load(restored) == 1
    assert restored.all() == source.all()
    assert path.with_name(f"{path.name}.lock").is_file()


def test_skill_lifecycle_checkpoint_refuses_secret_material(tmp_path: Path) -> None:
    manager = SkillLifecycleManager(repeated_success_threshold=2)
    for case_id in ("a", "b"):
        manager.observe_success(
            candidate_id="unsafe",
            pattern_key="unsafe",
            name="unsafe",
            description="d",
            procedure="use password: secret",
            case_id=case_id,
        )
    with pytest.raises(ValueError, match="unsafe"):
        SkillLifecycleFileStore(tmp_path / "skills.json").save(manager)


@pytest.mark.asyncio
async def test_due_lifecycle_flush_delegates_session_lock_to_memorize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler must not acquire the non-reentrant session lock twice."""

    from everos.service import background_flush

    record = build_lifecycle_record(
        app_id="a",
        project_id="p",
        session_id="s",
        track="memorize",
        message_ids=["m-1"],
        revision=1,
        should_wait=True,
        authority_state="published",
        observed_at=NOW,
    )
    store = SimpleNamespace(list_due=AsyncMock(return_value=[record]))
    memorize_mock = AsyncMock()
    monkeypatch.setattr(background_flush, "BoundaryLifecycleStore", lambda: store)
    monkeypatch.setattr(background_flush, "memorize", memorize_mock)

    def unexpected_lock(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("run_due_lifecycle_flushes must not nest session locks")

    monkeypatch.setattr(background_flush, "scoped_session_lock", unexpected_lock)
    assert await background_flush.run_due_lifecycle_flushes() == 1
    memorize_mock.assert_awaited_once_with(
        {
            "session_id": "s",
            "app_id": "a",
            "project_id": "p",
            "messages": [],
        },
        is_final=True,
        background_worker=True,
    )


def test_knowledge_surfaces_export_low_cardinality_metrics() -> None:
    from everos.core.observability.metrics import (
        generate_metrics_response,
        reset_metrics_registry,
        set_metrics_registry,
    )

    registry = CollectorRegistry()
    set_metrics_registry(registry)
    try:
        TruthAwareRetriever().retrieve(
            [_claim("metric", "metric deployment")],
            TruthAwareQuery(query="deployment", scope=SCOPE),
        )
        WikiCompiler().compile(
            snapshot_id="metric-snapshot",
            title="Metrics",
            claims=[_claim("metric", "metric deployment")],
        )
        score_promotion(PromotionObservation(candidate_id="metric"))
        evaluate_boundary_lifecycle(
            build_lifecycle_record(
                app_id="a",
                project_id="p",
                session_id="s",
                track="memorize",
                message_ids=["m"],
                revision=1,
                should_wait=True,
                authority_state="published",
                observed_at=NOW,
            ),
            now=NOW,
        )
        payload = generate_metrics_response().decode("utf-8")
        assert "everos_knowledge_retrieval_duration_seconds_count" in payload
        assert "everos_knowledge_wiki_build_duration_seconds_count" in payload
        assert "everos_knowledge_promotion_recommendations_total" in payload
        assert "everos_boundary_lifecycle_decisions_total" in payload
    finally:
        reset_metrics_registry()
