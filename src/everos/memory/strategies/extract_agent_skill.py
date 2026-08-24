"""extract_agent_skill strategy — distil / update an AgentSkill per case.

Triggered by :class:`SkillClusterUpdated` after ``trigger_skill_clustering``
has assigned the fresh case to its cluster. The strategy:

1. Loads the target case **markdown-first**
   (:func:`_load_target_case`). The previous implementation probed
   ``agent_case_repo.find_by_owner_entry`` for the freshly-written case
   and raised a retry-class error when cascade hadn't indexed it yet,
   which under sustained cascade lag meant the run died after
   ``max_retries`` and OME dead-lettered it — the case existed durably in
   markdown but was never distilled into a skill (production:
   ``success ≈ 370 / failed ≈ 465 / dead_letter ≈ 165``, real error
   ``_CaseNotYetIndexedError: AgentCase ... not in LanceDB yet``). The
   principle is **markdown durable source > LanceDB projection**: when
   LanceDB lags, the case body is read back from the daily-log md entry
   and the run proceeds; the raise survives only for a case that exists
   *nowhere* (genuine same-tick race — OME retry will catch up).
2. Selects the ``existing_relevant_skills`` slice for this cluster,
   **md-first** (:func:`_select_existing_skills`):

   * ``AgentSkillReader.list_by_cluster`` is the source of truth for
     "which skills exist in this cluster" — md is strongly consistent,
     LanceDB is cascade-lagged and must not be used for existence checks
     (a stale index previously made the LLM emit ``add()`` for a skill
     that already existed in md, silently clobbering it on write-back);
   * cluster size ``≤ MAX_SKILLS_IN_PROMPT`` → every md skill is used
     (ranking would be pointless on a fully-inclusive set);
   * cluster size ``> MAX_SKILLS_IN_PROMPT`` and a query vector is
     obtainable for the target → LanceDB ranks by cosine relevance, md
     hydrates the winning ids' content (LanceDB is a ranking index here,
     never an existence check);
   * cluster size ``> MAX_SKILLS_IN_PROMPT`` but no vector signal is
     obtainable → md ordering capped at K (logged warning so truncation
     without ranking is observable).
3. Hydrates ``supporting_cases`` from the chosen skills'
   ``source_case_ids`` lineage. The algo prompt joins each existing
   skill to its ``source_case_ids`` via the ``supporting_cases`` map;
   cases that do not back any of the chosen skills would just inflate
   the prompt without informing the LLM. Hydrated cases are then
   ranked ``(quality_score desc, timestamp desc)`` and capped at
   ``MAX_SUPPORTING_CASES`` to keep the prompt bounded as a cluster
   grows. Unlike the target case and existing skills, this lineage read
   stays LanceDB-backed: an un-indexed supporting case only means a
   thinner prompt this run (non-corrupting; the next run catches up),
   not a wrong write.
4. Feeds the target + existing + supporting trio to
   :class:`everalgo.agent_memory.AgentSkillExtractor`, then writes the
   emitted skills back via :class:`AgentSkillWriter` and reaps the
   directory an update left behind when it renamed a skill (see
   :func:`_reap_renamed_skills`).

**Retire is not implemented.** ``AgentSkillExtractor.aextract`` returns a
flat ``list[AgentSkill]`` with no op discriminator; its retire branch
(``skill_ops._apply_update``, taken when ``confidence <
retire_confidence``, default ``0.1``) is an ordinary skill carrying a
lowered confidence and nothing else. This strategy writes every emitted
skill back the same way, so a retirement persists as a normal skill: it
stays in markdown, stays in the next run's prompt, and stays searchable.
Honouring it means choosing between deleting the directory — handing an
LLM-produced confidence score the authority to destroy the source of
truth — and a ``retired`` frontmatter flag, which only works if the
enumeration, cascade, and search all learn to filter on it. That is a
design decision, not an omission to patch over, so it is deferred and
stated here rather than left implied by a docstring listing three ops.

Per-case granularity (one strategy run per fresh case) — algo
short-circuits low-quality cases internally via its own
``skip_quality_threshold``; the strategy trusts that gate.
``cluster_id`` is stamped onto each emitted skill before persistence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from everalgo.agent_memory import AgentSkillExtractor
from everalgo.types import AgentCase as AlgoAgentCase
from everalgo.types import AgentSkill as AlgoAgentSkill

from everos.component.embedding import (
    EmbeddingError,
    EmbeddingNotConfiguredError,
    get_embedder,
)
from everos.component.llm import get_llm_client
from everos.component.utils.datetime import (
    from_iso_format,
    to_timestamp_ms,
)
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot, StructuredEntry
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.lancedb import (
    AgentCase as LanceAgentCase,
)
from everos.infra.persistence.lancedb import (
    agent_case_repo,
    agent_skill_repo,
)
from everos.infra.persistence.markdown import (
    AgentCaseReader,
    AgentSkillFrontmatter,
    AgentSkillReader,
    AgentSkillWriter,
)
from everos.infra.persistence.sqlite import cluster_repo
from everos.memory.events import SkillClusterUpdated
from everos.memory.prompt_slots import PromptLoader
from everos.memory.strategies._partition_locks import get_partition_lock

logger = get_logger(__name__)

MAX_SKILLS_IN_PROMPT = 10
"""Upper bound on ``existing_relevant_skills`` fed to the algo per run.

The algo library expects the caller to pre-filter
``existing_relevant_skills`` to a relevant subset (cosine top-K over
the target case's embedding) so the prompt stays bounded as a cluster
grows."""

MAX_SUPPORTING_CASES = 9
"""Upper bound on ``supporting_cases`` after lineage hydration.

Mirrors ``AgentSkillExtractor.aextract``'s ``max_case_history`` default
so the algo's per-skill ``supporting_cases`` slot is never starved by a
too-aggressive cap here, nor overfilled by an unbounded lineage union
when many top-K skills each carry a distinct ``source_case_ids`` list.
Ranking is ``(quality_score desc, timestamp desc)`` — same ordering
opensource ``AgentSkillExtractor._load_case_history`` applies."""


class _ClusterMissingError(RuntimeError):
    """Race with the cluster strategy; OME retry will catch up."""


class _CaseNotYetIndexedError(RuntimeError):
    """The target case exists nowhere yet — neither LanceDB nor markdown.

    Only raised after *both* stores miss: a case durably present in
    markdown must never dead-letter merely because its LanceDB projection
    lags (see :func:`_load_target_case`).
    """


@dataclass(frozen=True)
class _TargetCase:
    """What this strategy needs from the target AgentCase, source-agnostic.

    Produced either from the LanceDB row (:func:`_lance_to_target`) or,
    when that row has not been projected yet, straight from the durable
    markdown entry (:func:`_md_entry_to_target`). ``vector`` is ``[]``
    whenever no embedding signal is available — callers treat that as
    "rank without a vector".
    """

    entry_id: str
    timestamp_ms: int
    task_intent: str
    approach: str
    quality_score: float
    key_insight: str
    vector: list[float] = field(default_factory=list)


_writer: AgentSkillWriter | None = None
_reader: AgentSkillReader | None = None
_case_reader: AgentCaseReader | None = None
_prompt_loader: PromptLoader | None = None

_SKILL_SAFETY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key_material",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.I),
    ),
    ("openai_style_secret", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")),
    ("github_token", re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("sshpass_plaintext_password", re.compile(r"\bsshpass\s+-p\b", re.I)),
    (
        "ssh_host_verification_disabled",
        re.compile(r"StrictHostKeyChecking\s*=\s*no", re.I),
    ),
    (
        "ssh_known_hosts_disabled",
        re.compile(r"UserKnownHostsFile\s*=\s*/dev/null", re.I),
    ),
    ("world_writable_permissions", re.compile(r"\bchmod\s+777\b", re.I)),
    (
        "tls_verification_disabled",
        re.compile(
            r"NODE_TLS_REJECT_UNAUTHORIZED\s*=\s*0|"
            r"\bverify\s*=\s*False\b|--no-check-certificate",
            re.I,
        ),
    ),
)


def _get_writer() -> AgentSkillWriter:
    global _writer
    if _writer is None:
        _writer = AgentSkillWriter(root=MemoryRoot.default())
    return _writer


def _get_reader() -> AgentSkillReader:
    global _reader
    if _reader is None:
        _reader = AgentSkillReader(root=MemoryRoot.default())
    return _reader


def _get_case_reader() -> AgentCaseReader:
    global _case_reader
    if _case_reader is None:
        _case_reader = AgentCaseReader(root=MemoryRoot.default())
    return _case_reader


def _get_prompt_loader() -> PromptLoader:
    global _prompt_loader
    if _prompt_loader is None:
        config_root = Path(__file__).resolve().parents[2] / "config"
        _prompt_loader = PromptLoader(config_root)
    return _prompt_loader


def _skill_rejection_reason(skill: AlgoAgentSkill) -> str | None:
    """Return a content-free rule id when a generated skill is unsafe."""
    text = "\n".join((skill.name, skill.description, skill.content))
    for rule_id, pattern in _SKILL_SAFETY_PATTERNS:
        if pattern.search(text):
            return rule_id
    return None


@offline_strategy(
    name="extract_agent_skill",
    trigger=Immediate(on=[SkillClusterUpdated]),
    emits=[],
    max_retries=3,
)
async def extract_agent_skill(event: SkillClusterUpdated, ctx: StrategyContext) -> None:
    # Serialise on agent_id: SKILL.md is addressed by (agent_id, skill_name)
    # — concurrent runs across different clusters of the same agent can
    # both decide to add the same skill_name and clobber the file. Different
    # agents run fully in parallel.
    # Lock per (app, project, agent): SKILL.md is addressed by (agent, name)
    # within a space; different spaces run in parallel.
    partition = f"{event.app_id}:{event.project_id}:{event.agent_id}"
    async with get_partition_lock("extract_agent_skill", partition):
        # 1. Check the cluster row exists.
        await _ensure_cluster_exists(event.cluster_id, event.case_entry_id)

        # 2. Load the target AgentCase — LanceDB first, durable markdown as
        #    the cascade-lag rescue (see module docstring, point 1).
        target = await _load_target_case(
            event.agent_id,
            event.case_entry_id,
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 3. Pick the top-K relevant existing skills in this cluster, md-first.
        #    (Cluster-scoped queries are implicitly space-scoped: cluster_id
        #    is globally unique to one (app, project, owner) cluster set.)
        existing_skills = await _select_existing_skills(
            agent_id=event.agent_id,
            cluster_id=event.cluster_id,
            target=target,
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 4. Pull the supporting cases referenced by those skills.
        supporting_lance = await _select_supporting_cases(
            existing_skills,
            agent_id=event.agent_id,
            exclude_entry_id=event.case_entry_id,
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 5. Run the LLM extractor. Emits add / update ops; a retire op comes
        #    back as an ordinary skill with confidence < retire_confidence and
        #    is NOT honoured here — see the module docstring.
        extractor = AgentSkillExtractor(llm=get_llm_client())
        prompt_loader = _get_prompt_loader()
        emitted_skills = await extractor.aextract(
            _target_to_algo(target),
            existing_relevant_skills=existing_skills,
            supporting_cases=[_to_algo_case(c) for c in supporting_lance],
            prompt_success=prompt_loader.load("agent_skill_success"),
            prompt_failure=prompt_loader.load("agent_skill_failure"),
        )

        # 6. Write each emitted skill back to its SKILL.md, then reap the
        #    directories that a rename left behind.
        writer = _get_writer()
        written_names: dict[str, str] = {}
        persisted = 0
        rejected = 0
        for skill in emitted_skills:
            rejection_reason = _skill_rejection_reason(skill)
            if rejection_reason is not None:
                rejected += 1
                logger.warning(
                    "agent_skill_rejected_by_safety_gate",
                    case_entry_id=event.case_entry_id,
                    cluster_id=event.cluster_id,
                    agent_id=event.agent_id,
                    safety_rule=rejection_reason,
                )
                continue
            written_names[skill.id] = await _persist_skill(
                writer,
                skill,
                agent_id=event.agent_id,
                cluster_id=event.cluster_id,
                app_id=event.app_id,
                project_id=event.project_id,
            )
            persisted += 1
        await _reap_renamed_skills(
            writer,
            written_names,
            existing_skills=existing_skills,
            agent_id=event.agent_id,
            app_id=event.app_id,
            project_id=event.project_id,
        )
    logger.info(
        "agent_skills_extracted",
        case_entry_id=event.case_entry_id,
        cluster_id=event.cluster_id,
        agent_id=event.agent_id,
        emitted=len(emitted_skills),
        persisted=persisted,
        rejected=rejected,
    )


# ── orchestration helpers ────────────────────────────────────────────────


async def _ensure_cluster_exists(cluster_id: str, case_entry_id: str) -> None:
    """Bail with a retry-class error when the cluster row is not yet there."""
    cluster = await cluster_repo.get_with_members(cluster_id)
    if cluster is None:
        # Same-transaction race with trigger_skill_clustering; OME retries.
        raise _ClusterMissingError(
            f"cluster_id={cluster_id} not found yet for case {case_entry_id}; retrying"
        )


async def _load_target_case(
    agent_id: str,
    case_entry_id: str,
    *,
    app_id: str,
    project_id: str,
) -> _TargetCase:
    """Load the target case, rescuing from markdown when LanceDB lags.

    Order of preference:

    1. The LanceDB row — the exact projection the recall path uses.
    2. The durable markdown entry. The AgentCase is written to the
       daily-log md *before* ``AgentCaseExtracted`` fires, so an entry
       missing from LanceDB is a projection backlog, not a missing case:
       reading the body back from md lets extraction proceed instead of
       burning the retry budget and dead-lettering (the production defect
       this closes). Markdown is the durable source; LanceDB is a
       rebuildable projection.
    3. Nowhere → raise :class:`_CaseNotYetIndexedError`. Still a genuine
       same-tick race (the md append has not landed yet either); OME
       retry will catch up.
    """
    lance = await agent_case_repo.find_by_owner_entry(
        agent_id, case_entry_id, app_id=app_id, project_id=project_id
    )
    if lance is not None:
        return _lance_to_target(lance)

    rescued = await _rescue_target_from_markdown(
        agent_id, case_entry_id, app_id=app_id, project_id=project_id
    )
    if rescued is not None:
        return rescued

    raise _CaseNotYetIndexedError(
        f"AgentCase entry_id={case_entry_id} found neither in LanceDB nor "
        "in markdown yet; retrying"
    )


async def _rescue_target_from_markdown(
    agent_id: str,
    case_entry_id: str,
    *,
    app_id: str,
    project_id: str,
) -> _TargetCase | None:
    """Reconstruct the target case from its daily-log markdown entry.

    Returns ``None`` when the entry does not exist (caller falls through
    to the retry-class error). Fields the entry lacks are degraded to
    safe empties with a warning rather than raised on — a hand-edited or
    partially-written entry must not reintroduce the dead-letter this
    rescue exists to prevent; the algo's own quality gate absorbs an
    empty-intent case.
    """
    try:
        entry = await _get_case_reader().find_structured(
            agent_id, case_entry_id, app_id=app_id, project_id=project_id
        )
    except ValueError:
        # Malformed entry id — no date bucket to look in; treat as absent.
        return None
    if entry is None:
        return None

    target = _structured_entry_to_target(entry, entry_id=case_entry_id)
    missing = [
        name
        for name, value in (
            ("TaskIntent", target.task_intent),
            ("Approach", target.approach),
            ("timestamp", entry.inline.get("timestamp", "")),
        )
        if not value
    ]
    if missing:
        logger.warning(
            "agent_skill_md_case_fields_missing",
            case_entry_id=case_entry_id,
            agent_id=agent_id,
            missing=missing,
        )

    logger.info(
        "agent_skill_target_case_rescued_from_markdown",
        case_entry_id=case_entry_id,
        agent_id=agent_id,
        app_id=app_id,
        project_id=project_id,
    )
    return target


async def _select_existing_skills(
    *,
    agent_id: str,
    cluster_id: str,
    target: _TargetCase,
    app_id: str,
    project_id: str,
) -> list[AlgoAgentSkill]:
    """Pick at most ``MAX_SKILLS_IN_PROMPT`` existing skills for the prompt.

    md is the source of truth for existence — this avoids the stale-index
    clobber where a skill was written last run but hadn't been indexed
    into LanceDB yet, causing the LLM to see no existing skill and emit
    ``add()`` for one that already exists. LanceDB is only consulted for
    relevance ordering when the cluster's md skill count exceeds
    ``MAX_SKILLS_IN_PROMPT`` and a query vector is obtainable; when it
    isn't, fall back to md ordering.

    ``list_by_cluster`` returns each skill's frontmatter *and* body
    together, so there is no second, name-based read to hydrate
    ``content`` — such a re-read would re-derive (and re-sanitize) a path
    from ``fm.name`` and could silently miss a skill whose on-disk
    directory suffix isn't itself a sanitizer fixpoint. See
    ``AgentSkillReader.list_by_cluster``'s docstring.
    """
    reader = _get_reader()
    md_skills = await reader.list_by_cluster(
        agent_id, cluster_id, app_id=app_id, project_id=project_id
    )

    if len(md_skills) <= MAX_SKILLS_IN_PROMPT:
        selected = md_skills
    else:
        query_vector = await _resolve_query_vector(target)
        if query_vector:
            selected = await _rank_skills_by_relevance(
                md_skills,
                agent_id=agent_id,
                cluster_id=cluster_id,
                query_vector=query_vector,
            )
        else:
            logger.warning(
                "agent_skill_topk_no_query_vector_md_fallback",
                agent_id=agent_id,
                cluster_id=cluster_id,
                md_count=len(md_skills),
            )
            selected = md_skills[:MAX_SKILLS_IN_PROMPT]

    return [_md_to_algo_skill(fm, body) for fm, body in selected]


async def _rank_skills_by_relevance(
    md_skills: list[tuple[AgentSkillFrontmatter, str]],
    *,
    agent_id: str,
    cluster_id: str,
    query_vector: list[float],
) -> list[tuple[AgentSkillFrontmatter, str]]:
    """Ask LanceDB to rank the md skills by cosine relevance, capped at K.

    LanceDB is used purely as a ranking index here, never as the
    existence check — the candidate set is always the md list. A LanceDB
    row with no matching md name is stale and skipped; md skills LanceDB
    didn't return (also stale index) then backfill in md order until the
    ``MAX_SKILLS_IN_PROMPT`` budget is full. That backfill keeps a lagging
    index from *under*-filling the prompt; it does not make the selection
    lossless — this function only runs when the cluster already holds more
    skills than the budget admits, so skills beyond K are dropped by
    design either way.
    """
    md_by_name = {fm.name: (fm, body) for fm, body in md_skills}
    ranked_lance = await agent_skill_repo.find_topk_relevant_in_cluster(
        owner_id=agent_id,
        cluster_id=cluster_id,
        query_vector=query_vector,
        top_k=MAX_SKILLS_IN_PROMPT,
    )
    selected: list[tuple[AgentSkillFrontmatter, str]] = []
    seen_names: set[str] = set()
    for lance_row in ranked_lance:
        pair = md_by_name.get(lance_row.name)
        if pair is not None and pair[0].name not in seen_names:
            selected.append(pair)
            seen_names.add(pair[0].name)
    for fm, body in md_skills:
        if fm.name not in seen_names and len(selected) < MAX_SKILLS_IN_PROMPT:
            selected.append((fm, body))
            seen_names.add(fm.name)
    return selected


async def _resolve_query_vector(target: _TargetCase) -> list[float]:
    """Return a usable query vector for cosine top-K, ``[]`` if unobtainable.

    Order of preference:

    1. ``target.vector`` if the cascade-projected row carried one — this
       is the exact vector the recall path uses, so reusing it keeps
       ranking semantics identical across reads. Always ``[]`` on the
       md-rescue path (the projection, and hence the persisted vector,
       does not exist yet).
    2. Compute on the fly from ``target.task_intent`` via the configured
       embedder — matches the cascade handler's own vectorisation
       contract (``cascade/handlers/agent_case.py``), so the two paths
       agree on what "the case embedding" means.

    Returns ``[]`` only when both options are unavailable (no persisted
    vector, no ``task_intent`` text, or the embedder is not configured /
    fails). The caller decides the policy for that case.
    """
    if target.vector:
        return list(target.vector)
    if not target.task_intent:
        return []
    try:
        embedder = get_embedder()
        return list(await embedder.embed(target.task_intent))
    except (EmbeddingNotConfiguredError, EmbeddingError) as exc:
        logger.warning(
            "agent_skill_query_embed_failed",
            case_entry_id=target.entry_id,
            error=str(exc),
        )
        return []


async def _select_supporting_cases(
    skills: list[AlgoAgentSkill],
    *,
    agent_id: str,
    exclude_entry_id: str,
    app_id: str,
    project_id: str,
) -> list[LanceAgentCase]:
    """Hydrate, rank, and cap supporting cases from skills' lineage.

    ``exclude_entry_id`` drops the target case's own entry id so the
    algo never sees the new case as one of its own supporting cases.
    Ranking ``(quality_score desc, timestamp desc)`` mirrors opensource
    ``AgentSkillExtractor._load_case_history``; the cap matches
    :data:`MAX_SUPPORTING_CASES`.
    """
    # 1. Collect source case ids from the chosen skills (dedup, drop target).
    entry_ids = _collect_supporting_entry_ids(skills, exclude=exclude_entry_id)
    if not entry_ids:
        return []

    # 2. Bulk-fetch those cases from LanceDB (scoped to space).
    hydrated = await agent_case_repo.find_by_owner_entries(
        agent_id, entry_ids, app_id=app_id, project_id=project_id
    )

    # 3. Sort by (quality, timestamp) desc, then cap.
    hydrated.sort(
        key=lambda c: (c.quality_score or 0.0, c.timestamp),
        reverse=True,
    )
    return hydrated[:MAX_SUPPORTING_CASES]


def _collect_supporting_entry_ids(
    skills: list[AlgoAgentSkill], *, exclude: str
) -> list[str]:
    """Dedup ``source_case_ids`` across ``skills``, preserving first-seen order."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for skill in skills:
        for cid in skill.source_case_ids or []:
            if not cid or cid == exclude or cid in seen_set:
                continue
            seen.append(cid)
            seen_set.add(cid)
    return seen


# ── algo / persistence projection ────────────────────────────────────────


def _lance_to_target(lance: LanceAgentCase) -> _TargetCase:
    """Project the LanceDB row onto the source-agnostic target shape."""
    return _TargetCase(
        entry_id=lance.entry_id,
        timestamp_ms=int(lance.timestamp.timestamp() * 1000),
        task_intent=lance.task_intent,
        approach=lance.approach,
        quality_score=lance.quality_score,
        key_insight=lance.key_insight or "",
        vector=list(lance.vector) if lance.vector else [],
    )


def _target_to_algo(target: _TargetCase) -> AlgoAgentCase:
    """Project the resolved target onto the algo-side AgentCase type."""
    return AlgoAgentCase(
        id=target.entry_id,
        timestamp=target.timestamp_ms,
        task_intent=target.task_intent,
        approach=target.approach,
        quality_score=target.quality_score,
        key_insight=target.key_insight or "",
    )


def _structured_entry_to_target(
    entry: StructuredEntry, *, entry_id: str
) -> _TargetCase:
    """Parse a daily-log audit-form entry into the target shape.

    Mirrors the field layout ``extract_agent_case._agent_case_to_entry_body``
    writes: ``TaskIntent`` / ``Approach`` / optional ``KeyInsight``
    sections, ``quality_score`` / ``timestamp`` inline. Missing values
    degrade to safe empties (callers warn; see
    :func:`_rescue_target_from_markdown`).
    """
    try:
        quality_score = float(entry.inline.get("quality_score", ""))
    except ValueError:
        quality_score = 0.0
    try:
        timestamp_ms = to_timestamp_ms(
            from_iso_format(entry.inline.get("timestamp", ""))
        )
    except ValueError:
        timestamp_ms = 0
    return _TargetCase(
        entry_id=entry_id,
        timestamp_ms=timestamp_ms,
        task_intent=(entry.sections.get("TaskIntent") or "").strip(),
        approach=(entry.sections.get("Approach") or "").strip(),
        quality_score=quality_score,
        key_insight=(entry.sections.get("KeyInsight") or "").strip(),
        vector=[],
    )


def _to_algo_case(lance: LanceAgentCase) -> AlgoAgentCase:
    """Project the LanceDB row onto the algo-side AgentCase type.

    Used only for ``supporting_cases`` — that lineage read stays
    LanceDB-backed (see module docstring, point 3).
    """
    return AlgoAgentCase(
        id=lance.entry_id,
        timestamp=int(lance.timestamp.timestamp() * 1000),
        task_intent=lance.task_intent,
        approach=lance.approach,
        quality_score=lance.quality_score,
        key_insight=lance.key_insight or "",
    )


def _md_to_algo_skill(fm: AgentSkillFrontmatter, body: str) -> AlgoAgentSkill:
    """Project a SKILL.md frontmatter + body onto everalgo's AgentSkill type.

    ``body`` populates ``AlgoAgentSkill.content`` so the extractor prompt's
    existing-skills block carries the real skill definition, not an empty
    placeholder — without it the LLM cannot distinguish "add new" from
    "update existing".
    """
    return AlgoAgentSkill(
        id=fm.id,
        cluster_id=fm.cluster_id or "",
        name=fm.name,
        description=fm.description,
        content=body,
        confidence=fm.confidence,
        maturity_score=fm.maturity_score,
        source_case_ids=list(fm.source_case_ids),
    )


async def _reap_renamed_skills(
    writer: AgentSkillWriter,
    written_names: Mapping[str, str],
    *,
    existing_skills: Sequence[AlgoAgentSkill],
    agent_id: str,
    app_id: str,
    project_id: str,
) -> None:
    """Delete the directory an update left behind when it renamed a skill.

    everalgo treats a name change as a first-class update
    (``skill_ops._apply_update`` computes ``name_changed`` and returns
    ``prior.model_copy(update={"name": eff_name, ...})``), so the emitted
    skill keeps ``prior.id`` while carrying the new name. ``_persist_skill``
    writes it to ``skill_<new_name>/`` and the old directory would survive
    with the same ``cluster_id``.

    That matters more here than it looks. Since existing skills are read
    from markdown rather than LanceDB, a surviving pre-rename directory
    does not merely sit on disk — it comes back in the next run's
    ``existing_relevant_skills`` as a duplicate of a skill the LLM already
    renamed. Shown its own stale copy, the LLM emits ``add`` for something
    that exists, and ``write_main`` full-replaces — precisely the clobber
    the md-first selection set out to close. Left unreaped, every rename
    adds another one.

    Identity comes from ``skill.id``, which is the only thing that survives
    a rename: ``_apply_update`` preserves ``prior.id`` while ``_apply_add``
    mints a fresh ``uuid4().hex``, so an id present in the enumerated set is
    an update by construction and a new skill can never match.

    A prior name that some *other* emitted skill just claimed is never
    deleted — with two ops in one batch (rename ``a``→``b`` while another
    op writes ``a``) the reap would otherwise remove a file written
    moments earlier in the same loop.
    """
    claimed = set(written_names.values())
    for prior in existing_skills:
        new_name = written_names.get(prior.id)
        prior_name = AgentSkillFrontmatter.sanitize_skill_name(prior.name)
        if new_name is None or new_name == prior_name or prior_name in claimed:
            continue
        removed = await writer.delete_skill(
            agent_id, prior_name, app_id=app_id, project_id=project_id
        )
        logger.info(
            "agent_skill_renamed_directory_reaped",
            agent_id=agent_id,
            skill_id=prior.id,
            old_name=prior_name,
            new_name=new_name,
            removed=removed,
        )


async def _persist_skill(
    writer: AgentSkillWriter,
    skill: AlgoAgentSkill,
    *,
    agent_id: str,
    cluster_id: str,
    app_id: str,
    project_id: str,
) -> str:
    """Write one ``SKILL.md`` with the post-stamped ``cluster_id``.

    Returns the sanitized name it wrote under, so the caller can
    reconcile renames without re-deriving it.

    ``skill.name`` is LLM output and is sanitized once, up front, via
    :meth:`AgentSkillFrontmatter.sanitize_skill_name` — the same helper
    :meth:`AgentSkillFrontmatter.skill_dir_name` uses for the directory
    segment. Sanitizing here (rather than handing the raw name to the
    frontmatter constructor) keeps ``frontmatter.name`` byte-identical to
    the on-disk directory name, and means a traversal-shaped LLM name
    (reachable via prompt injection, since the LLM's input is user
    conversation content) is made filesystem-safe *before* it reaches
    ``AgentSkillFrontmatter``, instead of tripping the read-side traversal
    validator and dead-lettering the whole extraction run for a name the
    writer would have sanitized safely anyway.
    """
    sanitized_name = AgentSkillFrontmatter.sanitize_skill_name(skill.name)
    frontmatter = AgentSkillFrontmatter(
        id=f"{agent_id}_{sanitized_name}",
        agent_id=agent_id,
        name=sanitized_name,
        description=skill.description,
        confidence=skill.confidence,
        maturity_score=skill.maturity_score,
        source_case_ids=list(skill.source_case_ids),
        cluster_id=cluster_id,
    )
    await writer.write_main(
        agent_id,
        sanitized_name,
        frontmatter=frontmatter,
        body=skill.content,
        app_id=app_id,
        project_id=project_id,
    )
    return sanitized_name
