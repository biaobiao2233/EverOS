"""extract_atomic_facts strategy — derive AtomicFacts from a fresh MemCell.

One LLM call per memcell, then md-level fan-out to every user sender.
Mirrors :class:`UserMemoryPipeline`'s Episode handling: the algo
prompt is subject-agnostic (``INPUT_TEXT`` + ``TIME`` only, no
``sender_id`` placeholder — see
``everalgo.user_memory.atomic_fact.AtomicFactExtractor.aextract``), so
calling it once per sender would waste LLM tokens and let non-
determinism drift the per-sender md files apart. Instead, run the
extractor once with ``sender_id=None`` (algo's "generic owner"
signal) and rebroadcast the same fact list under each user sender.

Per-owner batching: each sender's full fact list is appended in one
batched ``append_entries`` call rather than ``len(algo_facts)`` single
appends, dropping the per-cell IO complexity from ``O(N²)`` to
``O(N)`` (one read + one write per owner instead of N of each) and
narrowing the per-path lock window from N read-modify-write cycles to
one.

Note ``extract_foresight`` does run per-sender because its prompt
template *does* condition on the target sender; do not collapse that
strategy in the same way without re-checking the prompt.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from everalgo.user_memory import AtomicFactExtractor

from everos.component.llm import get_llm_client
from everos.component.utils.datetime import from_timestamp, to_iso_format
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.markdown import AtomicFactWriter
from everos.memory.events import UserPipelineStarted
from everos.memory.models import AtomicFact
from everos.memory.prompt_slots import PromptLoader

logger = get_logger(__name__)

MAX_FACTS_PER_MEMCELL = 12
"""Hard ceiling after the prompt's normal 3-8 fact target.

The extra headroom preserves complex architecture and incident conversations,
while preventing a single verbose extraction from flooding retrieval.
"""

_writer: AtomicFactWriter | None = None
_prompt_loader: PromptLoader | None = None


def _get_writer() -> AtomicFactWriter:
    global _writer
    if _writer is None:
        _writer = AtomicFactWriter(root=MemoryRoot.default())
    return _writer


def _get_prompt_loader() -> PromptLoader:
    global _prompt_loader
    if _prompt_loader is None:
        config_root = Path(__file__).resolve().parents[2] / "config"
        _prompt_loader = PromptLoader(config_root)
    return _prompt_loader


@offline_strategy(
    name="extract_atomic_facts",
    trigger=Immediate(on=[UserPipelineStarted]),
    emits=[],
    max_retries=2,
)
async def extract_atomic_facts(
    event: UserPipelineStarted, ctx: StrategyContext
) -> None:
    # 1. List the user senders in this memcell; bail early if there are none.
    memcell = event.memcell
    sender_ids = sorted({m.sender_id for m in memcell.items if m.role == "user"})
    if not sender_ids:
        logger.info(
            "atomic_facts_extracted",
            memcell_id=event.memcell_id,
            session_id=event.session_id,
            count=0,
            owner_ids=[],
        )
        return

    # 2. Run the LLM extractor once (algo prompt is subject-agnostic).
    extractor = AtomicFactExtractor(llm=get_llm_client())
    prompt = _get_prompt_loader().load("atomic_fact_extract")
    algo_facts = await extractor.aextract(
        memcell,
        sender_id=None,
        prompt=prompt,
    )
    if len(algo_facts) > MAX_FACTS_PER_MEMCELL:
        logger.warning(
            "atomic_facts_capped",
            memcell_id=event.memcell_id,
            session_id=event.session_id,
            extracted=len(algo_facts),
            kept=MAX_FACTS_PER_MEMCELL,
        )
        algo_facts = algo_facts[:MAX_FACTS_PER_MEMCELL]

    # 3. Fan the fact list out to one domain AtomicFact per (sender, algo_fact).
    facts: list[AtomicFact] = [
        AtomicFact.from_algo(
            algo_fact,
            owner_id=sid,
            session_id=event.session_id,
            parent_id=event.memcell_id,
        )
        for sid in sender_ids
        for algo_fact in algo_facts
    ]

    # 4. Group facts by owner so each sender's full list lands in one
    #    batched write.
    by_owner: dict[str, list[tuple[Mapping[str, object], Mapping[str, str]]]] = (
        defaultdict(list)
    )
    for fact in facts:
        by_owner[fact.owner_id].append(_atomic_fact_to_entry_body(fact))

    # 5. Write each owner's full list once for this source memcell.
    writer = _get_writer()
    source_date = from_timestamp(event.memcell.timestamp).date()
    for owner_id, items in by_owner.items():
        await writer.append_entries_once(
            owner_id,
            items,
            parent_id=event.memcell_id,
            date=source_date,
            app_id=event.app_id,
            project_id=event.project_id,
        )

    logger.info(
        "atomic_facts_extracted",
        memcell_id=event.memcell_id,
        session_id=event.session_id,
        count=len(facts),
        owner_ids=sender_ids,
    )


def _atomic_fact_to_entry_body(
    fact: AtomicFact,
) -> tuple[dict[str, object], dict[str, str]]:
    """Split a domain AtomicFact into ``(inline, sections)`` for md rendering.

    Mirrors ``_episode_to_entry_body`` in the user_memory pipeline. Lives in
    the memory layer (strategy module) rather than the writer (infra)
    because it depends on :class:`everos.memory.AtomicFact` — infra is
    not allowed to import memory per the layered architecture contract.
    """
    inline: dict[str, object] = {
        "owner_id": fact.owner_id,
        "session_id": fact.session_id,
        "timestamp": to_iso_format(from_timestamp(fact.timestamp)),
        "parent_type": "memcell",
        "parent_id": fact.parent_id,
    }
    sections = {"Fact": fact.fact}
    return inline, sections
