"""Strategy: Extract and update user profiles with dynamic lossless batching."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import structlog
from everalgo.prompts import render_prompt
from everalgo.types import (
    ChatMessage,
)
from everalgo.types import (
    MemCell as AlgoMemCell,
)
from everalgo.types import (
    Profile as AlgoProfile,
)
from everalgo.user_memory import ProfileExtractor
from everalgo.user_memory.profile import (
    PROFILE_INITIAL_EXTRACTION_PROMPT,
    PROFILE_UPDATE_PROMPT,
    _render_conversation,
    _render_profile_for_update,
    format_message_timestamp,
)

from everos.component.llm import get_llm_client
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.markdown import (
    ProfileReader,
    ProfileWriter,
    UserProfileFrontmatter,
)
from everos.infra.persistence.sqlite import cluster_repo, memcell_repo
from everos.memory.events import ProfileClusterUpdated
from everos.memory.strategies._partition_locks import get_partition_lock

logger = structlog.get_logger(__name__)

EXCLUDED_USER_PROFILE_APP_IDS: frozenset[str] = frozenset({"trae"})

HARD_MAX_PROMPT_CHARS: int = 40_000
DEFAULT_MAX_PROMPT_CHARS: int = 40_000
DEFAULT_MAX_BATCH_MEMCELLS: int = 25
DEFAULT_MAX_SINGLE_MESSAGE_CHARS: int = 15_000

_writer: ProfileWriter | None = None
_reader: ProfileReader | None = None


class BoundedProfileLLMClient:
    """Wrapper around LLMClient that strictly enforces prompt character limit."""

    def __init__(
        self,
        delegate: Any,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    ) -> None:
        if max_prompt_chars > HARD_MAX_PROMPT_CHARS:
            raise ValueError(
                f"max_prompt_chars ({max_prompt_chars}) cannot exceed "
                f"hard safety limit of {HARD_MAX_PROMPT_CHARS}"
            )
        self._delegate = delegate
        self._max_prompt_chars = max_prompt_chars

    async def chat(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        if messages:
            content = getattr(messages[0], "content", None)
            if content is None and isinstance(messages[0], dict):
                content = messages[0].get("content", "")
            content_len = len(str(content or ""))
            if content_len > self._max_prompt_chars:
                logger.error(
                    "profile_llm_prompt_budget_exceeded",
                    prompt_len=content_len,
                    max_prompt_chars=self._max_prompt_chars,
                )
                raise ValueError(
                    f"Profile LLM prompt length ({content_len} chars) "
                    f"exceeds hard budget ({self._max_prompt_chars} chars). "
                    "Fail closed without calling provider."
                )
        return await self._delegate.chat(messages, **kwargs)


def _get_writer() -> ProfileWriter:
    global _writer
    if _writer is None:
        _writer = ProfileWriter(root=MemoryRoot.default())
    return _writer


def _get_reader() -> ProfileReader:
    global _reader
    if _reader is None:
        _reader = ProfileReader(root=MemoryRoot.default())
    return _reader


def _to_algo_profile(frontmatter: UserProfileFrontmatter) -> AlgoProfile:
    """Convert persisted UserProfileFrontmatter into everalgo AlgoProfile."""
    algo_profile = AlgoProfile.model_validate(
        {
            "owner_id": frontmatter.user_id,
            "summary": frontmatter.summary,
            "timestamp": frontmatter.profile_timestamp_ms,
            "explicit_info": list(frontmatter.explicit_info or []),
            "implicit_traits": list(frontmatter.implicit_traits or []),
        }
    )
    algo_profile.profile_watermark_memcell_ids = list(
        getattr(frontmatter, "profile_watermark_memcell_ids", []) or []
    )
    return algo_profile


def _to_frontmatter(
    profile: AlgoProfile,
    *,
    owner_id: str,
    app_id: str,
    project_id: str,
) -> UserProfileFrontmatter:
    """Convert everalgo AlgoProfile into persisted UserProfileFrontmatter."""
    extras = getattr(profile, "model_extra", {}) or {}
    explicit_info = getattr(profile, "explicit_info", None)
    if explicit_info is None:
        explicit_info = extras.get("explicit_info", [])
    implicit_traits = getattr(profile, "implicit_traits", None)
    if implicit_traits is None:
        implicit_traits = extras.get("implicit_traits", [])

    watermark_memcell_ids = getattr(profile, "profile_watermark_memcell_ids", None)
    if watermark_memcell_ids is None:
        watermark_memcell_ids = extras.get("profile_watermark_memcell_ids", [])

    return UserProfileFrontmatter(
        id=f"profile_{owner_id}",
        app_id=app_id,
        project_id=project_id,
        user_id=owner_id,
        summary=profile.summary,
        explicit_info=list(explicit_info or []),
        implicit_traits=list(implicit_traits or []),
        profile_timestamp_ms=profile.timestamp,
        profile_watermark_memcell_ids=list(watermark_memcell_ids or []),
    )


def render_full_prompt(
    memcells: Sequence[AlgoMemCell],
    old_profile: AlgoProfile | None,
) -> str:
    """Render the exact prompt that ProfileExtractor passes to LLM."""
    conversation_text = _render_conversation(memcells)
    if old_profile is None:
        return render_prompt(
            PROFILE_INITIAL_EXTRACTION_PROMPT,
            None,
            conversation_text=conversation_text,
        )
    else:
        current_profile_text = _render_profile_for_update(old_profile)
        return render_prompt(
            PROFILE_UPDATE_PROMPT,
            None,
            current_profile=current_profile_text,
            conversations=conversation_text,
        )


def split_memcell_losslessly(
    cell: AlgoMemCell,
    *,
    old_profile: AlgoProfile | None = None,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> list[AlgoMemCell]:
    """Split a MemCell into sub-MemCells without dropping any message or text.

    Guarantees that each resulting sub-MemCell, when rendered in the prompt,
    strictly satisfies:
    len(render_full_prompt([sub_cell], old_profile)) <= max_prompt_chars.

    If metadata overhead alone exceeds max_prompt_chars, fails closed with
    ValueError.
    """
    # 1. If the entire cell already fits within prompt budget, return as is
    if len(render_full_prompt([cell], old_profile)) <= max_prompt_chars:
        return [cell]

    # Calculate prompt overhead without any conversation text
    base_prompt_overhead = len(render_full_prompt([], old_profile))
    if base_prompt_overhead >= max_prompt_chars:
        raise ValueError(
            f"Prompt overhead with current profile ({base_prompt_overhead} chars) "
            f"exceeds max_prompt_chars ({max_prompt_chars}). Cannot process."
        )

    avail_conv_chars = max_prompt_chars - base_prompt_overhead - 50

    # 2. Decompose items, slicing content if needed
    decomposed_items: list[Any] = []
    for item in cell.items:
        if isinstance(item, ChatMessage) or (
            hasattr(item, "role") and hasattr(item, "content")
        ):
            time_str = format_message_timestamp(
                getattr(item, "timestamp", cell.timestamp)
            )
            speaker = getattr(item, "sender_name", None) or getattr(
                item, "sender_id", ""
            )
            user_id = getattr(item, "sender_id", "") or ""
            prefix = f"[{time_str}] {speaker}(user_id:{user_id}): "
            meta_overhead = len(prefix) + 1  # include newline

            if meta_overhead > avail_conv_chars:
                # Metadata alone exceeds available conversation budget -> fail closed!
                raise ValueError(
                    f"ChatMessage metadata length ({meta_overhead} chars) "
                    f"exceeds available prompt budget ({avail_conv_chars} chars). "
                    "Fail closed without truncating metadata."
                )

            max_content_chunk = max(100, avail_conv_chars - meta_overhead)
            content_str = str(getattr(item, "content", "") or "")

            if len(content_str) > max_content_chunk:
                for idx, i in enumerate(range(0, len(content_str), max_content_chunk)):
                    slice_msg = ChatMessage(
                        id=f"{getattr(item, 'id', 'm')}_slice_{idx}",
                        role=getattr(item, "role", "user"),
                        content=content_str[i : i + max_content_chunk],
                        timestamp=getattr(item, "timestamp", cell.timestamp),
                        sender_id=getattr(item, "sender_id", ""),
                        sender_name=getattr(item, "sender_name", None),
                    )
                    decomposed_items.append(slice_msg)
            else:
                decomposed_items.append(item)
        else:
            decomposed_items.append(item)

    # 3. Pack decomposed_items into bounded sub-MemCells
    sub_cells: list[AlgoMemCell] = []
    current_items: list[Any] = []

    for item in decomposed_items:
        test_cell = AlgoMemCell(items=current_items + [item], timestamp=cell.timestamp)
        if (
            current_items
            and len(render_full_prompt([test_cell], old_profile)) > max_prompt_chars
        ):
            sub_cells.append(AlgoMemCell(items=current_items, timestamp=cell.timestamp))
            current_items = [item]
        else:
            current_items.append(item)

    if current_items:
        sub_cells.append(AlgoMemCell(items=current_items, timestamp=cell.timestamp))

    # Final assertion on all generated sub-cells
    for sc in sub_cells:
        sc_len = len(render_full_prompt([sc], old_profile))
        if sc_len > max_prompt_chars:
            raise ValueError(
                f"Generated sub-cell prompt length ({sc_len} chars) "
                f"exceeds max_prompt_chars ({max_prompt_chars})."
            )

    return sub_cells


@dataclasses.dataclass(frozen=True)
class NextBatch:
    """Next dynamic batch of MemCells with watermark commit info."""

    memcells: list[AlgoMemCell]
    completed_watermark: int | None
    completed_memcell_ids: list[str] = dataclasses.field(default_factory=list)


# PendingItem: (sub_cell, stable_memcell_id, orig_ts, is_group_final, group_ids)
PendingItem = tuple[AlgoMemCell, str, int, bool, list[str]]


def prepare_pending_items(
    valid_items: Sequence[tuple[str, AlgoMemCell]],
    *,
    old_profile: AlgoProfile | None = None,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> list[PendingItem]:
    """Tag each sub-cell with timestamp group membership and commit boundaries."""
    if not valid_items:
        return []

    # 1. Group items by timestamp preserving stable order
    by_timestamp: dict[int, list[tuple[str, AlgoMemCell]]] = {}
    for stable_id, mc in valid_items:
        by_timestamp.setdefault(mc.timestamp, []).append((stable_id, mc))

    pending_items: list[PendingItem] = []

    for ts, group_cells in by_timestamp.items():
        group_member_ids = [stable_id for stable_id, _ in group_cells]
        total_cells_in_group = len(group_cells)

        for cell_idx, (stable_id, mc) in enumerate(group_cells):
            sub_cells = split_memcell_losslessly(
                mc, old_profile=old_profile, max_prompt_chars=max_prompt_chars
            )
            is_last_cell_in_group = cell_idx == (total_cells_in_group - 1)
            total_sub_cells = len(sub_cells)

            for sub_idx, sub_cell in enumerate(sub_cells):
                is_last_sub = sub_idx == (total_sub_cells - 1)
                is_group_final = is_last_cell_in_group and is_last_sub
                pending_items.append(
                    (sub_cell, stable_id, ts, is_group_final, group_member_ids)
                )

    return pending_items


def plan_next_step(
    pending_items: Sequence[PendingItem],
    current_profile: AlgoProfile | None,
    *,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    max_batch_memcells: int = DEFAULT_MAX_BATCH_MEMCELLS,
) -> tuple[NextBatch, Sequence[PendingItem]]:
    """Select the next batch dynamically against the latest current_profile.

    Watermark is emitted ONLY when the entire timestamp group is completed.
    """
    if not pending_items:
        return (
            NextBatch(
                memcells=[],
                completed_watermark=None,
                completed_memcell_ids=[],
            ),
            [],
        )

    (
        first_mc,
        first_id,
        first_orig_ts,
        first_is_group_final,
        first_group_ids,
    ) = pending_items[0]

    # Re-slice first cell if profile growth caused it to exceed budget
    if len(render_full_prompt([first_mc], current_profile)) > max_prompt_chars:
        sub_cells = split_memcell_losslessly(
            first_mc,
            old_profile=current_profile,
            max_prompt_chars=max_prompt_chars,
        )
        if len(sub_cells) > 1:
            total_new_subs = len(sub_cells)
            re_sliced: list[PendingItem] = []
            for s_idx, sc in enumerate(sub_cells):
                is_last_sub = s_idx == (total_new_subs - 1)
                is_final = first_is_group_final and is_last_sub
                re_sliced.append(
                    (sc, first_id, first_orig_ts, is_final, first_group_ids)
                )
            pending_items = list(re_sliced) + list(pending_items[1:])
            (
                first_mc,
                first_id,
                first_orig_ts,
                first_is_group_final,
                first_group_ids,
            ) = pending_items[0]

    # Greedily accumulate following cells up to max_batch_memcells
    batch_cells: list[AlgoMemCell] = []
    consumed = 0

    for idx, (
        mc,
        _stable_id,
        _orig_ts,
        _is_group_final,
        _group_ids,
    ) in enumerate(pending_items[:max_batch_memcells]):
        test_batch = batch_cells + [mc]
        if len(render_full_prompt(test_batch, current_profile)) <= max_prompt_chars:
            batch_cells.append(mc)
            consumed = idx + 1
        else:
            break

    if not batch_cells:
        batch_cells = [first_mc]
        consumed = 1

    last_consumed_item = pending_items[consumed - 1]
    (
        _last_mc,
        _last_id,
        last_orig_ts,
        last_is_group_final,
        last_group_ids,
    ) = last_consumed_item

    # Watermark is committed ONLY if the last item in this batch completes a group
    if last_is_group_final:
        completed_watermark = last_orig_ts
        completed_memcell_ids = list(last_group_ids)
    else:
        completed_watermark = None
        completed_memcell_ids = []

    return (
        NextBatch(
            memcells=batch_cells,
            completed_watermark=completed_watermark,
            completed_memcell_ids=completed_memcell_ids,
        ),
        pending_items[consumed:],
    )


async def _persist_profile(
    profile: AlgoProfile,
    *,
    owner_id: str,
    app_id: str,
    project_id: str,
) -> None:
    """Persist updated profile frontmatter and summary markdown to disk."""
    writer = _get_writer()
    frontmatter = _to_frontmatter(
        profile,
        owner_id=owner_id,
        app_id=app_id,
        project_id=project_id,
    )
    body = profile.summary.strip() + "\n" if profile.summary else "\n"
    await writer.write(
        owner_id,
        frontmatter=frontmatter,
        body=body,
        app_id=app_id,
        project_id=project_id,
    )
    logger.info(
        "profile_persisted_to_disk",
        owner_id=owner_id,
        app_id=app_id,
        project_id=project_id,
        watermark_ts=profile.timestamp,
        watermark_memcell_ids=getattr(frontmatter, "profile_watermark_memcell_ids", []),
    )


async def get_unprocessed_memcells_for_owner(
    owner_id: str,
    *,
    app_id: str = "default",
    project_id: str = "default",
    last_profile_ts: int = 0,
    processed_memcell_ids_at_ts: Sequence[str] | None = None,
) -> tuple[list[tuple[str, AlgoMemCell]], int, int]:
    """Retrieve candidate MemCells with timestamp >= last_profile_ts."""
    user_clusters = await cluster_repo.list_for_owner(
        owner_id,
        "user_memory",
        app_id=app_id,
        project_id=project_id,
    )
    if not user_clusters:
        return [], 0, 0

    processed_set = set(processed_memcell_ids_at_ts or [])
    has_cursor = bool(processed_memcell_ids_at_ts)

    target_clusters: list[Any] = [
        c
        for c in user_clusters
        if c.last_ts > last_profile_ts
        or (
            c.last_ts == last_profile_ts
            and (
                last_profile_ts == 0
                or (has_cursor and any(m not in processed_set for m in c.members))
            )
        )
    ]

    if not target_clusters:
        return [], 0, 0

    # Deduplicate member IDs across all fresh clusters in stable order
    candidate_member_ids = list(
        dict.fromkeys(
            member_id for cluster in target_clusters for member_id in cluster.members
        )
    )

    if not candidate_member_ids:
        return [], 0, 0

    rows = await memcell_repo.find_by_ids(candidate_member_ids)
    if not rows:
        return [], 0, 0

    valid_algo_memcells: list[tuple[str, AlgoMemCell]] = []
    excluded_count = 0
    max_excluded_ts = 0

    for row in rows:
        algo_mc = AlgoMemCell.model_validate_json(row.payload_json)
        mc_ts = algo_mc.timestamp

        # Candidate check: mc_ts > watermark OR unmerged same-timestamp cell
        if mc_ts < last_profile_ts:
            continue
        if mc_ts == last_profile_ts and (
            last_profile_ts > 0 and (not has_cursor or row.memcell_id in processed_set)
        ):
            continue

        if row.app_id in EXCLUDED_USER_PROFILE_APP_IDS:
            excluded_count += 1
            if mc_ts > max_excluded_ts:
                max_excluded_ts = mc_ts
            continue

        valid_algo_memcells.append((row.memcell_id, algo_mc))

    # Strict ascending sort by (timestamp, stable_memcell_id)
    valid_algo_memcells.sort(key=lambda item: (item[1].timestamp, item[0]))

    return valid_algo_memcells, excluded_count, max_excluded_ts


@offline_strategy(
    name="extract_user_profile",
    trigger=Immediate(on=[ProfileClusterUpdated]),
    emits=[],
    max_retries=3,
)
async def extract_user_profile(
    event: ProfileClusterUpdated, ctx: StrategyContext | None = None
) -> None:
    """Strategy to extract/update user profile on profile cluster updates."""
    owner_id = event.owner_id
    app_id = event.app_id
    project_id = event.project_id
    partition = f"{app_id}:{project_id}:{owner_id}"

    async with get_partition_lock("extract_user_profile", partition):
        reader = _get_reader()
        existing = await reader.read(
            owner_id,
            schema=UserProfileFrontmatter,
            app_id=app_id,
            project_id=project_id,
        )

        last_profile_ts = existing[0].profile_timestamp_ms if existing else 0
        current_cursor_ids = (
            list(getattr(existing[0], "profile_watermark_memcell_ids", []) or [])
            if existing
            else []
        )
        is_init = existing is None
        current_profile = _to_algo_profile(existing[0]) if existing else None

        # Fast no-op check for stale cluster events before querying memcells
        if event.cluster_id:
            user_clusters = await cluster_repo.list_for_owner(
                owner_id, "user_memory", app_id=app_id, project_id=project_id
            )
            matching_cluster = next(
                (
                    c
                    for c in user_clusters
                    if getattr(c, "id", None) == event.cluster_id
                    or getattr(c, "cluster_id", None) == event.cluster_id
                ),
                None,
            )
            if matching_cluster:
                if matching_cluster.last_ts < last_profile_ts:
                    logger.info(
                        "profile_extraction_noop_stale_cluster",
                        owner_id=owner_id,
                        cluster_id=event.cluster_id,
                        cluster_last_ts=matching_cluster.last_ts,
                        watermark=last_profile_ts,
                    )
                    return
                elif matching_cluster.last_ts == last_profile_ts:
                    processed_set = set(current_cursor_ids)
                    if event.memcell_id and event.memcell_id in processed_set:
                        logger.info(
                            "profile_extraction_noop_stale_cluster_member_processed",
                            owner_id=owner_id,
                            cluster_id=event.cluster_id,
                            memcell_id=event.memcell_id,
                            watermark=last_profile_ts,
                        )
                        return
                    elif not event.memcell_id and all(
                        m in processed_set for m in matching_cluster.members
                    ):
                        logger.info(
                            "profile_extraction_noop_all_members_processed",
                            owner_id=owner_id,
                            cluster_id=event.cluster_id,
                            watermark=last_profile_ts,
                        )
                        return

        (
            valid_memcells,
            excluded_count,
            max_excluded_ts,
        ) = await get_unprocessed_memcells_for_owner(
            owner_id,
            app_id=app_id,
            project_id=project_id,
            last_profile_ts=last_profile_ts,
            processed_memcell_ids_at_ts=current_cursor_ids if existing else None,
        )

        if not valid_memcells:
            if max_excluded_ts > last_profile_ts and existing:
                updated_profile = _to_algo_profile(existing[0])
                updated_profile.timestamp = max(last_profile_ts, max_excluded_ts)
                updated_profile.profile_watermark_memcell_ids = []
                await _persist_profile(
                    updated_profile,
                    owner_id=owner_id,
                    app_id=app_id,
                    project_id=project_id,
                )
                logger.info(
                    "profile_watermark_advanced_for_excluded_apps",
                    owner_id=owner_id,
                    excluded_count=excluded_count,
                    new_watermark=updated_profile.timestamp,
                )
            else:
                logger.info(
                    "profile_extraction_noop_no_unprocessed_cells",
                    owner_id=owner_id,
                    watermark=last_profile_ts,
                )
            return

        pending_items = prepare_pending_items(
            valid_memcells, old_profile=current_profile
        )
        bounded_llm = BoundedProfileLLMClient(
            get_llm_client(), max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS
        )
        extractor = ProfileExtractor(llm=bounded_llm)
        current_watermark = last_profile_ts
        current_watermark_ids = list(current_cursor_ids)
        batch_idx = 1

        while pending_items:
            batch, pending_items = plan_next_step(
                pending_items,
                current_profile,
                max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS,
                max_batch_memcells=DEFAULT_MAX_BATCH_MEMCELLS,
            )

            logger.info(
                "profile_extraction_batch_processing",
                batch_index=batch_idx,
                batch_cell_count=len(batch.memcells),
            )

            new_profile = await extractor.aextract(
                batch.memcells,
                sender_id=owner_id,
                old_profile=current_profile,
            )

            current_profile = new_profile

            # Commit watermark to disk only when an entire timestamp group has completed
            if batch.completed_watermark is not None:
                new_watermark = batch.completed_watermark
                if new_watermark > current_watermark:
                    current_watermark_ids = list(batch.completed_memcell_ids)
                elif new_watermark == current_watermark:
                    current_watermark_ids = sorted(
                        list(
                            set(current_watermark_ids)
                            | set(batch.completed_memcell_ids)
                        )
                    )
                new_watermark = max(current_watermark, new_watermark)
                new_profile.timestamp = new_watermark
                new_profile.profile_watermark_memcell_ids = current_watermark_ids

                await _persist_profile(
                    new_profile,
                    owner_id=owner_id,
                    app_id=app_id,
                    project_id=project_id,
                )
                current_watermark = new_watermark
                logger.info(
                    "profile_extraction_batch_committed",
                    batch_index=batch_idx,
                    watermark_advanced_to=new_watermark,
                    watermark_memcell_ids=current_watermark_ids,
                )

            batch_idx += 1

        logger.info(
            "user_profile_extracted",
            owner_id=owner_id,
            mode="INIT" if is_init else "UPDATE",
            app_id=app_id,
            project_id=project_id,
            total_processed_memcells=len(valid_memcells),
            final_watermark=current_watermark,
        )
