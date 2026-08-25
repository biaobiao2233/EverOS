"""``everos profile`` CLI command group for status inspection.

Provides safe offline recovery with dynamic prompt budgeting.
"""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import typer
from everalgo.user_memory import ProfileExtractor
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from everos.component.llm import get_llm_client
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
)
from everos.core.persistence.locking import LockError, memory_root_lock
from everos.infra.persistence.markdown import (
    ProfileReader,
    UserProfileFrontmatter,
)
from everos.infra.persistence.sqlite import cluster_repo
from everos.infra.persistence.sqlite import sqlite_manager as sm
from everos.memory.strategies._partition_locks import get_partition_lock
from everos.memory.strategies.extract_user_profile import (
    DEFAULT_MAX_BATCH_MEMCELLS,
    DEFAULT_MAX_PROMPT_CHARS,
    BoundedProfileLLMClient,
    _persist_profile,
    _to_algo_profile,
    get_unprocessed_memcells_for_owner,
    plan_next_step,
    prepare_pending_items,
    render_full_prompt,
    rewrite_profile_for_quality,
)

app = typer.Typer(
    name="profile",
    help="Inspect and safely recover user profile extractions",
    no_args_is_help=True,
)


def is_everos_service_active() -> bool:
    """Check if everos systemd service or API server process is active."""
    # 1. Check systemctl if available
    systemctl = shutil.which("systemctl")
    if systemctl:
        for svc in ("everos-v2", "everos", "everos.service", "everos-v2.service"):
            try:
                res = subprocess.run(
                    [systemctl, "is-active", "--quiet", svc],
                    capture_output=True,
                    timeout=2,
                )
                if res.returncode == 0:
                    return True
            except Exception:
                pass

    # 2. Check if local API port is actively listening
    try:
        from everos.config import load_settings

        settings = load_settings()
        port = settings.api.port or 1996
        host = settings.api.host or "127.0.0.1"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return True
    except Exception:
        pass

    return False


def _create_readonly_engine(db_path: Path) -> AsyncEngine:
    """Create an async SQLAlchemy engine with PRAGMA query_only=ON."""
    url = f"sqlite+aiosqlite:///{db_path.as_posix()}?mode=ro"
    engine = create_async_engine(url, future=True)

    @event.listens_for(engine.sync_engine, "connect")
    def _apply_ro_pragmas(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA query_only=ON;")
            cursor.execute("PRAGMA busy_timeout=5000;")
        finally:
            cursor.close()

    return engine


@asynccontextmanager
async def _readonly_runtime():  # type: ignore[no-untyped-def]
    """Stand up a strictly read-only SQLite connection without DDL or write PRAGMAs."""
    old_engine = sm._engine
    old_sf = sm._session_factory

    memory_root = MemoryRoot.default()
    ro_engine = _create_readonly_engine(memory_root.system_db)
    ro_sf = create_session_factory(ro_engine)

    sm._engine = ro_engine
    sm._session_factory = ro_sf
    try:
        yield
    finally:
        await ro_engine.dispose()
        sm._engine = old_engine
        sm._session_factory = old_sf


@app.command("clean")
def clean(
    owner_id: Annotated[
        str,
        typer.Option("--owner-id", "-o", help="Owner ID whose profile to clean."),
    ],
    app_id: Annotated[
        str,
        typer.Option("--app-id", "-a", help="App ID scope partition."),
    ],
    project_id: Annotated[
        str,
        typer.Option("--project-id", "-p", help="Project ID scope partition."),
    ],
) -> None:
    """Offline one-shot quality rewrite of the Profile at the same watermark."""

    async def _run() -> None:
        if is_everos_service_active():
            typer.echo(
                "Error: EverOS server is currently active; "
                "stop it before profile clean.",
                err=True,
            )
            raise typer.Exit(code=1)

        memory_root = MemoryRoot.default()
        partition = f"{app_id}:{project_id}:{owner_id}"
        try:
            async with (
                memory_root_lock(memory_root, blocking=False),
                get_partition_lock("extract_user_profile", partition),
            ):
                reader = ProfileReader(root=memory_root)
                existing = await reader.read(
                    owner_id,
                    schema=UserProfileFrontmatter,
                    app_id=app_id,
                    project_id=project_id,
                )
                if not existing:
                    typer.echo("No existing Profile found to clean.")
                    return

                original = _to_algo_profile(existing[0])
                bounded_llm = BoundedProfileLLMClient(
                    get_llm_client(), max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS
                )
                cleaned = await rewrite_profile_for_quality(
                    original, bounded_llm, force=True
                )
                await _persist_profile(
                    cleaned,
                    owner_id=owner_id,
                    app_id=app_id,
                    project_id=project_id,
                )
                typer.echo(
                    "Profile quality clean complete at unchanged watermark "
                    f"{cleaned.timestamp}."
                )
        except LockError:
            typer.echo(
                "Error: another process holds the EverOS memory-root lock.", err=True
            )
            raise typer.Exit(code=1) from None

    asyncio.run(_run())


@app.command("status")
def status(
    owner_id: Annotated[
        str,
        typer.Option(
            "--owner-id",
            "-o",
            help="Owner ID (e.g. 'user') to inspect.",
        ),
    ],
    app_id: Annotated[
        str,
        typer.Option(
            "--app-id",
            "-a",
            help="App ID scope partition (e.g. 'codex').",
        ),
    ],
    project_id: Annotated[
        str,
        typer.Option(
            "--project-id",
            "-p",
            help="Project ID scope partition (e.g. 'solo').",
        ),
    ],
) -> None:
    """Print profile status and unprocessed MemCell summary for an owner."""

    async def _run() -> None:
        async with _readonly_runtime():
            reader = ProfileReader(root=MemoryRoot.default())
            existing = await reader.read(
                owner_id,
                schema=UserProfileFrontmatter,
                app_id=app_id,
                project_id=project_id,
            )
            last_ts = existing[0].profile_timestamp_ms if existing else 0
            cursor_ids = (
                list(getattr(existing[0], "profile_watermark_memcell_ids", []) or [])
                if existing
                else None
            )

            clusters = await cluster_repo.list_for_owner(
                owner_id, "user_memory", app_id=app_id, project_id=project_id
            )
            total_clustered_cells = sum(c.count for c in clusters)

            (
                valid_memcells,
                excluded_count,
                _,
            ) = await get_unprocessed_memcells_for_owner(
                owner_id,
                app_id=app_id,
                project_id=project_id,
                last_profile_ts=last_ts,
                processed_memcell_ids_at_ts=cursor_ids,
            )

            current_profile = _to_algo_profile(existing[0]) if existing else None
            pending_items = prepare_pending_items(
                valid_memcells, old_profile=current_profile, owner_id=owner_id
            )
            est_batches = 0
            while pending_items:
                batch, pending_items = plan_next_step(
                    pending_items,
                    current_profile,
                    owner_id=owner_id,
                    max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS,
                    max_batch_memcells=DEFAULT_MAX_BATCH_MEMCELLS,
                )
                est_batches += 1

            typer.echo(
                f"Profile Status for '{owner_id}' (app={app_id}, project={project_id}):"
            )
            typer.echo(f"  Current Watermark:     {last_ts}")
            typer.echo(f"  Total Clusters:        {len(clusters)}")
            typer.echo(f"  Total Clustered Cells: {total_clustered_cells}")
            typer.echo(f"  Pending MemCells:      {len(valid_memcells)}")
            typer.echo(f"  Excluded (e.g. Trae):  {excluded_count}")
            typer.echo(f"  Estimated Batches:     {est_batches}")

    asyncio.run(_run())


@app.command("recover")
def recover(
    owner_id: Annotated[
        str,
        typer.Option(
            "--owner-id",
            "-o",
            help="Owner ID (e.g. 'user') whose profile to recover.",
        ),
    ],
    app_id: Annotated[
        str,
        typer.Option(
            "--app-id",
            "-a",
            help="App ID scope partition (e.g. 'codex').",
        ),
    ],
    project_id: Annotated[
        str,
        typer.Option(
            "--project-id",
            "-p",
            help="Project ID scope partition (e.g. 'solo').",
        ),
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Simulate recovery without LLM calls or disk writes.",
        ),
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Max MemCells per batch.",
        ),
    ] = DEFAULT_MAX_BATCH_MEMCELLS,
    max_prompt_chars: Annotated[
        int,
        typer.Option(
            "--max-prompt-chars",
            help="Max total prompt character budget per batch.",
        ),
    ] = DEFAULT_MAX_PROMPT_CHARS,
    max_checkpoints: Annotated[
        int,
        typer.Option(
            "--max-checkpoints",
            help=(
                "Stop after this many successful profile checkpoint writes. "
                "Use 0 for unlimited recovery."
            ),
        ),
    ] = 0,
) -> None:
    """Recover pending MemCells into user profile safely."""
    if max_prompt_chars > DEFAULT_MAX_PROMPT_CHARS:
        typer.echo(
            f"Error: --max-prompt-chars ({max_prompt_chars}) exceeds hard safety limit "
            f"of {DEFAULT_MAX_PROMPT_CHARS}.",
            err=True,
        )
        raise typer.Exit(code=1)
    if max_checkpoints < 0:
        typer.echo(
            "Error: --max-checkpoints must be 0 (unlimited) or a positive integer.",
            err=True,
        )
        raise typer.Exit(code=1)

    async def _run() -> None:
        memory_root = MemoryRoot.default()

        # 1. Dry Run Mode: strictly read-only, no file locks or writes
        if dry_run:
            async with _readonly_runtime():
                reader = ProfileReader(root=memory_root)
                existing = await reader.read(
                    owner_id,
                    schema=UserProfileFrontmatter,
                    app_id=app_id,
                    project_id=project_id,
                )
                last_profile_ts = existing[0].profile_timestamp_ms if existing else 0
                cursor_ids = (
                    list(
                        getattr(existing[0], "profile_watermark_memcell_ids", []) or []
                    )
                    if existing
                    else None
                )

                (
                    valid_memcells,
                    excluded_count,
                    _,
                ) = await get_unprocessed_memcells_for_owner(
                    owner_id,
                    app_id=app_id,
                    project_id=project_id,
                    last_profile_ts=last_profile_ts,
                    processed_memcell_ids_at_ts=cursor_ids,
                )

                current_algo_profile = (
                    _to_algo_profile(existing[0]) if existing else None
                )

                pending_items = prepare_pending_items(
                    valid_memcells,
                    old_profile=current_algo_profile,
                    owner_id=owner_id,
                )
                simulated_batches: list[tuple[int, int, int, int, int | None]] = []

                while pending_items:
                    batch, pending_items = plan_next_step(
                        pending_items,
                        current_algo_profile,
                        owner_id=owner_id,
                        max_prompt_chars=max_prompt_chars,
                        max_batch_memcells=batch_size,
                    )
                    p_chars = len(
                        render_full_prompt(
                            batch.memcells,
                            current_algo_profile,
                            owner_id=owner_id,
                        )
                    )
                    b_min_ts = min(mc.timestamp for mc in batch.memcells)
                    b_max_ts = max(mc.timestamp for mc in batch.memcells)
                    simulated_batches.append(
                        (
                            len(batch.memcells),
                            p_chars,
                            b_min_ts,
                            b_max_ts,
                            batch.completed_watermark,
                        )
                    )

                typer.echo(f"=== DRY-RUN Profile Recovery for '{owner_id}' ===")
                typer.echo(f"App / Project:         {app_id} / {project_id}")
                typer.echo(f"Current Watermark:     {last_profile_ts}")
                typer.echo(f"Pending MemCells:      {len(valid_memcells)}")
                typer.echo(f"Excluded MemCells:     {excluded_count}")
                typer.echo(f"Estimated Batches:     {len(simulated_batches)}")

                if simulated_batches:
                    typer.echo("Estimated Batch Breakdown:")
                    for idx, (
                        b_cells,
                        p_chars,
                        b_min_ts,
                        b_max_ts,
                        comp_wm,
                    ) in enumerate(simulated_batches, start=1):
                        completed_mark = (
                            f"-> advances to {comp_wm}"
                            if comp_wm is not None
                            else "(sub-chunk, watermark held)"
                        )
                        typer.echo(
                            f"  Batch {idx:02d}/{len(simulated_batches):02d}: "
                            f"{b_cells:2d} cells, "
                            f"{p_chars:5d} prompt chars, "
                            f"ts [{b_min_ts} .. {b_max_ts}] {completed_mark}"
                        )
                else:
                    typer.echo("  No pending batches to execute.")
                return

        # 2. Execution Mode: check if active service is running
        if is_everos_service_active():
            typer.echo(
                "Error: EverOS server is currently active "
                "(detected running service or listening port).",
                err=True,
            )
            typer.echo(
                "Cannot perform offline profile recovery while server is active. "
                "Please stop the EverOS service before recovery.",
                err=True,
            )
            raise typer.Exit(code=1)

        # 3. Execution Mode: acquire cross-process lock
        try:
            partition = f"{app_id}:{project_id}:{owner_id}"
            async with (
                memory_root_lock(memory_root, blocking=False),
                get_partition_lock("extract_user_profile", partition),
                _readonly_runtime(),
            ):
                reader = ProfileReader(root=memory_root)
                existing = await reader.read(
                    owner_id,
                    schema=UserProfileFrontmatter,
                    app_id=app_id,
                    project_id=project_id,
                )
                last_profile_ts = existing[0].profile_timestamp_ms if existing else 0
                current_cursor_ids = (
                    list(
                        getattr(existing[0], "profile_watermark_memcell_ids", []) or []
                    )
                    if existing
                    else []
                )

                (
                    valid_memcells,
                    excluded_count,
                    max_excluded_ts,
                ) = await get_unprocessed_memcells_for_owner(
                    owner_id,
                    app_id=app_id,
                    project_id=project_id,
                    last_profile_ts=last_profile_ts,
                    processed_memcell_ids_at_ts=current_cursor_ids
                    if existing
                    else None,
                )

                if not valid_memcells:
                    if max_excluded_ts > last_profile_ts and existing:
                        updated_profile = _to_algo_profile(existing[0])
                        updated_profile.timestamp = max_excluded_ts
                        updated_profile.profile_watermark_memcell_ids = []
                        await _persist_profile(
                            updated_profile,
                            owner_id=owner_id,
                            app_id=app_id,
                            project_id=project_id,
                        )
                        typer.echo(
                            f"Watermark advanced to {max_excluded_ts} for "
                            f"{excluded_count} excluded cells."
                        )
                    else:
                        typer.echo(
                            "No unprocessed MemCells found. Profile is up to date."
                        )
                    return

                current_algo_profile = (
                    _to_algo_profile(existing[0]) if existing else None
                )
                pending_items = prepare_pending_items(
                    valid_memcells,
                    old_profile=current_algo_profile,
                    owner_id=owner_id,
                )

                typer.echo(
                    f"Starting Profile Recovery for '{owner_id}': "
                    f"{len(valid_memcells)} cells pending..."
                )

                bounded_llm = BoundedProfileLLMClient(
                    get_llm_client(), max_prompt_chars=max_prompt_chars
                )
                extractor = ProfileExtractor(llm=bounded_llm)
                current_profile = current_algo_profile
                current_watermark = last_profile_ts
                current_watermark_ids = list(current_cursor_ids)
                batch_idx = 1
                checkpoints_written = 0

                while pending_items:
                    batch, pending_items = plan_next_step(
                        pending_items,
                        current_profile,
                        owner_id=owner_id,
                        max_prompt_chars=max_prompt_chars,
                        max_batch_memcells=batch_size,
                    )

                    p_chars = len(
                        render_full_prompt(
                            batch.memcells,
                            current_profile,
                            owner_id=owner_id,
                        )
                    )
                    typer.echo(
                        f"  [Batch {batch_idx}] "
                        f"Processing {len(batch.memcells)} cells "
                        f"({p_chars} prompt chars)..."
                    )

                    new_profile = await extractor.aextract(
                        batch.memcells,
                        sender_id=owner_id,
                        old_profile=current_profile,
                    )

                    new_profile = await rewrite_profile_for_quality(
                        new_profile, bounded_llm
                    )

                    current_profile = new_profile

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
                        new_profile.profile_watermark_memcell_ids = (
                            current_watermark_ids
                        )

                        await _persist_profile(
                            new_profile,
                            owner_id=owner_id,
                            app_id=app_id,
                            project_id=project_id,
                        )

                        current_watermark = new_watermark
                        checkpoints_written += 1
                        typer.echo(
                            f"  [Batch {batch_idx}] Done. "
                            f"Watermark advanced to {new_watermark} "
                            f"(cursor items: {len(current_watermark_ids)})."
                        )

                        if (
                            max_checkpoints > 0
                            and checkpoints_written >= max_checkpoints
                        ):
                            typer.echo(
                                "Recovery stopped after "
                                f"{checkpoints_written} checkpoint(s) as requested. "
                                f"Current watermark: {current_watermark}."
                            )
                            return
                    else:
                        typer.echo(
                            f"  [Batch {batch_idx}] Done sub-chunk. "
                            f"Watermark held at {current_watermark}."
                        )

                    batch_idx += 1

                typer.echo(
                    f"Recovery Complete: processed {len(valid_memcells)} cells. "
                    f"Final watermark: {current_watermark}."
                )
        except LockError:
            typer.echo(
                "Error: EverOS server is currently active "
                "(or another process holds the memory-root lock).",
                err=True,
            )
            typer.echo(
                "Cannot perform offline profile recovery concurrently. "
                "Please stop the EverOS service before recovery.",
                err=True,
            )
            raise typer.Exit(code=1) from None

    asyncio.run(_run())
