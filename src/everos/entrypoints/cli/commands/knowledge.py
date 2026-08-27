"""Explicit Knowledge Access maintenance commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from everos.core.persistence.memory_root import MemoryRoot
from everos.memory.knowledge.materializer import materialize_truth_view

app = typer.Typer(
    name="knowledge",
    help="Build and inspect rebuildable Knowledge Access artifacts",
    no_args_is_help=True,
)


@app.command("wiki-build")
def wiki_build(
    truth_view: Annotated[
        Path,
        typer.Option(
            "--truth-view",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Reviewed Local Brain Truth View v1 JSON file",
        ),
    ],
    memory_root: Annotated[
        Path | None,
        typer.Option(
            "--memory-root",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Derived output MemoryRoot (defaults to configured MemoryRoot)",
        ),
    ] = None,
) -> None:
    """Materialize a versioned read-only Wiki snapshot from explicit Truth."""

    payload = json.loads(truth_view.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(
            "truth view must be a JSON object", param_hint="--truth-view"
        )
    manifest = materialize_truth_view(
        payload,
        memory_root=MemoryRoot(memory_root) if memory_root is not None else None,
    )
    typer.echo(
        json.dumps(
            {
                "snapshot_id": manifest.snapshot_id,
                "compiler_version": manifest.compiler_version,
                "page_count": len(manifest.pages),
                "verification": sorted(
                    {page.verification_status for page in manifest.pages}
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
