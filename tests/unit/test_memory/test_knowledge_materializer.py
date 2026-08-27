from __future__ import annotations

import json
from pathlib import Path

import pytest

from everos.core.persistence.memory_root import MemoryRoot
from everos.memory.knowledge import WikiCompiler
from everos.memory.knowledge.materializer import (
    WikiMaterializationError,
    load_current_manifest,
    load_current_page,
    materialize_truth_view,
)


def _record(
    record_id: str,
    statement: str,
    memory_type: str,
    *,
    project_key: str = "project-a",
    truth_status: str = "CURRENT",
    status: str = "accepted",
    superseded_by: str | None = None,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "memory_type": memory_type,
        "subject": f"subject.{record_id}",
        "statement": statement,
        "state_key": record_id,
        "status": status,
        "scope": {"kind": "project", "project_key": project_key},
        "valid_from": "2026-08-27T10:00:00+08:00",
        "valid_to": None,
        "superseded_by": superseded_by,
        "truth_status": truth_status,
        "source_kinds": ["memo"],
        "provenance": [
            {
                "kind": "memo",
                "source_id": "project-spine",
                "file_sha256": "a" * 64,
                "line": 42,
            }
        ],
    }


def _truth_view() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": "derived-rebuildable-truth-view",
        "generated_at": "2026-08-27T10:30:00+08:00",
        "truth_view_sha256": "b" * 64,
        "policy": {
            "rebuildable": True,
            "conflicts_fail_closed": True,
        },
        "current_facts": [
            _record("state-1", "Current state", "PROJECT_STATE"),
            # User facts are valid Truth but are outside the project-Wiki v1
            # section contract and must not be silently reclassified.
            _record("user-fact", "User fact", "USER_FACT"),
        ],
        "active_goals": [_record("goal-1", "Ship it", "GOAL")],
        "active_decisions": [],
        "active_constraints": [],
        "open_todos": [],
        "history": [
            _record(
                "old-state",
                "Previous state",
                "PROJECT_STATE",
                truth_status="HISTORY",
                status="superseded",
                superseded_by="state-1",
            ),
            _record(
                "rejected-history",
                "Rejected history",
                "PROJECT_STATE",
                truth_status="HISTORY",
                status="rejected",
            ),
        ],
        "conflicts": [_record("conflict-1", "Must never render", "PROJECT_STATE")],
        "review_required": [_record("candidate-1", "Must never render either", "GOAL")],
    }


def test_materializer_is_deterministic_and_fail_closed(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path / "memory")
    payload = _truth_view()

    first = materialize_truth_view(payload, memory_root=root)
    second = materialize_truth_view(
        json.loads(json.dumps(payload)),
        memory_root=root,
    )

    assert first.snapshot_id == second.snapshot_id
    assert len(first.pages) == 1
    summary = first.pages[0]
    assert summary.project_key == "project-a"
    assert summary.rendered_claim_count == 3
    assert summary.backed_claim_count == 3
    assert summary.unsupported_claim_count == 0
    assert summary.stale_claim_count == 0
    assert summary.verification_status == "VERIFIED"

    page = load_current_page(summary.slug, root)
    assert page is not None
    markdown = str(page["markdown"])
    assert "## Current State" in markdown
    assert "Current state" in markdown
    assert "## Goals" in markdown
    assert "Ship it" in markdown
    assert "## History" in markdown
    assert "Previous state" in markdown
    assert "User fact" not in markdown
    assert "Must never render" not in markdown
    assert "Rejected history" not in markdown
    assert page["agent_view"]["claim_set_sha256"] == summary.claim_set_sha256

    snapshot_dirs = list((root.index_dir / "wiki" / "snapshots").iterdir())
    assert snapshot_dirs == [root.index_dir / "wiki" / "snapshots" / first.snapshot_id]


def test_materializer_snapshot_changes_with_compiler_version(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path / "memory")
    first = materialize_truth_view(_truth_view(), memory_root=root)
    second = materialize_truth_view(
        _truth_view(),
        memory_root=root,
        compiler=WikiCompiler(version="wiki-compiler-v2-test"),
    )
    assert first.snapshot_id != second.snapshot_id
    assert load_current_manifest(root) == second


def test_active_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path / "memory")
    manifest = materialize_truth_view(_truth_view(), memory_root=root)
    path = (
        root.index_dir / "wiki" / "snapshots" / manifest.snapshot_id / "manifest.json"
    )
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(WikiMaterializationError, match="checksum mismatch"):
        load_current_manifest(root)


def test_page_tamper_fails_closed(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path / "memory")
    manifest = materialize_truth_view(_truth_view(), memory_root=root)
    page = manifest.pages[0]
    path = (
        root.index_dir
        / "wiki"
        / "snapshots"
        / manifest.snapshot_id
        / f"{page.artifact_id}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["markdown"] = "# tampered\n"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WikiMaterializationError, match="markdown checksum mismatch"):
        load_current_page(page.slug, root)


def test_replay_refuses_valid_but_drifted_snapshot(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path / "memory")
    manifest = materialize_truth_view(_truth_view(), memory_root=root)
    path = (
        root.index_dir / "wiki" / "snapshots" / manifest.snapshot_id / "manifest.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pages"][0]["title"] = "drifted title"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        WikiMaterializationError,
        match="does not match deterministic build",
    ):
        materialize_truth_view(_truth_view(), memory_root=root)


def test_project_slug_collision_fails_closed(tmp_path: Path) -> None:
    payload = _truth_view()
    current_facts = payload["current_facts"]
    assert isinstance(current_facts, list)
    current_facts.append(
        _record(
            "state-collision",
            "Other current state",
            "PROJECT_STATE",
            project_key="project a",
        )
    )
    with pytest.raises(WikiMaterializationError, match="slug collision"):
        materialize_truth_view(payload, memory_root=tmp_path / "memory")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "something-else"),
        ("schema_version", 2),
    ],
)
def test_materializer_rejects_unknown_truth_view_contract(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _truth_view()
    payload[field] = value
    with pytest.raises(WikiMaterializationError):
        materialize_truth_view(payload, memory_root=tmp_path / "memory")
