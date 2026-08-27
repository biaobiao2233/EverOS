from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.memory.knowledge.materializer import materialize_truth_view


def _payload() -> dict[str, object]:
    record = {
        "record_id": "state-1",
        "memory_type": "PROJECT_STATE",
        "subject": "project.state",
        "statement": "Wiki API is read only",
        "status": "accepted",
        "scope": {"kind": "project", "project_key": "wiki-api"},
        "valid_from": "2026-08-27T10:00:00+08:00",
        "valid_to": None,
        "superseded_by": None,
        "truth_status": "CURRENT",
        "provenance": [{"kind": "memo", "source_id": "test"}],
    }
    return {
        "schema_version": 1,
        "source": "derived-rebuildable-truth-view",
        "generated_at": "2026-08-27T10:30:00+08:00",
        "policy": {"rebuildable": True, "conflicts_fail_closed": True},
        "current_facts": [record],
        "active_goals": [],
        "active_decisions": [],
        "active_constraints": [],
        "open_todos": [],
        "history": [],
        "conflicts": [],
        "review_required": [],
    }


def _client(monkeypatch, root: Path) -> TestClient:
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(root))
    monkeypatch.delenv("EVEROS_API_TOKEN", raising=False)
    monkeypatch.delenv("EVEROS_API_TOKEN_FILE", raising=False)
    load_settings.cache_clear()
    return TestClient(create_app(lifespan_providers=[]))


def test_wiki_index_reports_unavailable_before_materialization(
    tmp_path: Path, monkeypatch
) -> None:
    with _client(monkeypatch, tmp_path / "memory") as client:
        response = client.get("/api/v1/memory/wiki")
    assert response.status_code == 200
    assert response.json()["data"] == {
        "available": False,
        "snapshot_id": None,
        "compiler_version": None,
        "materialized_at": None,
        "pages": [],
    }


def test_wiki_index_and_page_return_same_compiled_view(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "memory"
    materialize_truth_view(_payload(), memory_root=root)
    with _client(monkeypatch, root) as client:
        index = client.get("/api/v1/memory/wiki")
        assert index.status_code == 200
        data = index.json()["data"]
        assert data["available"] is True
        assert len(data["pages"]) == 1
        slug = data["pages"][0]["slug"]
        page = client.get(f"/api/v1/memory/wiki/{slug}")
        missing = client.get("/api/v1/memory/wiki/not-a-page")

    assert page.status_code == 200
    page_data = page.json()["data"]
    assert "Wiki API is read only" in page_data["markdown"]
    assert (
        page_data["agent_view"]["claim_set_sha256"]
        == data["pages"][0]["claim_set_sha256"]
    )
    assert missing.status_code == 404
