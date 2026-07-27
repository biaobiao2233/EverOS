"""Focused Linux route/auth tests for the private read-only admin API."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import fcntl  # noqa: F401
except ImportError:
    pytest.skip(
        "EverOS production persistence requires POSIX fcntl",
        allow_module_level=True,
    )

from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app


@pytest.fixture
def memory_root(tmp_path: Path) -> Path:
    root = tmp_path / "memory-root"
    root.mkdir()
    return root


async def test_admin_routes_fail_closed_without_a_configured_token(
    memory_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVEROS_API_TOKEN", raising=False)
    monkeypatch.delenv("EVEROS_API_TOKEN_FILE", raising=False)
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(memory_root))
    load_settings.cache_clear()
    app = create_app(lifespan_providers=[])

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        health = await client.get("/health")
        admin = await client.get("/api/v1/admin/memory-files")
        preflight = await client.options(
            "/api/v1/admin/memory-files",
            headers={
                "Origin": "http://localhost",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert health.status_code == 200
    assert admin.status_code == 503
    assert admin.json() == {"detail": "Admin API requires EVEROS_API_TOKEN"}
    assert preflight.status_code != 503


async def test_configured_token_requires_the_exact_bearer(
    memory_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_API_TOKEN", "test-admin-token")
    monkeypatch.delenv("EVEROS_API_TOKEN_FILE", raising=False)
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(memory_root))
    load_settings.cache_clear()
    app = create_app(lifespan_providers=[])

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        missing = await client.get("/api/v1/admin/pipeline/status")
        wrong = await client.get(
            "/api/v1/admin/pipeline/status",
            headers={"Authorization": "Bearer wrong"},
        )
        accepted = await client.get(
            "/api/v1/admin/pipeline/status",
            headers={"Authorization": "Bearer test-admin-token"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert accepted.status_code == 200


async def test_list_content_filters_and_response_envelopes(
    memory_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_API_TOKEN", "test-admin-token")
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(memory_root))
    load_settings.cache_clear()
    episodes = memory_root / "episodes"
    episodes.mkdir()
    (episodes / "alpha.md").write_text("# alpha", encoding="utf-8")
    (episodes / "beta.md").write_text("# beta", encoding="utf-8")
    (memory_root / "user.md").write_text("# profile", encoding="utf-8")
    percent = memory_root / "file%2e.md"
    percent.write_text("# percent", encoding="utf-8")
    app = create_app(lifespan_providers=[])
    headers = {"Authorization": "Bearer test-admin-token"}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        listing = await client.get(
            "/api/v1/admin/memory-files",
            params={"kind": "episode", "query": "alpha", "page_size": 1},
            headers=headers,
        )
        content = await client.get(
            "/api/v1/admin/memory-files/content",
            params={"path": "episodes/alpha.md"},
            headers=headers,
        )
        percent_content = await client.get(
            "/api/v1/admin/memory-files/content",
            params={"path": "file%2e.md"},
            headers=headers,
        )

    assert listing.status_code == 200
    assert set(listing.json()) == {"data"}
    assert listing.json()["data"]["total_count"] == 1
    assert listing.json()["data"]["items"][0]["path"] == "episodes/alpha.md"
    assert content.status_code == 200
    assert set(content.json()["data"]) == {
        "path",
        "kind",
        "size_bytes",
        "modified_at",
        "content",
    }
    assert content.json()["data"]["content"] == "# alpha"
    assert percent_content.status_code == 200
    assert percent_content.json()["data"]["path"] == "file%2e.md"


async def test_encoded_traversal_is_rejected_without_recursive_decoding(
    memory_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_API_TOKEN", "test-admin-token")
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(memory_root))
    load_settings.cache_clear()
    app = create_app(lifespan_providers=[])
    headers = {"Authorization": "Bearer test-admin-token"}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        single_decoded = await client.get(
            "/api/v1/admin/memory-files/content?path=%2e%2e/user.md",
            headers=headers,
        )
        double_encoded = await client.get(
            "/api/v1/admin/memory-files/content?path=%252e%252e/user.md",
            headers=headers,
        )

    assert single_decoded.status_code == 400
    assert double_encoded.status_code == 404


async def test_openapi_registers_admin_and_preserves_existing_routes(
    memory_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENV", "DEV")
    monkeypatch.setenv("EVEROS_API_TOKEN", "test-admin-token")
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(memory_root))
    load_settings.cache_clear()
    app = create_app(lifespan_providers=[])

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/openapi.json",
            headers={"Authorization": "Bearer test-admin-token"},
        )

    assert response.status_code == 200
    paths = response.json()["paths"]
    for expected in (
        "/health",
        "/metrics",
        "/api/v1/memory/add",
        "/api/v1/memory/search",
        "/api/v1/memory/get",
        "/api/v1/admin/memory-files",
        "/api/v1/admin/memory-files/content",
        "/api/v1/admin/pipeline/status",
    ):
        assert expected in paths
