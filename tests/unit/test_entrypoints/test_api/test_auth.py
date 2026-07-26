"""Optional bearer authentication for the HTTP API."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from everos.entrypoints.api.app import create_app


async def test_configured_token_is_required_even_for_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_API_TOKEN", "test-token")
    app = create_app(lifespan_providers=[])
    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.get("/metrics")
        wrong = await client.get(
            "/metrics", headers={"Authorization": "Bearer wrong-token"}
        )
        accepted = await client.get(
            "/metrics", headers={"Authorization": "Bearer test-token"}
        )
        health = await client.get("/health")

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert accepted.status_code == 200
    assert health.status_code == 200


async def test_token_file_is_supported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "everos.token"
    token_file.write_text("file-token\n", encoding="utf-8")
    monkeypatch.delenv("EVEROS_API_TOKEN", raising=False)
    monkeypatch.setenv("EVEROS_API_TOKEN_FILE", str(token_file))
    app = create_app(lifespan_providers=[])
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/metrics", headers={"Authorization": "Bearer file-token"}
        )

    assert response.status_code == 200


def test_explicit_missing_token_file_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("EVEROS_API_TOKEN", raising=False)
    monkeypatch.setenv("EVEROS_API_TOKEN_FILE", str(tmp_path / "missing"))

    with pytest.raises(RuntimeError, match="cannot read EVEROS_API_TOKEN_FILE"):
        create_app(lifespan_providers=[])
