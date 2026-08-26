"""Focused API contract tests for deferred memorize endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from everos.entrypoints.api.app import create_app
from everos.service.memorize import BackgroundFlushResult, PublishResult, StageResult


def _message() -> dict[str, object]:
    return {
        "sender_id": "u_test",
        "role": "user",
        "timestamp": 1_700_000_000_000,
        "content": "hello",
    }


def test_stage_route_returns_staged_receipt(monkeypatch) -> None:
    routes = __import__(
        "everos.entrypoints.api.routes.memorize",
        fromlist=["stage"],
    )
    operation_id = "evop1-stage-" + "a" * 64
    stage_mock = AsyncMock(
        return_value=StageResult(
            message_count=1,
            operation_id=operation_id,
        )
    )
    monkeypatch.setattr(routes, "stage", stage_mock)
    app = create_app(lifespan_providers=[])

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/memory/stage",
            json={
                "session_id": "s1",
                "messages": [_message()],
                "operation_id": operation_id,
            },
        )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "message_count": 1,
        "status": "staged",
        "operation_id": operation_id,
        "replayed": False,
        "inserted_count": 0,
        "updated_count": 0,
        "duplicate_count": 0,
        "stale_count": 0,
        "consumed_replay_count": 0,
    }
    stage_mock.assert_awaited_once()


def test_publish_route_returns_content_free_authority_receipt(monkeypatch) -> None:
    routes = __import__(
        "everos.entrypoints.api.routes.memorize",
        fromlist=["publish"],
    )
    operation_id = "evop1-publish-" + "c" * 64
    publish_mock = AsyncMock(
        return_value=PublishResult(
            message_count=1,
            operation_id=operation_id,
            published_count=1,
        )
    )
    monkeypatch.setattr(routes, "publish", publish_mock)
    app = create_app(lifespan_providers=[])

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/memory/publish",
            json={
                "session_id": "s1",
                "messages": [
                    {"source": "web", "external_ref": "gpt-msg-1", "revision": 2}
                ],
                "authority_ref": "ledger:commit-1",
                "operation_id": operation_id,
            },
        )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "message_count": 1,
        "status": "published",
        "operation_id": operation_id,
        "replayed": False,
        "published_count": 1,
        "already_published_count": 0,
        "consumed_count": 0,
    }
    publish_mock.assert_awaited_once()


def test_web_stage_requires_external_ref() -> None:
    app = create_app(lifespan_providers=[])
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/memory/stage",
            json={
                "session_id": "s1",
                "messages": [{**_message(), "source": "web"}],
                "operation_id": "evop1-stage-" + "d" * 64,
            },
        )
    assert response.status_code == 422


def test_background_flush_returns_processing_and_wakes_worker(monkeypatch) -> None:
    routes = __import__(
        "everos.entrypoints.api.routes.memorize",
        fromlist=["queue_background_flush"],
    )
    operation_id = "evop1-flush-" + "b" * 64
    queue_mock = AsyncMock(
        return_value=BackgroundFlushResult(operation_id=operation_id)
    )

    class _Scheduler:
        woke = False

        def wake(self) -> None:
            self.woke = True

    scheduler = _Scheduler()
    monkeypatch.setattr(routes, "queue_background_flush", queue_mock)
    monkeypatch.setattr(
        routes,
        "get_background_flush_scheduler",
        lambda: scheduler,
    )
    app = create_app(lifespan_providers=[])

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/memory/flush",
            json={
                "session_id": "s1",
                "operation_id": operation_id,
                "background": True,
            },
        )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "status": "processing",
        "operation_id": operation_id,
        "replayed": False,
    }
    assert scheduler.woke is True
    queue_mock.assert_awaited_once()


def test_background_flush_requires_operation_id() -> None:
    app = create_app(lifespan_providers=[])
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/memory/flush",
            json={"session_id": "s1", "background": True},
        )
    assert response.status_code == 422
