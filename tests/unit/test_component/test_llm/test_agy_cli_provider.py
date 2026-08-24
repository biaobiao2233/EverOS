"""Security and contract tests for the official Antigravity CLI provider."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from everos.component.llm.agy_cli_provider import (
    AgyCLIProvider,
    _validate_structured_output,
)
from everos.component.llm.protocol import ChatMessage, LLMError


class Extraction(BaseModel):
    title: str
    score: int


def test_command_contains_no_prompt_and_enforces_read_only_mode(tmp_path) -> None:
    provider = AgyCLIProvider(
        executable="/opt/agy",
        workdir=tmp_path,
        model="flash",
        timeout_seconds=42,
    )
    command = provider._command(None)
    assert command == [
        "/opt/agy",
        "--print-timeout",
        "42s",
        "--mode",
        "plan",
        "--sandbox",
        "--agent",
        "everos-text",
        "--model",
        "flash",
    ]
    assert "private user text" not in " ".join(command)


def test_prompt_serializes_roles_and_text_as_untrusted_json(tmp_path) -> None:
    provider = AgyCLIProvider(workdir=tmp_path)
    prompt = provider._build_prompt(
        [
            ChatMessage(role="system", content="system text"),
            ChatMessage(role="user", content="private user text"),
        ],
        Extraction,
    )
    assert '"role":"system","content":"system text"' in prompt
    assert '"role":"user","content":"private user text"' in prompt
    assert '"title"' in prompt
    assert "Do not use tools" in prompt


def test_structured_output_accepts_plain_object_or_one_json_fence() -> None:
    plain, parsed_plain = _validate_structured_output(
        '{"title":"alpha","score":9}', Extraction
    )
    fenced, parsed_fenced = _validate_structured_output(
        '```json\n{"title":"beta","score":8}\n```', Extraction
    )
    assert plain == '{"title":"alpha","score":9}'
    assert parsed_plain.title == "alpha"
    assert fenced == '{"title":"beta","score":8}'
    assert parsed_fenced.score == 8


@pytest.mark.parametrize(
    "output",
    [
        'prefix {"title":"alpha","score":9}',
        '{"title":"alpha","score":9} suffix',
        '[{"title":"alpha","score":9}]',
        '```\n{"title":"alpha","score":9}\n```\nextra',
        '```yaml\n{"title":"alpha","score":9}\n```',
    ],
)
def test_structured_output_rejects_non_exact_object(output: str) -> None:
    with pytest.raises(LLMError):
        _validate_structured_output(output, Extraction)


async def test_chat_returns_validated_model_without_stderr_or_raw(tmp_path) -> None:
    provider = AgyCLIProvider(workdir=tmp_path)
    provider._run = AsyncMock(  # type: ignore[method-assign]
        return_value=(0, b'{"title":"alpha","score":9}')
    )
    response = await provider.chat(
        [ChatMessage(role="user", content="extract this")],
        response_format=Extraction,
    )
    assert response.content == '{"title":"alpha","score":9}'
    assert isinstance(response.parsed, Extraction)
    assert response.raw is None


async def test_chat_fails_closed_on_cli_error(tmp_path) -> None:
    provider = AgyCLIProvider(workdir=tmp_path)
    provider._run = AsyncMock(return_value=(7, b"secret stderr is never returned"))  # type: ignore[method-assign]
    with pytest.raises(LLMError, match="status 7") as exc_info:
        await provider.chat([ChatMessage(role="user", content="private prompt")])
    message = str(exc_info.value)
    assert "private prompt" not in message
    assert "secret stderr" not in message


async def test_chat_round_robins_configured_account_homes(tmp_path) -> None:
    home_a = tmp_path / "account-a"
    home_b = tmp_path / "account-b"
    provider = AgyCLIProvider(
        workdir=tmp_path / "work",
        homes=[home_a, home_b],
    )
    provider._run = AsyncMock(return_value=(0, b"ok"))  # type: ignore[method-assign]

    await provider.chat([ChatMessage(role="user", content="first")])
    await provider.chat([ChatMessage(role="user", content="second")])

    assert provider._run.await_args_list[0].kwargs["home"] == home_a
    assert provider._run.await_args_list[1].kwargs["home"] == home_b


async def test_chat_fails_over_to_next_home_on_cli_nonzero(tmp_path) -> None:
    home_a = tmp_path / "account-a"
    home_b = tmp_path / "account-b"
    provider = AgyCLIProvider(
        workdir=tmp_path / "work",
        homes=[home_a, home_b],
    )
    provider._run = AsyncMock(  # type: ignore[method-assign]
        side_effect=[(1, b""), (0, b"ok")]
    )

    response = await provider.chat([ChatMessage(role="user", content="retry")])

    assert response.content == "ok"
    assert provider._run.await_count == 2
    assert provider._run.await_args_list[0].kwargs["home"] == home_a
    assert provider._run.await_args_list[1].kwargs["home"] == home_b
