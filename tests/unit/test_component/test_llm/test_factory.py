"""``build_llm_provider`` — settings validation + provider build."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from everos.component.llm import build_llm_provider
from everos.component.llm.agy_cli_provider import AgyCLIProvider
from everos.component.llm.openai_provider import OpenAIProvider
from everos.config.settings import LLMSettings


def test_raises_when_api_key_missing() -> None:
    s = LLMSettings(model="m", api_key=None, base_url="https://x")
    with pytest.raises(ValueError, match="EVEROS_LLM__API_KEY"):
        build_llm_provider(s)


def test_raises_when_base_url_missing() -> None:
    s = LLMSettings(model="m", api_key=SecretStr("k"), base_url=None)
    with pytest.raises(ValueError, match="EVEROS_LLM__BASE_URL"):
        build_llm_provider(s)


def test_builds_openai_provider() -> None:
    s = LLMSettings(model="m", api_key=SecretStr("k"), base_url="https://x")
    p = build_llm_provider(s)
    assert isinstance(p, OpenAIProvider)


def test_builds_agy_cli_without_http_credentials(tmp_path) -> None:
    s = LLMSettings(
        provider="agy_cli",
        api_key=None,
        base_url=None,
        agy_workdir=tmp_path,
    )
    p = build_llm_provider(s)
    assert isinstance(p, AgyCLIProvider)
