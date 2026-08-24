"""Factory for building an LLM provider from :class:`LLMSettings`."""

from __future__ import annotations

from everos.config import LLMSettings

from .agy_cli_provider import AgyCLIProvider
from .openai_provider import OpenAIProvider
from .protocol import LLMClient


def build_llm_provider(settings: LLMSettings) -> LLMClient:
    """Build the configured text LLM provider.

    Unwraps :class:`pydantic.SecretStr` here so downstream callers never
    touch the raw key directly. Fails fast if either ``api_key`` or
    ``base_url`` is missing — caller is expected to set them via
    ``.env`` / user toml / programmatic init before calling.

    Args:
        settings: The :class:`LLMSettings` slice from
            :func:`everos.config.load_settings`.

    Returns:
        A provider that structurally satisfies
        :class:`everalgo.llm.LLMClient` and can be passed to everalgo
        operators via ``llm=``.

    Raises:
        ValueError: If ``api_key`` or ``base_url`` is unset.
    """
    if settings.provider == "agy_cli":
        return AgyCLIProvider(
            executable=settings.agy_executable,
            workdir=settings.agy_workdir,
            homes=settings.agy_homes,
            agent=settings.agy_agent,
            model=settings.agy_model,
            timeout_seconds=settings.agy_timeout_seconds,
            max_concurrency=settings.agy_max_concurrency,
        )

    if settings.provider == "openai":
        if settings.api_key is None:
            raise ValueError(
                "LLM api_key is not configured "
                "(set EVEROS_LLM__API_KEY or [llm] api_key in user toml)"
            )
        if not settings.base_url:
            raise ValueError(
                "LLM base_url is not configured "
                "(set EVEROS_LLM__BASE_URL or [llm] base_url in user toml)"
            )
        return OpenAIProvider(
            model=settings.model,
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
        )

    raise ValueError(f"unsupported LLM provider: {settings.provider}")
