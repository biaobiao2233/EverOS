"""Process-wide LLM client accessor.

Lazy singleton — first call reads settings and builds the algo LLM
client; subsequent calls return the cached instance. Raises
:class:`LLMNotConfiguredError` when no credentials are present so
misconfiguration surfaces at app startup (via the LLM lifespan
provider) instead of silently failing per-request downstream.
"""

from __future__ import annotations

from everalgo.llm import build_client
from everalgo.llm.config import LLMConfig
from everalgo.llm.protocols import LLMClient

from everos.config import load_settings
from everos.core.observability.logging import get_logger

from .factory import build_llm_provider

logger = get_logger(__name__)


class LLMNotConfiguredError(RuntimeError):
    """Raised when ``settings.llm`` is missing ``api_key`` or ``base_url``."""


_llm_client: LLMClient | None = None
_multimodal_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the singleton algo LLM client.

    Raises:
        LLMNotConfiguredError: When ``settings.llm.api_key`` or
            ``settings.llm.base_url`` is unset.
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    llm_cfg = load_settings().llm
    try:
        _llm_client = build_llm_provider(llm_cfg)
    except ValueError as exc:
        raise LLMNotConfiguredError(str(exc)) from exc
    logger.info(
        "llm_client_built",
        provider=llm_cfg.provider,
        model=(
            llm_cfg.agy_model or "antigravity-default"
            if llm_cfg.provider == "agy_cli"
            else llm_cfg.model
        ),
    )
    return _llm_client


def get_multimodal_llm_client() -> LLMClient:
    """Return the singleton multimodal LLM client (for everalgo.parser).

    Reads the flat ``[multimodal]`` config — kept separate from the main
    ``[llm]`` so parsing can target a vision/audio-capable endpoint.

    Raises:
        LLMNotConfiguredError: When ``settings.multimodal.api_key`` or
            ``settings.multimodal.base_url`` is unset.
    """
    global _multimodal_client
    if _multimodal_client is not None:
        return _multimodal_client

    cfg = load_settings().multimodal
    api_key = cfg.api_key.get_secret_value() if cfg.api_key is not None else None
    if not api_key or not cfg.base_url:
        raise LLMNotConfiguredError(
            "Multimodal LLM is required for parsing; set "
            "EVEROS_MULTIMODAL__API_KEY + EVEROS_MULTIMODAL__BASE_URL"
        )
    _multimodal_client = build_client(
        LLMConfig(
            model=cfg.model,
            api_key=api_key,
            base_url=cfg.base_url,
        )
    )
    logger.info("multimodal_llm_client_built", model=cfg.model)
    return _multimodal_client
