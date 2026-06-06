"""Tests for LlmCustomSetting DTO and LlmCustomSettingModel persistence layer.

Both layers must expose the `profile` scene so that user/group profile
extraction can be routed to a different provider/model from boundary or
extraction. Regression: prior to the fix, EXTRACT_SCENES was
("boundary", "extraction", "profile") but the DTO/Model only carried
boundary + extraction, so profile scenes were silently dropped via
Pydantic field-stripping.
"""

import pytest
from pydantic import ValidationError

from api_specs.dtos.settings import (
    LlmCustomSetting,
    LlmProviderConfig,
    UpdateSettingsRequest,
)
from infra_layer.adapters.out.persistence.document.memory.global_settings import (
    LlmCustomSettingModel,
    LlmProviderConfigModel,
)


class TestLlmCustomSettingDto:
    """DTO-side: api_specs/dtos/settings.py"""

    def test_default_all_scenes_none(self) -> None:
        cs = LlmCustomSetting()
        assert cs.boundary is None
        assert cs.extraction is None
        assert cs.profile is None
        assert cs.extra is None

    def test_profile_field_present_in_dict(self) -> None:
        cs = LlmCustomSetting(
            profile=LlmProviderConfig(provider="openai", model="gpt-4o")
        )
        d = cs.model_dump()
        assert "profile" in d
        assert d["profile"]["provider"] == "openai"
        assert d["profile"]["model"] == "gpt-4o"

    def test_all_three_scenes_round_trip(self) -> None:
        cs = LlmCustomSetting(
            boundary=LlmProviderConfig(provider="openai", model="gpt-4.1-mini"),
            extraction=LlmProviderConfig(
                provider="openrouter", model="qwen/qwen3-235b-a22b-2507"
            ),
            profile=LlmProviderConfig(provider="openai", model="gpt-4o"),
        )
        d = cs.model_dump()
        assert d["boundary"]["model"] == "gpt-4.1-mini"
        assert d["extraction"]["model"] == "qwen/qwen3-235b-a22b-2507"
        assert d["profile"]["model"] == "gpt-4o"

    def test_update_settings_request_carries_profile(self) -> None:
        req = UpdateSettingsRequest(
            llm_custom_setting=LlmCustomSetting(
                profile=LlmProviderConfig(provider="openai", model="gpt-4o")
            )
        )
        assert req.llm_custom_setting.profile.model == "gpt-4o"


class TestLlmCustomSettingModel:
    """Persistence-side: infra_layer/adapters/out/persistence/document/memory/global_settings.py"""

    def test_from_dict_round_trip(self) -> None:
        data = {
            "boundary": {"provider": "openai", "model": "gpt-4.1-mini"},
            "extraction": {
                "provider": "openrouter",
                "model": "qwen/qwen3-235b-a22b-2507",
            },
            "profile": {"provider": "openai", "model": "gpt-4o"},
        }
        m = LlmCustomSettingModel.from_any(data)
        assert m is not None
        assert m.boundary.model == "gpt-4.1-mini"
        assert m.extraction.model == "qwen/qwen3-235b-a22b-2507"
        assert m.profile.model == "gpt-4o"

    def test_to_dict_includes_profile(self) -> None:
        m = LlmCustomSettingModel(
            profile=LlmProviderConfigModel(provider="openai", model="gpt-4o")
        )
        d = m.to_dict()
        assert d is not None
        assert "profile" in d
        assert d["profile"]["model"] == "gpt-4o"

    def test_from_dto_object(self) -> None:
        """from_any must also accept the DTO Pydantic object, not just dicts."""
        dto = LlmCustomSetting(
            profile=LlmProviderConfig(provider="openai", model="gpt-4o")
        )
        m = LlmCustomSettingModel.from_any(dto)
        assert m is not None
        assert m.profile is not None
        assert m.profile.model == "gpt-4o"
        assert m.profile.provider == "openai"

    def test_from_dict_legacy_2_fields_still_works(self) -> None:
        """Existing 2-field docs (no profile) must still parse cleanly."""
        data = {
            "boundary": {"provider": "openai", "model": "gpt-4.1-mini"},
            "extraction": {"provider": "openai", "model": "gpt-4o"},
        }
        m = LlmCustomSettingModel.from_any(data)
        assert m is not None
        assert m.profile is None
        assert m.boundary.model == "gpt-4.1-mini"
        assert m.extraction.model == "gpt-4o"

    def test_from_empty_dict_returns_none(self) -> None:
        assert LlmCustomSettingModel.from_any({}) is None
        assert LlmCustomSettingModel.from_any(None) is None
