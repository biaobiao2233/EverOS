"""ApiKeyRotator unit tests."""

import pytest

from memory_layer.llm.api_key_rotator import ApiKeyRotator


@pytest.fixture(autouse=True)
def _reset_shared_rotator():
    """Ensure each test starts with a clean shared-instance cache."""
    ApiKeyRotator.clear_shared()
    yield
    ApiKeyRotator.clear_shared()


class TestApiKeyRotator:
    """Unit tests for ApiKeyRotator core rotation logic."""

    def test_single_key_always_returns_same(self) -> None:
        rotator = ApiKeyRotator(["key-a"])
        results = [rotator.get_next() for _ in range(5)]
        assert results == ["key-a"] * 5

    def test_multiple_keys_round_robin(self) -> None:
        rotator = ApiKeyRotator(["key-a", "key-b", "key-c"])
        results = [rotator.get_next() for _ in range(3)]
        assert results == ["key-a", "key-b", "key-c"]

    def test_multiple_keys_wraps_around(self) -> None:
        rotator = ApiKeyRotator(["key-a", "key-b", "key-c"])
        results = [rotator.get_next() for _ in range(6)]
        assert results == ["key-a", "key-b", "key-c", "key-a", "key-b", "key-c"]

    def test_size_property(self) -> None:
        assert ApiKeyRotator(["key-a"]).size == 1
        assert ApiKeyRotator(["key-a", "key-b", "key-c"]).size == 3

    def test_empty_keys_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="At least one API key is required"):
            ApiKeyRotator([])

    def test_repr(self) -> None:
        rotator = ApiKeyRotator(["key-a", "key-b"])
        assert repr(rotator) == "ApiKeyRotator(size=2)"

    def test_keys_are_immutable(self) -> None:
        original = ["key-a", "key-b"]
        rotator = ApiKeyRotator(original)
        original.append("key-c")
        assert rotator.size == 2


class TestApiKeyRotatorGetOrCreate:
    """Tests for get_or_create: parsing + singleton behavior."""

    def test_single_key(self) -> None:
        rotator = ApiKeyRotator.get_or_create("key-a")
        assert rotator.size == 1
        assert rotator.get_next() == "key-a"

    def test_multiple_keys_comma_separated(self) -> None:
        rotator = ApiKeyRotator.get_or_create("key-a,key-b,key-c")
        assert rotator.size == 3
        assert rotator.get_next() == "key-a"
        assert rotator.get_next() == "key-b"
        assert rotator.get_next() == "key-c"

    def test_strips_whitespace(self) -> None:
        rotator = ApiKeyRotator.get_or_create(" key-a , key-b , key-c ")
        assert rotator.size == 3
        assert rotator.get_next() == "key-a"

    def test_ignores_trailing_comma(self) -> None:
        rotator = ApiKeyRotator.get_or_create("key-a,key-b,")
        assert rotator.size == 2

    def test_ignores_empty_segments(self) -> None:
        rotator = ApiKeyRotator.get_or_create("key-a,,key-b")
        assert rotator.size == 2

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="At least one API key is required"):
            ApiKeyRotator.get_or_create("")

    def test_only_commas_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="At least one API key is required"):
            ApiKeyRotator.get_or_create(",,,")

    def test_returns_same_instance(self) -> None:
        r1 = ApiKeyRotator.get_or_create("key-a,key-b")
        r2 = ApiKeyRotator.get_or_create("key-a,key-b")
        assert r1 is r2

    def test_shared_counter_across_calls(self) -> None:
        r1 = ApiKeyRotator.get_or_create("key-a,key-b")
        assert r1.get_next() == "key-a"
        r2 = ApiKeyRotator.get_or_create("key-a,key-b")
        assert r2.get_next() == "key-b"

    def test_new_instance_after_clearing_shared(self) -> None:
        r1 = ApiKeyRotator.get_or_create("key-a,key-b")
        ApiKeyRotator.clear_shared()
        r2 = ApiKeyRotator.get_or_create("key-a,key-b")
        assert r1 is not r2

    def test_distinct_key_sets_get_distinct_instances(self) -> None:
        """Per-key-set cache: two different key sets must not share a rotator.

        Regression: previously the singleton was a single instance pinned
        to the first key set, so e.g. default Qwen and custom Agnes
        providers would both end up using whichever key set was registered
        first, silently breaking the second provider (HTTP 401).
        """
        r_qwen = ApiKeyRotator.get_or_create("qwen-key")
        r_agnes = ApiKeyRotator.get_or_create("agnes-key")
        r_gemini = ApiKeyRotator.get_or_create("gemini-key-a,gemini-key-b")
        assert r_qwen is not r_agnes
        assert r_qwen is not r_gemini
        assert r_agnes is not r_gemini
        assert r_qwen.size == 1
        assert r_agnes.size == 1
        assert r_gemini.size == 2
        # Cycle state is independent per rotator
        assert r_qwen.get_next() == "qwen-key"
        assert r_agnes.get_next() == "agnes-key"

    def test_same_key_set_returns_same_instance(self) -> None:
        r1 = ApiKeyRotator.get_or_create("key-a,key-b")
        r2 = ApiKeyRotator.get_or_create(" key-a , key-b ")  # whitespace normalized
        r3 = ApiKeyRotator.get_or_create("key-a,key-b,")  # trailing comma ignored
        assert r1 is r2 is r3

    def test_clear_shared_empties_all_instances(self) -> None:
        ApiKeyRotator.get_or_create("key-a")
        ApiKeyRotator.get_or_create("key-b")
        assert len(ApiKeyRotator._shared) == 2
        ApiKeyRotator.clear_shared()
        assert ApiKeyRotator._shared == {}
