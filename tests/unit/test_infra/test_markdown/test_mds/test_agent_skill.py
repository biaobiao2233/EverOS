"""Tests for :class:`AgentSkillFrontmatter` — the AgentSkill schema.

Lives under ``test_infra`` because :class:`AgentSkillFrontmatter` itself
lives under ``infra/.../mds`` (it carries business fields + the
directory-shape ClassVars). The schema-agnostic chassis tests live
under ``test_core/test_persistence/test_markdown/``.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError

from everos.infra.persistence.markdown import AgentSkillFrontmatter


def _kwargs(**overrides: object) -> dict[str, object]:
    """Minimal valid kwargs for AgentSkillFrontmatter."""
    base: dict[str, object] = {
        "id": "skill_contract_risk_scan",
        "agent_id": "agent_zhang_legal",
        "name": "contract_risk_scan",
        "description": "Scan a contract draft for risk clauses.",
        "confidence": 0.5,
        "maturity_score": 0.5,
    }
    base.update(overrides)
    return base


def test_skill_inherits_agent_scope() -> None:
    """Skills always live under ``agents/`` — track + SCOPE_DIR confirm."""
    assert AgentSkillFrontmatter.SCOPE_DIR == "agents"
    fm = AgentSkillFrontmatter(**_kwargs())  # type: ignore[arg-type]
    assert fm.track == "agent"
    assert fm.type == "agent_skill"


def test_skill_requires_name_and_description() -> None:
    """Tier-1 prompt injection demands both fields — schema enforces."""
    bad = _kwargs()
    del bad["name"]
    with pytest.raises(ValidationError):
        AgentSkillFrontmatter(**bad)  # type: ignore[arg-type]

    bad = _kwargs()
    del bad["description"]
    with pytest.raises(ValidationError):
        AgentSkillFrontmatter(**bad)  # type: ignore[arg-type]


def test_skill_requires_confidence_and_maturity_score() -> None:
    """LLM-emitted score fields are required (no default)."""
    bad = _kwargs()
    del bad["confidence"]
    with pytest.raises(ValidationError):
        AgentSkillFrontmatter(**bad)  # type: ignore[arg-type]

    bad = _kwargs()
    del bad["maturity_score"]
    with pytest.raises(ValidationError):
        AgentSkillFrontmatter(**bad)  # type: ignore[arg-type]


def test_skill_optional_fields_default() -> None:
    """``source_case_ids`` defaults to empty list; ``cluster_id`` to None."""
    fm = AgentSkillFrontmatter(**_kwargs())  # type: ignore[arg-type]
    assert fm.source_case_ids == []
    assert fm.cluster_id is None


def test_skill_lineage_fields_round_trip() -> None:
    """``source_case_ids`` + ``cluster_id`` round-trip through model_dump."""
    fm = AgentSkillFrontmatter(
        **_kwargs(
            source_case_ids=["case_a", "case_b"],
            cluster_id="cl_x",
        ),  # type: ignore[arg-type]
    )
    dumped = fm.model_dump()
    assert dumped["source_case_ids"] == ["case_a", "case_b"]
    assert dumped["cluster_id"] == "cl_x"


def test_skill_extra_fields_still_allowed() -> None:
    """L2 system metadata (md_sha256 / last_indexed_at) rides along."""
    fm = AgentSkillFrontmatter(
        **_kwargs(
            md_sha256="deadbeef",
            last_indexed_at="2026-05-07T08:00:00Z",
        ),  # type: ignore[arg-type]
    )
    dumped = fm.model_dump()
    assert dumped["md_sha256"] == "deadbeef"
    assert dumped["last_indexed_at"] == "2026-05-07T08:00:00Z"


def test_skill_directory_shape_classvars() -> None:
    """Path-shape ClassVars pin the wiki layout for the writer/reader pair."""
    assert AgentSkillFrontmatter.SKILLS_CONTAINER_NAME == "skills"
    assert AgentSkillFrontmatter.SKILL_DIR_PREFIX == "skill_"
    assert AgentSkillFrontmatter.SKILL_MAIN_FILENAME == "SKILL.md"
    assert AgentSkillFrontmatter.SKILL_REFERENCES_DIR_NAME == "references"
    assert AgentSkillFrontmatter.SKILL_SCRIPTS_DIR_NAME == "scripts"


# ── path safety: traversal validator ─────────────────────────────────────


@pytest.mark.parametrize(
    "bad_name",
    [
        "../../../etc/passwd",
        "skills/../../escape",
        "a/b",
        "a\\b",
        "..",
    ],
)
def test_skill_name_rejects_path_traversal(bad_name: str) -> None:
    """Defence in depth: a hand-edited ``SKILL.md`` with a traversal-shaped
    ``name`` is caught on parse rather than silently relocating the skill
    on the next write (the write path pre-sanitizes via
    :meth:`AgentSkillFrontmatter.sanitize_skill_name`, so this validator
    only fires for names that bypassed it).
    """
    with pytest.raises(ValidationError, match="path separators"):
        AgentSkillFrontmatter(**_kwargs(name=bad_name))  # type: ignore[arg-type]


def test_skill_name_allows_cjk_and_spaces() -> None:
    """Non-ASCII / whitespace names are legitimate — only traversal shapes
    are rejected."""
    fm = AgentSkillFrontmatter(**_kwargs(name="修复 Django 自动重载问题"))  # type: ignore[arg-type]
    assert fm.name == "修复 Django 自动重载问题"


# ── path safety: sanitizer semantics ─────────────────────────────────────


@pytest.mark.parametrize("raw", ["..", "../", "/../", ".", "./"])
def test_degenerate_fixpoints_fall_back_instead_of_escaping(raw: str) -> None:
    """A short input that strips down to exactly ``".."`` or ``"."`` must
    fall back, not be returned as-is.

    ``"."`` is a *safe* character (kept, not stripped), so ``"../"`` and
    ``"."``+``"/"`` both collapse to a degenerate fixpoint once the
    separator is removed. Without the fallback, the sanitized result would
    resolve one directory level up instead of into a new child.
    """
    assert AgentSkillFrontmatter.sanitize_dirname(raw, "unnamed") == "unnamed"


def test_traversal_payload_has_no_separator_and_stays_under_root(
    tmp_path: Path,
) -> None:
    payload = "../" * 8 + "tmp/pwned"

    sanitized = AgentSkillFrontmatter.sanitize_dirname(payload, "unnamed")

    assert "/" not in sanitized
    assert "\\" not in sanitized
    assert sanitized != ".."
    # The skill_ prefix absorbs nothing here — the segment alone is safe.
    resolved = (tmp_path / sanitized).resolve()
    assert resolved.is_relative_to(tmp_path.resolve())


def test_nfc_normalizes_decomposed_accents() -> None:
    """An NFD-decomposed accented character (base letter + combining mark)
    must sanitize to the same result as its NFC (precomposed) form —
    without normalization, the combining mark is not ``\\w`` and gets
    silently stripped, losing the accent instead of preserving it.
    """
    nfc = "café"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd  # sanity: the two forms really are distinct strings

    sanitized_nfc = AgentSkillFrontmatter.sanitize_dirname(nfc, "unnamed")
    sanitized_nfd = AgentSkillFrontmatter.sanitize_dirname(nfd, "unnamed")

    assert sanitized_nfc == sanitized_nfd == "café"


def test_nfc_does_not_help_composition_exclusions() -> None:
    """Pins the documented exception: for Unicode "composition exclusion"
    codepoints, NFC normalization does not help — it decomposes an
    already-precomposed character, and the resulting combining mark is
    stripped either way. Best-effort for the common case, not a guarantee
    for every script.
    """
    precomposed = "क़ख़"
    assert unicodedata.normalize("NFC", precomposed) != precomposed

    sanitized = AgentSkillFrontmatter.sanitize_dirname(precomposed, "unnamed")

    # The nukta (combining mark, U+093C) is lost either way.
    assert sanitized == "कख"


def test_cjk_and_space_input_preserved_readably() -> None:
    raw = "修复 Django 自动重载问题"

    sanitized = AgentSkillFrontmatter.sanitize_dirname(raw, "unnamed")

    assert "修复" in sanitized
    assert "Django" in sanitized
    assert "_" in sanitized  # spaces became underscores, not stripped
    assert " " not in sanitized


@pytest.mark.parametrize(
    "raw",
    [
        "../" * 8 + "tmp/pwned",
        "修复 Django 自动重载问题",
        "normal_skill",
        "../../etc/passwd",
        "   ",
        "!!!@@@###",
        "..",
        "../",
        "/../",
        ".",
        "./",
    ],
)
def test_sanitize_is_idempotent(raw: str) -> None:
    """Idempotency is what lets a reader deriving a name from an on-disk
    directory and a writer deriving it from raw input agree on one path."""
    once = AgentSkillFrontmatter.sanitize_dirname(raw, "unnamed")
    twice = AgentSkillFrontmatter.sanitize_dirname(once, "unnamed")
    assert once == twice


def test_empty_result_falls_back_and_long_input_truncates() -> None:
    assert AgentSkillFrontmatter.sanitize_dirname("!!!@@@###", "unnamed") == "unnamed"
    assert len(AgentSkillFrontmatter.sanitize_dirname("a" * 200, "unnamed")) == 50


# ── path safety: presanitized boundary names through the model ───────────


@pytest.mark.parametrize(
    "raw_name",
    [
        "..",
        "../",
        "/../",
        ".",
        "./",
        "!!!",  # sanitizes to empty -> fallback
        "a" * 200,  # truncation
        "修复 Django 自动重载问题",  # CJK + space
        "../" * 8 + "tmp/pwned",
    ],
)
def test_frontmatter_accepts_presanitized_boundary_names(raw_name: str) -> None:
    """Mirrors ``extract_agent_skill._persist_skill``'s write path: the
    caller sanitizes ``skill_name`` via
    :meth:`AgentSkillFrontmatter.sanitize_skill_name` *before* constructing
    the frontmatter, so a traversal-shaped or degenerate LLM name never
    reaches the validator as a raw, unsanitized string — construction
    succeeds rather than dead-lettering the extraction run for a name the
    sanitizer handles safely anyway.
    """
    sanitized = AgentSkillFrontmatter.sanitize_skill_name(raw_name)

    assert "/" not in sanitized
    assert "\\" not in sanitized
    assert sanitized not in ("", ".", "..")

    fm = AgentSkillFrontmatter(**_kwargs(name=sanitized))

    assert fm.name == sanitized


def test_skill_dir_name_prefixes_the_sanitized_name() -> None:
    """``skill_dir_name`` is the prefixed single sanitization point shared
    by the writer and reader path resolvers."""
    cjk_raw = "修复 Django 自动重载问题"
    traversal_raw = "../" * 8 + "tmp/pwned"
    assert AgentSkillFrontmatter.skill_dir_name(cjk_raw) == (
        f"skill_{AgentSkillFrontmatter.sanitize_skill_name(cjk_raw)}"
    )
    assert AgentSkillFrontmatter.skill_dir_name(traversal_raw) == (
        f"skill_{AgentSkillFrontmatter.sanitize_skill_name(traversal_raw)}"
    )
