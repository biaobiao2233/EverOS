"""AgentSkill frontmatter — single SKILL.md inside a skill directory.

Path: ``agents/<scope_id>/skills/skill_<name>/SKILL.md`` (plus sibling
``references/*.md`` and ``scripts/*.<ext>`` files that are not part of
the frontmatter contract).

Skills are *named entities* rather than daily-log entries: the
LanceDB primary key is ``<owner_id>_<skill_name>`` (no date / seq).
Upserts replace the file wholesale; the cascade daemon recomputes the
``content`` index column by concatenating ``SKILL.md`` body with every
``references/*.md`` sibling.

Five directory-shape ClassVars pin the layout in one place so the
writer / reader pair reads off them — no duplicated string literals.

**Path safety.** ``name`` is LLM output (see
``memory.strategies.extract_agent_skill``), so it must never reach the
filesystem unsanitized (CWE-22). :meth:`sanitize_skill_name` /
:meth:`skill_dir_name` are the single sanitization point both
:class:`AgentSkillWriter` and :class:`AgentSkillReader` derive their
``skill_<name>`` directory segment from, and that the strategy uses to
sanitize a name *before* constructing this model — so on the production
write path ``frontmatter.name`` is byte-identical to the directory
suffix. The :meth:`_reject_path_traversal` validator is defence in depth
for names that bypass the writer (e.g. a hand-edited ``SKILL.md``).
"""

from __future__ import annotations

import datetime as _dt
import re
import unicodedata
from typing import ClassVar, Literal

from pydantic import field_validator

from everos.core.persistence.markdown import (
    AgentScopedFrontmatter,
    SkillPathMixin,
)

_MAX_DIRNAME_LEN = 50
_SAFE_CHARS = re.compile(r"[^\w\-.]", re.UNICODE)
_DEGENERATE = frozenset({"", ".", ".."})


def _sanitize_dirname(raw: str, fallback: str) -> str:
    """Produce a safe directory/file name segment from free-text input.

    * NFC-normalize first: for an ordinary decomposed (NFD) input — a base
      letter plus a combining mark — this collapses to the precomposed form
      before the character filter runs, so the accent survives (a combining
      mark alone is not ``\\w`` and would otherwise be silently stripped).
      Best-effort for the common case, not a guarantee for every script:
      Unicode *composition exclusion* codepoints decompose under NFC and
      lose their mark either way.
    * Replace spaces with underscores.
    * Strip characters outside ``[a-zA-Z0-9_\\-.]`` (``\\w`` is
      Unicode-aware, so CJK and other non-ASCII scripts survive readably).
      ``.`` is a safe character and is not stripped.
    * Truncate to 50 characters.
    * Fall back to *fallback* when the result is empty, ``"."``, or ``".."``
      — the only single components that resolve to no new child of the
      directory they are concatenated into.

    Every path separator (``/``, ``\\``) is stripped by the character-class
    filter, so the result is always exactly one path component. The
    function is lossy and not injective: distinct inputs can collapse onto
    the same output (dropped characters, space/underscore collapse,
    truncation); callers that need distinct outputs must disambiguate
    themselves. Idempotent: ``f(f(x)) == f(x)`` — which is what lets a
    reader deriving a name from an already-sanitized on-disk directory and
    a writer deriving it from raw input agree on the same path.
    """
    slug = unicodedata.normalize("NFC", raw)
    slug = slug.replace(" ", "_")
    slug = _SAFE_CHARS.sub("", slug)
    slug = slug[:_MAX_DIRNAME_LEN]
    return slug if slug not in _DEGENERATE else fallback


class AgentSkillFrontmatter(SkillPathMixin, AgentScopedFrontmatter):
    """Frontmatter for ``agents/<scope>/skills/skill_<name>/SKILL.md``."""

    SKILLS_CONTAINER_NAME: ClassVar[str] = "skills"
    SKILL_DIR_PREFIX: ClassVar[str] = "skill_"
    SKILL_MAIN_FILENAME: ClassVar[str] = "SKILL.md"
    SKILL_REFERENCES_DIR_NAME: ClassVar[str] = "references"
    SKILL_SCRIPTS_DIR_NAME: ClassVar[str] = "scripts"

    type: Literal["agent_skill"] = "agent_skill"

    name: str
    """Skill identifier — also the directory suffix
    (``skills/skill_<name>/``, sanitized via
    :meth:`skill_dir_name`). Keep snake_case so it stays readable and
    ID-stable; the directory segment is sanitized regardless."""

    @field_validator("name")
    @classmethod
    def _reject_path_traversal(cls, value: str) -> str:
        """Catch a frontmatter ``name`` that bypassed the writer's sanitizer.

        The normal write path
        (``memory.strategies.extract_agent_skill._persist_skill``) sanitizes
        LLM-emitted ``skill_name`` via :meth:`sanitize_skill_name` *before*
        constructing this model, so ``name`` is traversal-free by the time
        it gets here on that path — this validator should not normally fire
        for LLM output at all. It exists for the case that does bypass the
        writer: a hand-edited ``SKILL.md`` (or any other direct
        ``AgentSkillFrontmatter`` construction that skips pre-sanitization)
        whose ``name`` contains a path separator, or is exactly ``".."`` —
        raise loudly rather than silently relocating the skill on next
        write.

        The check is deliberately narrower than "contains ``..``": a
        sanitized name may legitimately contain a run of literal dots
        (:func:`_sanitize_dirname` keeps ``.`` as a safe character, so
        ``"../" * 8 + "tmp/pwned"`` sanitizes to
        ``"................tmppwned"``, which still contains the substring
        ``".."`` many times over). With no path separator left, that string
        is one opaque filename component, not a ``..`` traversal segment —
        rejecting on substring containment would make this validator reject
        the sanitizer's own safe output.
        """
        if "/" in value or "\\" in value or value == "..":
            raise ValueError(
                f"skill name {value!r} must not contain path separators, "
                "and must not be exactly '..'"
            )
        return value

    description: str
    """One-line summary surfaced at Tier-1 prompt injection. Short — the
    agent's startup-time scanner reads ``(name, description)`` for every
    skill, so the token budget is tight."""

    confidence: float
    """LLM-emitted confidence in the skill's correctness, 0.0–1.0."""

    maturity_score: float
    """LLM-emitted maturity score, 0.0–1.0. The retrieval-time threshold
    (``maturity_threshold``) lives in MemorizeConfig, not on this file."""

    source_case_ids: list[str] = []
    """AgentCase ids that fed into this skill's synthesis (lineage)."""

    cluster_id: str | None = None
    """Optional MemScene clustering tag; may be unset early on."""

    created_at: _dt.datetime | None = None
    updated_at: _dt.datetime | None = None

    # ── Path-safety API ───────────────────────────────────────────────────

    @staticmethod
    def sanitize_dirname(raw: str, fallback: str) -> str:
        """Sanitize a free-text filesystem segment (see :func:`_sanitize_dirname`).

        The general primitive behind :meth:`sanitize_skill_name`. Also used
        by :class:`AgentSkillWriter` / :class:`AgentSkillReader` for the
        segments appended *after* the skill directory — reference filenames
        and script filenames — which ``skill_dir_name`` does not cover.
        """
        return _sanitize_dirname(raw, fallback)

    @classmethod
    def sanitize_skill_name(cls, skill_name: str) -> str:
        """Bare sanitized skill name (no ``skill_`` prefix).

        The single sanitization point for a skill's ``name`` value itself —
        as opposed to :meth:`skill_dir_name`, which additionally prefixes it
        for the directory segment. Callers building
        ``AgentSkillFrontmatter.name`` from LLM output (see
        ``memory.strategies.extract_agent_skill._persist_skill``) route
        through this *before* constructing the frontmatter, so
        ``frontmatter.name`` ends up byte-identical to the directory-derived
        name rather than merely idempotent-if-resanitized.

        Lossy and not injective (see :func:`_sanitize_dirname`): distinct
        raw names can collapse onto the same sanitized name — dropped
        punctuation, space/underscore collapse, the 50-character cap, every
        combining mark regardless of script, and case on case-insensitive
        filesystems. Accepted deliberately: detecting-and-raising would let
        LLM output decide whether an extraction run dead-letters, and a
        disambiguating suffix needs a collision probe plus a case-folding
        rule that is deferred to a deliberate design pass.
        """
        return cls.sanitize_dirname(skill_name, "unnamed")

    @classmethod
    def skill_dir_name(cls, skill_name: str) -> str:
        """Sanitized ``skill_<name>`` directory segment (traversal-safe).

        Idempotent in ``skill_name``: calling this again on an already
        sanitized name (e.g. one recovered by walking the directory tree)
        returns the same segment, so a reader deriving ``skill_name`` from
        the on-disk directory and a writer deriving it from raw LLM output
        land on the same path.
        """
        return f"{cls.SKILL_DIR_PREFIX}{cls.sanitize_skill_name(skill_name)}"
