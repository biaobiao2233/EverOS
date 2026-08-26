"""Explicit Agent Case → Skill candidate lifecycle.

This module is deliberately procedural-memory-only. It can recommend or
accept a reusable procedure, but it cannot create a semantic Truth claim.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

import portalocker


class SkillCandidateState(StrEnum):
    OBSERVED = "observed"
    CANDIDATE = "candidate"
    REVIEW = "review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    RETIRED = "retired"


_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.I),
    re.compile(r"\b(?:sk|ghp|github_pat|xoxb|xoxp)-[A-Za-z0-9_-]{12,}\b", re.I),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:password|token|authorization|cookie)\s*[:=]"),
)
_PATH_PATTERN = re.compile(r"(?:[A-Za-z]:\\|/(?:home|root|Users|opt)/|\\\\)")


@dataclass(frozen=True, slots=True)
class SkillCandidate:
    candidate_id: str
    pattern_key: str
    name: str
    description: str
    procedure: str
    state: SkillCandidateState
    success_case_ids: tuple[str, ...] = ()
    failure_case_ids: tuple[str, ...] = ()
    project_id: str | None = None
    environment: str | None = None
    source_episode_ids: tuple[str, ...] = ()
    version: int = 1
    # Candidate ID of the predecessor this version replaces.
    supersedes: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class SkillLifecycleManager:
    def __init__(self, *, repeated_success_threshold: int = 3) -> None:
        if repeated_success_threshold < 2:
            raise ValueError("a Skill needs at least two successful cases")
        self.threshold = repeated_success_threshold
        self._items: dict[str, SkillCandidate] = {}

    def observe_success(
        self,
        *,
        candidate_id: str,
        pattern_key: str,
        name: str,
        description: str,
        procedure: str,
        case_id: str,
        project_id: str | None = None,
        environment: str | None = None,
        episode_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> SkillCandidate:
        current = self._items.get(candidate_id)
        if current is not None and current.state in {
            SkillCandidateState.REJECTED,
            SkillCandidateState.RETIRED,
            SkillCandidateState.SUPERSEDED,
        }:
            raise ValueError("terminal Skill candidate requires a new candidate id")
        if current is not None and current.pattern_key != pattern_key:
            raise ValueError("candidate id is already bound to another pattern")
        cases = list(current.success_case_ids if current else ())
        if case_id not in cases:
            cases.append(case_id)
        state = (
            current.state
            if current is not None and current.state == SkillCandidateState.ACCEPTED
            else (
                SkillCandidateState.CANDIDATE
                if len(cases) >= self.threshold
                else SkillCandidateState.OBSERVED
            )
        )
        candidate = self._build(
            candidate_id=candidate_id,
            pattern_key=pattern_key,
            name=name,
            description=description,
            procedure=procedure,
            state=state,
            success_case_ids=tuple(cases),
            failure_case_ids=current.failure_case_ids if current else (),
            project_id=project_id,
            environment=environment,
            source_episode_ids=_append_unique(
                current.source_episode_ids if current else (), episode_id
            ),
            version=current.version if current else 1,
            metadata={
                **(current.metadata if current else {}),
                **(metadata or {}),
            },
        )
        self._items[candidate_id] = candidate
        return candidate

    def observe_failure(self, candidate_id: str, case_id: str) -> SkillCandidate | None:
        current = self._items.get(candidate_id)
        if current is None:
            return None
        if current.state == SkillCandidateState.ACCEPTED:
            # A later failed Case is review evidence, not an automatic
            # demotion or deletion of an already reviewed Skill.
            return current
        failures = _append_unique(current.failure_case_ids, case_id)
        # A counterexample invalidates this candidate's promotion path. It is
        # retained as provenance for review, but cannot later be accepted or
        # revived under the same identity.
        state = SkillCandidateState.REJECTED if failures else current.state
        updated = self._build_from(current, state=state, failure_case_ids=failures)
        self._items[candidate_id] = updated
        return updated

    def submit_for_review(self, candidate_id: str) -> SkillCandidate:
        candidate = self._require(candidate_id)
        if candidate.state != SkillCandidateState.CANDIDATE:
            raise ValueError("only repeated-success candidates may enter review")
        return self._replace(candidate, state=SkillCandidateState.REVIEW)

    def accept(self, candidate_id: str, *, reusable: bool = True) -> SkillCandidate:
        candidate = self._require(candidate_id)
        if candidate.state != SkillCandidateState.REVIEW:
            raise ValueError("Skill acceptance requires explicit review state")
        if reusable and candidate.project_id is not None:
            raise ValueError("project-specific procedure cannot become reusable Skill")
        if _unsafe_content(candidate.name, candidate.description, candidate.procedure):
            raise ValueError("skill content contains secret or private-path material")
        return self._replace(candidate, state=SkillCandidateState.ACCEPTED)

    def supersede(self, candidate_id: str, replacement_id: str) -> SkillCandidate:
        candidate = self._require(candidate_id)
        replacement = self._require(replacement_id)
        if candidate_id == replacement_id:
            raise ValueError("a Skill cannot supersede itself")
        if replacement.state not in {
            SkillCandidateState.REVIEW,
            SkillCandidateState.ACCEPTED,
        }:
            raise ValueError("replacement must be reviewed or accepted")
        # ``supersedes`` belongs on the new version and points backwards to
        # its predecessor. The predecessor is terminally marked separately;
        # no acceptance is inferred for a replacement still in REVIEW.
        updated_replacement = self._replace(
            replacement,
            supersedes=candidate.candidate_id,
            version=max(replacement.version, candidate.version + 1),
        )
        self._replace(candidate, state=SkillCandidateState.SUPERSEDED)
        return updated_replacement

    def retire(self, candidate_id: str) -> SkillCandidate:
        candidate = self._require(candidate_id)
        if candidate.state not in {
            SkillCandidateState.ACCEPTED,
            SkillCandidateState.SUPERSEDED,
        }:
            raise ValueError("only accepted or superseded Skills can be retired")
        return self._replace(candidate, state=SkillCandidateState.RETIRED)

    def get(self, candidate_id: str) -> SkillCandidate | None:
        return self._items.get(candidate_id)

    def all(self) -> tuple[SkillCandidate, ...]:
        return tuple(sorted(self._items.values(), key=lambda item: item.candidate_id))

    def _build(self, **kwargs: object) -> SkillCandidate:
        raw_name = str(kwargs.pop("name"))
        raw_description = str(kwargs.pop("description"))
        raw_procedure = str(kwargs.pop("procedure"))
        return SkillCandidate(
            **kwargs,
            name=sanitize_skill_name(raw_name),
            description=_redact_private_path(raw_description),
            procedure=_redact_private_path(raw_procedure),
        )

    def _build_from(self, current: SkillCandidate, **changes: object) -> SkillCandidate:
        data = {
            field: getattr(current, field)
            for field in SkillCandidate.__dataclass_fields__
        }
        data.update(changes)
        return SkillCandidate(**data)

    def _replace(self, current: SkillCandidate, **changes: object) -> SkillCandidate:
        updated = self._build_from(current, **changes)
        self._items[current.candidate_id] = updated
        return updated

    def _require(self, candidate_id: str) -> SkillCandidate:
        candidate = self._items.get(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        return candidate


class SkillLifecycleFileStore:
    """Crash-safe, secret-free checkpoint for candidate lifecycle state.

    Markdown ``SKILL.md`` remains the durable source for accepted skills. This
    small sidecar is only for the review queue (observed/candidate/review and
    terminal lifecycle records), so a restart does not forget repeated-case
    evidence. Writes use a sibling lock and same-directory ``os.replace``;
    loading validates the complete snapshot before replacing manager state.
    """

    _SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_name(f"{path.name}.lock")

    def save(self, manager: SkillLifecycleManager) -> None:
        items = manager.all()
        serialized = []
        for item in items:
            if _unsafe_content(item.name, item.description, item.procedure):
                raise ValueError("refusing to persist unsafe skill candidate")
            serialized.append(_candidate_to_dict(item))
        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "items": serialized,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a", timeout=5):
            temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, self.path)
            finally:
                if temporary.exists():
                    temporary.unlink()

    def load(self, manager: SkillLifecycleManager) -> int:
        if not self.path.is_file():
            return 0
        with portalocker.Lock(str(self.lock_path), mode="a", timeout=5):
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != self._SCHEMA_VERSION
        ):
            raise ValueError("unsupported Skill lifecycle snapshot")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("Skill lifecycle snapshot items must be a list")
        restored: dict[str, SkillCandidate] = {}
        for raw in raw_items:
            candidate = _candidate_from_dict(raw)
            if _unsafe_content(
                candidate.name, candidate.description, candidate.procedure
            ):
                raise ValueError("refusing to load unsafe skill candidate")
            prior = restored.get(candidate.candidate_id)
            if prior is not None and prior.pattern_key != candidate.pattern_key:
                raise ValueError("candidate id is already bound to another pattern")
            restored[candidate.candidate_id] = candidate
        manager._items = restored
        return len(restored)


def sanitize_skill_name(raw: str) -> str:
    value = unicodedata.normalize("NFC", raw).strip().casefold()
    value = re.sub(r"[^\w\-.]+", "_", value, flags=re.UNICODE)
    value = value.strip("._")[:50]
    return value or "unnamed"


def _unsafe_content(*values: str) -> bool:
    joined = "\n".join(values)
    return any(pattern.search(joined) for pattern in _SECRET_PATTERNS) or bool(
        _PATH_PATTERN.search(joined)
    )


def _redact_private_path(value: str) -> str:
    return _PATH_PATTERN.sub("<private-path>", value)


def _append_unique(values: Iterable[str], value: str | None) -> tuple[str, ...]:
    result = list(values)
    if value is not None and value not in result:
        result.append(value)
    return tuple(result)


def _candidate_to_dict(candidate: SkillCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "pattern_key": candidate.pattern_key,
        "name": candidate.name,
        "description": candidate.description,
        "procedure": candidate.procedure,
        "state": candidate.state.value,
        "success_case_ids": list(candidate.success_case_ids),
        "failure_case_ids": list(candidate.failure_case_ids),
        "project_id": candidate.project_id,
        "environment": candidate.environment,
        "source_episode_ids": list(candidate.source_episode_ids),
        "version": candidate.version,
        "supersedes": candidate.supersedes,
        "metadata": dict(sorted(candidate.metadata.items())),
    }


def _candidate_from_dict(raw: object) -> SkillCandidate:
    if not isinstance(raw, dict):
        raise ValueError("Skill lifecycle item must be an object")

    def required_text(name: str) -> str:
        value = raw.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Skill lifecycle field {name!r} must be non-empty")
        return value

    def string_tuple(name: str) -> tuple[str, ...]:
        value = raw.get(name, [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ValueError(f"Skill lifecycle field {name!r} must be string list")
        return tuple(dict.fromkeys(value))

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError("Skill lifecycle metadata must be a string map")
    version = raw.get("version", 1)
    if not isinstance(version, int) or version < 1:
        raise ValueError("Skill lifecycle version must be a positive integer")
    try:
        state = SkillCandidateState(required_text("state"))
    except ValueError as exc:
        raise ValueError("unknown Skill lifecycle state") from exc
    project_id = raw.get("project_id")
    environment = raw.get("environment")
    supersedes = raw.get("supersedes")
    if any(
        value is not None and not isinstance(value, str)
        for value in (project_id, environment, supersedes)
    ):
        raise ValueError("Skill lifecycle optional fields must be strings")
    return SkillCandidate(
        candidate_id=required_text("candidate_id"),
        pattern_key=required_text("pattern_key"),
        name=required_text("name"),
        description=required_text("description"),
        procedure=required_text("procedure"),
        state=state,
        success_case_ids=string_tuple("success_case_ids"),
        failure_case_ids=string_tuple("failure_case_ids"),
        project_id=project_id,
        environment=environment,
        source_episode_ids=string_tuple("source_episode_ids"),
        version=version,
        supersedes=supersedes,
        metadata=dict(metadata),
    )
