"""Versioned, rebuildable materialisation for deterministic Memory Wiki pages.

The materializer consumes the already-reviewed Local Brain Truth View v1
artifact. It never searches, calls an LLM, mutates Truth, or writes into the
user-visible Markdown authority tree. Output lives under ``.index/wiki`` and
is therefore explicitly derived/rebuildable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from everos.component.utils.datetime import ensure_utc
from everos.core.persistence.memory_root import MemoryRoot

from .truth import AuthorityState, TruthClaim, TruthClass
from .wiki import WikiCompiler, WikiPage

TRUTH_VIEW_SOURCE = "derived-rebuildable-truth-view"
TRUTH_VIEW_SCHEMA_VERSION = 1
WIKI_ARTIFACT_SCHEMA_VERSION = 2
_SUPPORTED_CURRENT_KINDS = {
    "PROJECT_STATE",
    "GOAL",
    "DECISION",
    "CONSTRAINT",
    "TODO",
}
_SAFE_SNAPSHOT = re.compile(r"^[a-f0-9]{64}$")


class WikiMaterializationError(ValueError):
    """Raised when an input Truth View cannot be safely materialized."""


@dataclass(frozen=True, slots=True)
class MaterializedWikiPage:
    slug: str
    title: str
    project_key: str
    artifact_id: str
    rendered_claim_count: int
    backed_claim_count: int
    unsupported_claim_count: int
    stale_claim_count: int
    verification_status: str
    claim_set_sha256: str
    agent_view_sha256: str
    markdown_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "slug": self.slug,
            "title": self.title,
            "project_key": self.project_key,
            "artifact_id": self.artifact_id,
            "rendered_claim_count": self.rendered_claim_count,
            "backed_claim_count": self.backed_claim_count,
            "unsupported_claim_count": self.unsupported_claim_count,
            "stale_claim_count": self.stale_claim_count,
            "verification_status": self.verification_status,
            "claim_set_sha256": self.claim_set_sha256,
            "agent_view_sha256": self.agent_view_sha256,
            "markdown_sha256": self.markdown_sha256,
        }


@dataclass(frozen=True, slots=True)
class MaterializedWikiManifest:
    snapshot_id: str
    source_truth_view_sha256: str | None
    compiler_version: str
    materialized_at: str
    pages: tuple[MaterializedWikiPage, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": WIKI_ARTIFACT_SCHEMA_VERSION,
            "snapshot_id": self.snapshot_id,
            "source_truth_view_sha256": self.source_truth_view_sha256,
            "compiler_version": self.compiler_version,
            "materialized_at": self.materialized_at,
            "pages": [page.as_dict() for page in self.pages],
        }


def materialize_truth_view(
    payload: Mapping[str, Any],
    *,
    memory_root: MemoryRoot | Path | str | None = None,
    compiler: WikiCompiler | None = None,
) -> MaterializedWikiManifest:
    """Compile and atomically publish a versioned Wiki snapshot."""

    _validate_truth_view(payload)
    compiler = compiler or WikiCompiler()
    root = _coerce_memory_root(memory_root)
    wiki_root = root.index_dir / "wiki"
    snapshots_root = wiki_root / "snapshots"
    source_digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    snapshot_id = hashlib.sha256(
        (
            f"{source_digest}:{compiler.version}:"
            f"artifact-v{WIKI_ARTIFACT_SCHEMA_VERSION}"
        ).encode()
    ).hexdigest()
    grouped = _claims_by_project(payload)

    pages: list[tuple[MaterializedWikiPage, WikiPage, dict[str, object], str]] = []
    page_by_slug: dict[str, str] = {}
    for project_key in sorted(grouped):
        title = project_key if project_key != "__global__" else "Global Memory"
        page = compiler.compile(
            snapshot_id=snapshot_id,
            title=title,
            claims=grouped[project_key],
            scope={"project_key": project_key},
        )
        if not page.claims:
            continue
        previous_project = page_by_slug.get(page.slug)
        if previous_project is not None and previous_project != project_key:
            raise WikiMaterializationError(
                "wiki slug collision between project keys "
                f"{previous_project!r} and {project_key!r}"
            )
        page_by_slug[page.slug] = project_key
        agent_view = page.agent_view()
        markdown = page.human_markdown()
        artifact_id = hashlib.sha256(page.slug.encode("utf-8")).hexdigest()[:24]
        pages.append(
            (
                MaterializedWikiPage(
                    slug=page.slug,
                    title=page.title,
                    project_key=project_key,
                    artifact_id=artifact_id,
                    rendered_claim_count=page.rendered_claim_count,
                    backed_claim_count=page.backed_claim_count,
                    unsupported_claim_count=page.unsupported_claim_count,
                    stale_claim_count=page.stale_claim_count,
                    verification_status=page.verification_status,
                    claim_set_sha256=page.claim_set_sha256,
                    agent_view_sha256=hashlib.sha256(
                        _canonical_json(agent_view)
                    ).hexdigest(),
                    markdown_sha256=_sha256_text(markdown),
                ),
                page,
                agent_view,
                markdown,
            )
        )

    materialized_at = _source_timestamp(payload) or dt.datetime.now(dt.UTC).isoformat()
    manifest = MaterializedWikiManifest(
        snapshot_id=snapshot_id,
        source_truth_view_sha256=_optional_string(payload.get("truth_view_sha256")),
        compiler_version=compiler.version,
        materialized_at=materialized_at,
        pages=tuple(item[0] for item in pages),
    )

    snapshots_root.mkdir(parents=True, exist_ok=True)
    target = snapshots_root / snapshot_id
    if not target.exists():
        staging = snapshots_root / f".{snapshot_id}.tmp-{uuid.uuid4().hex}"
        try:
            staging.mkdir(parents=False, exist_ok=False)
            for summary, _page, agent_view, markdown in pages:
                page_payload = {
                    "schema_version": WIKI_ARTIFACT_SCHEMA_VERSION,
                    "snapshot_id": snapshot_id,
                    "page": summary.as_dict(),
                    "agent_view": agent_view,
                    "markdown": markdown,
                }
                _write_json_atomic(
                    staging / f"{summary.artifact_id}.json", page_payload
                )
                _write_text_atomic(staging / f"{summary.artifact_id}.md", markdown)
            _write_json_atomic(staging / "manifest.json", manifest.as_dict())
            os.replace(staging, target)
        except FileExistsError:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    _validate_snapshot_against_manifest(target, manifest)
    wiki_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        wiki_root / "current.json",
        {
            "schema_version": WIKI_ARTIFACT_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "manifest_sha256": _sha256_file(target / "manifest.json"),
        },
    )
    return _read_manifest(target / "manifest.json")


def load_current_manifest(
    memory_root: MemoryRoot | Path | str | None = None,
) -> MaterializedWikiManifest | None:
    root = _coerce_memory_root(memory_root)
    wiki_root = root.index_dir / "wiki"
    pointer_path = wiki_root / "current.json"
    if not pointer_path.exists():
        return None
    pointer = _read_json(pointer_path)
    snapshot_id = str(pointer.get("snapshot_id", ""))
    if not _SAFE_SNAPSHOT.fullmatch(snapshot_id):
        raise WikiMaterializationError("invalid active wiki snapshot id")
    target = wiki_root / "snapshots" / snapshot_id
    _validate_snapshot_dir(target, snapshot_id)
    manifest_path = target / "manifest.json"
    if str(pointer.get("manifest_sha256", "")) != _sha256_file(manifest_path):
        raise WikiMaterializationError("active wiki manifest checksum mismatch")
    manifest = _read_manifest(manifest_path)
    if manifest.snapshot_id != snapshot_id:
        raise WikiMaterializationError("active wiki manifest snapshot mismatch")
    return manifest


def load_current_page(
    slug: str,
    memory_root: MemoryRoot | Path | str | None = None,
) -> dict[str, Any] | None:
    manifest = load_current_manifest(memory_root)
    if manifest is None:
        return None
    summary = next((page for page in manifest.pages if page.slug == slug), None)
    if summary is None:
        return None
    root = _coerce_memory_root(memory_root)
    snapshot_dir = root.index_dir / "wiki" / "snapshots" / manifest.snapshot_id
    return _read_page_artifact(snapshot_dir, manifest.snapshot_id, summary)


def _claims_by_project(payload: Mapping[str, Any]) -> dict[str, tuple[TruthClaim, ...]]:
    grouped: defaultdict[str, list[TruthClaim]] = defaultdict(list)
    for section in (
        "current_facts",
        "active_goals",
        "active_decisions",
        "active_constraints",
        "open_todos",
    ):
        for record in _records(payload, section):
            kind = str(record.get("memory_type", ""))
            if kind not in _SUPPORTED_CURRENT_KINDS:
                continue
            if (
                record.get("truth_status") != "CURRENT"
                or record.get("status") != "accepted"
            ):
                continue
            claim = _record_to_claim(record, kind=kind, truth_class=TruthClass.CURRENT)
            if claim is not None:
                grouped[str(claim.scope["project_key"])].append(claim)

    for record in _records(payload, "history"):
        if record.get("truth_status") != "HISTORY":
            continue
        if record.get("status") not in {"accepted", "superseded"}:
            continue
        claim = _record_to_claim(record, kind="HISTORY", truth_class=TruthClass.HISTORY)
        if claim is not None:
            grouped[str(claim.scope["project_key"])].append(claim)
    return {key: tuple(value) for key, value in grouped.items()}


def _record_to_claim(
    record: Mapping[str, Any], *, kind: str, truth_class: TruthClass
) -> TruthClaim | None:
    claim_id = _optional_string(record.get("record_id") or record.get("memory_id"))
    text = _optional_string(record.get("statement"))
    if not claim_id or not text:
        return None
    raw_scope = record.get("scope")
    if not isinstance(raw_scope, Mapping):
        return None
    project_key = _optional_string(raw_scope.get("project_key")) or "__global__"
    source_refs = _source_refs(record)
    if not source_refs:
        return None
    return TruthClaim(
        claim_id=claim_id,
        text=text,
        kind=kind,
        scope={"project_key": project_key},
        authority=AuthorityState.ACCEPTED,
        truth_class=truth_class,
        superseded_by=_optional_string(record.get("superseded_by")),
        valid_from=_parse_datetime(record.get("valid_from")),
        valid_until=_parse_datetime(record.get("valid_to")),
        source_refs=source_refs,
        created_at=_parse_datetime(record.get("valid_from")),
        metadata={
            "memory_type": _optional_string(record.get("memory_type")),
            "subject": _optional_string(record.get("subject")),
            "state_key": _optional_string(record.get("state_key")),
            "source_kinds": tuple(_string_sequence(record.get("source_kinds"))),
            "truth_status": _optional_string(record.get("truth_status")),
        },
    )


def _source_refs(record: Mapping[str, Any]) -> tuple[str, ...]:
    refs: set[str] = set()
    provenance = record.get("provenance")
    if isinstance(provenance, Sequence) and not isinstance(provenance, (str, bytes)):
        for source in provenance:
            if not isinstance(source, Mapping):
                continue
            kind = _optional_string(source.get("kind")) or "source"
            source_id = _optional_string(source.get("source_id"))
            file_hash = _optional_string(source.get("file_sha256"))
            line = source.get("line")
            evidence_refs = _string_sequence(source.get("evidence_refs"))
            if source_id:
                refs.add(f"{kind}:{source_id}")
            if file_hash:
                suffix = f":L{line}" if isinstance(line, int) and line > 0 else ""
                refs.add(f"{kind}:sha256:{file_hash}{suffix}")
            refs.update(f"{kind}:evidence:{item}" for item in evidence_refs)
    return tuple(sorted(refs))


def _validate_truth_view(payload: Mapping[str, Any]) -> None:
    if payload.get("source") != TRUTH_VIEW_SOURCE:
        raise WikiMaterializationError("unsupported truth view source")
    if payload.get("schema_version") != TRUTH_VIEW_SCHEMA_VERSION:
        raise WikiMaterializationError("unsupported truth view schema version")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise WikiMaterializationError("truth view policy is missing")
    if (
        policy.get("rebuildable") is not True
        or policy.get("conflicts_fail_closed") is not True
    ):
        raise WikiMaterializationError("truth view does not satisfy fail-closed policy")


def _validate_snapshot_dir(path: Path, snapshot_id: str) -> None:
    if not _SAFE_SNAPSHOT.fullmatch(snapshot_id):
        raise WikiMaterializationError("invalid wiki snapshot id")
    if not path.is_dir() or not (path / "manifest.json").is_file():
        raise WikiMaterializationError("wiki snapshot is incomplete")


def _validate_snapshot_against_manifest(
    path: Path,
    expected: MaterializedWikiManifest,
) -> None:
    """Treat a same-id snapshot as immutable deterministic output."""

    _validate_snapshot_dir(path, expected.snapshot_id)
    actual = _read_manifest(path / "manifest.json")
    if actual.as_dict() != expected.as_dict():
        raise WikiMaterializationError(
            "existing wiki snapshot does not match deterministic build"
        )
    for summary in expected.pages:
        payload = _read_page_artifact(path, expected.snapshot_id, summary)
        markdown_path = path / f"{summary.artifact_id}.md"
        try:
            markdown = markdown_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise WikiMaterializationError(
                f"cannot read wiki markdown artifact: {exc}"
            ) from exc
        if markdown != payload["markdown"]:
            raise WikiMaterializationError("wiki markdown artifact mismatch")


def _read_page_artifact(
    snapshot_dir: Path,
    snapshot_id: str,
    summary: MaterializedWikiPage,
) -> dict[str, Any]:
    payload = _read_json(snapshot_dir / f"{summary.artifact_id}.json")
    if payload.get("snapshot_id") != snapshot_id:
        raise WikiMaterializationError("wiki page snapshot mismatch")
    if payload.get("page") != summary.as_dict():
        raise WikiMaterializationError("wiki page summary mismatch")
    agent_view = payload.get("agent_view")
    if not isinstance(agent_view, Mapping):
        raise WikiMaterializationError("wiki agent view must be an object")
    if (
        hashlib.sha256(_canonical_json(agent_view)).hexdigest()
        != summary.agent_view_sha256
    ):
        raise WikiMaterializationError("wiki agent view checksum mismatch")
    if agent_view.get("claim_set_sha256") != summary.claim_set_sha256:
        raise WikiMaterializationError("wiki agent claim-set mismatch")
    markdown = payload.get("markdown")
    if not isinstance(markdown, str):
        raise WikiMaterializationError("wiki markdown must be text")
    if _sha256_text(markdown) != summary.markdown_sha256:
        raise WikiMaterializationError("wiki markdown checksum mismatch")
    return payload


def _read_manifest(path: Path) -> MaterializedWikiManifest:
    payload = _read_json(path)
    if payload.get("schema_version") != WIKI_ARTIFACT_SCHEMA_VERSION:
        raise WikiMaterializationError("unsupported wiki artifact schema")
    snapshot_id = str(payload.get("snapshot_id", ""))
    if not _SAFE_SNAPSHOT.fullmatch(snapshot_id):
        raise WikiMaterializationError("invalid wiki manifest snapshot id")
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        raise WikiMaterializationError("invalid wiki manifest pages")
    pages = tuple(
        MaterializedWikiPage(
            slug=str(item["slug"]),
            title=str(item["title"]),
            project_key=str(item["project_key"]),
            artifact_id=str(item["artifact_id"]),
            rendered_claim_count=int(item["rendered_claim_count"]),
            backed_claim_count=int(item["backed_claim_count"]),
            unsupported_claim_count=int(item["unsupported_claim_count"]),
            stale_claim_count=int(item["stale_claim_count"]),
            verification_status=str(item["verification_status"]),
            claim_set_sha256=str(item["claim_set_sha256"]),
            agent_view_sha256=str(item["agent_view_sha256"]),
            markdown_sha256=str(item["markdown_sha256"]),
        )
        for item in raw_pages
        if isinstance(item, Mapping)
    )
    if any(not re.fullmatch(r"[a-f0-9]{24}", page.artifact_id) for page in pages):
        raise WikiMaterializationError("invalid wiki page artifact id")
    if any(
        not re.fullmatch(r"[a-f0-9]{64}", page.claim_set_sha256)
        or not re.fullmatch(r"[a-f0-9]{64}", page.agent_view_sha256)
        or not re.fullmatch(r"[a-f0-9]{64}", page.markdown_sha256)
        for page in pages
    ):
        raise WikiMaterializationError("invalid wiki page checksum")
    if len({page.slug for page in pages}) != len(pages):
        raise WikiMaterializationError("duplicate wiki page slug in manifest")
    return MaterializedWikiManifest(
        snapshot_id=snapshot_id,
        source_truth_view_sha256=_optional_string(
            payload.get("source_truth_view_sha256")
        ),
        compiler_version=str(payload.get("compiler_version", "")),
        materialized_at=str(payload.get("materialized_at", "")),
        pages=pages,
    )


def _coerce_memory_root(value: MemoryRoot | Path | str | None) -> MemoryRoot:
    if value is None:
        return MemoryRoot.default()
    if isinstance(value, MemoryRoot):
        return value
    return MemoryRoot(value)


def _records(payload: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise WikiMaterializationError(f"truth view section {key!r} must be a list")
    return tuple(item for item in value if isinstance(item, Mapping))


def _parse_datetime(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ensure_utc(parsed)


def _source_timestamp(payload: Mapping[str, Any]) -> str | None:
    value = _parse_datetime(payload.get("generated_at"))
    return value.isoformat() if value is not None else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WikiMaterializationError(f"cannot read wiki artifact: {exc}") from exc
    if not isinstance(value, dict):
        raise WikiMaterializationError("wiki artifact must be a JSON object")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
