"""Non-destructive LanceDB upgrade evidence and copy-migration probe.

The production index is rebuildable from Markdown, but that is not a
rollback proof. This module therefore keeps engine upgrades outside the
runtime path and provides a disposable-copy probe that checks rows, schema,
FTS query results, isolated maintenance, and readability of the untouched
source snapshot after the probe.
"""

from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LanceDBMigrationProbeError(RuntimeError):
    """Raised when a disposable-copy migration probe cannot prove parity."""


@dataclass(frozen=True, slots=True)
class LanceDBApiProbe:
    package_version: str
    has_sync_connect: bool
    has_async_connect: bool
    has_fts_index: bool
    has_optimize: bool
    has_index_stats: bool


@dataclass(frozen=True, slots=True)
class LanceDBCopyEvidence:
    source_dir: str
    isolated_dir: str
    table_name: str
    source_schema: tuple[str, ...]
    isolated_schema: tuple[str, ...]
    source_row_count: int
    isolated_row_count: int
    source_query_ids: tuple[str, ...]
    isolated_query_ids: tuple[str, ...]
    rollback_query_ids: tuple[str, ...]
    isolated_maintenance_ok: bool

    @property
    def rows_match(self) -> bool:
        return (
            self.source_schema == self.isolated_schema
            and self.source_row_count == self.isolated_row_count
        )

    @property
    def query_parity(self) -> bool:
        return self.source_query_ids == self.isolated_query_ids

    @property
    def rollback_proven(self) -> bool:
        return self.rollback_query_ids == self.source_query_ids

    @property
    def passed(self) -> bool:
        return (
            self.rows_match
            and self.query_parity
            and self.rollback_proven
            and self.isolated_maintenance_ok
        )


@dataclass(frozen=True, slots=True)
class LanceDBUpgradeAssessment:
    current_version: str
    target_version: str
    benefit_evidence: str
    schema_compatible: bool
    api_compatible: bool
    index_parity: bool
    rollback_proven: bool
    concurrent_soak_passed: bool
    destructive: bool
    decision: str
    missing_gates: tuple[str, ...]

    @property
    def ready_for_isolated_migration(self) -> bool:
        return self.decision == "READY_FOR_ISOLATED_MIGRATION"


def probe_local_api() -> LanceDBApiProbe:
    """Report only local package/API facts; never changes a database."""

    import lancedb

    try:
        import lancedb.table as lancedb_table

        lance_table_type = getattr(lancedb_table, "LanceTable", None)
    except ImportError:  # pragma: no cover - defensive across future releases
        lance_table_type = None
    return LanceDBApiProbe(
        package_version=str(getattr(lancedb, "__version__", "unknown")),
        has_sync_connect=callable(getattr(lancedb, "connect", None)),
        has_async_connect=callable(getattr(lancedb, "connect_async", None)),
        has_fts_index=_has_fts_index_api(lance_table_type),
        has_optimize=bool(lance_table_type and hasattr(lance_table_type, "optimize")),
        has_index_stats=bool(
            lance_table_type and hasattr(lance_table_type, "index_stats")
        ),
    )


def probe_isolated_copy(
    source_dir: Path,
    isolated_dir: Path,
    *,
    table_name: str,
    query: str,
    fts_column: str | None = None,
) -> LanceDBCopyEvidence:
    """Copy a closed disposable DB and prove read/query parity.

    ``source_dir`` is read-only from this function's perspective. The
    destination must not exist, avoiding accidental overwrite of an operator
    supplied path. FTS rebuild and optimize, when requested, run only on the
    copied database.
    """

    if not source_dir.is_dir():
        raise LanceDBMigrationProbeError(
            f"source directory does not exist: {source_dir}"
        )
    if isolated_dir.exists():
        raise LanceDBMigrationProbeError(
            f"isolated destination already exists: {isolated_dir}"
        )

    import lancedb

    source_db = lancedb.connect(source_dir)
    try:
        source_table = source_db.open_table(table_name)
        source_schema = tuple(source_table.schema.names)
        source_rows = source_table.to_arrow().to_pylist()
        source_query_ids = _query_ids(source_table, query)
    finally:
        close = getattr(source_db, "close", None)
        if callable(close):
            close()

    shutil.copytree(source_dir, isolated_dir)
    isolated_db = lancedb.connect(isolated_dir)
    isolated_row_count = 0
    try:
        isolated_table = isolated_db.open_table(table_name)
        isolated_schema = tuple(isolated_table.schema.names)
        maintenance_ok = True
        if fts_column is not None:
            try:
                _rebuild_fts_index(isolated_table, fts_column)
                isolated_table.optimize()
            except Exception as exc:  # pragma: no cover - engine-specific detail
                maintenance_ok = False
                raise LanceDBMigrationProbeError(
                    f"isolated FTS rebuild/optimize failed: {exc}"
                ) from exc
        isolated_query_ids = _query_ids(isolated_table, query)
        isolated_row_count = len(isolated_table.to_arrow().to_pylist())
    finally:
        close = getattr(isolated_db, "close", None)
        if callable(close):
            close()

    # Re-open the untouched source after isolated maintenance. This is the
    # rollback proof: the pre-migration snapshot remains queryable and has not
    # been rewritten by the experiment.
    rollback_db = lancedb.connect(source_dir)
    try:
        rollback_query_ids = _query_ids(rollback_db.open_table(table_name), query)
    finally:
        close = getattr(rollback_db, "close", None)
        if callable(close):
            close()

    return LanceDBCopyEvidence(
        source_dir=str(source_dir),
        isolated_dir=str(isolated_dir),
        table_name=table_name,
        source_schema=source_schema,
        isolated_schema=isolated_schema,
        source_row_count=len(source_rows),
        isolated_row_count=isolated_row_count,
        source_query_ids=source_query_ids,
        isolated_query_ids=isolated_query_ids,
        rollback_query_ids=rollback_query_ids,
        isolated_maintenance_ok=maintenance_ok,
    )


def assess_engine_upgrade(
    *,
    current_version: str,
    target_version: str,
    benefit_evidence: str,
    schema_compatible: bool,
    api_compatible: bool,
    index_parity: bool,
    rollback_proven: bool,
    concurrent_soak_passed: bool,
    destructive: bool = False,
) -> LanceDBUpgradeAssessment:
    """Turn migration evidence into a conservative, non-deploying verdict."""

    checks = {
        "benefit_evidence": bool(benefit_evidence.strip()),
        "schema_compatible": schema_compatible,
        "api_compatible": api_compatible,
        "index_parity": index_parity,
        "rollback_proven": rollback_proven,
        "concurrent_soak_passed": concurrent_soak_passed,
        "non_destructive": not destructive,
    }
    missing = tuple(name for name, passed in checks.items() if not passed)
    decision = (
        "READY_FOR_ISOLATED_MIGRATION"
        if not missing
        else "LANCEDB_ENGINE_UPGRADE_DEFERRED_WITH_EVIDENCE"
    )
    return LanceDBUpgradeAssessment(
        current_version=current_version,
        target_version=target_version,
        benefit_evidence=benefit_evidence,
        schema_compatible=schema_compatible,
        api_compatible=api_compatible,
        index_parity=index_parity,
        rollback_proven=rollback_proven,
        concurrent_soak_passed=concurrent_soak_passed,
        destructive=destructive,
        decision=decision,
        missing_gates=missing,
    )


def _query_ids(table: Any, query: str) -> tuple[str, ...]:
    try:
        rows = table.search(query, query_type="fts").limit(50).to_list()
    except Exception as exc:
        raise LanceDBMigrationProbeError(
            "query parity requires an FTS-searchable table"
        ) from exc
    ids: list[str] = []
    for row in rows:
        value = row.get("id") if isinstance(row, dict) else None
        if value is not None:
            ids.append(str(value))
    return tuple(ids)


def _rebuild_fts_index(table: Any, column: str) -> None:
    """Rebuild FTS across the pre/post-0.33 LanceDB API boundary."""

    create_index = getattr(table, "create_index", None)
    if callable(create_index) and _accepts_config(create_index):
        try:
            from lancedb.index import FTS
        except ImportError:  # pragma: no cover - defensive for partial installs
            pass
        else:
            create_index(column, config=FTS(), replace=True)
            return
    create_fts_index = getattr(table, "create_fts_index", None)
    if not callable(create_fts_index):
        raise LanceDBMigrationProbeError("local LanceDB has no FTS index API")
    create_fts_index(column, replace=True)


def _has_fts_index_api(table_type: Any) -> bool:
    if table_type is None:
        return False
    if callable(getattr(table_type, "create_fts_index", None)):
        return True
    create_index = getattr(table_type, "create_index", None)
    return callable(create_index) and _accepts_config(create_index)


def _accepts_config(callable_object: Any) -> bool:
    try:
        return "config" in inspect.signature(callable_object).parameters
    except (TypeError, ValueError):  # pragma: no cover - extension methods
        return False


def canonical_rows(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Expose deterministic row fingerprints for external probe reports."""

    return tuple(
        json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) for row in rows
    )
