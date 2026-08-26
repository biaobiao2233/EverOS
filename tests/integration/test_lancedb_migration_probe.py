"""Disposable LanceDB migration evidence; never opens an EverOS production path."""

from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor

import lancedb
import pytest

from everos.infra.persistence.lancedb import (
    LanceDBMigrationProbeError,
    assess_engine_upgrade,
    probe_isolated_copy,
)


def test_isolated_copy_rebuild_optimize_query_parity_and_rollback(tmp_path) -> None:
    source = tmp_path / "source"
    isolated = tmp_path / "isolated"
    db = lancedb.connect(source)
    table = db.create_table(
        "records",
        [
            {"id": "a", "text": "alpha release checklist"},
            {"id": "b", "text": "beta unrelated note"},
        ],
    )
    _create_fts_index(table, "text")
    close = getattr(db, "close", None)
    if callable(close):
        close()

    evidence = probe_isolated_copy(
        source,
        isolated,
        table_name="records",
        query="release",
        fts_column="text",
    )

    assert evidence.passed
    assert evidence.rows_match
    assert evidence.query_parity
    assert evidence.rollback_proven


def test_upgrade_assessment_defers_without_rollback_and_soak_evidence() -> None:
    assessment = assess_engine_upgrade(
        current_version="0.30.2",
        target_version="0.33.0",
        benefit_evidence="upstream release exists; compatibility not yet demonstrated",
        schema_compatible=True,
        api_compatible=True,
        index_parity=True,
        rollback_proven=False,
        concurrent_soak_passed=False,
    )

    assert assessment.decision == "LANCEDB_ENGINE_UPGRADE_DEFERRED_WITH_EVIDENCE"
    assert "rollback_proven" in assessment.missing_gates
    assert "concurrent_soak_passed" in assessment.missing_gates


def test_current_engine_concurrent_read_write_smoke(tmp_path) -> None:
    """Exercise the current engine's local reader/writer behavior separately.

    This is evidence for the accepted engine only; it is intentionally not
    promoted to target-version migration evidence by the assessment helper.
    """

    source = tmp_path / "concurrent"
    db = lancedb.connect(source)
    db.create_table(
        "records",
        [{"id": "seed", "text": "seed"}],
    )
    close = getattr(db, "close", None)
    if callable(close):
        close()

    def read_rows() -> None:
        for _ in range(10):
            reader = lancedb.connect(source)
            try:
                assert reader.open_table("records").to_arrow().num_rows >= 1
            finally:
                close_reader = getattr(reader, "close", None)
                if callable(close_reader):
                    close_reader()

    def append_rows() -> None:
        writer = lancedb.connect(source)
        try:
            writer.open_table("records").add(
                [{"id": f"write-{index}", "text": "write"} for index in range(10)]
            )
        finally:
            close_writer = getattr(writer, "close", None)
            if callable(close_writer):
                close_writer()

    with ThreadPoolExecutor(max_workers=2) as pool:
        read_future = pool.submit(read_rows)
        write_future = pool.submit(append_rows)
        read_future.result()
        write_future.result()

    final_db = lancedb.connect(source)
    try:
        assert final_db.open_table("records").to_arrow().num_rows == 11
    finally:
        close = getattr(final_db, "close", None)
        if callable(close):
            close()


def test_probe_failure_injection_leaves_source_readable(tmp_path) -> None:
    source = tmp_path / "source"
    isolated = tmp_path / "isolated"
    db = lancedb.connect(source)
    table = db.create_table(
        "records",
        [{"id": "a", "text": "alpha release checklist"}],
    )
    _create_fts_index(table, "text")
    close = getattr(db, "close", None)
    if callable(close):
        close()

    with pytest.raises(LanceDBMigrationProbeError):
        probe_isolated_copy(
            source,
            isolated,
            table_name="records",
            query="release",
            fts_column="missing_column",
        )

    rollback_db = lancedb.connect(source)
    try:
        rows = (
            rollback_db.open_table("records")
            .search("release", query_type="fts")
            .limit(10)
            .to_list()
        )
        assert [row["id"] for row in rows] == ["a"]
    finally:
        close = getattr(rollback_db, "close", None)
        if callable(close):
            close()


def _create_fts_index(table, column: str) -> None:
    """Use the current API on new LanceDB and the compatibility API on old."""

    create_index = getattr(table, "create_index", None)
    if (
        callable(create_index)
        and "config" in inspect.signature(create_index).parameters
    ):
        from lancedb.index import FTS

        create_index(column, config=FTS())
        return
    table.create_fts_index(column)
