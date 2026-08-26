# EV-MEMORY-01 — LanceDB engine upgrade decision

Status: `LANCEDB_ENGINE_UPGRADE_DEFERRED_WITH_EVIDENCE`

Date: 2026-08-26

## Current and investigated versions

- Accepted EverOS baseline: `lancedb==0.30.2`.
- Local runtime verification: `lancedb.__version__ == 0.30.2`; the existing
  API exposes sync/async connection, `create_fts_index`, `optimize`, and
  `index_stats` capabilities used by EverOS.
- PyPI currently lists `0.37.1` as the newest published wheel visible to this
  review, with Python 3.12 and Windows/Linux artifacts. The official release
  history also records API and write-path changes after the accepted baseline.
  This is a candidate for a future isolated trial, not an accepted runtime
  dependency.

Primary references:

- [LanceDB releases](https://github.com/lancedb/lancedb/releases)
- [LanceDB Python package index](https://pypi.org/simple/lancedb/)
- [Reindexing and index maintenance](https://docs.lancedb.com/indexing/reindexing)
- [Full-text search and index freshness](https://docs.lancedb.com/search/full-text-search)

## Evidence and decision

The candidate branch includes a disposable-copy probe in
`everos.infra.persistence.lancedb.migration`. It copies a closed database to
a new path, compares schema/row counts and FTS query IDs, rebuilds FTS and
runs `optimize` only on the copy, then reopens the untouched source and checks
the same query again. The probe is covered by an integration test and does not
touch any EverOS production or staging path.

That proves the mechanics of the probe on the accepted local engine. It does
not prove that a different engine version can read/write the existing Lance
format, preserve every index, or be rolled back after it has written new
versions. A target-version install, multi-table corpus soak, concurrent
reader/writer test, and explicit rollback proof are still missing. The
assessment helper therefore refuses a migration verdict until all of those
gates are supplied.

The current EverOS safeguards remain required for either future trial:

1. snapshot and checksum the source directory while the writer is quiescent;
2. migrate only an isolated copy;
3. compare all business schemas, row counts, representative exact/FTS/vector
   results, index statistics, and cascade rebuild output;
4. run concurrent read/write soak and failure injection;
5. prove the original snapshot remains readable and restore it without
   changing production pointers;
6. only then prepare a separately reviewed runtime dependency change.

No dependency pin, production index, schema, release pointer, or LanceDB
engine was changed by this work. Storage-engine migration stays a separate
rollback domain from the remaining knowledge-access features.
