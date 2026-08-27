# Production v2 status

Updated: 2026-08-27

## Exact accepted source

- Accepted Stage 3 runtime commit: `0ffd67c9418e34c3dc1ea5156ad4c9c5c6456f50`
- Exact-source tag: `production-v2-stage3-20260826`
- Historical upstream base: `8f175d3f8f222fc1a26c943402ce8bb4ea877a1d`
- Relationship to upstream 1.2.x: **not yet converged / not yet claimed compatible**

The branch may contain documentation commits after the exact runtime snapshot. Use the tag above when you need the byte-for-byte Git source boundary that was accepted for the Stage 3 runtime.

## Stage 3 contract

```text
stage
  ↓
pending_publish
  ↓ explicit publish authority
published
  ↓ boundary / flush
consumed
```

The accepted source provides:

- stable caller-owned identity for staged messages;
- revision-aware supersession;
- stale-revision no-op/rejection and same-revision payload conflict detection;
- explicit publish authority for an exact revision;
- exclusion of pending messages from extraction and search;
- durable consumed receipts so late same-revision replay does not re-enter extraction;
- content-free operation/receipt metadata rather than a second raw-message ledger;
- backward-compatible handling for the older ingest path where retained by this source line.

## Earlier production-hardening lineage

This line also carries earlier work around durable idempotency, crash recovery, append-once Markdown persistence, safer extraction, OME/Cascade reliability, LanceDB maintenance, path containment, and deployment/API safety boundaries. Historical details are in [PRODUCTION_OPTIMIZED.md](PRODUCTION_OPTIMIZED.md).

## What this branch does not claim

- It is not upstream EverOS 1.2.3 with a small patch set.
- It does not claim the old production hardening is still needed unchanged on current upstream.
- It does not include private deployment credentials, personal memory content, server addresses, or runtime databases.
- It does not automatically include the later `memory-roadmap-candidate`; that work remains a separate candidate branch.

For the safe forward-port plan, see `main` → [docs/UPSTREAM_CONVERGENCE.md](https://github.com/biaobiao2233/EverOS/blob/main/docs/UPSTREAM_CONVERGENCE.md).
