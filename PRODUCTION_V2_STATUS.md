# Production v2 status

Updated: 2026-09-07

## Current accepted source lineage

- Latest accepted source commit: `563ab9f19fabb871723478c39b21da1cc4362924`
- Immutable source tag: `production-v2-source-20260830`
- Stage 3 boundary: `0ffd67c9418e34c3dc1ea5156ad4c9c5c6456f50`
- Stage 3 tag: `production-v2-stage3-20260826`
- Historical upstream base: `8f175d3f8f222fc1a26c943402ce8bb4ea877a1d`
- Relationship to upstream `v1.3.0`: **not converged / not claimed compatible**

The accepted lineage is linear:

```text
0ffd67c  Stage 3 deferred ingest + explicit Publish authority
   ↓
638dcd6  production Wiki-only surface
   ↓
64c0ff7  Publish identity / recovery contract repair
   ↓
563ab9f  Cascade idle-prune health repair
```

`production-v2` publishes that reviewed source lineage. It is **not** a byte-for-byte
snapshot of the production server filesystem: production was rolled forward using
reviewed file sets, and runtime configuration such as provider/model selection lives
outside Git.

## What was accepted and rolled out

### Stage 3 ingest

The Stage 3 contract separates durable staging from explicit publication:

```text
stage
  ↓
pending_publish
  ↓ explicit publish authority
published
  ↓ boundary / flush
consumed
```

Key properties include stable caller-owned identity, revision-aware supersession,
same-revision conflict detection, exclusion of unpublished rows from extraction and
search, and durable consumed receipts for idempotent replay.

### Deterministic Memory Wiki

`638dcd6` adds the production Wiki-only surface and deterministic materialization
path without promoting the broader `memory-roadmap-candidate`. The later candidate
branch remains separate and is not implied to be live by this production branch.

### Ingest identity repair

`64c0ff7` repairs the Stage 3 Publish identity/recovery boundary. The source candidate
passed independent review before the bounded production rollout and 3-session pilot.

### Cascade health repair

`563ab9f` fixes false Cascade degradation caused by stale prune-age accounting after
idle periods. Focused worker/health tests passed, the first independent review found
the stale clean→dirty clock edge case, and the repaired candidate subsequently passed
independent re-review before the production file replacement.

## What this branch does not contain or claim

- It does not contain personal conversations, memory databases, credentials, server
  addresses, rollback archives, or production `.env` values.
- It does not claim that the accepted production deltas have been ported to or fully
  tested against upstream EverOS `v1.3.0`.
- It does not include workstation-only EV-INGEST-02/03 upload clients as EverOS server
  runtime code; those are separate local ingest tooling.
- It does not promote `memory-roadmap-candidate` (`913e345`) to production.

Current upstream tracking and the convergence plan live on `main` in
`FORK_STATUS.md` and `docs/UPSTREAM_CONVERGENCE.md`.
