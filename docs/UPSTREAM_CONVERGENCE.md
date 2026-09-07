# Upstream convergence plan

Updated: 2026-09-07

Goal: converge the fork's accepted `production-v2` capabilities onto current upstream EverOS **without** treating an old production branch as if it were already compatible with upstream `v1.3.0`.

## Baselines

- Current upstream anchor: `e8612b9` (2026-09-07; official tag `v1.3.0`).
- Current accepted production source lineage: `0ffd67c → 638dcd6 → 64c0ff7 → 563ab9f`.
- Stage 3 boundary: `0ffd67c`; latest accepted source: `563ab9f`.
- Memory-roadmap candidate: `913e345`, based on Stage 3 and not yet production.
- Historical production-v2 common upstream base: `8f175d3`.

## Port order

### 1. Stage 1 production-hardening classification

Compare the old production hardening against current upstream through `v1.3.0`. Upstream has added substantial Cascade/LanceDB hardening, path-safety work, retry/backoff, background-loop supervision, Agent Skill rescue, health readiness, rebuild/backfill tooling, capability degradation, a derived-index abstraction, and optional Milvus support.

For each old Stage 1 delta, classify it as:

- `DROP_AS_UPSTREAMED` — upstream now provides an equivalent or stronger fix;
- `PORT` — still unique and valid;
- `REDESIGN` — intent remains useful but upstream architecture changed;
- `DO_NOT_PORT` — private deployment behavior or obsolete workaround.

Do not bulk cherry-pick Stage 1.

### 2. Stage 2 EverAlgo/profile compatibility

Re-evaluate the accepted EverAlgo/profile semantics against the versions and contracts used by current upstream. Preserve owner/speaker/recovery invariants, but do not pin an older dependency merely to reproduce the historical branch.

### 3. Stage 3 deferred ingest / publish authority

Port the durable receipt and explicit publish-authority model as a bounded feature slice:

- caller-owned stable message identity;
- revision monotonicity/conflict handling;
- `pending_publish → published → consumed` authority state;
- pending messages excluded from extraction/search;
- consumed late-replay no-op;
- content-free receipt metadata;
- backward-compatible legacy ingest behavior where still supported.

This slice is expected to remain materially unique to the fork and should receive dedicated API/SQLite/integration tests on the new upstream baseline.

### 4. Wiki, ingest-repair, and production-only operational extensions

Reassess the deterministic Wiki surface, the Publish-identity/recovery repair, the Cascade-health repair, and older fork-only additions such as private admin APIs, optional bearer protection, local CLI provider integration, and deployment-specific recovery helpers. Keep generic reusable capability separate from workstation/server-specific configuration.

### 5. Memory roadmap only after runtime convergence

Do not port `memory-roadmap-candidate` first. Truth-aware retrieval, deterministic Wiki, promotion scoring, and Case→Skill lifecycle depend on the runtime/search/storage contracts underneath them. Rebase/reimplement them only after the new upstream-based Stage 3 candidate is stable.

## Gates before any new production claim

- public/private-path and secret scan;
- focused changed-scope tests;
- full useful regression suite on Windows and Linux where applicable;
- schema/openapi/static checks;
- isolated staging with a non-production memory root;
- independent source review;
- explicit rollout authorization;
- production pre-image/rollback and post-rollout verification.

Until those gates pass, `main`, `production-v2`, and `memory-roadmap-candidate` remain separate truthful branches.
