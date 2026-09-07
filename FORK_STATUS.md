# biaobiao2233/EverOS fork status

Updated: 2026-09-07

This fork intentionally keeps **upstream tracking**, **current production-derived code**, and **future memory candidates** on separate branches. They are related, but they are not the same source baseline and must not be presented as one merged release.

## Branch map

| Branch | Meaning | Current anchor | Release meaning |
|---|---|---|---|
| `main` | Current upstream-tracking line plus fork documentation | upstream `e8612b9` (2026-09-07; official `v1.3.0`) | Best branch for following current official EverOS |
| `production-v2` | Current accepted production source lineage | source `563ab9f` | Stage 3 → Wiki production → ingest identity repair → Cascade health repair; based on older upstream `8f175d3`, not claimed as `v1.3.0`-compatible |
| `memory-roadmap-candidate` | Next memory-quality candidate built on `production-v2` | `913e345` | Candidate/reviewed source only; **not production** |
| `production-optimized` | First-generation production hardening published in July 2026 | `3ee75b5` | Historical/legacy comparison branch |

## What `production-v2` adds

`production-v2` carries the production-hardening lineage plus later accepted Stage 1/2/3, Wiki, ingest-repair, and Cascade-health work. The current source lineage is `0ffd67c → 638dcd6 → 64c0ff7 → 563ab9f`. The most important Stage 3 contract is **deferred ingest with explicit publish authority**:

```text
stage
  ↓
pending_publish
  ↓ explicit publish authority
published
  ↓ boundary / flush
consumed
```

Key properties include:

- stable caller-owned message identity across re-batching;
- revision-aware supersession and stale-revision rejection;
- same-revision payload conflicts fail closed;
- unpublished staged messages stay out of extraction and search;
- consumed receipts make same-revision late replay a no-op;
- operation/receipt metadata is content-free rather than a second copy of raw conversation text.

The branch also includes the deterministic Wiki-only production surface, the reviewed Publish-identity/recovery repair, and the Cascade idle-prune health repair. Production was rolled forward with bounded reviewed file sets, so the branch represents the accepted source lineage rather than a byte-for-byte server filesystem snapshot. Runtime `.env` configuration remains outside Git.

## What `memory-roadmap-candidate` adds

This branch starts from the accepted Stage 3 source and experiments with the next memory-quality layer:

- truth-aware hybrid retrieval with authority/temporal/conflict gates before ranking;
- deterministic Memory Wiki compilation from accepted, source-backed claims;
- evidence-aware promotion scoring that does not silently mutate authority;
- Case → Skill lifecycle with repeated-success thresholds and explicit review/accept;
- collision/secret/private-path guards and retirement/supersession handling.

It is intentionally labeled **candidate**. Publishing the source does not mean it is running in production.

## Why these branches are not merged yet

The accepted production line and current upstream line have an old common base (`8f175d3`). Upstream has since progressed through the 1.2.x releases to `v1.3.0`, including substantial persistence/index refactoring, optional Milvus derived indexes, dependency fixes, security/reliability changes, and broader integration coverage. A blind merge/cherry-pick would make it very easy to reintroduce fixed defects or claim compatibility that has not actually been tested.

The safe convergence rule is therefore:

1. keep `main` current with upstream;
2. preserve the exact accepted production source separately;
3. classify each production-only capability as **already upstream / still needed / needs redesign**;
4. port only the still-needed deltas onto the current upstream baseline;
5. run focused + full regression gates and independent review;
6. only then create a new production candidate.

The working plan is documented in [docs/UPSTREAM_CONVERGENCE.md](docs/UPSTREAM_CONVERGENCE.md).

## Project Continuity and EverOS Control Center

- [Project Continuity](https://github.com/biaobiao2233/project-continuity) keeps per-project current state, handoff, ownership, and acceptance authority.
- EverOS remains the derived historical-memory layer; cross-project clues can be discovered, but `PASS`, authorization, ownership, and production state do not propagate automatically.
- [EverOS Control Center](https://github.com/biaobiao2233/everos-control-center) now includes the deterministic Wiki view plus multi-source sync/run-history recovery fixes. **Sync remains the most mature day-to-day workflow**; memory reading/search/pipeline/history surfaces are still evolving.
