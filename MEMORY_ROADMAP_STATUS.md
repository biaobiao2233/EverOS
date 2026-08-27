# Memory roadmap candidate status

Updated: 2026-08-27

## Status

- Exact reviewed candidate: `913e3452c1cc4bc76fdeb41f7d6c7f44155927c5`
- Exact-source tag: `memory-roadmap-candidate-20260826`
- Base: accepted Stage 3 source `0ffd67c9418e34c3dc1ea5156ad4c9c5c6456f50`
- Review state: candidate review passed / rollout-ready source candidate
- Production state: **not rolled out**

This branch may have documentation-only commits above the exact candidate. Use the tag when you need the exact reviewed source boundary.

## Candidate capabilities

### Truth-aware retrieval

Retrieval applies scope, authority, temporal, supersession, and contradiction/conflict gates before later ranking/prompt stages. The intent is to avoid a high-similarity but stale or lower-authority memory outranking the current accepted truth.

### Deterministic Memory Wiki

The candidate compiles source-backed accepted claims into a deterministic representation so human Markdown and agent-facing structured output share the same claim set. Unsupported/stale claims are excluded rather than rendered as if accepted.

### Promotion scoring

Dreaming/promotion output is a recommendation layer with evidence and contradiction/correction penalties. It has no automatic authority to mutate accepted truth.

### Case → Skill lifecycle

- repeated-success threshold before promotion;
- durable candidate sidecar;
- explicit review/accept step;
- collision, secret, and private-path guards;
- supersession/retirement handling;
- no first-success auto-creation of an accepted `SKILL.md`.

## Boundary

This is intentionally a candidate branch, not a release announcement. It should be re-evaluated only after the accepted production runtime is converged onto the current upstream baseline. The convergence plan lives on `main` at [docs/UPSTREAM_CONVERGENCE.md](https://github.com/biaobiao2233/EverOS/blob/main/docs/UPSTREAM_CONVERGENCE.md).
