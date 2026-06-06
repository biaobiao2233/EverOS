> ⚠️ **This is a personal archive fork, not an active contribution.**
>
> The fix in branch `fix/api-key-rotator-and-profile-scene` was developed
> for the **pre-rewrite** `EverMind-AI/EverOS` codebase (commits up to
> `773e19b`, before the upstream force-push that rewrote the architecture
> in mid-2026). After the rewrite, the `main` branch here was automatically
> resynced to the new upstream `ab23e40` (EverOS 1.0.0), which no longer
> contains the code paths the fix targets.
>
> **Status**: No PR is open. The fix is local-only and runs on the
> author's VPS. See branch `fix/api-key-rotator-and-profile-scene` for
> the actual code changes.
>
> Created: 2026-06-06.

---

# biaobiao2233 / EverOS (archive fork)

A personal fork of [`EverMind-AI/EverOS`](https://github.com/EverMind-AI/EverOS)
maintained only to preserve a single bug-fix branch developed before an
upstream rewrite.

## Branches

| Branch | Base | What it is |
|--------|------|------------|
| `main` | `EverMind-AI/EverOS@ab23e40` | Mirror of upstream main, post-rewrite |
| `fix/api-key-rotator-and-profile-scene` | `0f14d05` (pre-rewrite) | Two bug fixes for the old `methods/EverCore/...` architecture |

## What's in the fix branch

`fix/api-key-rotator-and-profile-scene` (single commit `6a596dd`) targets
two bugs in the pre-rewrite `EverMind-AI/EverOS` architecture (the version
where the application lived under `methods/EverCore/` and used a
`ClassVar`-based `ApiKeyRotator`):

1. **Bug 1 — `ApiKeyRotator` singleton locks first key set forever.**
   The first caller's key tuple would win forever; subsequent callers
   with different keys silently got the first instance. Forced an
   `.env` workaround (`OPENAI_*` mirrored to `AGNES_*`) to make keys
   coincide.
2. **Bug 2 — `LlmCustomSetting.profile` silently dropped by Pydantic.**
   `EXTRACT_SCENES` contained `('boundary', 'extraction', 'profile')`
   but the DTO/Model only declared the first two. Pydantic's
   `model_dump()` / `from_any` stripped unknown fields on round-trip.

Both fixes were verified on a live VPS deployment (Singapore, port 1995)
end-to-end (add → flush → new episode persisted by Gemini-3.1-pro-preview).

## Why no PR is open

The upstream repo was force-pushed between PR #252 and #256. The
pre-rewrite code path the fix targets (`methods/EverCore/src/memory_layer/llm/api_key_rotator.py`,
`LlmCustomSetting` DTO, etc.) **no longer exists in upstream `main`**.
The rewrite implicitly resolved both bugs by removing the problematic
abstractions:

- The new `OpenAIProvider` in `src/everos/component/llm/` has no
  `ApiKeyRotator` at all; the docstring even says "no multi-key
  rotation, no scenario-level routing, no token-usage collector —
  those are deployment concerns layered on top."
- The new settings live in TOML (`src/everos/config/settings.py`),
  not a Pydantic DTO+DB-backed model, so there's no Pydantic-stripping
  of unknown fields.

GitHub refused the PR with `422: branch has no history in common with
EverMind-AI:main`, which is consistent with the force-push.

## Authorship

Branch was authored by `HylaruCoder <twocucao@gmail.com>` and pushed via
the `biaobiao2233` GitHub account (Personal Access Token stored in the
author's private Memo vault).
