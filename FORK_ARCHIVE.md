# EverOS fork archive

This personal fork preserves one bug-fix branch developed before the upstream
repository rewrote its architecture.

## Branches

| Branch | Base | Purpose |
| --- | --- | --- |
| `main` | Current `EverMind-AI/EverOS` main | Upstream-compatible mirror with this archive notice |
| `fix/api-key-rotator-and-profile-scene` | `0f14d05` (pre-rewrite) | Two fixes for the old `methods/EverCore/...` architecture |

## Preserved fixes

Commit `6a596dd` on `fix/api-key-rotator-and-profile-scene` addresses two
issues in the pre-rewrite codebase:

1. `ApiKeyRotator` behaved as a singleton whose first key tuple remained in
   use for later callers with different keys.
2. `LlmCustomSetting.profile` was not declared in the DTO/model and could be
   dropped during Pydantic serialization.

The fixes were verified against the historical deployment, but the relevant
code paths no longer exist in current upstream `main`.

## Why there is no pull request

Upstream rewrote the repository history and architecture after this branch was
created. GitHub therefore reports no common history between the preserved fix
branch and current upstream `main`, and the old implementation cannot be
merged meaningfully into the new codebase.

The branch remains only as a historical implementation reference.
