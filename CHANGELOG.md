# Changelog

All notable changes to **EverOS** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Optional durable `operation_id` receipts and a content-free operation-status
  endpoint for safely retrying `/memory/add` and `/memory/flush`.
- Cross-process scoped session locks and atomic MemCell/buffer/checkpoint
  commits.
- Append-once Markdown persistence keyed by source MemCell, with corruption
  checks and replay repair.
- Optional bearer authentication backed by an environment value or token file.
- A sandboxed local official Antigravity `agy` CLI text provider alongside the
  existing OpenAI-compatible provider.
- Evidence-focused Atomic Fact and Agent Skill prompts, a 12-fact hard cap, and
  a pre-persistence skill safety gate.

### Changed

- OME starts APScheduler paused, uses stable engine identity, safely rebinds
  persisted jobs, and recovers orphaned runs before resuming.
- Cascade excludes index data from recursive watches, avoids re-enqueuing an
  unchanged file version, and no longer manufactures idle optimize work.
- Markdown writes now reject resolved paths outside the configured memory root.

See [PRODUCTION_OPTIMIZED.md](PRODUCTION_OPTIMIZED.md) for the complete
upstream comparison and deployment boundaries.

## [1.0.0] - 2026-06-03

First public release of EverOS — a Markdown-first memory extraction framework
for AI agents.

### Added

- **Markdown as source of truth** — all memory persists as plain `.md` files you
  can open, edit, grep, and version with Git.
- **Lightweight three-piece storage** — Markdown (truth) + SQLite (state / queue
  / audit) + LanceDB (vector + BM25 + scalar index). No external services
  required.
- **Hybrid retrieval** — BM25, vector, and scalar filtering in a single LanceDB
  query.
- **Cascade index sync** — editing a `.md` file triggers a file watcher →
  entry-level diff → sub-second LanceDB sync.
- **Dual-track memory** — user-track (Episodes / Profiles) and agent-track
  (Cases / Skills).
- **Multi-source extraction** — conversations, workflows, agent traces, and file
  knowledge.
- **CLI + HTTP API** — the `everos` command-line tool and a FastAPI server,
  async-first throughout.
- **Pluggable providers** — LLM / embedding / rerank via the OpenAI-compatible
  protocol (works with OpenAI, OpenRouter, vLLM, Ollama, …).
- **Decoupled algorithms** — memory extraction algorithms live in the standalone
  `everalgo-*` libraries published on PyPI.

[Unreleased]: https://github.com/EverMind-AI/EverOS/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/EverMind-AI/EverOS/releases/tag/v1.0.0
