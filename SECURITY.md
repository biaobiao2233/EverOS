# Security Policy

## Supported Versions

EverOS is in active alpha development. This fork publishes production
hardening on the `production-optimized` branch; upstream versions follow the
upstream project's own support policy.

| Line | Supported here |
|------|----------------|
| `production-optimized` | ✅ |
| Fork `main` / upstream releases | See upstream |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

For a vulnerability introduced by this fork, use GitHub's private
**Security → Advisories → Report a vulnerability** flow when available.
For an upstream vulnerability, follow the
[upstream security policy](https://github.com/EverMind-AI/EverOS/security/policy).
Do not include credentials or private memory content in a public issue.

Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a proof-of-concept
- The affected version / commit
- Any suggested mitigation, if you have one

## Scope & Threat Model

EverOS runs as a **local-first service** for single users or small teams
(Markdown + SQLite + LanceDB on the local filesystem). Please keep the
following in mind:

- The server binds to `127.0.0.1` by default (env `EVEROS_API__HOST`). Keep it
  loopback-only whenever possible. This fork adds an optional shared bearer
  token through `EVEROS_API_TOKEN` or `EVEROS_API_TOKEN_FILE`; once configured,
  every route except `/health` and preflight `OPTIONS` requires it, including
  requests arriving from loopback. This avoids trusting a reverse proxy's
  source address.
- Exposing the HTTP API to an untrusted network still requires TLS, firewall
  rules, rate limiting, and a mature gateway. The shared token is not a
  multi-tenant authorization system.
- Secrets (LLM / embedding API keys) live in your local `.env`; protect that
  file as you would any credential. EverOS never transmits them anywhere except
  the providers you configure.
- The optional `agy_cli` provider relies on the official locally installed and
  authenticated Antigravity CLI. Do not publish its login state or wrap the
  account in a public reverse proxy/token bridge.
- Memory content is stored as plaintext `.md` files; apply OS-level file
  permissions or disk encryption if your data is sensitive.
