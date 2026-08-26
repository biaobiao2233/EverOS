"""Errors for durable deferred-message identity / authority state."""

from __future__ import annotations


class MemoryMessageConflictError(ValueError):
    """A logical message/revision conflicts with already persisted identity."""


class MemoryMessageRecoveryError(RuntimeError):
    """Durable message receipt and buffer state are internally inconsistent."""
