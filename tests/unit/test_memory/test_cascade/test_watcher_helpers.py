"""Unit tests for the pure helpers in :mod:`everos.memory.cascade.watcher`.

The :class:`CascadeWatcher` itself needs a running event loop + real
filesystem to test end-to-end (see ``tests/integration/``). The pure
helpers can be exercised in isolation.
"""

from __future__ import annotations

from pathlib import Path

from everos.memory.cascade.watcher import (
    _is_app_root,
    _relative_to_root,
    _safe_mtime,
    _watch_roots,
)


def test_relative_to_root_within(tmp_path: Path) -> None:
    target = tmp_path / "users" / "u1" / "x.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x")
    assert _relative_to_root(tmp_path, str(target)) == "users/u1/x.md"


def test_relative_to_root_outside(tmp_path: Path) -> None:
    """A path outside the memory root returns ``None``."""
    outside = tmp_path.parent / "completely-different" / "y.md"
    assert _relative_to_root(tmp_path, str(outside)) is None


def test_safe_mtime_missing_path_returns_zero(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.md"
    assert _safe_mtime(str(missing)) == 0.0


def test_safe_mtime_existing_path_returns_positive(tmp_path: Path) -> None:
    f = tmp_path / "f.md"
    f.write_text("ok")
    assert _safe_mtime(str(f)) > 0


def test_watch_roots_excludes_hidden_index_and_legacy_lancedb(tmp_path: Path) -> None:
    for name in ("codex", "claude", ".index", ".tmp", "lancedb"):
        (tmp_path / name).mkdir()
    (tmp_path / "not-a-directory").write_text("x")
    assert [path.name for path in _watch_roots(tmp_path)] == ["claude", "codex"]


def test_is_app_root_accepts_only_allowed_direct_children(tmp_path: Path) -> None:
    assert _is_app_root(tmp_path, tmp_path / "codex")
    assert not _is_app_root(tmp_path, tmp_path / ".index")
    assert not _is_app_root(tmp_path, tmp_path / "lancedb")
    assert not _is_app_root(tmp_path, tmp_path / "codex" / "nested")
