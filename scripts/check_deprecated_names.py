"""Block deprecated product names in tracked repository text."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEPRECATED_NAME_RE = re.compile(r"\bever[\s_-]*core\b", flags=re.IGNORECASE)
_ROOT = Path(__file__).resolve().parents[1]
SKIP_SUFFIXES = frozenset(
    {
        ".avif",
        ".bmp",
        ".gif",
        ".heic",
        ".heif",
        ".icns",
        ".ico",
        ".jpeg",
        ".jpg",
        ".mov",
        ".mp4",
        ".png",
        ".webp",
    }
)


@dataclass(frozen=True)
class Violation:
    path: str
    line_number: int
    line: str


def find_violations(files: Iterable[tuple[str, str]]) -> list[Violation]:
    violations: list[Violation] = []
    for path, text in files:
        for line_number, line in enumerate(text.splitlines(), start=1):
            if DEPRECATED_NAME_RE.search(line):
                violations.append(
                    Violation(path=path, line_number=line_number, line=line.strip())
                )
    return violations


def _tracked_paths() -> list[Path]:
    env = os.environ.copy()
    git_control = _ROOT / ".git"
    if git_control.is_file():
        raw = git_control.read_text(encoding="utf-8").strip()
        prefix = "gitdir:"
        if raw.lower().startswith(prefix):
            worktree_gitdir = _resolve_git_path(raw[len(prefix) :].strip())
            commondir_file = worktree_gitdir / "commondir"
            if commondir_file.is_file():
                common_gitdir = (
                    worktree_gitdir / commondir_file.read_text(encoding="utf-8").strip()
                ).resolve()
                if common_gitdir.is_dir():
                    # WSL cannot consume the Windows absolute path stored by
                    # Git for a linked worktree. Supplying the common dir and
                    # worktree index explicitly keeps this read-only checker
                    # portable without rewriting Git metadata.
                    env.update(
                        {
                            "GIT_DIR": str(common_gitdir),
                            "GIT_WORK_TREE": str(_ROOT),
                            "GIT_INDEX_FILE": str(worktree_gitdir / "index"),
                        }
                    )
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
        text=False,
        cwd=_ROOT,
        env=env,
    )
    return [
        _ROOT / Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw
    ]


def _resolve_git_path(raw: str) -> Path:
    """Resolve Git's linked-worktree path on Windows and WSL."""
    candidate = Path(raw)
    if candidate.exists():
        return candidate
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", raw):
        drive = raw[0].lower()
        return Path(f"/mnt/{drive}") / raw[3:].replace("\\", "/")
    return candidate


def _tracked_text_files() -> Iterable[tuple[str, str]]:
    for path in _tracked_paths():
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        yield path.relative_to(_ROOT).as_posix(), text


def main() -> int:
    violations = find_violations(_tracked_text_files())
    if not violations:
        print("Deprecated-name check passed.")
        return 0

    print(
        "Deprecated-name check failed.\n"
        "Use EverOS or EverMind Cloud. Do not use deprecated product naming.\n"
    )
    for violation in violations:
        print(f"- {violation.path}:{violation.line_number}: {violation.line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
