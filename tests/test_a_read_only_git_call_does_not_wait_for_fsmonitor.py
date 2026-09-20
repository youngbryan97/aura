"""Every read-only git call, and the daemon that makes it take thirty seconds.

`git status --porcelain --untracked-files=all` takes 0.06s on this checkout
and 19.6s through the fsmonitor daemon, which is running. `reqproof capture`
bounds its git probes at 30 seconds and wedged there — twelve proofs refused
by the clock rather than by anything they measured. The flag costs nothing
when no daemon is running and is the difference between a bounded read and a
wedged one when there is.

Read-only only: a write (`add`, `commit`, `checkout`) wants the daemon's index
cache and is not bounded the same way.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = (
    "tools", "core", "interface", "skills", "security", "llm", "executors",
    # Tests too. One `git ls-files artifacts` in a test wedged a 44-file
    # run for twenty minutes against no CPU, which is the same cost as the
    # production one and lands on whoever runs the suite.
    "tests",
)

#: Subcommands that only read. A write is left alone deliberately.
READ_ONLY = frozenset({
    "status", "rev-parse", "ls-files", "log", "diff", "show", "grep",
    "cat-file", "symbolic-ref", "describe", "for-each-ref", "merge-base",
    "name-rev", "rev-list", "ls-tree", "branch", "remote", "shortlog",
    "diff-tree", "diff-index", "check-ignore", "var", "count-objects",
})

#: Options that take a value, so the subcommand is not the token after them.
_TAKES_A_VALUE = frozenset({"-C", "-c", "--git-dir", "--work-tree"})


def _argv(node: ast.AST) -> list[str | None] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    return [
        item.value if isinstance(item, ast.Constant) and isinstance(item.value, str) else None
        for item in node.elts
    ]


def _subcommand(argv: list[str | None]) -> str | None:
    skip = 0
    for token in argv[1:]:
        if skip:
            skip -= 1
            continue
        if token in _TAKES_A_VALUE:
            skip = 1
            continue
        if token and not token.startswith("-"):
            return token
    return None


def _read_only_git_calls(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):  # pragma: no cover - unreadable file
        return []
    found = []
    for node in ast.walk(tree):
        argv = _argv(node)
        if not argv or not argv[0] or argv[0].rsplit("/", 1)[-1] != "git":
            continue
        subcommand = _subcommand(argv)
        if subcommand in READ_ONLY and "core.fsmonitor=false" not in argv:
            found.append((node.lineno, subcommand))
    return found


def test_no_read_only_git_call_waits_for_the_daemon() -> None:
    waiting = [
        f"{path.relative_to(ROOT)}:{line}: git {subcommand}"
        for root in SCANNED
        for path in sorted((ROOT / root).rglob("*.py"))
        if "__pycache__" not in str(path)
        for line, subcommand in _read_only_git_calls(path)
    ]

    assert waiting == [], (
        f"{len(waiting)} read-only git call(s) can block on the fsmonitor "
        'daemon; pass "-c", "core.fsmonitor=false" right after "git":\n'
        + "\n".join(waiting[:40])
    )


def test_the_detector_can_see_one() -> None:
    """The null: a gate that cannot match reports green forever."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        sample = Path(folder) / "sample.py"
        sample.write_text('run(["git", "status", "--porcelain"])\n')
        assert _read_only_git_calls(sample) == [(1, "status")]

        sample.write_text(
            'run(["git", "-c", "core.fsmonitor=false", "status", "--porcelain"])\n'
        )
        assert _read_only_git_calls(sample) == []


def test_a_write_is_left_alone() -> None:
    """A commit wants the daemon's index cache; only reads are covered."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        sample = Path(folder) / "sample.py"
        sample.write_text('run(["git", "commit", "-m", "x"])\n')
        assert _read_only_git_calls(sample) == []


def test_a_repository_named_with_its_own_path_still_finds_its_subcommand() -> None:
    """`-C <path>` takes a value, so the subcommand is two tokens later."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        sample = Path(folder) / "sample.py"
        sample.write_text('run(["git", "-C", "/somewhere", "rev-parse", "HEAD"])\n')
        assert _read_only_git_calls(sample) == [(1, "rev-parse")]
