#!/usr/bin/env python3
"""Strict mypy on the files this branch touched.

``make typecheck`` runs strict mypy against ``config/mypy_strict_files.txt``,
which holds 78 of the repository's 6,615 Python files. Everything not on that
list is configured strictly and checked never. The list only grows, which is
the right shape, but nothing makes it grow — so it grows when somebody
remembers.

This makes it grow by default, as a ratchet rather than a wall. A file this
branch changed may not have MORE type errors than the same file at the merge
base, and a file this branch ADDED must have none. A file that reaches zero is
added to the allowlist in the same run.

The difference matters. "Every touched file must pass" sounds stronger and is
weaker: a one-line change — converting a raw lock to a checked one, say, across
twenty-six modules — would demand typing twenty-six modules that the change
never looked at, and a rule with that cost is a rule people route around. "No
worse than before, and new code is clean" is enforceable on every branch, which
is what closes the surface.

    python tools/typecheck_changed.py                # check what changed
    python tools/typecheck_changed.py --adopt        # and record what reaches zero
    python tools/typecheck_changed.py --base main
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT / "config" / "mypy_strict_files.txt"

PRODUCTION_ROOTS = (
    "core/",
    "interface/",
    "skills/",
    "security/",
    "llm/",
    "executors/",
    "infrastructure/",
    "tools/",
)
MYPY_FLAGS = ("--follow-imports=skip", "--explicit-package-bases")


def _git(*args: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *args], capture_output=True, text=True, cwd=ROOT, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def changed_files(base: str) -> list[str]:
    """Production Python files this branch changed, relative to ``base``."""
    merge_base = _git("merge-base", "HEAD", base) or base
    names = _git("diff", "--name-only", "--diff-filter=ACMR", merge_base, "HEAD")
    staged = _git("diff", "--name-only", "--diff-filter=ACMR", "--cached")
    working = _git("diff", "--name-only", "--diff-filter=ACMR")
    candidates = set(filter(None, (names + "\n" + staged + "\n" + working).splitlines()))
    return sorted(
        name
        for name in candidates
        if name.endswith(".py")
        and name.startswith(PRODUCTION_ROOTS)
        and (ROOT / name).exists()
    )


def allowlisted() -> set[str]:
    if not ALLOWLIST.exists():
        return set()
    return {
        line.strip()
        for line in ALLOWLIST.read_text("utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def run_mypy(paths: list[str]) -> tuple[bool, str]:
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "mypy", *MYPY_FLAGS, *paths],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


_ERROR_LINE = re.compile(r"^(?P<path>[^:]+):\d+: error: ", re.MULTILINE)


def error_count(output: str) -> int:
    return len(_ERROR_LINE.findall(output))


def errors_at(revision: str, relative: str) -> int | None:
    """How many errors the same file had at ``revision``, or None if absent.

    The file is checked out into a mirror of its own path so mypy resolves the
    same module name; checking it at a flat temporary path changes the module
    and, with it, the errors.
    """
    shown = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "show", f"{revision}:{relative}"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if shown.returncode != 0:
        return None
    with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
        mirrored = Path(scratch) / relative
        mirrored.parent.mkdir(parents=True, exist_ok=True)
        mirrored.write_text(shown.stdout, encoding="utf-8")
        _ok, output = run_mypy([str(mirrored.relative_to(ROOT))])
    return error_count(output)


def adopt(passing: list[str]) -> None:
    """Record the files that reached zero, if the whole list still passes.

    Clean on its own is not clean in company. `make typecheck` runs mypy over
    every allowlisted file in ONE invocation, and with
    `--follow-imports=skip` a module that another listed file imports resolves
    differently there than it does alone — `core/brain/lane_admission.py` was
    adopted as clean and then failed the combined run on a value that arrives
    as Any only when its callee is visible. So the combined run is the test,
    and an addition that breaks it is taken back out.
    """
    if not passing:
        return
    existing = allowlisted()
    fresh = [p for p in passing if p not in existing]
    if not fresh:
        return

    original = ALLOWLIST.read_text("utf-8") if ALLOWLIST.exists() else ""
    with ALLOWLIST.open("a", encoding="utf-8") as handle:
        handle.write("\n# Adopted by tools/typecheck_changed.py\n")
        for path in fresh:
            handle.write(f"{path}\n")

    combined_ok, output = run_mypy(sorted(allowlisted()))
    if not combined_ok:
        ALLOWLIST.write_text(original, encoding="utf-8")
        print(
            f"not adopting {len(fresh)} file(s): clean alone, and the combined "
            "run fails with them in. Fix the combined errors first:"
        )
        print("\n".join(f"   {line}" for line in output.splitlines()[:8]))
        return
    print(f"adopted {len(fresh)} file(s) into {ALLOWLIST.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--adopt", action="store_true")
    parser.add_argument("--files", nargs="*", default=None)
    args = parser.parse_args(argv)

    files = args.files if args.files is not None else changed_files(args.base)
    if not files:
        print("✅ no production Python files changed")
        return 0

    known = allowlisted()
    merge_base = _git("merge-base", "HEAD", args.base).strip() or args.base
    print(f"strict mypy over {len(files)} changed file(s), against {merge_base[:12]}")

    failures: list[str] = []
    passing: list[str] = []
    for path in files:
        ok, output = run_mypy([path])
        now = error_count(output)
        if ok and now == 0:
            passing.append(path)
            continue
        was = errors_at(merge_base, path)
        if was is None:
            failures.append(f"{path}: NEW file with {now} error(s)")
        elif now > was:
            failures.append(f"{path}: {was} -> {now} error(s)")
        else:
            continue
        marker = "already on the allowlist" if path in known else "not yet typed"
        print(f"\n❌ {path} ({marker})")
        print("\n".join(f"   {line}" for line in output.splitlines()[:12]))

    if args.adopt:
        adopt(passing)

    if failures:
        print(
            f"\n{len(failures)} of {len(files)} changed file(s) got worse, or are "
            "new and not clean:"
        )
        for line in failures:
            print(f"   • {line}")
        print(
            "\nA file you touched may not gain a type error, and a file you added "
            "may not have one."
        )
        return 1

    clean = len(passing)
    print(
        f"✅ {len(files)} changed file(s): {clean} clean, "
        f"{len(files) - clean} unchanged in error count"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
