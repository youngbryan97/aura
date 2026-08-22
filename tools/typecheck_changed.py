#!/usr/bin/env python3
"""Strict mypy on the files this branch touched.

``make typecheck`` runs strict mypy against ``config/mypy_strict_files.txt``,
which holds 78 of the repository's 6,615 Python files. Everything not on that
list is configured strictly and checked never. The list only grows, which is
the right shape, but nothing makes it grow — so it grows when somebody
remembers.

This makes it grow by default. Every production file a branch changes must
pass strict mypy, and a file that passes is added to the allowlist in the same
run. Touch a file, type a file. A million lines do not have to convert
overnight for that rule to close the surface a commit at a time.

    python tools/typecheck_changed.py                # check what changed
    python tools/typecheck_changed.py --adopt        # and record what passes
    python tools/typecheck_changed.py --base main
"""

from __future__ import annotations

import argparse
import subprocess
import sys
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


def adopt(passing: list[str]) -> None:
    """Record the newly clean files. The list only ever grows."""
    if not passing:
        return
    existing = allowlisted()
    fresh = [p for p in passing if p not in existing]
    if not fresh:
        return
    with ALLOWLIST.open("a", encoding="utf-8") as handle:
        handle.write("\n# Adopted by tools/typecheck_changed.py\n")
        for path in fresh:
            handle.write(f"{path}\n")
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
    print(f"strict mypy over {len(files)} changed file(s)")

    failures: list[str] = []
    passing: list[str] = []
    for path in files:
        ok, output = run_mypy([path])
        if ok:
            passing.append(path)
            continue
        failures.append(path)
        marker = "already on the allowlist" if path in known else "not yet typed"
        print(f"\n❌ {path} ({marker})")
        print("\n".join(f"   {line}" for line in output.splitlines()[:15]))

    if args.adopt:
        adopt(passing)

    if failures:
        print(
            f"\n{len(failures)} of {len(files)} changed file(s) do not pass strict "
            "mypy. A file you touched is a file you can type."
        )
        return 1

    print(f"✅ all {len(files)} changed file(s) pass strict mypy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
