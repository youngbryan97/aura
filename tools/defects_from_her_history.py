#!/usr/bin/env python3
"""Defects with known answers, from her own history: fix commits whose tests fail before them and pass after.

For each commit on main whose subject starts "fix" and which changed both
source and tests, the tests it added or changed are run on a copy of its
parent with the commit's test files put in. Kept are the commits where those
tests fail there, and pass once the commit's source files are put in as well:
a defect, the files its fix touched, and the change a person made. Each is a
case for tools/repair_by_selection.py --at PARENT --tests-from COMMIT.

Usage:
    defects_from_her_history.py --since DATE --out FILE [--minutes M]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIT = ["git", "-C", str(ROOT), "-c", "core.fsmonitor=false"]
_RESULT = re.compile(r"^(PASSED|FAILED|ERROR)\s+(\S+)", re.MULTILINE)


def git(*args: str) -> str:
    return subprocess.run([*GIT, *args], check=True, capture_output=True, text=True).stdout


def changed_tests(commit: str, path: str) -> list[str]:
    """Top-level test functions the commit added or changed in one test file, as pytest node ids."""
    diff = git("diff", "-U0", f"{commit}^", commit, "--", path)
    after = git("show", f"{commit}:{path}").splitlines()
    starts = [(i + 1, m.group(1)) for i, line in enumerate(after) if (m := re.match(r"(?:async )?def (test_\w+)", line))]
    names = set()
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.MULTILINE):
        first, count = int(m.group(1)), int(m.group(2) or 1)
        for line in range(first, first + max(count, 1)):
            owner = [name for start, name in starts if start <= line]
            if owner:
                names.add(owner[-1])
    return [f"{path}::{n}" for n in sorted(names)]


def run(copy: Path, tests: list[str], seconds: float) -> dict[str, bool]:
    env = {k: v for k, v in os.environ.items() if k != "AURA_TESTING"}
    env.update(AURA_STATE_ROOT=str(copy.parent / "state"), AURA_LOG_DIR=str(copy.parent / "logs"),
               PYTHONDONTWRITEBYTECODE="1", VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1")
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", *tests],
                             cwd=copy, env=env, capture_output=True, text=True, timeout=seconds).stdout
    except subprocess.TimeoutExpired:
        return {}
    seen = {node: word == "PASSED" for word, node in _RESULT.findall(out)}
    return {t: seen.get(t, False) for t in tests}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minutes", type=float, required=True, help="the whole harvest's budget")
    args = parser.parse_args()
    deadline = time.time() + args.minutes * 60
    commits = git("log", "--no-merges", "--format=%H %s", f"--since={args.since}", "origin/main").splitlines()
    where = Path(tempfile.mkdtemp(prefix="defects-from-history-"))
    copy = where / "copy"
    kept, looked = [], 0
    try:
        for row in commits:
            if time.time() >= deadline:
                break
            commit, subject = row.split(" ", 1)
            if not subject.lower().startswith("fix"):
                continue
            names = [line.split("\t") for line in git("diff-tree", "--no-commit-id", "--name-status", "-r",
                                                       commit).splitlines()]
            sources = [p for s, p, *_ in names if p.endswith(".py") and not p.startswith("tests/") and s == "M"]
            test_files = [p for s, p, *_ in names if p.startswith("tests/") and p.endswith(".py") and s in "AM"]
            if not sources or not test_files:
                continue
            tests = [t for f in test_files for t in changed_tests(commit, f)]
            if not tests:
                continue
            looked += 1
            subprocess.run([*GIT, "worktree", "add", "--detach", "--force", str(copy), f"{commit}^"],
                           check=True, capture_output=True)
            try:
                for f in test_files:
                    (copy / f).parent.mkdir(parents=True, exist_ok=True)
                    (copy / f).write_text(git("show", f"{commit}:{f}"), encoding="utf-8")
                before = run(copy, tests, max(1.0, deadline - time.time()))
                failing = [t for t, ok in before.items() if not ok]
                if not failing:
                    continue
                for f in sources:
                    (copy / f).write_text(git("show", f"{commit}:{f}"), encoding="utf-8")
                after = run(copy, failing, max(1.0, deadline - time.time()))
                mended = [t for t in failing if after.get(t)]
                if mended:
                    kept.append({"commit": commit, "parent": git("rev-parse", f"{commit}^").strip(),
                                 "subject": subject, "files": sources, "test_files": test_files, "tests": mended})
                    print(json.dumps(kept[-1]), flush=True)
            finally:
                subprocess.run([*GIT, "worktree", "remove", "--force", str(copy)], capture_output=True)
    finally:
        args.out.write_text(json.dumps({"since": args.since, "fix_commits_with_tests": looked, "defects": kept},
                                       indent=1))
    print(json.dumps({"looked": looked, "kept": len(kept), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
