#!/usr/bin/env python3
"""Prove a checkpoint exists on pushed main with no local-only obligation."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_OPERATION_MARKERS = (
    "BISECT_LOG",
    "CHERRY_PICK_HEAD",
    "MERGE_HEAD",
    "REVERT_HEAD",
    "index.lock",
    "rebase-apply",
    "rebase-merge",
    "sequencer",
)


def _git(root: Path, *arguments: str) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 126, "", f"{type(exc).__name__}: {exc}"
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _required_git_value(
    root: Path,
    issues: list[str],
    label: str,
    *arguments: str,
) -> str:
    returncode, stdout, stderr = _git(root, *arguments)
    if returncode != 0 or not stdout:
        detail = stderr or f"exit {returncode} with no output"
        issues.append(f"cannot resolve {label}: {detail}")
        return ""
    return stdout


def audit(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    issues: list[str] = []
    top_level = _required_git_value(root, issues, "repository root", "rev-parse", "--show-toplevel")
    head = _required_git_value(root, issues, "HEAD", "rev-parse", "HEAD")
    origin_main = _required_git_value(
        root,
        issues,
        "origin/main",
        "rev-parse",
        "refs/remotes/origin/main",
    )
    branch = _required_git_value(
        root,
        issues,
        "current branch",
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
    )
    upstream = _required_git_value(
        root,
        issues,
        "upstream",
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    )
    git_dir_text = _required_git_value(
        root,
        issues,
        "worktree git directory",
        "rev-parse",
        "--path-format=absolute",
        "--git-dir",
    )
    common_dir_text = _required_git_value(
        root,
        issues,
        "common git directory",
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
    )

    if top_level and Path(top_level).resolve() != root:
        issues.append(f"audit root {root} is not git top-level {top_level}")
    if head and not _SHA_RE.fullmatch(head):
        issues.append("HEAD is not a full commit identity")
    if origin_main and not _SHA_RE.fullmatch(origin_main):
        issues.append("origin/main is not a full commit identity")
    if head and origin_main and head != origin_main:
        _, counts, _ = _git(root, "rev-list", "--left-right", "--count", "HEAD...origin/main")
        issues.append(f"HEAD is not exactly pushed origin/main (ahead/behind={counts or 'unknown'})")
    if upstream and upstream != "origin/main":
        issues.append(f"checkpoint upstream is {upstream!r}, expected 'origin/main'")

    status_code, status, status_error = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status_code != 0:
        issues.append(f"cannot inspect worktree status: {status_error or status_code}")
    elif status:
        entries = [line for line in status.splitlines() if line.strip()]
        issues.append(f"worktree has {len(entries)} tracked or untracked obligation(s)")

    diff_code, diff_output, diff_error = _git(root, "diff", "--check", "HEAD")
    if diff_code != 0 or diff_output:
        issues.append(f"git diff --check failed: {diff_output or diff_error or diff_code}")

    marker_roots = {
        Path(path).resolve()
        for path in (git_dir_text, common_dir_text)
        if path
    }
    active_markers = sorted(
        {
            str((marker_root / marker).relative_to(marker_root))
            for marker_root in marker_roots
            for marker in _OPERATION_MARKERS
            if (marker_root / marker).exists()
        }
    )
    if active_markers:
        issues.append(f"git operation or lock is still active: {active_markers!r}")

    return {
        "schema": "aura.closeout.checkpoint_hygiene.v1",
        "passed": not issues,
        "root": str(root),
        "branch": branch,
        "upstream": upstream,
        "head": head,
        "origin_main": origin_main,
        "clean": status_code == 0 and not status,
        "head_is_pushed_main": bool(head and head == origin_main),
        "active_git_markers": active_markers,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    report = audit(arguments.root)
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "passed" if report["passed"] else "failed"
        print(
            f"Checkpoint hygiene {status}: branch={report['branch']} "
            f"upstream={report['upstream']} clean={report['clean']}"
        )
        for issue in report["issues"]:
            print(f"- {issue}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
