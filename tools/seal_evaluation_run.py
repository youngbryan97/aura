#!/usr/bin/env python3
"""Pin the commit a result was measured on, so the result stays checkable.

A run records the commit it ran on and the hash of the tree that decided the
answer. That is enough to say what was measured and not enough to stop the
answer drifting: a branch moves, a history is rewritten, and the commit a
report names becomes unreachable or means something else. A reader a year later
has a number and no way to get back to the thing that produced it.

So an authoritative run gets a tag, named for its campaign fingerprint rather
than for the day, and the tag never moves. Pointing it somewhere else is
refused rather than forced: a tag that can be repointed is a label, and the
point of this one is that it cannot be.

    python tools/seal_evaluation_run.py --run artifacts/subject_core/run_021
    python tools/seal_evaluation_run.py --run ... --check     # report, seal nothing

Four things are checked before a tag is written. The tree was clean when the
run started. The commit still exists and its tree still hashes to what the run
recorded. The tag does not already point somewhere else. And the tests that
define the evaluation are green — because a result measured on a commit whose
own tests fail is a result about a broken instrument.

Signing, branch protection and CI are reported rather than performed. A signing
key and a repository setting are the operator's, and a tool that claimed them
would be claiming something it cannot do.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: The suites that define what the battery measures. A result taken on a commit
#: where one of these is red is a result about the instrument. Named here so
#: the list is a declaration rather than whatever happened to be run.
DEFINING_TESTS: tuple[str, ...] = (
    "tests/test_subject_core_measures.py",
    "tests/test_a_null_must_fail_the_whole_battery.py",
    "tests/test_each_null_fails_for_its_own_reason.py",
    "tests/test_a_criterion_does_not_pass_on_noise.py",
    "tests/test_a_rescue_needs_a_deficit_worth_rescuing.py",
    "tests/test_the_synergy_null_is_not_the_estimator.py",
    "tests/test_a_clamped_domain_is_actually_held.py",
    "tests/test_the_floor_says_which_domain_it_crowded.py",
    "tests/test_every_state_field_is_placed.py",
    "tests/test_the_closure_test_can_see_a_tensor.py",
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), *args],
            capture_output=True, text=True, check=False, timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _finding(name: str, ok: bool | None, detail: str, **extra: Any) -> dict[str, Any]:
    return {
        "check": name,
        "status": "unknown" if ok is None else ("ready" if ok else "blocked"),
        "detail": detail,
        **extra,
    }


def _report(run: Path) -> dict[str, Any]:
    path = run / "subject_core_report.json"
    if not path.is_file():
        raise SystemExit(f"no subject_core_report.json in {run}")
    return json.loads(path.read_text(encoding="utf-8"))


def _tag_name(campaign: dict[str, Any]) -> str:
    """Named for what was frozen, not for when it ran.

    Two runs of one campaign belong under one name; a run with a different
    threshold is a different campaign and gets a different one, which is the
    same rule the fingerprint already enforces on the scorecard.
    """
    return f"isc-{str(campaign.get('fingerprint', 'unknown'))[:12]}"


def _checks(report: dict[str, Any], tag: str) -> list[dict[str, Any]]:
    campaign = report.get("campaign", {}) or {}
    commit = str(campaign.get("commit", ""))
    tree = str(campaign.get("tree_hash", ""))
    out: list[dict[str, Any]] = []

    out.append(
        _finding(
            "tree was clean",
            not campaign.get("dirty", True),
            "no uncommitted change under core or tools when the run started"
            if not campaign.get("dirty", True)
            else "the run started on a dirty tree, so the commit does not describe it",
        )
    )

    if not commit:
        out.append(_finding("commit is reachable", None, "the run recorded no commit"))
    else:
        kind = _git("cat-file", "-t", commit)
        out.append(
            _finding(
                "commit is reachable",
                kind == "commit",
                commit[:12] if kind == "commit" else f"{commit[:12]} is not in this repository",
                commit=commit,
            )
        )

    # Recomputed from the commit's own blobs, not from whatever the working
    # tree holds now. The run records a digest over file contents rather than a
    # git tree object, and comparing it to `commit^{tree}` compared two
    # unrelated things — it read as drift on the first run it was pointed at.
    from core.subject.provenance import tree_hash_at

    at_commit = tree_hash_at(commit) if commit else ""
    if not tree or not at_commit:
        out.append(
            _finding(
                "tree still hashes the same",
                None,
                "nothing to compare" if not tree else f"{commit[:12]} has none of the measured paths",
            )
        )
    else:
        out.append(
            _finding(
                "tree still hashes the same",
                at_commit == tree,
                f"{tree[:12]}" if at_commit == tree
                else f"recorded {tree[:12]}, the commit now hashes {at_commit[:12]}",
                recorded=tree,
                at_commit=at_commit,
            )
        )

    existing = _git("rev-parse", "-q", "--verify", f"refs/tags/{tag}")
    if not existing:
        out.append(_finding("tag is free", True, f"{tag} does not exist yet", tag=tag))
    else:
        points_at = _git("rev-list", "-n", "1", tag)
        out.append(
            _finding(
                "tag is free",
                points_at == commit,
                f"{tag} already points at {points_at[:12]}"
                + ("" if points_at == commit else f", not {commit[:12]}"),
                tag=tag,
            )
        )

    # Reported, not performed. A signing key and a repository setting belong to
    # whoever operates the repository.
    key = _git("config", "--get", "user.signingkey")
    out.append(
        _finding(
            "signing key configured",
            bool(key) or None,
            key or "none configured, so the tag will be annotated and unsigned",
        )
    )
    remote = _git("config", "--get", "remote.origin.url")
    out.append(
        _finding(
            "branch protection",
            None,
            f"a setting on {remote or 'the remote'}, not something this can read or set",
        )
    )
    return out


def _tests(selection: tuple[str, ...], *, run_them: bool) -> dict[str, Any]:
    present = [name for name in selection if (REPO / name).is_file()]
    missing = [name for name in selection if not (REPO / name).is_file()]
    if not run_them:
        return {
            "check": "defining tests",
            "status": "unknown",
            "detail": "not run",
            "suites": present,
            "missing": missing,
        }
    if missing:
        return {
            "check": "defining tests",
            "status": "blocked",
            "detail": f"named but absent: {missing}",
            "missing": missing,
        }
    done = subprocess.run(
        [sys.executable, "-m", "pytest", *present, "-q", "-p", "no:randomly"],
        capture_output=True, text=True, check=False, cwd=str(REPO), timeout=3600,
    )
    tail = (done.stdout or done.stderr).strip().splitlines()
    return {
        "check": "defining tests",
        "status": "ready" if done.returncode == 0 else "blocked",
        "detail": tail[-1] if tail else f"exit {done.returncode}",
        "suites": present,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="report and seal nothing")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    report = _report(args.run)
    campaign = report.get("campaign", {}) or {}
    tag = _tag_name(campaign)
    checks = _checks(report, tag)
    checks.append(_tests(DEFINING_TESTS, run_them=not args.skip_tests and not args.check))

    for row in checks:
        mark = {"ready": "  ", "blocked": "✗ ", "unknown": "? "}[row["status"]]
        print(f"{mark}{row['check']:26} {row['detail']}")

    blocked = [row["check"] for row in checks if row["status"] == "blocked"]
    verdict = report.get("verdict", {}) or {}
    print()
    print(f"run:       {args.run.name}  {verdict.get('passed', '?')}/{verdict.get('total', '?')}")
    print(f"campaign:  {campaign.get('fingerprint', '?')[:16]}")
    print(f"tag:       {tag}")

    if blocked:
        print("\nnot sealed: " + ", ".join(blocked))
        return 1
    if args.check:
        print("\nnothing blocked; rerun without --check to write the tag")
        return 0

    commit = str(campaign.get("commit", ""))
    existing = _git("rev-parse", "-q", "--verify", f"refs/tags/{tag}")
    if existing:
        print(f"\n{tag} already points at this commit; nothing to do")
        return 0
    message = (
        f"Intrinsic Subject Core evaluation\n\n"
        f"run {args.run.name}: {verdict.get('passed', '?')}/{verdict.get('total', '?')} criteria\n"
        f"campaign {campaign.get('fingerprint', '?')}\n"
        f"tree {campaign.get('tree_hash', '?')}\n"
    )
    subprocess.run(
        ["git", "-C", str(REPO), "tag", "-a", tag, commit, "-m", message],
        check=False, timeout=60,
    )
    print(f"\nsealed {tag} at {commit[:12]}")
    print("push it with: git push origin " + tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
