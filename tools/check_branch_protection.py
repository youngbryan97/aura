#!/usr/bin/env python3
"""The gates only have authority if the branch requires them.

Measured before this existed: ``main`` returned ``protected: false``, no
required status checks, and no required review. Eight workflow files, nine
jobs, every ratchet in ``config/`` — all of them advisory. Anything could
land, and something red did: the enterprise gate was failing on ``main``
with seven categories above baseline while every one of those jobs was
configured to run.

A check that cannot block is a notification. This turns the wanted state
into a file, compares it against what GitHub actually has, and can apply it.

    python tools/check_branch_protection.py --offline   # policy is coherent
    python tools/check_branch_protection.py             # live settings match
    python tools/check_branch_protection.py --apply     # make them match

``--offline`` is what CI runs, because a CI job asserting its own branch
protection through the API needs a token with admin rights, and handing that
to every pull request is a worse problem than the one it checks. What it can
check without the network is the part that rots: that every required check
names a job that really runs on a pull request, and that no job runs on a
pull request without being required. Those two together are what stop the
list drifting away from the workflows beside it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "config" / "branch_protection_policy.json"
WORKFLOWS = ROOT / ".github" / "workflows"

_JOB_RE = re.compile(r"^  (?P<job>[A-Za-z0-9_-]+):\s*$")
_TOP_KEY_RE = re.compile(r"^(?P<key>[A-Za-z0-9_-]+):")


def workflow_jobs_on_pull_request() -> dict[str, str]:
    """Every job id that runs on a pull request, mapped to its workflow file.

    Parsed by structure rather than with a YAML dependency: this tool runs in
    a CI job that installs nothing but ruff, and a gate that needs a package
    to check whether the gates are required is a gate that will be skipped.
    """
    found: dict[str, str] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        lines = path.read_text("utf-8").splitlines()
        in_on = False
        on_pull_request = False
        in_jobs = False
        for line in lines:
            top = _TOP_KEY_RE.match(line)
            if top:
                key = top.group("key")
                in_on = key == "on"
                in_jobs = key == "jobs"
                continue
            if in_on and re.match(r"^\s*pull_request:", line):
                on_pull_request = True
            if in_jobs:
                job = _JOB_RE.match(line)
                if job:
                    found.setdefault(job.group("job"), path.name)
        if not on_pull_request:
            for job in [j for j, f in found.items() if f == path.name]:
                found.pop(job, None)
    return found


def check_offline(policy: dict) -> list[str]:
    problems: list[str] = []
    required = list(policy.get("required_checks", []))
    if not required:
        problems.append("the policy requires no checks at all")
    duplicates = {name for name in required if required.count(name) > 1}
    if duplicates:
        problems.append(f"required_checks lists {sorted(duplicates)} more than once")

    jobs = workflow_jobs_on_pull_request()
    for name in required:
        if name not in jobs:
            problems.append(
                f"required check {name!r} is not a job that runs on a pull request"
            )
    for job, workflow in sorted(jobs.items()):
        if job not in required:
            problems.append(
                f"{workflow} runs job {job!r} on pull requests and no branch "
                "protection rule requires it, so it cannot block anything"
            )

    if not policy.get("enforced"):
        if not policy.get("not_enforced_reason") or not policy.get("apply_with"):
            problems.append(
                "the policy is not enforced and records neither the reason nor "
                "the command that enforces it"
            )
    return problems


def live_protection(repo: str, branch: str) -> dict | None:
    """What GitHub has now, or None when the branch is unprotected."""
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["gh", "api", f"repos/{repo}/branches/{branch}/protection"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if "Branch not protected" in (result.stdout + result.stderr):
            return None
        raise RuntimeError(result.stderr.strip() or "gh api failed")
    return json.loads(result.stdout)


def current_repo() -> str:
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "gh repo view failed")
    return result.stdout.strip()


def compare_live(policy: dict, live: dict | None) -> list[str]:
    if live is None:
        return [f"branch {policy['branch']!r} is not protected at all"]

    problems: list[str] = []
    settings = policy["settings"]
    required = set(policy["required_checks"])

    checks = (live.get("required_status_checks") or {}).get("contexts") or []
    missing = required - set(checks)
    if missing:
        problems.append(f"these checks are not required: {sorted(missing)}")

    reviews = live.get("required_pull_request_reviews")
    if settings["require_pull_request"] and not reviews:
        problems.append("a pull request is not required")
    elif reviews:
        if reviews.get("required_approving_review_count", 0) < settings[
            "required_approving_review_count"
        ]:
            problems.append(
                "fewer approving reviews are required than the policy states"
            )
        if settings["require_code_owner_reviews"] and not reviews.get(
            "require_code_owner_reviews"
        ):
            problems.append("code owner review is not required")
        if settings["dismiss_stale_reviews"] and not reviews.get("dismiss_stale_reviews"):
            problems.append("stale reviews are not dismissed on a new push")

    for key, path in (
        ("enforce_admins", "enforce_admins"),
        ("required_linear_history", "required_linear_history"),
        ("required_conversation_resolution", "required_conversation_resolution"),
    ):
        want = settings[key]
        got = bool((live.get(path) or {}).get("enabled"))
        if want and not got:
            problems.append(f"{key} is off")

    for key in ("allow_force_pushes", "allow_deletions"):
        if not settings[key] and bool((live.get(key) or {}).get("enabled")):
            problems.append(f"{key} is on and the policy forbids it")

    return problems


def apply(policy: dict, repo: str) -> int:
    """Write the policy to GitHub. Only ever from an explicit --apply."""
    settings = policy["settings"]
    body = {
        "required_status_checks": {
            "strict": True,
            "contexts": sorted(policy["required_checks"]),
        },
        "enforce_admins": settings["enforce_admins"],
        "required_pull_request_reviews": (
            {
                "required_approving_review_count": settings[
                    "required_approving_review_count"
                ],
                "require_code_owner_reviews": settings["require_code_owner_reviews"],
                "dismiss_stale_reviews": settings["dismiss_stale_reviews"],
            }
            if settings["require_pull_request"]
            else None
        ),
        "restrictions": None,
        "required_linear_history": settings["required_linear_history"],
        "allow_force_pushes": settings["allow_force_pushes"],
        "allow_deletions": settings["allow_deletions"],
        "required_conversation_resolution": settings[
            "required_conversation_resolution"
        ],
    }
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            "gh",
            "api",
            "--method",
            "PUT",
            f"repos/{repo}/branches/{policy['branch']}/protection",
            "--input",
            "-",
        ],
        input=json.dumps(body),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return 1
    print(f"✅ applied branch protection to {repo}@{policy['branch']}")
    print(
        "   Now set \"enforced\": true in config/branch_protection_policy.json "
        "so the gate holds it there."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="check the policy only")
    parser.add_argument("--apply", action="store_true", help="write the policy to GitHub")
    parser.add_argument("--repo", default="")
    args = parser.parse_args(argv)

    policy = json.loads(POLICY.read_text("utf-8"))

    problems = check_offline(policy)
    if problems:
        print(f"❌ branch protection policy: {len(problems)} problem(s)")
        for problem in problems:
            print(f"   • {problem}")
        return 1

    if args.offline:
        state = "enforced" if policy.get("enforced") else "declared, not yet enforced"
        print(
            f"✅ branch protection policy is coherent "
            f"({len(policy['required_checks'])} checks, {state})"
        )
        if not policy.get("enforced"):
            print(f"   reason: {policy['not_enforced_reason']}")
            print(f"   enable: {policy['apply_with']}")
        return 0

    repo = args.repo or current_repo()
    if args.apply:
        return apply(policy, repo)

    live = live_protection(repo, policy["branch"])
    differences = compare_live(policy, live)
    if not differences:
        print(f"✅ {repo}@{policy['branch']} matches the policy")
        return 0
    if not policy.get("enforced"):
        print(f"⚠️  {repo}@{policy['branch']} does not match the policy, which is "
              "recorded as not yet enforced:")
        for difference in differences:
            print(f"   • {difference}")
        print(f"   reason: {policy['not_enforced_reason']}")
        print(f"   enable: {policy['apply_with']}")
        return 0
    print(f"❌ {repo}@{policy['branch']} does not match the enforced policy")
    for difference in differences:
        print(f"   • {difference}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
