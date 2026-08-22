#!/usr/bin/env python3
"""Ownership and review, checked rather than declared.

There was no CODEOWNERS, no pull-request template and no issue template in
this repository, and its own August architecture assessment measured why that
mattered: roughly 99.6% of commits were agent-authored and no independent
review existed to reject the four-thousand-line functions while they were
still small. The argument is not about who writes the code. An agent writing
and another invocation of the same agent reviewing share their assumptions,
so the second pass cannot see what the first one could not.

A CODEOWNERS file is only worth having while its paths still exist, so this
checks the file against the tree instead of trusting it:

1. every path pattern in CODEOWNERS matches something that exists;
2. the paths where a wrong assumption is expensive are all covered — the
   security boundary, the runtime foundation, persistent state, model
   execution, the interface, and the build and release contract;
3. every owner is a real handle or team reference;
4. the pull-request and issue templates exist and ask for the checks;
5. every job that runs on a pull request is on the branch protection
   policy's required list, so a new gate cannot be green and toothless.

    python tools/check_review_policy.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODEOWNERS = ROOT / ".github" / "CODEOWNERS"
PR_TEMPLATE = ROOT / ".github" / "pull_request_template.md"
ISSUE_TEMPLATES = ROOT / ".github" / "ISSUE_TEMPLATE"
PROTECTION_POLICY = ROOT / "config" / "branch_protection_policy.json"

#: The directories where an unreviewed wrong assumption is most expensive.
#: Each must be matched by at least one CODEOWNERS pattern more specific than
#: the catch-all, or ownership of it is "whoever the default is", which is the
#: state this file exists to end.
MUST_BE_OWNED = (
    "security/",
    "core/security/",
    "core/sandbox/",
    "core/governance/",
    "core/runtime/",
    "core/observability/",
    "core/verify/",
    "core/db/",
    "core/persistence/",
    "core/memory/",
    "core/identity/",
    "core/brain/llm/",
    "interface/",
    "requirements/",
    ".github/workflows/",
    "config/",
)

_OWNER_RE = re.compile(r"^@[A-Za-z0-9][A-Za-z0-9-]*(/[A-Za-z0-9._-]+)?$")


def parse_codeowners(text: str) -> list[tuple[str, list[str]]]:
    rules: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        rules.append((parts[0], parts[1:]))
    return rules


def pattern_matches_something(pattern: str) -> bool:
    """Does this CODEOWNERS pattern name anything that exists?"""
    cleaned = pattern.strip("/")
    if pattern == "*":
        return True
    if "*" in cleaned:
        parent = ROOT / str(Path(cleaned).parent)
        glob = Path(cleaned).name
        return parent.exists() and any(parent.glob(glob))
    return (ROOT / cleaned).exists()


def check() -> list[str]:
    problems: list[str] = []

    if not CODEOWNERS.exists():
        return [".github/CODEOWNERS does not exist"]

    rules = parse_codeowners(CODEOWNERS.read_text("utf-8"))
    if not rules:
        problems.append("CODEOWNERS has no rules")

    patterns = [pattern for pattern, _ in rules]
    for pattern, owners in rules:
        if not owners:
            problems.append(f"CODEOWNERS: {pattern} names no owner")
        for owner in owners:
            if not _OWNER_RE.match(owner):
                problems.append(f"CODEOWNERS: {owner!r} is not a handle or team")
        if not pattern_matches_something(pattern):
            problems.append(
                f"CODEOWNERS: {pattern} matches nothing in the tree; the path "
                "moved or was deleted"
            )

    specific = {p.strip("/") for p in patterns if p != "*"}
    for path in MUST_BE_OWNED:
        if not (ROOT / path).exists():
            continue
        wanted = path.strip("/")
        if not any(wanted == s or wanted.startswith(s + "/") for s in specific):
            problems.append(
                f"CODEOWNERS: {path} has no owner more specific than the "
                "catch-all, and is a path where a wrong assumption is expensive"
            )

    if not PR_TEMPLATE.exists():
        problems.append(".github/pull_request_template.md does not exist")
    else:
        body = PR_TEMPLATE.read_text("utf-8")
        for wanted in ("make", "test", "ratchet"):
            if wanted not in body:
                problems.append(
                    f"the pull request template never mentions {wanted!r}; it "
                    "should ask for the check, not the intent"
                )

    if not ISSUE_TEMPLATES.exists() or not list(ISSUE_TEMPLATES.glob("*.md")):
        problems.append(".github/ISSUE_TEMPLATE holds no templates")

    problems.extend(_check_every_gate_is_required())
    return problems


def _check_every_gate_is_required() -> list[str]:
    """A job that runs on a pull request and is required by nothing is a
    notification. The branch protection policy holds the list; this only
    checks that the workflows and the list agree."""
    if not PROTECTION_POLICY.exists():
        return ["config/branch_protection_policy.json does not exist"]
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_check_branch_protection", ROOT / "tools" / "check_branch_protection.py"
    )
    if spec is None or spec.loader is None:
        return ["tools/check_branch_protection.py could not be loaded"]
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    policy = json.loads(PROTECTION_POLICY.read_text("utf-8"))
    return module.check_offline(policy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    problems = check()
    if args.json:
        print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
        return 1 if problems else 0
    if problems:
        print(f"❌ review policy: {len(problems)} problem(s)")
        for problem in problems:
            print(f"   • {problem}")
        return 1
    print("✅ review policy holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
