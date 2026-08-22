#!/usr/bin/env python3
"""What must be true before a tag becomes a release.

The signing, notarization, SBOM and provenance steps in
`.github/workflows/release.yml` are real, and the repository has published no
releases at all — so none of it has run end to end. That is a gap this cannot
close. What it can close is the set of conditions that are checkable before
the tag, and that were not being checked: the release lane installed
`requirements/runtime.txt`, a file that has never existed, and continued past
the failure with `|| true` before signing and notarizing whatever happened to
be in the image.

Checks:

1. the version in pyproject.toml has a CHANGELOG entry;
2. the release workflow installs the lockfile, with --require-hashes;
3. no step in the release workflow swallows a failure;
4. the release workflow still signs, notarizes, staples and emits provenance;
5. the lifecycle document's channel table names gates that exist as make
   targets, so a channel cannot promise a gate nobody can run.

    python tools/check_release_ready.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
CHANGELOG = ROOT / "CHANGELOG.md"
LIFECYCLE = ROOT / "docs" / "RELEASE_LIFECYCLE.md"
PYPROJECT = ROOT / "pyproject.toml"
MAKEFILE = ROOT / "Makefile"

#: Steps a stable release cannot be a stable release without.
REQUIRED_STEPS = {
    "codesign": "the bundle is signed",
    "notarytool submit": "the bundle is notarized",
    "stapler staple": "the ticket is stapled",
    "build_provenance.py": "an SBOM and provenance are emitted",
    "--require-hashes": "dependencies are installed from the lockfile by hash",
}

_FAIL_OPEN_RE = re.compile(r"\|\|\s*true\b")
_MAKE_TARGET_RE = re.compile(r"^([a-z][a-z0-9-]*):", re.MULTILINE)
_CHANNEL_GATE_RE = re.compile(r"`make ([a-z][a-z0-9-]*)`")


def _code(path: Path) -> str:
    if not path.exists():
        return ""
    return "\n".join(
        line
        for line in path.read_text("utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def check() -> list[str]:
    problems: list[str] = []

    version = ""
    if PYPROJECT.exists():
        match = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text("utf-8"), re.M)
        version = match.group(1) if match else ""
    if not version:
        problems.append("pyproject.toml declares no version")

    if not CHANGELOG.exists():
        problems.append("CHANGELOG.md does not exist")
    elif version:
        body = CHANGELOG.read_text("utf-8")
        month = ".".join(version.split(".")[:2]).replace(".", "-")
        if version not in body and month not in body:
            problems.append(
                f"CHANGELOG.md has no entry for {version} (or its month {month}); "
                "a release with no entry is a release nobody can read"
            )

    workflow = _code(WORKFLOW)
    if not workflow:
        problems.append(".github/workflows/release.yml does not exist")
    else:
        for marker, meaning in sorted(REQUIRED_STEPS.items()):
            if marker not in workflow:
                problems.append(f"the release workflow no longer ensures that {meaning}")
        for match in _FAIL_OPEN_RE.finditer(workflow):
            line = workflow[: match.start()].count("\n") + 1
            problems.append(
                f"release.yml line {line}: a step continues after failure; a "
                "release pipeline stops"
            )
        for named in re.findall(r"pip install[^\n|&;]*-r\s+(\S+)", workflow):
            if not (ROOT / named).exists():
                problems.append(f"release.yml installs {named}, which does not exist")

    if not LIFECYCLE.exists():
        problems.append("docs/RELEASE_LIFECYCLE.md does not exist")
    else:
        targets = set(_MAKE_TARGET_RE.findall(MAKEFILE.read_text("utf-8")))
        for gate in sorted(set(_CHANNEL_GATE_RE.findall(LIFECYCLE.read_text("utf-8")))):
            if gate not in targets:
                problems.append(
                    f"the lifecycle promises `make {gate}` for a channel and the "
                    "Makefile has no such target"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    problems = check()
    if args.json:
        print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
        return 1 if problems else 0
    if problems:
        print(f"❌ release readiness: {len(problems)} problem(s)")
        for problem in problems:
            print(f"   • {problem}")
        return 1
    print("✅ the release lane is fail-closed and its channels name gates that exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
