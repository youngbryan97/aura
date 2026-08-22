#!/usr/bin/env python3
"""Every pinned version, asked about at OSV before the build asks.

The `Dependency Vulnerability Scan` job runs `osv-scanner` over
`requirements_lock.txt` in CI, which is the right place for it and the wrong
place to find out: the answer arrives after the push, and it arrived saying
eight advisories against Pillow and one against setuptools in a lock that had
not moved in months.

This asks the same database from a workstation, over the same lockfile, so a
dependency bump is a thing you do before the branch exists rather than after
the gate turns red. `tools/bump_locked_package.py` is what fixes what this
finds.

    python tools/check_lock_advisories.py
    python tools/check_lock_advisories.py --json
    python tools/check_lock_advisories.py --offline   # skip when there is no network
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCK = ROOT / "requirements_lock.txt"
BATCH = "https://api.osv.dev/v1/querybatch"
ADVISORY = "https://api.osv.dev/v1/vulns/"

_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==\s*([^\s\;]+)")


def pinned(lock: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in lock.read_text("utf-8").splitlines():
        match = _PIN_RE.match(line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def query(packages: dict[str, str], *, timeout: float = 60.0) -> dict[str, list[str]]:
    body = json.dumps(
        {
            "queries": [
                {"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
                for name, version in packages.items()
            ]
        }
    ).encode()
    request = urllib.request.Request(  # noqa: S310 - fixed https host
        BATCH, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.load(response)

    findings: dict[str, list[str]] = {}
    for name, result in zip(packages, payload.get("results", []), strict=False):
        ids = [v["id"] for v in (result.get("vulns") or [])]
        if ids:
            findings[name] = ids
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lockfile", default=str(DEFAULT_LOCK))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    lock = Path(args.lockfile)
    packages = pinned(lock)
    if args.offline:
        print(f"skipped: {len(packages)} pinned versions, no query made")
        return 0

    try:
        findings = query(packages)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        # A scan that cannot reach the database has not cleared anything, and
        # saying so is the difference between "no advisories" and "no answer".
        print(f"❌ could not reach OSV: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"pinned": len(packages), "findings": findings}, indent=2))
        return 1 if findings else 0

    if findings:
        total = sum(len(v) for v in findings.values())
        print(f"❌ {total} advisory/advisories against {len(findings)} pinned package(s)")
        for name, ids in sorted(findings.items()):
            print(f"   • {name}=={packages[name]}: {', '.join(ids[:6])}")
            print(
                f"     python tools/bump_locked_package.py --package {name} --latest"
            )
        return 1
    print(f"✅ no advisories against {len(packages)} pinned versions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
