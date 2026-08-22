#!/usr/bin/env python3
"""Move one pinned package to a new version, hashes and all.

A hashed lockfile is the right thing to have and the wrong thing to edit by
hand: every distribution for a version carries its own SHA-256, there are
often ninety of them, and getting one wrong makes `pip install
--require-hashes` fail at build time with a message about a file nobody
recognises. So the lock tends to go stale, which is how an advisory scan
started reporting eight CVEs in Pillow and one in setuptools against a lock
that had not moved.

This rewrites one entry from the index's own metadata: the version, every
distribution hash for it, and the `# via` line that says who wanted it. The
rest of the file is untouched, so the diff is one package.

    python tools/bump_locked_package.py --package pillow --version 12.3.0
    python tools/bump_locked_package.py --package pillow --latest
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
INDEX = "https://pypi.org/pypi"


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.split("[", 1)[0]).lower()


def fetch(package: str, version: str | None) -> tuple[str, list[str]]:
    """(version, sorted sha256 hashes) straight from the index."""
    url = f"{INDEX}/{package}/json" if version is None else f"{INDEX}/{package}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - fixed https host
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise SystemExit(f"could not read {url}: {exc}") from exc
    resolved = payload["info"]["version"]
    digests = sorted(
        entry["digests"]["sha256"]
        for entry in payload["urls"]
        if entry.get("digests", {}).get("sha256")
    )
    if not digests:
        raise SystemExit(f"{package} {resolved} publishes no sha256 digests")
    return resolved, digests


def entry_bounds(lines: list[str], package: str) -> tuple[int, int]:
    """The line range of one package's entry, comments included."""
    target = canonical(package)
    start = -1
    for index, line in enumerate(lines):
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==", line)
        if match and canonical(match.group(1)) == target:
            start = index
            break
    if start < 0:
        raise SystemExit(f"{package} is not pinned in the lockfile")
    end = start + 1
    while end < len(lines) and (
        lines[end].startswith((" ", "\t")) or lines[end].lstrip().startswith("#")
    ):
        end += 1
    return start, end


def render(name_as_written: str, version: str, digests: list[str], via: list[str]) -> list[str]:
    lines = [f"{name_as_written}=={version} \\\n"]
    for position, digest in enumerate(digests):
        suffix = " \\\n" if position < len(digests) - 1 else "\n"
        lines.append(f"    --hash=sha256:{digest}{suffix}")
    lines.extend(via)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--version", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--lockfile", default=str(DEFAULT_LOCK))
    args = parser.parse_args(argv)

    if not args.version and not args.latest:
        parser.error("give --version or --latest")

    lock = Path(args.lockfile)
    lines = lock.read_text("utf-8").splitlines(keepends=True)
    start, end = entry_bounds(lines, args.package)

    written_name = re.match(
        r"^([A-Za-z0-9][A-Za-z0-9._-]*(?:\[[^\]]*\])?)", lines[start]
    ).group(1)
    was = lines[start].split("==", 1)[1].split()[0].rstrip(" \\")
    via = [line for line in lines[start:end] if line.lstrip().startswith("#")]

    version, digests = fetch(canonical(args.package), None if args.latest else args.version)
    if version == was:
        print(f"{written_name} is already at {version}")
        return 0

    lines[start:end] = render(written_name, version, digests, via)
    lock.write_text("".join(lines), encoding="utf-8")
    print(f"{written_name}: {was} -> {version} ({len(digests)} hashes)")
    print("   regenerate the derived files: make lockfiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
