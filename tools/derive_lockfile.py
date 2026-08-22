#!/usr/bin/env python3
"""Cut a smaller lockfile out of a bigger one, keeping the pins and the hashes.

A repository with one requirements file needs one lock. This one has five
install surfaces — the desktop release, the runtime image, the code-sandbox
image, CI and a workstation — and only the first had a lock. The others
installed a range specifier and got whatever the index served that morning,
which is the reproducibility hole `requirements_lock.txt` was created to close
and then closed for exactly one of the five.

Resolving each surface independently needs `pip-compile` on that surface's
platform. Deriving them does not. `requirements_lock.txt` already records the
resolver's answer — the version chosen for every package, every distribution
hash for that version (`--generate-hashes` records the whole file set, not
just the one wheel this machine downloaded), and, in its `# via` comments, the
dependency edges it used to get there. A subset of that answer is still that
answer, so the closure of a smaller root set can be read straight out of it.

What this does NOT do is resolve. If a target's requirements name a package
the source lock has never seen, that is a resolution the source lock cannot
answer and this refuses rather than guessing. Run `pip-compile` and commit the
result instead.

    python tools/derive_lockfile.py \\
        --source requirements_lock.txt \\
        --requirements requirements/core.txt \\
        --out requirements/lock/container-runtime.txt
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# `uvicorn[standard]==0.46.0` is how pip-compile spells an extra, and the
# project is still `uvicorn`. Missing that bracket made this tool report the
# ASGI server as unpinned.
_ENTRY_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==\s*(?P<version>[^\s\;]+)"
)
_VIA_RE = re.compile(r"^\s*#\s+via\s*(?P<rest>.*)$")
_REQ_NAME_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)")


def canonical(name: str) -> str:
    """PEP 503 normalisation, with extras dropped.

    `charset_normalizer` and `charset-normalizer` are the same project and a
    lockfile may spell either; so are `uvicorn` and `uvicorn[standard]`.
    """
    name = name.split("[", 1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


class SourceLock:
    """A fully hashed lockfile, parsed into entries and forward edges."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.text = path.read_text("utf-8")
        self.entries: dict[str, str] = {}
        self.versions: dict[str, str] = {}
        self.requires: dict[str, set[str]] = {}
        self._parse()

    def _parse(self) -> None:
        lines = self.text.splitlines()
        current: str | None = None
        buffer: list[str] = []
        for line in lines:
            match = _ENTRY_RE.match(line)
            if match:
                self._flush(current, buffer)
                current = canonical(match.group("name"))
                self.versions[current] = match.group("version")
                buffer = [line]
                continue
            if current is None:
                continue
            via = _VIA_RE.match(line)
            if via or line.startswith("    #"):
                self._record_via(current, line)
                buffer.append(line)
                continue
            if line.startswith(("    ", "\t")) or line.strip().startswith("--hash"):
                buffer.append(line)
                continue
            self._flush(current, buffer)
            current, buffer = None, []
        self._flush(current, buffer)

    def _record_via(self, package: str, line: str) -> None:
        via = _VIA_RE.match(line)
        text = via.group("rest") if via else line.strip().lstrip("#").strip()
        for token in text.replace(",", " ").split():
            token = token.strip()
            if not token or token.startswith("-"):
                continue
            if token.endswith(".txt") or token.endswith(".in"):
                # A root of the source lock, not a package.
                continue
            dependant = canonical(token)
            self.requires.setdefault(dependant, set()).add(package)

    def _flush(self, name: str | None, buffer: list[str]) -> None:
        if name and buffer:
            self.entries[name] = "\n".join(buffer).rstrip() + "\n"

    def closure(self, roots: list[str]) -> tuple[list[str], list[str]]:
        """Every package the roots need, and the roots that are not here."""
        missing = [r for r in roots if canonical(r) not in self.entries]
        seen: set[str] = set()
        stack = [canonical(r) for r in roots if canonical(r) in self.entries]
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            for dependency in sorted(self.requires.get(name, ())):
                if dependency in self.entries and dependency not in seen:
                    stack.append(dependency)
        return sorted(seen), missing


def requirement_roots(path: Path) -> list[str]:
    """The distributions a requirements file asks for by name."""
    roots: list[str] = []
    for raw in path.read_text("utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        # `uvicorn[standard]` asks for uvicorn; extras are resolved in the
        # source lock already, so the closure picks them up through `# via`.
        line = line.split(";", 1)[0].strip()
        name = _REQ_NAME_RE.match(line)
        if name:
            roots.append(name.group("name"))
    return roots


def _repo_relative(path: Path) -> str:
    """Paths in the header are repository-relative.

    An absolute path here records the machine that generated the file, which
    makes the same content differ between checkouts and turns the freshness
    check into a false alarm.
    """
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def render(source: SourceLock, names: list[str], *, requirements: Path, out: Path) -> str:
    source_digest = hashlib.sha256(source.path.read_bytes()).hexdigest()
    req_digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    header = [
        "#",
        f"# Derived by tools/derive_lockfile.py from {_repo_relative(source.path)}.",
        "# Do not edit by hand: run `make lockfiles`.",
        "#",
        f"#   requirements: {_repo_relative(requirements)}",
        f"#   requirements-sha256: {req_digest}",
        f"#   source-lock: {_repo_relative(source.path)}",
        f"#   source-lock-sha256: {source_digest}",
        "#",
        "# Install with --require-hashes. Every version and every hash below is",
        "# the source lock's, unchanged.",
        "#",
        "",
    ]
    body = [source.entries[name] for name in names]
    return "\n".join(header) + "\n".join(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the file on disk differs from what would be written",
    )
    args = parser.parse_args(argv)

    source = SourceLock(ROOT / args.source)
    requirements = ROOT / args.requirements
    roots = requirement_roots(requirements)
    names, missing = source.closure(roots)
    if missing:
        print(
            f"error: {args.source} does not pin {missing}, which "
            f"{args.requirements} requires. Run pip-compile for this target "
            "instead of deriving it.",
            file=sys.stderr,
        )
        return 2

    rendered = render(source, names, requirements=requirements, out=ROOT / args.out)
    out = ROOT / args.out
    if args.check:
        if not out.exists():
            print(f"error: {args.out} is missing; run `make lockfiles`", file=sys.stderr)
            return 1
        if out.read_text("utf-8") != rendered:
            print(
                f"error: {args.out} is stale — {args.requirements} or "
                f"{args.source} changed since it was generated. Run "
                "`make lockfiles`.",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {args.out} matches ({len(names)} pinned)")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}: {len(names)} pinned distributions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
