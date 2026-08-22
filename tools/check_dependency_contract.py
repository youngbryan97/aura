#!/usr/bin/env python3
"""What each build installs, declared once and checked against the builds.

The state this replaces, measured rather than described:

* The production Dockerfile copied ``requirements_lock.txt``, carried the
  comment "Production builds MUST succeed with the lockfile. No best-effort
  fallback", and then installed ``requirements/core.txt``. The lockfile was
  never read.
* The release workflow ran ``pip install -r requirements/runtime.txt || pip
  install -r requirements.txt || true``. ``requirements/runtime.txt`` does not
  exist, so the first install always failed; the ``|| true`` meant the second
  one was allowed to fail as well, and the job went on to sign and notarize a
  bundle built against whatever happened to be installed.
* ``docker/Dockerfile`` — the image untrusted code executes in — ran ``pip
  install numpy pandas requests`` with no bounds at all.
* ``make setup`` swallowed both installs with ``2>/dev/null || ... || echo``.
* The image labelled itself ``org.opencontainers.image.licenses="MIT"`` while
  ``LICENSE`` reserves all rights.

None of these are exotic. Each one is a claim in one file that the file next
to it contradicts, which is what a contract plus a gate exists to stop.

Checks, in order of what they would have caught:

1. every ``pip install -r FILE`` in a Dockerfile, workflow or the Makefile
   names a file that EXISTS and that some target declares;
2. no install on a release or image build is fail-open (``|| true``,
   ``|| pip install``, ``2>/dev/null ||``);
3. a target whose ``lock_status`` is ``present`` installs its lockfile, that
   lockfile is fully hashed, and the install passes ``--require-hashes``;
4. a lockfile derived from another is not stale — the digests in its header
   still match the files it was derived from;
5. a target whose ``lock_status`` is ``pending`` records the command that
   closes it;
6. the image's declared license matches ``LICENSE``.

    python tools/check_dependency_contract.py
    python tools/check_dependency_contract.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "config" / "dependency_contract.json"

#: Files whose install commands this gate governs. A build surface that is not
#: scanned is a build surface with no contract.
SCANNED = (
    "Dockerfile",
    "docker/Dockerfile",
    "Makefile",
)
WORKFLOW_DIR = ROOT / ".github" / "workflows"

_INSTALL_RE = re.compile(r"pip\s+install\b(?P<args>[^\n|&;]*)")
_DASH_R_RE = re.compile(r"-r\s+(?P<path>[^\s]+)")
_CONSTRAINT_RE = re.compile(r"(?:-c|--constraint)\s+(?P<path>[^\s]+)")

#: Fail-open shapes. Each one lets a build continue after the dependency
#: install it depends on has failed.
_FAIL_OPEN_RE = re.compile(r"pip\s+install[^\n]*?(\|\|\s*true|2>/dev/null\s*\|\||\|\|\s*pip\s+install)")

#: Surfaces where a failed install must stop the build. CI gate jobs are
#: allowed to be resilient; a build that ships an artifact is not.
_MUST_FAIL_CLOSED = ("Dockerfile", "docker/Dockerfile", ".github/workflows/release.yml")

_HEADER_DIGEST_RE = re.compile(r"^#\s+(?P<key>[a-z-]+-sha256):\s*(?P<value>[0-9a-f]{64})\s*$", re.M)
_HEADER_PATH_RE = re.compile(r"^#\s+(?P<key>requirements|source-lock):\s*(?P<value>\S+)\s*$", re.M)


class Finding(str):
    """A failure, rendered as its own message."""


def _read(rel: str) -> str:
    path = ROOT / rel
    return path.read_text("utf-8") if path.exists() else ""


def _code(rel: str) -> str:
    """The file with its comment-only lines removed.

    A gate that reads comments as code cannot be used to explain, in a
    comment, the command it just made illegal — and the explanation is the
    part a reader needs most. Dockerfiles, workflow `run:` blocks and
    Makefiles all comment with a leading `#`.
    """
    return "\n".join(
        line for line in _read(rel).splitlines() if not line.lstrip().startswith("#")
    )


def _scan_targets() -> list[str]:
    files = [f for f in SCANNED if (ROOT / f).exists()]
    if WORKFLOW_DIR.exists():
        files.extend(
            str(p.relative_to(ROOT)) for p in sorted(WORKFLOW_DIR.glob("*.yml"))
        )
    return files


def installed_requirement_files(body: str) -> list[str]:
    """Every ``-r FILE`` a pip install in this text names."""
    found: list[str] = []
    for install in _INSTALL_RE.finditer(body):
        for match in _DASH_R_RE.finditer(install.group("args")):
            found.append(match.group("path"))
    return found


def _lock_is_fully_hashed(text: str) -> tuple[bool, list[str]]:
    """Every pinned requirement carries at least one hash."""
    unhashed: list[str] = []
    blocks = re.split(r"\n(?=[A-Za-z0-9])", text)
    for block in blocks:
        head = block.splitlines()[0] if block.splitlines() else ""
        if "==" not in head or head.lstrip().startswith("#"):
            continue
        if "--hash=sha256:" not in block:
            unhashed.append(head.split()[0])
    return (not unhashed), unhashed


def _check_derived_lock_freshness(lock_rel: str) -> list[Finding]:
    """A derived lock records what it was derived from. Re-check it."""
    text = _read(lock_rel)
    if not text:
        return [Finding(f"{lock_rel}: declared present but the file is missing")]
    digests = {m.group("key"): m.group("value") for m in _HEADER_DIGEST_RE.finditer(text)}
    paths = {m.group("key"): m.group("value") for m in _HEADER_PATH_RE.finditer(text)}
    if not digests:
        return []  # A hand-compiled lock records no derivation; nothing to re-check.
    findings: list[Finding] = []
    for key, rel in (("requirements-sha256", "requirements"), ("source-lock-sha256", "source-lock")):
        recorded = digests.get(key)
        named = paths.get(rel)
        if not recorded or not named:
            continue
        target = ROOT / named
        if not target.exists():
            findings.append(Finding(f"{lock_rel}: was derived from {named}, which no longer exists"))
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != recorded:
            findings.append(
                Finding(
                    f"{lock_rel} is stale: {named} changed since it was generated. "
                    "Run `make lockfiles`."
                )
            )
    return findings


def check(contract: dict) -> list[Finding]:
    findings: list[Finding] = []
    targets = contract["targets"]

    declared_requirements = {
        t["requirements"] for t in targets.values() if t.get("requirements")
    }
    declared_locks = {
        t["lockfile"] for t in targets.values() if t.get("lockfile")
    }
    installable = declared_requirements | declared_locks

    # 1 + 2: every install names a declared, existing file, and build surfaces
    # that ship an artifact do not continue past a failed install.
    for rel in _scan_targets():
        body = _code(rel)
        for named in installed_requirement_files(body):
            if not (ROOT / named).exists():
                findings.append(
                    Finding(f"{rel}: installs {named}, which does not exist in the repository")
                )
            elif named not in installable:
                findings.append(
                    Finding(
                        f"{rel}: installs {named}, which no target in "
                        f"{CONTRACT.name} declares"
                    )
                )
        if rel in _MUST_FAIL_CLOSED:
            for match in _FAIL_OPEN_RE.finditer(body):
                findings.append(
                    Finding(
                        f"{rel}: dependency install continues after failure "
                        f"({match.group(1).strip()}); a build that ships must stop"
                    )
                )

    # 3 + 4 + 5: locks.
    for name, target in sorted(targets.items()):
        status = target.get("lock_status")
        lock = target.get("lockfile")
        if status == "present":
            if not lock:
                findings.append(Finding(f"{name}: lock_status is present with no lockfile"))
                continue
            text = _read(lock)
            if not text:
                findings.append(Finding(f"{name}: lockfile {lock} is missing"))
                continue
            hashed, unhashed = _lock_is_fully_hashed(text)
            if not hashed:
                findings.append(
                    Finding(f"{name}: {lock} pins {unhashed[:5]} with no hash")
                )
            findings.extend(_check_derived_lock_freshness(lock))
            for installer in target.get("installed_by", []):
                body = _code(installer)
                if lock not in body:
                    findings.append(
                        Finding(f"{name}: {installer} does not install {lock}")
                    )
                elif target.get("require_hashes") and "--require-hashes" not in body:
                    findings.append(
                        Finding(
                            f"{name}: {installer} installs {lock} without --require-hashes"
                        )
                    )
        elif status == "pending":
            if not target.get("pending_reason") or not target.get("closed_by"):
                findings.append(
                    Finding(
                        f"{name}: lock_status is pending without a pending_reason "
                        "and the command that closes it"
                    )
                )
            constraint = target.get("constraint")
            if constraint:
                for installer in target.get("installed_by", []):
                    body = _code(installer)
                    if constraint not in body:
                        findings.append(
                            Finding(
                                f"{name}: {installer} does not constrain the install "
                                f"with {constraint}"
                            )
                        )
        elif status == "unlocked":
            if not target.get("unlocked_reason"):
                findings.append(Finding(f"{name}: unlocked with no reason recorded"))
        else:
            findings.append(Finding(f"{name}: unknown lock_status {status!r}"))

    # 6: the license the image claims is the license the repository grants.
    findings.extend(_check_license_metadata())

    # 7: a requirements file changed without its digest being refreshed means
    # the contract is describing a state that no longer exists.
    for rel, recorded in sorted(contract.get("source_digests", {}).items()):
        path = ROOT / rel
        if not path.exists():
            findings.append(Finding(f"{CONTRACT.name}: records a digest for missing {rel}"))
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != recorded:
            findings.append(
                Finding(
                    f"{rel} changed since {CONTRACT.name} recorded it. Regenerate the "
                    "locks that depend on it and refresh the digest in the same commit."
                )
            )
    return findings


def _declared_license() -> str:
    """The SPDX identifier the repository's LICENSE actually grants."""
    text = _read("LICENSE").lower()
    if "all rights reserved" in text and "may not" in text:
        # Source-available with no grant. SPDX has no identifier for this;
        # the OCI spec allows any expression, and "NONE" is the honest one.
        return "NONE"
    if "mit license" in text:
        return "MIT"
    if "apache license" in text:
        return "Apache-2.0"
    return ""


def _check_license_metadata() -> list[Finding]:
    expected = _declared_license()
    if not expected:
        return []
    findings: list[Finding] = []
    for rel in ("Dockerfile", "docker/Dockerfile"):
        body = _read(rel)
        for match in re.finditer(
            r"org\.opencontainers\.image\.licenses\s*=\s*\"(?P<value>[^\"]*)\"", body
        ):
            if match.group("value") != expected:
                findings.append(
                    Finding(
                        f"{rel}: labels the image "
                        f"licenses=\"{match.group('value')}\" while LICENSE grants "
                        f"{expected}"
                    )
                )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    contract = json.loads(CONTRACT.read_text("utf-8"))
    findings = check(contract)

    if args.json:
        print(json.dumps({"ok": not findings, "findings": list(findings)}, indent=2))
        return 1 if findings else 0

    if findings:
        print(f"❌ dependency contract: {len(findings)} problem(s)")
        for finding in findings:
            print(f"   • {finding}")
        return 1
    pending = [
        name for name, t in contract["targets"].items() if t.get("lock_status") == "pending"
    ]
    print(f"✅ dependency contract holds ({len(contract['targets'])} targets)")
    if pending:
        print(f"   pending locks (may only shrink): {', '.join(sorted(pending))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
