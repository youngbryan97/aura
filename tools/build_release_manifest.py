#!/usr/bin/env python3
"""Build the release manifest: one signed-by-checksum picture of a release.

Emits artifacts/release/RELEASE_MANIFEST.json carrying:
- git commit, branch, dirty flag, tag (if any), build timestamp
- pinned Python version (.python-version) and the interpreter that built it
- host platform/hardware class
- SHA-256 checksums for: claim documents (supported / not supported /
  matrix), dependency manifests, the architecture map, and any proof
  artifacts found under artifacts/
- the gate commands a reviewer runs to reproduce the evidence

The manifest refuses to mark itself clean if the worktree is dirty or a
referenced core document is missing — a release manifest that papers
over missing evidence would be the exact failure mode it exists to
prevent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLAIM_DOCUMENTS = (
    "CLAIMS_SUPPORTED.md",
    "CLAIMS_NOT_SUPPORTED.md",
    "CLAIMS_MATRIX.md",
    "AI_SYSTEM_CARD.md",
)

DEPENDENCY_MANIFESTS = (
    "requirements.txt",
    "requirements/core.txt",
    "requirements/dev.txt",
    "requirements/ml.txt",
    "requirements/senses.txt",
    "requirements/voice.txt",
    ".python-version",
)

CORE_DOCUMENTS = (
    "docs/ARCHITECTURE_MAP.md",
    "docs/evidence/CLOSEOUT.md",
)

GATE_COMMANDS = (
    "make doctor",
    "make lint",
    "make compile",
    "make source-hygiene",
    "make test            # chunked: tools/run_test_chunks.py",
    "make enterprise-gate",
    "make production-gate",
    "make security",
    "make governance-lint",
)

PROOF_ARTIFACT_GLOBS = (
    "artifacts/closeout/semantic_review/SEMANTIC_REVIEW_LEDGER.jsonl",
    "artifacts/behavioral_proof/**/*.json",
    "artifacts/certification/**/*.json",
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-c", "core.fsmonitor=false", *args], cwd=ROOT, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checksum_entry(rel: str) -> dict | None:
    path = ROOT / rel
    if not path.is_file():
        return None
    return {
        "path": rel,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "mtime_unix": path.stat().st_mtime,
    }


def build_manifest() -> tuple[dict, list[str]]:
    problems: list[str] = []

    commit = _git("rev-parse", "HEAD")
    if not commit:
        problems.append("not a git checkout: commit identity unavailable")
    dirty = bool(_git("status", "--porcelain"))
    if dirty:
        problems.append("worktree dirty: manifest does not describe a clean commit")

    claim_docs = {}
    for rel in CLAIM_DOCUMENTS:
        entry = _checksum_entry(rel)
        if entry is None:
            problems.append(f"missing claim document: {rel}")
        else:
            claim_docs[rel] = entry

    dependency_docs = {}
    for rel in DEPENDENCY_MANIFESTS:
        entry = _checksum_entry(rel)
        if entry is None:
            problems.append(f"missing dependency manifest: {rel}")
        else:
            dependency_docs[rel] = entry

    core_docs = {}
    for rel in CORE_DOCUMENTS:
        entry = _checksum_entry(rel)
        if entry is None:
            problems.append(f"missing core document: {rel}")
        else:
            core_docs[rel] = entry

    proof_artifacts = []
    for pattern in PROOF_ARTIFACT_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                rel = str(path.relative_to(ROOT))
                entry = _checksum_entry(rel)
                if entry:
                    proof_artifacts.append(entry)

    pinned_python = ""
    pin_path = ROOT / ".python-version"
    if pin_path.is_file():
        pinned_python = pin_path.read_text().strip()
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    if pinned_python and not running.startswith(pinned_python):
        problems.append(
            f"interpreter {running} does not match pinned {pinned_python}"
        )

    manifest = {
        "schema": "aura.release_manifest.v1",
        "generated_at_unix": time.time(),
        "git": {
            "commit": commit,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "tag": _git("describe", "--tags", "--exact-match") or None,
            "dirty": dirty,
        },
        "python": {
            "pinned": pinned_python,
            "build_interpreter": sys.version.split()[0],
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "claims": {
            "documents": claim_docs,
            "note": (
                "Supported claims live in CLAIMS_SUPPORTED.md; consciousness, "
                "qualia, sentience, and personhood are strictly unsupported "
                "per CLAIMS_NOT_SUPPORTED.md."
            ),
        },
        "dependencies": dependency_docs,
        "core_documents": core_docs,
        "proof_artifacts": proof_artifacts,
        "gate_commands": list(GATE_COMMANDS),
        "problems": problems,
        "clean": not problems,
    }
    return manifest, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "release" / "RELEASE_MANIFEST.json",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="exit 0 even when the manifest records problems (dev builds)",
    )
    args = parser.parse_args(argv)

    manifest, problems = build_manifest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Release manifest written to {args.out}")
    if problems:
        for problem in problems:
            print(f"  ⚠️  {problem}")
        if not args.allow_dirty:
            print("❌ manifest not clean; rerun from a clean tagged checkout")
            return 1
    else:
        print("✅ manifest clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
