"""core/evaluation/source_attestation.py — is the code that produced this artifact public?

THE CRITICISM
"The public repository is not the deployed code." It is a fair charge and it
cannot be answered by asserting otherwise: an artifact claiming a result about
Aura is only checkable by a reader if the code that produced it is code the
reader can obtain. An evidence bundle generated from a working tree with
uncommitted edits is a measurement of something that exists on exactly one
machine.

`tools/reqproof/capture.py` already refuses to capture unless HEAD is exact
pushed main with a clean tree. That check is correct, and it was reachable by
precisely one evidence producer. Every other artifact in `artifacts/` — the
ablation scorecards included — asserted results about Aura while saying
nothing about whether the code behind them had ever left this laptop.

WHAT THIS ADDS
A statement, embeddable in any artifact, that answers the question honestly in
all three of its states rather than two:

    published   HEAD is pushed to the tracked upstream and the tree is clean.
                A reader can obtain this exact code.
    divergent   it is not, and the attestation names what differs — how many
                unpushed commits, which files are dirty.
    unknown     git could not be consulted.

`unknown` is deliberately not folded into either. A provenance check that
cannot run is not a provenance check that passed, and the failure mode this
guards against is a missing git binary silently producing artifacts stamped
as reproducible.

WHAT IT DELIBERATELY DOES NOT DO
It does not refuse to run anything, and it does not gate a launch. The runtime
is developed in place; a build that refused to start on a dirty tree would
make the repository undevelopable, and a check whose cost is that high gets
switched off. What gets refused is the CLAIM: an artifact may still be
produced from a divergent tree, and it will say so on its face.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "aura.source_attestation.v1"

#: Enough to identify the divergence without turning an artifact into a diff.
_MAX_DIRTY_LISTED = 20

_GIT_ERRORS = (
    OSError,
    subprocess.SubprocessError,
    subprocess.TimeoutExpired,
    ValueError,
)


@dataclass(frozen=True)
class SourceAttestation:
    """Whether the code that produced an artifact can be obtained by a reader."""

    verdict: str  # "published" | "divergent" | "unknown"
    head_sha: str
    upstream_ref: str
    upstream_sha: str
    is_clean: bool
    is_pushed: bool
    unpushed_commits: int
    dirty_files: tuple[str, ...]
    dirty_count: int
    detail: str

    @property
    def reproducible_from_public_source(self) -> bool:
        return self.verdict == "published"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "verdict": self.verdict,
            "reproducible_from_public_source": self.reproducible_from_public_source,
            "head_sha": self.head_sha,
            "upstream_ref": self.upstream_ref,
            "upstream_sha": self.upstream_sha,
            "is_clean": self.is_clean,
            "is_pushed": self.is_pushed,
            "unpushed_commits": self.unpushed_commits,
            "dirty_count": self.dirty_count,
            "dirty_files": list(self.dirty_files),
            "detail": self.detail,
        }


def _git(root: Path, *args: str) -> tuple[int, str]:
    try:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        proc = get_subprocess_gateway().run(
            ["/usr/bin/git", "-c", "core.fsmonitor=false", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
            read_only=True,
            source="core.evaluation.source_attestation",
            accelerator_capability="none",
        )
    except _GIT_ERRORS:
        return 1, ""
    return proc.returncode, (proc.stdout or "").strip()


def attest(root: Path | str | None = None, *, upstream: str = "") -> SourceAttestation:
    """Describe the publication state of the tree that is producing this artifact.

    Never raises. A provenance statement that can crash the run producing it
    gets removed from the run, and then artifacts carry no provenance at all —
    which is the state this module exists to end. Every failure path lands in
    `unknown`, which is not a pass.
    """
    base = Path(root or Path(__file__).resolve().parents[2])

    code, head = _git(base, "rev-parse", "HEAD")
    if code != 0 or not head:
        return SourceAttestation(
            verdict="unknown",
            head_sha="",
            upstream_ref="",
            upstream_sha="",
            is_clean=False,
            is_pushed=False,
            unpushed_commits=0,
            dirty_files=(),
            dirty_count=0,
            detail="git could not resolve HEAD; publication state is unknown, not clean",
        )

    upstream_ref = upstream
    if not upstream_ref:
        code, tracked = _git(base, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        upstream_ref = tracked if code == 0 and tracked else "origin/main"

    code, upstream_sha = _git(base, "rev-parse", upstream_ref)
    if code != 0 or not upstream_sha:
        upstream_sha = ""

    code, status = _git(base, "status", "--porcelain", "--untracked-files=all")
    # A status call that fails is not a clean tree.
    is_clean = code == 0 and not status
    dirty_lines = [line.strip() for line in status.splitlines() if line.strip()] if code == 0 else []
    dirty_files = tuple(line.split(maxsplit=1)[-1] for line in dirty_lines[:_MAX_DIRTY_LISTED])

    unpushed = 0
    if upstream_sha:
        code, count = _git(base, "rev-list", "--count", f"{upstream_sha}..{head}")
        if code == 0 and count.isdigit():
            unpushed = int(count)

    is_pushed = bool(upstream_sha) and unpushed == 0

    if is_pushed and is_clean:
        verdict = "published"
        detail = f"HEAD {head[:12]} is {upstream_ref} and the tree is clean"
    else:
        verdict = "divergent"
        reasons: list[str] = []
        if not upstream_sha:
            reasons.append(f"no upstream {upstream_ref!r} to compare against")
        elif unpushed:
            reasons.append(f"{unpushed} unpushed commit(s) ahead of {upstream_ref}")
        if not is_clean:
            reasons.append(f"{len(dirty_lines)} uncommitted change(s)")
        detail = (
            "this artifact was produced by code a reader cannot obtain: "
            + "; ".join(reasons or ["tree differs from the published source"])
        )

    return SourceAttestation(
        verdict=verdict,
        head_sha=head,
        upstream_ref=upstream_ref,
        upstream_sha=upstream_sha,
        is_clean=is_clean,
        is_pushed=is_pushed,
        unpushed_commits=unpushed,
        dirty_files=dirty_files,
        dirty_count=len(dirty_lines),
        detail=detail,
    )


__all__ = ["SCHEMA", "SourceAttestation", "attest"]
