"""tools/agi_gauntlet/protocol.py — the freeze, and what it is worth.

An evaluation whose tasks existed before the system did tells you about the
overlap between the two. The control that removes that is the only one that
cannot be added afterwards: fix the system, then build the tasks.

    H = the commit                W = the model weights
    C = the configuration         S = the seed every environment comes from

S is derived from H, W and C, and from nothing else. So an environment in this
gauntlet could not have existed when the commit was written, because its shape
is a function of the commit's own hash — change one line of the organism and
every environment changes with it. That is weaker than an outside team
inventing tasks in a room Aura has never been in, and it is stronger than a
fixture checked in beside the code, and the difference between those three is
worth saying out loud rather than blurring.

What this file will not do
--------------------------
It will not report a score for something it did not run. A gate needing the
ARC-AGI-2 private set, a GAIA holdout, an OSWorld image or a human baseline is
marked NOT RUN with the protocol for running it, because a benchmark harness
that quietly substitutes a proxy for the thing it names is how a system gets
credited with a capability nobody measured.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Freeze",
    "Receipt",
    "seed_for",
    "take_the_freeze",
]


def _run(*command: str) -> str:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _digest_of_tree(root: Path, *, suffixes: tuple[str, ...] = (".py",)) -> str:
    """A hash of the organism's own source, so an uncommitted edit shows.

    The commit hash alone says what was committed. This says what was run,
    and the two differ exactly when somebody edits a file mid-evaluation —
    which is the failure this whole protocol exists to make visible.
    """

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix not in suffixes or not path.is_file():
            continue
        # Relative to the root, not absolute. Checking the absolute parts
        # excluded every file whenever the checkout itself sat under a dotted
        # directory — which every worktree here does — so the digest of the
        # whole organism came back as the digest of nothing, and every freeze
        # from a worktree had the same seed.
        inside = path.relative_to(root)
        if any(
            part.startswith(".") or part == "__pycache__" for part in inside.parts
        ):
            continue
        digest.update(str(inside).encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()


@dataclass(frozen=True)
class Freeze:
    """What was frozen, and when. Everything downstream hangs off this."""

    commit: str
    dirty: bool
    source_digest: str
    weights: str
    config: str
    at: float = field(default_factory=time.time)
    host: str = field(default_factory=lambda: platform.platform())

    @property
    def seed(self) -> int:
        """The one number every sealed environment is generated from."""

        return seed_for(self.commit, self.source_digest, self.weights, self.config)

    @property
    def trustworthy(self) -> bool:
        """Whether the freeze means what it says.

        A dirty tree is not frozen. The commit hash names something other than
        what ran, so an environment derived from it is derived from a
        description of the system rather than the system.
        """

        return not self.dirty and bool(self.commit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "dirty": self.dirty,
            "source_digest": self.source_digest,
            "weights": self.weights,
            "config": self.config,
            "seed": self.seed,
            "trustworthy": self.trustworthy,
            "at": self.at,
            "host": self.host,
        }


def seed_for(*parts: str) -> int:
    digest = hashlib.sha256("|".join(str(one) for one in parts).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


def _weights_digest(root: Path) -> str:
    """What model is behind this, named rather than hashed byte by byte.

    Hashing twenty gigabytes of weights before every run is not a control, it
    is a delay. The pointer and its size and mtime identify the file; a
    reviewer reproducing the run checks the pointer and hashes it once.
    """

    pointer = os.environ.get("AURA_MODEL_PATH") or ""
    if not pointer:
        for candidate in ("config/model_registry.json", "config/models.json"):
            where = root / candidate
            if where.exists():
                try:
                    return hashlib.sha256(where.read_bytes()).hexdigest()[:32]
                except OSError:
                    continue
        return "unnamed"
    where = Path(pointer)
    if not where.exists():
        return f"missing:{pointer}"
    try:
        stat = where.stat()
        return hashlib.sha256(
            f"{where.name}:{stat.st_size}:{int(stat.st_mtime)}".encode()
        ).hexdigest()[:32]
    except OSError:
        return f"unreadable:{pointer}"


def take_the_freeze(root: Path | None = None) -> Freeze:
    """Record what is about to be evaluated, before anything is generated."""

    here = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    commit = _run("git", "-C", str(here), "rev-parse", "HEAD")
    # What makes a freeze untrustworthy is source that differs from the
    # commit. The receipts this harness writes are not source, and counting
    # them made every run report itself untrustworthy for having been run.
    status = "\n".join(
        line
        for line in _run("git", "-C", str(here), "status", "--porcelain").splitlines()
        if any(
            line[3:].startswith(where)
            for where in ("core/", "interface/", "skills/", "llm/", "executors/",
                          "security/", "tools/", "config/", "tests/")
        )
    )
    config = ""
    where = here / "config"
    if where.exists():
        config = _digest_of_tree(where, suffixes=(".json", ".yaml", ".yml"))[:32]
    return Freeze(
        commit=commit,
        dirty=bool(status.strip()),
        source_digest=_digest_of_tree(here / "core")[:32],
        weights=_weights_digest(here),
        config=config,
    )


@dataclass
class Receipt:
    """What one gate did, written down so somebody else can check it.

    Every number a gate reports carries the run that produced it. A gauntlet
    whose output is a table of scores is a claim; one whose output is a table
    of scores with the trajectories behind them is evidence.
    """

    gate: str
    freeze: Freeze
    ran: bool = False
    why_not: str = ""
    passed: bool | None = None
    measurements: dict[str, Any] = field(default_factory=dict)
    trajectories: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    #: Whether the clocks were held while this gate ran. A number measured
    #: with time frozen is a different number, and reporting it as an ordinary
    #: one is the dishonest part — Soar can run for N decisions and that is why
    #: its experiments reproduce, so the mode goes on the receipt.
    clocks_held: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "ran": self.ran,
            "why_not": self.why_not,
            "passed": self.passed,
            "measurements": self.measurements,
            "trajectories": self.trajectories[:200],
            "trajectories_total": len(self.trajectories),
            "seconds": round(self.seconds, 2),
            "clocks_held": self.clocks_held,
            "freeze": self.freeze.to_dict(),
        }

    def write(self, into: Path) -> Path:
        into.mkdir(parents=True, exist_ok=True)
        where = into / f"{self.gate.replace(' ', '_')}.json"
        where.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return where
