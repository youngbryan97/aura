"""What was measured, on what code, with which numbers fixed in advance.

A battery result is worth nothing without the campaign around it. Two runs of
the same command on two different heads produce two numbers that look
comparable and are not, and a threshold that can be edited between the run and
the reading is not a threshold. So every artifact carries the commit it ran on,
the hash of the working tree, the hash of every value that could change the
answer, and the schema the state was read through.

The hash is the point. It is cheap to say a campaign was frozen and hard to
prove it; a fingerprint over the thresholds, the conditions, the displacement
size, the injection points, the null list, the seeds and the schema turns that
claim into something a reader can check against a second run. Two runs with the
same fingerprint were measured the same way. Two with different fingerprints
belong to different campaigns however similar the command looked.

A dirty tree is recorded as dirty rather than refused. Refusing would make the
tool unusable during the work it exists to support, and a reader who sees
`dirty: true` knows exactly how much weight the run can take.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["campaign", "fingerprint", "next_run_directory"]

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def _tree_hash() -> str:
    """A hash over the tracked files that decide the answer.

    The whole tree would change with every artifact written, so this covers the
    measurement code and the organism it measures, which is what a second run
    would have to match to be the same campaign.
    """
    paths = _git(
        "ls-files",
        "core/subject",
        "core/consciousness",
        "core/phases",
        "core/agency",
        "core/ontogeny",
        "tools/run_subject_core.py",
    ).splitlines()
    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(paths):
        target = REPO / name
        try:
            digest.update(name.encode())
            digest.update(target.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()


def fingerprint(frozen: dict[str, Any]) -> str:
    """One short hash over everything that was fixed before the run."""
    blob = json.dumps(frozen, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()


def campaign(*, seed: int, rounds: int, trials: int, turns: int) -> dict[str, Any]:
    """Everything a second run would have to match to be the same measurement."""
    from core.subject.battery import THRESHOLDS
    from core.subject.causal import (
        DEFAULT_DELTA,
        DIVERGENCE_CEILING,
        EDGE_EFFECT,
        EDGE_QVALUE,
        EDGE_REPLICATION,
        SIGN_FLIP_DRAWS,
    )
    from core.subject.driver import CONDITIONS, SUBSTRATE_STEP_SECONDS
    from core.subject.irreducibility import COMPONENTS, FOLDS
    from core.subject.nulls import ARCHITECTURES
    from core.subject.state import DOMAINS, feature_names
    from core.subject.synergy import TRIPLES

    schema = feature_names()
    frozen: dict[str, Any] = {
        "thresholds": dict(sorted(THRESHOLDS.items())),
        "edge_rules": {
            "q_max": EDGE_QVALUE,
            "effect_min": EDGE_EFFECT,
            "replication_min": EDGE_REPLICATION,
            "sign_flip_draws": SIGN_FLIP_DRAWS,
        },
        "intervention": {
            "delta": DEFAULT_DELTA,
            "divergence_ceiling": DIVERGENCE_CEILING,
            "trials": trials,
            "turns_per_arm": turns,
            "substrate_step_seconds": SUBSTRATE_STEP_SECONDS,
        },
        "recording": {"rounds": rounds, "conditions": [c.name for c in CONDITIONS]},
        "estimator": {"components_per_domain": COMPONENTS, "folds": FOLDS},
        "nulls": {"architectures": list(ARCHITECTURES)},
        "synergy_triples": [list(t) for t in TRIPLES],
        "domains": list(DOMAINS),
        "schema": {"width": len(schema), "hash": hashlib.blake2b(
            "|".join(schema).encode(), digest_size=16
        ).hexdigest()},
        "seed": seed,
    }
    return {
        "frozen": frozen,
        "fingerprint": fingerprint(frozen),
        "commit": _git("rev-parse", "HEAD"),
        "commit_subject": _git("log", "-1", "--format=%s"),
        "tree_hash": _tree_hash(),
        "dirty": bool(_git("status", "--porcelain", "core", "tools")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "started_at": time.time(),
    }


def next_run_directory(root: Path) -> Path:
    """`run_001`, `run_002`, and never a name that already holds a report.

    Overwriting the previous run is how a campaign loses the run that did not
    come out well, and the run that did not come out well is the one a reader
    most needs.
    """
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in root.glob("run_*") if p.is_dir())
    index = 1
    if existing:
        try:
            index = int(existing[-1].split("_")[-1]) + 1
        except ValueError:
            index = len(existing) + 1
    while (root / f"run_{index:03d}").exists():
        index += 1
    return root / f"run_{index:03d}"
