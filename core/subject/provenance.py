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
import inspect
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.subject.clock import real_time

__all__ = [
    "TREE_PATHS",
    "campaign",
    "campaign_v25",
    "tree_hash_at",
    "environment",
    "fingerprint",
    "run_fingerprint",
    "manifest",
    "mind_identity",
    "next_run_directory",
]

REPO = Path(__file__).resolve().parents[2]


def _run_git(*args: str, text: bool = True, timeout: float = 30.0) -> Any | None:
    """Run git through its gateway. None when git could not answer.

    One call site, two shapes. A listing wants the text stripped; a blob about
    to be hashed wants the bytes exactly as git holds them, because a digest
    over a round-trip through str is a digest of something else. Splitting
    them into two gateway calls was two buckets of effect-ownership debt for
    one spawn, and the same receipt either way.
    """
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    try:
        out = get_subprocess_gateway().run(
            ["git", *args],
            cwd=REPO,
            capture_output=True,
            text=text,
            timeout=timeout,
            check=False,
            read_only=True,
            source="subject_core.provenance.git",
            accelerator_capability="none",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _git(*args: str) -> str:
    """What git printed, stripped. Empty when it could not answer."""
    printed = _run_git(*args)
    return "" if printed is None else str(printed or "").strip()


def _git_bytes(*args: str) -> bytes | None:
    """A blob exactly as git holds it, or None if git could not produce it."""
    raw = _run_git(*args, text=False, timeout=60.0)
    return None if raw is None else bytes(raw or b"")


#: What "the same code" means for a campaign. The whole tree would change with
#: every artifact written, so this is the measurement code and the organism it
#: measures — what a second run would have to match to be the same campaign.
TREE_PATHS: tuple[str, ...] = (
    "core/subject",
    "core/consciousness",
    "core/phases",
    "core/agency",
    "core/ontogeny",
    "tools/run_subject_core.py",
)


def _tree_hash() -> str:
    """A hash over the tracked files that decide the answer, as they are now."""
    paths = _git("ls-files", *TREE_PATHS).splitlines()
    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(paths):
        target = REPO / name
        try:
            digest.update(name.encode())
            digest.update(target.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()


def tree_hash_at(commit: str) -> str:
    """The same hash, over the files as they were at one commit.

    A run records the working-tree digest. Checking later that a result still
    describes its commit means recomputing the digest from that commit's blobs,
    not from whatever the tree holds now — those differ as soon as anyone
    commits anything, which is not the same fact as a rewritten history.
    """
    listing = _git("ls-tree", "-r", "--name-only", commit, "--", *TREE_PATHS)
    names = [line for line in listing.splitlines() if line]
    if not names:
        return ""
    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(names):
        blob = _git_bytes("-C", str(REPO), "show", f"{commit}:{name}")
        if blob is None:
            continue
        digest.update(name.encode())
        digest.update(blob)
    return digest.hexdigest()


def fingerprint(frozen: dict[str, Any]) -> str:
    """One short hash over the method: everything fixed before the run but the seed.

    Two runs differing only in their seed are the same experiment run twice,
    which is what a replicate is. Folding the seed in made every replicate its
    own campaign, so the one comparison a campaign exists to support — the same
    method, independently initialised — could not be made without the scorecard
    refusing to read across the runs.

    The seed is recorded beside this and enters `run_fingerprint`, so a
    particular run is still identified exactly.
    """
    method = {key: value for key, value in frozen.items() if key != "seed"}
    blob = json.dumps(method, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()


def run_fingerprint(frozen: dict[str, Any]) -> str:
    """The method and the seed: this exact run, reproducibly."""
    blob = json.dumps(frozen, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()


def campaign(
    *,
    seed: int,
    rounds: int,
    trials: int,
    turns: int,
    lesion_rounds: int = 0,
    lesion_cycles: int = 1,
) -> dict[str, Any]:
    """Everything a second run would have to match to be the same measurement."""
    from core.subject.battery import DEFICIT_SHARE, RECOVERY_TOLERANCE, THRESHOLDS
    from core.subject.causal import (
        DEFAULT_DELTA,
        DIVERGENCE_CEILING,
        EDGE_EFFECT,
        EDGE_QVALUE,
        EDGE_REPLICATION,
        SIGN_FLIP_DRAWS,
        SUSTAINED,
    )
    from core.subject.driver import CONDITIONS, SECONDS_PER_TURN, SUBSTRATE_BODY
    from core.subject.irreducibility import COMPONENTS, FOLDS
    from core.subject.nulls import ARCHITECTURES
    from core.subject.state import DOMAINS, feature_names
    from core.subject.steppable import LAYERS
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
            # Which domains are held at the displacement rather than pushed
            # once. `do(X)` holds X, and a pulse is a different intervention.
            "sustained": sorted(SUSTAINED),
        },
        # What the free-running layers run at, which is what makes a counted
        # schedule the same organism as a timed one. The rates are each
        # layer's own configuration, and the frame they are counted against
        # follows from one turn being worth one second and from how many
        # readings a turn takes — a property of the phase list, not of the
        # machine, so two machines give the organism the same life.
        "seconds_per_turn": SECONDS_PER_TURN,
        "layer_rates": {layer.name: layer.hz for layer in LAYERS},
        "layer_body": {
            layer.name: [
                call.method if call.every == 1 else f"{call.method}/{call.every}"
                for call in layer.body
            ]
            for layer in LAYERS
        },
        # The substrate is scheduled by the driver rather than through LAYERS,
        # and its body decides what organism the run measured just as much.
        "substrate_body": [
            name if every == 1 else f"{name}/{every}"
            for name, every, _ in SUBSTRATE_BODY
        ],
        "recording": {"rounds": rounds, "conditions": [c.name for c in CONDITIONS]},
        # How long the lesion and rescue arms live. It changes what those two
        # criteria are measured on, so it belongs in the fingerprint with
        # everything else that does.
        "lesion": {
            "rounds": lesion_rounds,
            # How many times the cut is made and released. One cycle gives one
            # reading of each arm and no way to tell a deficit from the noise
            # around it; the count is frozen here because spending the same
            # budget as three cycles instead of one changes what the lesion
            # and rescue criteria are measured on.
            "cycles": lesion_cycles,
            # How much of the deficit a rescue has to bring back, and how large
            # a deficit has to be before there is one to bring back. Fixed here
            # so a tolerance cannot be chosen after seeing which measure
            # recovered.
            "recovery_tolerance": RECOVERY_TOLERANCE,
            "deficit_share": DEFICIT_SHARE,
        },
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
        "run_fingerprint": run_fingerprint(frozen),
        "commit": _git("rev-parse", "HEAD"),
        "commit_subject": _git("log", "-1", "--format=%s"),
        "tree_hash": _tree_hash(),
        "dirty": bool(_git("status", "--porcelain", "core", "tools")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        # The machine's clock, not the experiment's: how long a run took is a
        # question about the host, and the experiment's clock is stopped.
        "started_at": real_time(),
    }


def campaign_v25(
    *,
    seed: int,
    rounds: int,
    anchors: int,
    history_turns: int,
    turns: int,
    cut_rounds: int,
    support: Sequence[str],
) -> dict[str, Any]:
    """The v25 fingerprint, which is a different campaign from the battery's.

    A methodological change after seeing a result has to start a new campaign
    or the scorecard is reading across two experiments. That rule applies to
    v25 on its own terms: what it freezes is the action basis, the frequency
    bank the signatures are read through, the horizon ladder and the rule for
    extending it, the estimator, the sequential stopping rule and the
    tolerances. Move any of them and the hash moves with it.
    """
    from core.subject.causal import SUSTAINED
    from core.subject.driver import CONDITIONS, SECONDS_PER_TURN
    from core.subject.state import DOMAINS, feature_names
    from core.subject.steppable import LAYERS
    from core.subject.v25_cut import ANCHOR_STEP, OPENING_ANCHORS

    # Imported from the runner so the frozen values are the ones in force
    # rather than a second copy that can drift away from them.
    from tools.run_subject_core_v25 import (  # noqa: PLC0415
        FREQUENCIES,
        FREQUENCY_SEED,
        INVARIANCE_TOLERANCE,
        LAG_CEILING,
        LAGS,
        SUFFICIENCY_TOLERANCE,
    )

    schema = feature_names()
    frozen: dict[str, Any] = {
        "generation": "v25",
        "target": "F_intrinsic = Phi_FR / tau, gated on closure and recurrence",
        "grain": {
            "signature": "characteristic function at preregistered frequencies",
            "frequencies_per_test": FREQUENCIES,
            "frequency_seed": FREQUENCY_SEED,
            "rank_rule": "parallel analysis against column-shuffled signatures",
            "sufficiency_tolerance": SUFFICIENCY_TOLERANCE,
            "history_turns": history_turns,
        },
        "metric": {
            "name": "Fisher-Rao",
            "estimator": "cross-fitted k-NN posterior -> Bhattacharyya -> 2 arccos",
            "floor": "sham against sham, same estimator",
            "lower_bound": "paired bootstrap, alpha 0.05",
            "p_value": "paired randomization over common forks",
        },
        "horizons": {
            "lags_frames": list(LAGS),
            "ceiling_frames": LAG_CEILING,
            "extension_rule": "double while the maximum sits in the last two bins",
            "reported": "the whole spectrum; tau-star is a summary, not a law",
        },
        "cuts": {
            "enumeration": "every bipartition, first domain fixed left",
            "opening_anchors": OPENING_ANCHORS,
            "anchor_step": ANCHOR_STEP,
            "rounds": cut_rounds,
            "stopping_rule": "a cut stops drawing once its lower bound clears zero",
            "score": "the weakest cut, as an intersection-union over all of them",
            "cut_construction": "clamp both sides in turn, compose the free halves",
        },
        "invariance_tolerance": INVARIANCE_TOLERANCE,
        "nulls": ["playback", "duplicate_coordinates", "invertible_recoding"],
        "intervention": {"sustained": sorted(SUSTAINED), "turns_per_arm": turns},
        "seconds_per_turn": SECONDS_PER_TURN,
        "layer_rates": {layer.name: layer.hz for layer in LAYERS},
        "recording": {"rounds": rounds, "conditions": [c.name for c in CONDITIONS]},
        "anchors": anchors,
        "support": list(support),
        "domains": list(DOMAINS),
        "schema": {
            "width": len(schema),
            "hash": hashlib.blake2b("|".join(schema).encode(), digest_size=16).hexdigest(),
        },
        "seed": seed,
    }
    return {
        "frozen": frozen,
        "fingerprint": fingerprint(frozen),
        "run_fingerprint": run_fingerprint(frozen),
        "commit": _git("rev-parse", "HEAD"),
        "commit_subject": _git("log", "-1", "--format=%s"),
        "tree_hash": _tree_hash(),
        "dirty": bool(_git("status", "--porcelain", "core", "tools")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "started_at": real_time(),
    }


def environment() -> dict[str, Any]:
    """What the run ran on, beyond the commit.

    A reader who wants to reproduce a number needs the machine and the
    dependencies as well as the source. None of these enter the fingerprint —
    two machines running the same campaign are the same campaign, and whether
    the numbers agree is the question a second machine is run to answer.
    """
    import importlib.metadata as metadata

    locks: list[str] = []
    for name in ("requirements.txt", "requirements-lock.txt", "uv.lock", "poetry.lock"):
        path = Path(__file__).resolve().parents[2] / name
        if path.is_file():
            locks.append(f"{name}:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")
    hardware: dict[str, Any] = {}
    try:
        # Through the observer, not through psutil. A run recorded under a
        # simulated observer has to say the machine the run believed it was
        # on, and reading the host directly here would write the real one
        # into the provenance of a run that never saw it.
        from core.runtime import resource_psutil

        hardware = {
            "cpus": resource_psutil.cpu_count(logical=True),
            "physical_cpus": resource_psutil.cpu_count(logical=False),
            "memory_gb": round(resource_psutil.virtual_memory().total / 1e9, 1),
        }
    except Exception:  # noqa: BLE001 - a machine that will not describe itself says so
        hardware = {"note": "the machine did not describe itself"}
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "torch", "scikit-learn"):
        try:
            packages[name] = metadata.version(name)
        except Exception:  # noqa: BLE001
            packages[name] = "absent"
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "hardware": hardware,
        "packages": packages,
        "dependency_locks": locks,
    }


def mind_identity(mind: Any) -> dict[str, Any]:
    """Which model answered, exactly.

    The battery installs a deterministic stub so two arms differ by the
    intervention rather than by decoding. That is a choice with consequences —
    no edge measured under it runs through language — so the run records which
    mind it had rather than leaving a reader to assume the cortex was up.
    """
    if mind is None:
        return {"kind": "absent"}
    identity: dict[str, Any] = {
        "kind": type(mind).__name__,
        "module": type(mind).__module__,
    }
    for name in ("model_name", "model_path", "tokenizer", "chat_template", "checksum"):
        value = getattr(mind, name, None)
        if isinstance(value, (str, int, float)):
            identity[name] = value
    source = ""
    try:
        source = inspect.getsource(type(mind))
    except (OSError, TypeError):
        source = ""
    if source:
        identity["code_sha256"] = hashlib.sha256(source.encode()).hexdigest()[:16]
    identity["deterministic"] = bool(getattr(mind, "deterministic", True))
    return identity


def manifest(directory: Path) -> dict[str, str]:
    """A SHA-256 for every file the run wrote, so a copy can be checked."""
    out: dict[str, str] = {}
    for item in sorted(Path(directory).rglob("*")):
        if not item.is_file() or item.name == "manifest.json":
            continue
        digest = hashlib.sha256()
        try:
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
        except OSError:
            continue
        out[str(item.relative_to(directory))] = digest.hexdigest()
    return out


def next_run_directory(root: Path) -> Path:
    """`run_001`, `run_002`, and never a name that already holds a report.

    Overwriting the previous run is how a campaign loses the run that did not
    come out well, and the run that did not come out well is the one a reader
    most needs.
    """
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("subject_core.provenance"):
        get_file_write_gateway().ensure_directory(root, source="subject_core.provenance")
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
