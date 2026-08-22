"""Whether a sealed learning artifact may be admitted, and why not.

The mathematics-memory tissue is admitted only when its manifest still matches
the evidence it was sealed with — including the SHA-256 of every source file the
training run was pinned against. That check is fail-closed and right to be.

What it could not do was say so to anybody. `neural_objective_producer` caught
the RuntimeError, fell back to the deterministic solver, and returned a receipt
of the same shape, so the answer arrived and looked ordinary. The only visible
effect of a sealed capability going offline was a registered runtime claim
quietly reporting False — with no way to tell "the seal is broken" from "the
tissue computed the wrong answer", which are different problems with different
owners.

Measured 2026-08-15: `core/learning/frontier_process_supervision.py` drifted
from its pinned hash in `8c48eec8d` (CP546, schema v1→v2, 317 lines). The
refusal is CORRECT — the code that produced the artifact is not the code on
disk — and re-sealing without re-running the canary would launder a real
provenance break. So this module does not fix the seal. It makes the break
legible, which is what was actually missing.

This lives outside `core/learning/recurrent_work_memory_tissue.py` deliberately:
that file is itself pinned, so adding a diagnostic to it would drift the very
hash the diagnostic reports on. A probe must not change what it measures.
"""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Learning.SealedAdmission")

SEALED_ADMISSION_SCHEMA = "aura.learning.sealed_artifact_admission.v1"

#: Sealed artifacts this runtime depends on, by name. Each entry is the module
#: attribute holding its directory, so a moved artifact is a resolution failure
#: here rather than a silently absent capability somewhere else.
_SEALED_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    (
        "mathematics_memory_tissue",
        "core.learning.recurrent_work_memory_tissue",
        "DEFAULT_MATHEMATICS_MEMORY_ARTIFACT",
    ),
)

# Health polls are frequent, while a strict artifact admission verifies every
# pinned source byte.  Cache only the *health projection*, and only while the
# complete dependency metadata signature is unchanged.  Real capability use
# continues through ``artifact_admission_status`` below and never consults this
# memo.  ``ctime_ns`` is included so preserving a file's mtime cannot conceal a
# rewrite on the local filesystem.
_HEALTH_ADMISSION_LOCK = checked_lock("core.learning.sealed_artifact_admission")
_HEALTH_ADMISSION_CACHE: dict[
    tuple[str, str, str], tuple[tuple[Any, ...], dict[str, Any]]
] = {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_metadata(path: Path) -> tuple[Any, ...]:
    """Cheap identity for one dependency, including missing/unreadable state."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (str(path), "missing")
    except OSError as exc:
        return (str(path), "unreadable", type(exc).__name__)
    return (
        str(path),
        stat.st_dev,
        stat.st_ino,
        stat.st_mode,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _artifact_dependency_signature(module: str, attribute: str) -> tuple[Any, ...]:
    """Metadata signature for everything a strict admission reads.

    This deliberately parses the small manifest on every health poll.  It
    avoids hashing model/source bytes, while still discovering a changed
    pinned-source set before a cached verdict can be reused.
    """
    try:
        import importlib

        directory = Path(getattr(importlib.import_module(module), attribute))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        return ("unresolvable", module, attribute, type(exc).__name__)

    manifest_path = directory / "manifest.json"
    dependencies: list[tuple[Any, ...]] = [
        ("artifact-root", *_path_metadata(directory)),
        ("manifest", *_path_metadata(manifest_path)),
    ]
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        dependencies.append(("manifest-error", type(exc).__name__))
        return tuple(dependencies)

    # The sealed loader may consume any file in the artifact directory.  Track
    # every descendant, including directories, so additions/removals and
    # replacements invalidate the verdict without reading large tensor files.
    try:
        for candidate in sorted(directory.rglob("*"), key=lambda item: str(item)):
            dependencies.append(("artifact-entry", *_path_metadata(candidate)))
    except OSError as exc:
        dependencies.append(("artifact-scan-error", type(exc).__name__))

    pinned = (manifest.get("canary") or {}).get("source_sha256s") or {}
    root = _repo_root()
    for relative in sorted(pinned):
        dependencies.append(("pinned-source", *_path_metadata(root / str(relative))))
    return tuple(dependencies)


def _health_artifact_admission_status(
    name: str,
    module: str,
    attribute: str,
) -> dict[str, Any]:
    """Strict verdict memoized until any consumed dependency changes."""
    key = (name, module, attribute)
    signature = _artifact_dependency_signature(module, attribute)
    with _HEALTH_ADMISSION_LOCK:
        cached = _HEALTH_ADMISSION_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            return deepcopy(cached[1])

        # Singleflight the expensive verification. Health polling is the only
        # caller of this helper; capability admission remains independently
        # strict on every use.
        status = artifact_admission_status(name, module, attribute)
        _HEALTH_ADMISSION_CACHE[key] = (signature, deepcopy(status))
        return status


def artifact_admission_status(name: str, module: str, attribute: str) -> dict[str, Any]:
    """One artifact's admission verdict, with the drifted sources named.

    Never raises. An availability probe that can fail is one every caller wraps
    in a broad except, and that is the silence this module exists to remove.
    """
    status: dict[str, Any] = {
        "artifact": name,
        "admitted": False,
        "reason": "",
        "drifted_sources": [],
        "pinned_source_count": 0,
    }
    try:
        import importlib

        directory = Path(getattr(importlib.import_module(module), attribute))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        status["reason"] = f"artifact location unresolvable: {type(exc).__name__}"
        return status

    try:
        manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    except (OSError, ValueError) as exc:
        status["reason"] = f"manifest unreadable: {type(exc).__name__}"
        return status

    pinned = (manifest.get("canary") or {}).get("source_sha256s") or {}
    status["pinned_source_count"] = len(pinned)
    root = _repo_root()
    drifted: list[str] = []
    for relative, expected in sorted(pinned.items()):
        candidate = root / str(relative)
        try:
            if not candidate.is_file():
                drifted.append(f"{relative} (missing)")
            elif _file_sha256(candidate) != expected:
                drifted.append(str(relative))
        except OSError:
            drifted.append(f"{relative} (unreadable)")
    status["drifted_sources"] = drifted

    try:
        from core.learning.recurrent_work_memory_tissue import (
            load_mathematics_memory_tissue,
        )

        load_mathematics_memory_tissue(directory)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, ImportError) as exc:
        status["reason"] = str(exc)[:200] or type(exc).__name__
        return status

    status["admitted"] = True
    return status


# Refusal sets already announced, so a polled health surface does not repeat
# the same warning on every call. Keyed by (artifact, reason) so a NEW reason
# for the same artifact is announced again.
_ANNOUNCED_REFUSALS: tuple[tuple[str, str], ...] = ()


def sealed_artifact_admission_report() -> dict[str, Any]:
    """Every sealed artifact, admitted or not, for a health surface to publish."""
    artifacts = [
        _health_artifact_admission_status(name, module, attribute)
        for name, module, attribute in _SEALED_ARTIFACTS
    ]
    refused = [a for a in artifacts if not a["admitted"]]
    # A capability that has gone offline because its provenance no longer holds
    # is worth saying out loud. It was previously visible only as a registered
    # claim reporting False. This is a query a health surface polls, so it
    # announces a refusal set once and again only when the set changes -- a
    # warning on every poll would bury the one that matters.
    signature = tuple(sorted((a["artifact"], a["reason"]) for a in refused))
    global _ANNOUNCED_REFUSALS
    if refused and signature != _ANNOUNCED_REFUSALS:
        _ANNOUNCED_REFUSALS = signature
        logger.warning(
            "🔒 Sealed artifact refused: %s",
            "; ".join(
                f"{a['artifact']} ({a['reason']}"
                + (f"; drifted: {', '.join(a['drifted_sources'])}" if a["drifted_sources"] else "")
                + ")"
                for a in refused
            ),
        )
    return {
        "schema": SEALED_ADMISSION_SCHEMA,
        "artifacts": artifacts,
        "admitted": sum(1 for a in artifacts if a["admitted"]),
        "declared": len(artifacts),
        "refused": [a["artifact"] for a in refused],
    }


def mathematics_memory_admitted() -> tuple[bool, str]:
    """Whether the sealed mathematics-memory tissue is usable right now.

    The single question `neural_objective_producer` and the runtime claim
    validator both need, and neither could ask.
    """
    status = artifact_admission_status(*_SEALED_ARTIFACTS[0])
    detail = status["reason"]
    if status["drifted_sources"]:
        detail = f"{detail}; drifted: {', '.join(status['drifted_sources'])}"
    return bool(status["admitted"]), detail


__all__ = [
    "SEALED_ADMISSION_SCHEMA",
    "artifact_admission_status",
    "mathematics_memory_admitted",
    "sealed_artifact_admission_report",
]


def _register_fragment() -> None:
    """Publish admission status to the runtime health surface."""
    try:
        from core.runtime.health_fragments import register_health_fragment

        register_health_fragment("sealed_artifacts", sealed_artifact_admission_report)
    except (ImportError, AttributeError):
        logger.debug("health fragment registry unavailable; admission not published")


_register_fragment()
