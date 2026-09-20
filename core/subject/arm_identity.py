"""What served an arm, hashed, so two arms can be shown to be the same system.

The paired design assumes every arm ran on one system and differed only in the
displacement. For the offline organism that is nearly free: the stub decoder is
deterministic and the code is whatever the process imported. Through the real
cortex it is not free at all. A fallback lane picking up one arm, a chat
template edited between arms, a pointer swung to a different checkpoint, a
module reloaded — each of those makes the two arms two systems, and the
displacement is then the smaller of the two things being measured.

None of it can be recovered afterwards from a report that did not record it. So
an arm pins what served it, cheaply and without loading anything: the pointer
and the digest of the files it names, the tokenizer and template that shape the
prompt, the decoding settings, the commit the code came from, and the import
paths of the packages that make the organism. Two pins that differ name their
differences, and a campaign whose arms do not share a pin has to say so instead
of reporting a rate.

Digests are taken from names, sizes and modification times rather than from
content. Reading twenty gigabytes of weights for every arm would cost more than
the arm, and the question here is whether the file changed between two arms
minutes apart, which a stat answers.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.subprocess_gateway import get_subprocess_gateway

__all__ = [
    "DECODING_KEYS",
    "ArmIdentity",
    "differences",
    "pin_arm_identity",
]

#: Sampler settings that have to be the same in both arms. A free sampler puts
#: uncontrolled variation between two arms that differ only in a displacement.
DECODING_KEYS: tuple[str, ...] = ("temperature", "top_p", "top_k", "seed")

#: How deep the model-artifact walk goes, and how many files it will take.
#: The resident 27B directory holds eleven files; one level covers the layouts
#: that put shards in a subdirectory, and the cap exists so an unresolved
#: pointer cannot set the walk loose on a filesystem.
MAX_ARTIFACT_DEPTH: int = 1
MAX_ARTIFACT_FILES: int = 4096

#: Files whose content shapes the prompt or the decode. Weights are covered by
#: the whole-directory stat digest; these are named because an edit to one of
#: them is the change most likely to go unnoticed.
SHAPING_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "config.json",
)


def _digest(parts: Any) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:32]


def _tree_digest(root: Path) -> tuple[str, int, bool]:
    """Names, sizes and times of the files that make up a model artifact.

    Depth-limited and capped. A model directory is flat — the resident 27B is
    eleven files beside a tokenizer and a template — so the walk needs one
    level for the layouts that put shards in a subdirectory and nothing more.
    The cap is there because the path comes from a registry that can hand back
    nothing: an unresolved pointer once left this walking the repository root,
    which does not finish. A walk that hits either bound says so rather than
    returning a digest of part of a directory as though it were the whole.
    """
    rows: list[tuple[str, int, int]] = []
    if not root.is_dir():
        return "", 0, False
    truncated = False
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for path in entries:
            if len(rows) >= MAX_ARTIFACT_FILES:
                return _digest(rows), len(rows), True
            try:
                if path.is_dir():
                    if depth < MAX_ARTIFACT_DEPTH:
                        stack.append((path, depth + 1))
                    else:
                        truncated = True
                    continue
                stat = path.stat()
            except OSError:
                continue
            rows.append(
                (str(path.relative_to(root)), int(stat.st_size), int(stat.st_mtime_ns))
            )
    return _digest(rows), len(rows), truncated


def _file_digests(root: Path) -> dict[str, str]:
    """The prompt-shaping files, hashed by content because they are small."""
    out: dict[str, str] = {}
    for name in SHAPING_FILES:
        path = root / name
        if not path.is_file():
            continue
        try:
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:32]
        except OSError:
            continue
    return out


def _commit() -> str:
    try:
        result = get_subprocess_gateway().run(
            ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            read_only=True,
            source="subject.arm_identity.commit",
            accelerator_capability="none",
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _foreign_modules(packages: tuple[str, ...]) -> tuple[dict[str, str], int]:
    """Organism modules loaded from outside this checkout, and how many were loaded.

    A hot reload that rebinds a class, or a second checkout on the path, makes
    two arms run two versions of one name, and the file a module was loaded from
    is what says so. Only the ones from somewhere else are named: a run imports
    more of itself as it goes, so a digest over every loaded module rises on
    every ordinary lazy import and reports a system change that did not happen.
    """
    root = str(Path(__file__).resolve().parents[2])
    foreign: dict[str, str] = {}
    seen = 0
    for name, module in sorted(sys.modules.items()):
        if not name.startswith(packages):
            continue
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if not origin:
            continue
        seen += 1
        if not str(origin).startswith(root):
            foreign[name] = str(origin)
    return foreign, seen


@dataclass(frozen=True)
class ArmIdentity:
    """Everything about the system an arm ran on, short of the weights bytes."""

    pointer: str = ""
    model_path: str = ""
    #: Names, sizes and times of every file under the model directory.
    weights_digest: str = ""
    weights_files: int = 0
    #: True when the walk hit its depth or file bound, so the digest covers
    #: part of the artifact. A pin that cannot see all of it must not read as
    #: one that looked and found nothing more.
    weights_truncated: bool = False
    #: The prompt-shaping files, hashed by content.
    shaping: dict[str, str] = field(default_factory=dict)
    decoding: dict[str, Any] = field(default_factory=dict)
    commit: str = ""
    #: Organism modules loaded from outside this checkout, by name.
    foreign_modules: dict[str, str] = field(default_factory=dict)
    #: How many organism modules were loaded. Recorded, never compared: a run
    #: imports more of itself as it goes.
    modules_loaded: int = 0
    #: The process the pin was taken in. Two pins from one process cannot
    #: differ in the code that is running, whatever the checkout does.
    process: int = 0
    #: True when the run was served by the deterministic stub rather than by a
    #: cortex. A substrate campaign is authoritative with this set; a
    #: cortex-inclusive claim is not.
    stubbed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "pointer": self.pointer,
            "model_path": self.model_path,
            "weights_digest": self.weights_digest,
            "weights_files": self.weights_files,
            "weights_truncated": self.weights_truncated,
            "shaping": dict(self.shaping),
            "decoding": dict(self.decoding),
            "commit": self.commit,
            "foreign_modules": dict(self.foreign_modules),
            "modules_loaded": self.modules_loaded,
            "process": self.process,
            "stubbed": self.stubbed,
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "ArmIdentity":
        return cls(
            pointer=str(blob.get("pointer", "")),
            model_path=str(blob.get("model_path", "")),
            weights_digest=str(blob.get("weights_digest", "")),
            weights_files=int(blob.get("weights_files", 0) or 0),
            weights_truncated=bool(blob.get("weights_truncated", False)),
            shaping=dict(blob.get("shaping", {}) or {}),
            decoding=dict(blob.get("decoding", {}) or {}),
            commit=str(blob.get("commit", "")),
            foreign_modules=dict(blob.get("foreign_modules", {}) or {}),
            modules_loaded=int(blob.get("modules_loaded", 0) or 0),
            process=int(blob.get("process", 0) or 0),
            stubbed=bool(blob.get("stubbed", True)),
        )


def pin_arm_identity(
    spec: Any = None,
    *,
    packages: tuple[str, ...] = ("core", "interface", "llm", "skills"),
) -> ArmIdentity:
    """Pin what is serving right now. Loads nothing.

    ``spec`` is the active cortex spec when there is one. Without it the run is
    on the stub, which is recorded rather than guessed at: a report that cannot
    say which of the two it was cannot support either claim.
    """
    if spec is None:
        try:
            from core.brain.llm.model_registry import get_active_cortex_spec

            spec = get_active_cortex_spec()
        except (ImportError, RuntimeError, OSError, ValueError):
            spec = None

    # An unresolved pointer hands back nothing, and Path("") is Path("."). The
    # first version of this walked the repository root for a run with no cortex
    # and did not come back.
    foreign, loaded = _foreign_modules(packages)
    named = str(getattr(spec, "model_path", "") or "") if spec is not None else ""
    model_path = Path(named) if named else None
    if model_path is None:
        weights_digest, weights_files, truncated = "", 0, False
    else:
        weights_digest, weights_files, truncated = _tree_digest(model_path)
    decoding: dict[str, Any] = {}
    for key in DECODING_KEYS:
        env = os.environ.get(f"AURA_{key.upper()}")
        if env is not None:
            decoding[key] = env
            continue
        try:
            from core.config import get_config

            value = getattr(get_config(), key, None)
        except (ImportError, AttributeError, RuntimeError):
            value = None
        if value is not None:
            decoding[key] = value
    # The registry's own identity for the pointer, not a display name. A tag
    # can be moved to different weights; these hashes cannot.
    pointer = " ".join(
        str(getattr(spec, field, "") or "")
        for field in ("base_model", "tag", "pointer_sha256", "descriptor_sha256")
    ).strip() if spec is not None else ""
    return ArmIdentity(
        pointer=pointer,
        model_path=named,
        weights_digest=weights_digest,
        weights_files=weights_files,
        weights_truncated=truncated,
        shaping=_file_digests(model_path) if model_path is not None else {},
        decoding=decoding,
        commit=_commit(),
        foreign_modules=foreign,
        modules_loaded=loaded,
        process=os.getpid(),
        stubbed=weights_files == 0,
    )


def differences(first: ArmIdentity, second: ArmIdentity) -> list[str]:
    """What changed between two arms, in the words a reader needs.

    Empty means the two arms ran on one system as far as anything short of
    reading the weights can tell.
    """
    out: list[str] = []
    if first.pointer != second.pointer:
        out.append(f"the cortex pointer moved: {first.pointer!r} then {second.pointer!r}")
    if first.model_path != second.model_path:
        out.append(f"the model path moved: {first.model_path!r} then {second.model_path!r}")
    if first.weights_digest != second.weights_digest:
        out.append(
            "the model directory changed between the arms "
            f"({first.weights_files} files then {second.weights_files})"
        )
    for name in sorted(set(first.shaping) | set(second.shaping)):
        if first.shaping.get(name) != second.shaping.get(name):
            out.append(f"{name} changed between the arms")
    for key in DECODING_KEYS:
        if first.decoding.get(key) != second.decoding.get(key):
            out.append(
                f"{key} changed between the arms: "
                f"{first.decoding.get(key)!r} then {second.decoding.get(key)!r}"
            )
    # The checkout can move under a running process without changing a line of
    # what that process is executing, so a commit is only evidence of two
    # systems when the two pins were taken in two processes — which is what a
    # resumed run is.
    if first.process != second.process and first.commit != second.commit:
        out.append(
            f"the arms ran in two processes on two commits: "
            f"{first.commit[:12]} then {second.commit[:12]}"
        )
    for name in sorted(set(first.foreign_modules) | set(second.foreign_modules)):
        was = first.foreign_modules.get(name)
        now = second.foreign_modules.get(name)
        if was != now:
            out.append(f"{name} was loaded from {was!r} and then from {now!r}")
    if first.weights_truncated or second.weights_truncated:
        out.append(
            "the model artifact was too deep or too large to read whole, so the "
            "arms were compared on part of it"
        )
    if first.stubbed != second.stubbed:
        out.append(
            "one arm ran on the stub and the other on a cortex, which is two systems"
        )
    return out
