"""What this process is RUNNING, versus what is on disk right now.

MEASURED live 2026-08-10. Three source files were edited and committed while
the runtime was up. The user clicked "Hot-reload Aura code from disk" and the
log said:

    ✅ [HOT-RELOAD] Hot reboot complete. 0 modules reloaded.

A success tick for doing nothing. Two things were wrong under it, and the
second is the interesting one.

The narrow fault: ``hot_reboot`` only ever considered modules named
``core.phases*`` or ``core.kernel*`` plus the classes of registered phases, so
``core.conversation.*`` and ``interface.routes.*`` — where the conversation
actually happens — were not candidates for reloading and never could be. The
button could not do what its own tooltip promised.

The wide fault: NOTHING in the runtime knew the source had moved underneath it.
``SourceBodyAwareness`` is the organ for "someone is operating on me", but it
watches the GIT DIRTY STATE — the difference between the working tree and the
last commit. That is a different quantity. Commit the edit and the tree is
clean while the running process is just as stale. So:

  * hot reload had no way to choose what to reload;
  * the health surface could not report that the process was behind its source;
  * and asked "are you running my latest fix?", she had no instrument to read.

This measures the missing quantity directly, without needing a boot-time hook.

CPython writes a bytecode cache whose header records the SOURCE mtime and size
at the moment it compiled (PEP 3147), or a hash of the source under PEP 552.
That header is the process's own record of what it compiled, and it is not
rewritten while nothing re-imports the module. Comparing it against the file on
disk answers exactly the right question: has this file changed since the
interpreter last compiled it?

Where no cache exists — ``sys.dont_write_bytecode``, a read-only tree, a
frozen or namespace module — a boot baseline recorded by ``record_baseline()``
is used instead, and modules with neither are reported as ``unknown`` rather
than silently counted as fresh. An unmeasurable module is not a clean one.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import marshal
import os
import struct
import sys
import threading
from pathlib import Path
from typing import Any, Iterable
from core.governance_context import local_internal_governed_scope
from core.runtime.dynamic_execution_gateway import get_dynamic_execution_gateway
from core.runtime.lockdep import LockRank, checked_lock

#: Path components that mean "not this project's source".
_VENDOR_PARTS = frozenset({".venv", "venv", "site-packages", "dist-packages", "node_modules", ".git"})

#: PEP 552 flag bit: the cache stores a source hash rather than mtime+size.
_PYC_HASH_BASED = 0b1
_PYC_HEADER_BYTES = 16


def _project_root() -> Path:
    """The repository this module lives in."""
    return Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class ModuleDrift:
    """One loaded module whose file no longer matches what was compiled."""

    module: str
    path: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"module": self.module, "path": self.path, "reason": self.reason}


@dataclasses.dataclass(frozen=True)
class DriftReport:
    """Every project module this process has loaded, and which have moved."""

    checked: int
    stale: tuple[ModuleDrift, ...]
    unknown: tuple[str, ...]

    @property
    def is_stale(self) -> bool:
        return bool(self.stale)

    @property
    def stale_modules(self) -> tuple[str, ...]:
        return tuple(drift.module for drift in self.stale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "stale_count": len(self.stale),
            "stale": [drift.to_dict() for drift in self.stale],
            "unknown_count": len(self.unknown),
            "unknown": list(self.unknown[:20]),
            "is_stale": self.is_stale,
        }

    def narrative(self) -> str:
        """One sentence a person can act on, in plain words."""
        if not self.stale:
            if self.unknown:
                return (
                    f"Running code matches the {self.checked} source files I can "
                    f"check; {len(self.unknown)} could not be checked."
                )
            return f"Running code matches all {self.checked} source files on disk."
        files = ", ".join(drift.module for drift in self.stale[:4])
        more = "" if len(self.stale) <= 4 else f", and {len(self.stale) - 4} more"
        return (
            f"{len(self.stale)} of {self.checked} loaded source files have changed "
            f"on disk since I compiled them ({files}{more}). I am running the older "
            "version of those until they are reloaded or I am restarted."
        )


def _read_cache_header(cache_path: Path) -> tuple[int, int, int] | None:
    """``(flags, mtime, size)`` from a bytecode cache header, or None."""
    try:
        with open(cache_path, "rb") as handle:
            header = handle.read(_PYC_HEADER_BYTES)
    except OSError:
        return None
    if len(header) < _PYC_HEADER_BYTES:
        return None
    try:
        _magic, flags, second, third = struct.unpack("<4sIII", header)
    except struct.error:
        return None
    return int(flags), int(second), int(third)


def _compiled_bodies_differ(source: Path, cache: Path) -> bool | None:
    """Does the current source compile to different bytecode than the cache?

    A timestamp is evidence of an edit, not of a CHANGE. ``git checkout``, a
    formatter that rewrites a file byte-identically, or restoring a backup all
    move the mtime while leaving the running code correct, and an instrument
    that cries stale on those is one nobody will trust by the third time.

    So a timestamp mismatch is only ever a suspicion here, confirmed against
    what the bytes actually compile to. Returns None when the comparison cannot
    be made, and the caller keeps the timestamp verdict rather than guessing.

    The comparison is between CODE OBJECTS, not between marshalled bytes.
    Marshal shares references for interned strings, so the same source
    serialises to byte strings that differ in layout while being equal in
    value — measured here at 11,821 bytes each, identical length, unequal
    bytes. Code objects compare by value, which is the question being asked,
    and it falls out that a whitespace-only edit correctly reads as clean.
    """

    try:
        cached_code = marshal.loads(cache.read_bytes()[_PYC_HEADER_BYTES:])
        # Compilation is still a dynamic-code surface even though this code
        # object is compared and never executed. Keep the raw builtin inside
        # the canonical owner and give this maintenance probe an explicit,
        # bounded internal governance scope.
        with local_internal_governed_scope(
            "loaded_source_drift.compare",
            domain="state_mutation",
        ):
            current_code = get_dynamic_execution_gateway().compile_source(
                source.read_bytes(),
                filename=str(source),
                mode="exec",
                source="loaded_source_drift.compare",
                dont_inherit=True,
            )
    except (OSError, SyntaxError, ValueError, TypeError, EOFError, MemoryError, RecursionError):
        return None
    if not isinstance(cached_code, type(current_code)):
        return None
    return cached_code != current_code


def _drift_reason(module_name: str, source: Path, cache: Path | None) -> str | None:
    """Why ``source`` differs from what was compiled, or None if it matches.

    Returns the sentinel ``"unknown"`` when the question cannot be answered,
    which the caller reports separately — an unmeasurable module must never be
    counted as a clean one.
    """

    try:
        stat = source.stat()
    except OSError:
        return "missing_on_disk"

    header = _read_cache_header(cache) if cache else None
    if header is not None and cache is not None:
        flags, second, third = header
        if flags & _PYC_HASH_BASED:
            try:
                digest = importlib.util.source_hash(source.read_bytes())
            except (OSError, ValueError):
                return "unknown"
            if digest != struct.pack("<II", second, third):
                return "source_hash"
            return None
        # PEP 3147 stores the source mtime truncated to 32 bits, and its size.
        suspected = None
        if int(stat.st_mtime) & 0xFFFFFFFF != second:
            suspected = "mtime"
        elif int(stat.st_size) & 0xFFFFFFFF != third:
            suspected = "size"
        if suspected is None:
            return None
        differ = _compiled_bodies_differ(source, cache)
        if differ is False:
            return None
        return suspected if differ is None else "content"

    baseline = _BASELINE.get(module_name)
    if baseline is None:
        return "unknown"
    if (int(stat.st_mtime), int(stat.st_size)) != baseline:
        return "mtime_or_size"
    return None


_BASELINE: dict[str, tuple[int, int]] = {}
_BASELINE_LOCK = checked_lock("loaded_source_drift.baseline", rank=LockRank.LEAF)


def _loaded_project_modules(root: Path) -> Iterable[tuple[str, Path, Path | None]]:
    """Every loaded module whose source file lives inside the project."""
    # A copy: importing during iteration mutates sys.modules under us, and this
    # runs on a live process that is importing all the time.
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        # `__file__` and `__cached__` are not guaranteed to be strings. A
        # torch operator namespace answers `getattr(module, "__cached__")` with
        # a `_OpNamespace` — it forwards unknown attributes — and Path() then
        # raises TypeError, which this loop did not catch, so ONE such module
        # in sys.modules turned the whole scan into a 500. Measured live
        # 2026-08-10 the first time the endpoint ran on the real process.
        #
        # An explicit isinstance is the fix rather than a wider except: a
        # module whose path is not a string has no source path to compare, and
        # guessing at one is how the wrong file gets blamed.
        file_name = getattr(module, "__file__", None)
        if not isinstance(file_name, (str, os.PathLike)) or not str(file_name).endswith(".py"):
            continue
        # A RELATIVE `__file__` carries no identity. `torch.classes` and
        # `torch.ops` report the bare strings "_classes.py" and "_ops.py";
        # `.resolve()` then joins them to the current working directory, which
        # is this repository, and two synthetic modules that have no source
        # file anywhere appeared as project files missing from disk.
        if not Path(file_name).is_absolute():
            continue
        try:
            source = Path(file_name).resolve()
        except (OSError, ValueError, TypeError):
            continue
        try:
            relative = source.relative_to(root)
        except ValueError:
            continue
        # The virtualenv lives INSIDE the project root here, so "under root"
        # alone swept in every installed dependency — 792 modules scanned, and
        # the only two reported stale were `torch.classes` and `torch.ops`,
        # synthetic modules whose `__file__` points into site-packages. This
        # instrument answers "am I running the code I just edited", and nobody
        # edits their dependencies in place; scanning them only adds noise and
        # cost to a question about this repository.
        if _VENDOR_PARTS.intersection(relative.parts):
            continue
        # Pytest rewrites assertion bytecode before executing test modules.
        # Their cache therefore cannot equal an ordinary compile of the same
        # source, and an edited test file was falsely reported as stale live
        # runtime code. Tests are not shipped runtime surfaces; exclude them
        # structurally instead of special-casing the rewritten cache format.
        if relative.parts and relative.parts[0] == "tests":
            continue
        cached = getattr(module, "__cached__", None)
        cache_path: Path | None = None
        if isinstance(cached, (str, os.PathLike)) and cached:
            try:
                candidate = Path(cached)
                cache_path = candidate if candidate.exists() else None
            except (OSError, ValueError, TypeError):
                cache_path = None
        yield name, source, cache_path


def record_baseline(root: Path | str | None = None) -> int:
    """Snapshot current mtime/size for loaded modules that have no cache file.

    Call once at boot. Only fills gaps: where a bytecode cache exists it is the
    better record, because it is what the interpreter itself compiled from.
    """

    base = Path(root).resolve() if root else _project_root()
    recorded = 0
    for name, source, cache in _loaded_project_modules(base):
        if cache is not None:
            continue
        try:
            stat = source.stat()
        except OSError:
            continue
        with _BASELINE_LOCK:
            if name not in _BASELINE:
                _BASELINE[name] = (int(stat.st_mtime), int(stat.st_size))
                recorded += 1
    return recorded


def scan_drift(root: Path | str | None = None) -> DriftReport:
    """Which loaded project modules no longer match their file on disk."""

    base = Path(root).resolve() if root else _project_root()
    stale: list[ModuleDrift] = []
    unknown: list[str] = []
    checked = 0
    for name, source, cache in _loaded_project_modules(base):
        checked += 1
        reason = _drift_reason(name, source, cache)
        if reason is None:
            continue
        if reason == "unknown":
            unknown.append(name)
            continue
        try:
            shown = str(source.relative_to(base))
        except ValueError:
            shown = str(source)
        stale.append(ModuleDrift(module=name, path=shown, reason=reason))
    stale.sort(key=lambda drift: drift.module)
    return DriftReport(checked=checked, stale=tuple(stale), unknown=tuple(sorted(unknown)))


__all__ = [
    "DriftReport",
    "ModuleDrift",
    "record_baseline",
    "scan_drift",
]
