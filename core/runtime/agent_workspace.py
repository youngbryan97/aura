"""core/runtime/agent_workspace.py — the surface an agent acts through, bounded.

When Aura edits a file, runs a command or reads a directory, she does it
through whatever path that particular skill happened to use. Some go through
the write gateway, some through a subprocess, some through the sandbox. The
consequence is that "what may this task touch" has as many answers as there are
call sites, and the answer nobody can give is the one an OS sandbox needs.

A :class:`Workspace` is one answer. It declares a root, a read set and a write
set, and every operation resolves against them before anything happens.
Resolution is the whole mechanism and it does three things a string comparison
does not:

* **Resolves symlinks first.** A path inside the root that points outside it is
  outside it. Checking the string before resolving is the classic escape.
* **Treats a missing parent as outside.** A path that cannot be resolved cannot
  be shown to be inside, and "probably fine" is how a write lands in a home
  directory.
* **Refuses by default.** A workspace with no write set is read-only, and a
  workspace with no root refuses everything. The empty case is the safe one.

Local and remote behind one shape
---------------------------------
:class:`Workspace` is an interface with a local implementation. A remote one
would implement the same four methods, which is what card A5.5 asks for -
without building remote execution, which Aura is local-first and does not want.
The value is that the boundary is now somewhere rather than nowhere.

Research and production
-----------------------
``purpose`` marks a workspace as experimental, and an experimental workspace
may not be rooted at the live tree. That is the structural half of separating
research from production: not a convention about where to put files, but a
refusal when an experiment reaches for the real one.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Purpose",
    "Access",
    "WorkspaceRefusal",
    "Workspace",
    "LocalWorkspace",
    "workspace_report",
]


class Purpose(StrEnum):
    #: Serves the live instance. May be rooted anywhere it is granted.
    PRODUCTION = "production"
    #: An experiment. May not be rooted at a production tree.
    RESEARCH = "research"


class Access(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


class WorkspaceRefusal(PermissionError):
    """An operation reached outside the workspace it was granted."""


@runtime_checkable
class Workspace(Protocol):
    """What an agent may touch. Local and remote implement the same four."""

    def resolve(self, path: str, access: Access) -> Path: ...

    def read(self, path: str) -> str: ...

    def write(self, path: str, content: str) -> Path: ...

    def listdir(self, path: str = ".") -> list[str]: ...


@dataclass
class LocalWorkspace:
    """A workspace rooted in this filesystem."""

    root: Path
    readable: tuple[Path, ...] = ()
    writable: tuple[Path, ...] = ()
    purpose: Purpose = Purpose.PRODUCTION
    #: Trees an experimental workspace may never be rooted at or reach into.
    production_trees: tuple[Path, ...] = ()
    refusals: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.readable = tuple(Path(p).resolve() for p in (self.readable or (self.root,)))
        self.writable = tuple(Path(p).resolve() for p in self.writable)
        self.production_trees = tuple(Path(p).resolve() for p in self.production_trees)
        if self.purpose is Purpose.RESEARCH:
            for tree in self.production_trees:
                if self._within(self.root, tree):
                    raise WorkspaceRefusal(
                        f"a research workspace cannot be rooted at {tree}; separating "
                        "research from production is a refusal, not a convention"
                    )

    @staticmethod
    def _within(candidate: Path, parent: Path) -> bool:
        try:
            candidate.relative_to(parent)
        except ValueError:
            return False
        return True

    def resolve(self, path: str, access: Access) -> Path:
        """Resolve and check. Symlinks first, missing parents refused."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            self._refuse(path, access, f"unresolvable: {exc}")
        # A path whose parent does not exist cannot be shown to be inside.
        if not resolved.parent.exists():
            self._refuse(path, access, "the parent directory does not exist")
        allowed = self.writable if access is not Access.READ else self.readable
        if not allowed:
            self._refuse(
                path, access,
                "no writable set was granted; a workspace with none is read-only"
                if access is not Access.READ else "no readable set was granted",
            )
        if not any(self._within(resolved, base) for base in allowed):
            self._refuse(path, access, f"resolves to {resolved}, outside the granted set")
        for tree in self.production_trees:
            if self.purpose is Purpose.RESEARCH and self._within(resolved, tree):
                self._refuse(path, access, f"a research workspace may not reach into {tree}")
        return resolved

    def _refuse(self, path: str, access: Access, reason: str) -> None:
        self.refusals.append({"path": str(path), "access": access.value, "reason": reason})
        raise WorkspaceRefusal(f"{access.value} {path!r} refused: {reason}")

    def read(self, path: str) -> str:
        return self.resolve(path, Access.READ).read_text()

    def write(self, path: str, content: str) -> Path:
        """Resolve, then write through the gateway that every write goes through.

        The bound checked in :meth:`resolve` decides WHETHER this may be
        written; the gateway decides HOW, and it is not optional — an on-loop
        fsync once froze the live event loop for twenty minutes, which is why
        there is one write path and a ratchet that only shrinks.
        """
        from core.runtime.file_write_gateway import get_file_write_gateway

        target = self.resolve(path, Access.WRITE)
        gateway = get_file_write_gateway()
        gateway.ensure_directory(target.parent, source="agent_workspace")
        gateway.write_text(target, content, source="agent_workspace")
        return target

    def listdir(self, path: str = ".") -> list[str]:
        target = self.resolve(path, Access.READ)
        return sorted(p.name for p in target.iterdir()) if target.is_dir() else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "purpose": self.purpose.value,
            "readable": [str(p) for p in self.readable],
            "writable": [str(p) for p in self.writable],
            "read_only": not self.writable,
            "refusals": self.refusals[-20:],
        }


def workspace_report(workspaces: Sequence[LocalWorkspace]) -> dict[str, Any]:
    """What the granted surfaces look like across every open workspace."""
    return {
        "workspaces": len(workspaces),
        "read_only": sum(1 for w in workspaces if not w.writable),
        "research": sum(1 for w in workspaces if w.purpose is Purpose.RESEARCH),
        "refusals": sum(len(w.refusals) for w in workspaces),
        "by_root": {str(w.root): w.to_dict() for w in workspaces},
    }
