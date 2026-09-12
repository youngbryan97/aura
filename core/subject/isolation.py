"""Every run in its own state root.

A campaign builds the whole organism, and the organism persists. Its world
model checkpoints every two minutes, and its self model, substrate, ontogeny,
goals and ledgers are written within minutes of starting. None of the tools
that build it named a state root, so under the test profile every run shared
`~/.aura-test` with every earlier run and every test process. run_023's world
model arrived with 170,936 steps and 87,613 training updates and most of its
latent already saturated, which is a fact about the runs before it. Two runs
with one fingerprint were two different experiments.

`isolate_state` gives a run its own root, inside its own directory, before
anything opens a store. `state_leaks` checks afterwards that no loaded module
kept a path into the root the process would otherwise have used: fifty-eight
modules build such a path when they are imported, and one imported before the
root moved would keep writing where it started.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path, PurePath
from typing import Any

__all__ = ["STATE_DIR", "StateIsolationError", "isolate_state", "state_leaks"]

#: Where a run's own state lives, under its run directory.
STATE_DIR: str = "state"

#: The root this process would have used had nothing been injected, taken the
#: first time a run is isolated. Once the override is set there is no asking
#: for it again.
_DEFAULT_ROOT: str | None = None


class StateIsolationError(RuntimeError):
    """A run could not be given its own state, so it must not run."""


def _real(path: Any) -> str:
    return os.path.realpath(str(path))


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def isolate_state(run_dir: Path) -> Path:
    """Point every state store at this run's own directory.

    Call it before the organism is built. The path is checked after it is set,
    because a root that did not move is the failure this exists to stop, and
    it looks exactly like success until the ledgers are compared.
    """
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.runtime.state_ownership import state_root

    global _DEFAULT_ROOT
    if _DEFAULT_ROOT is None:
        _DEFAULT_ROOT = _real(state_root())
    root = Path(run_dir) / STATE_DIR
    with local_internal_governed_scope("subject_core.isolation"):
        get_file_write_gateway().ensure_directory(root, source="subject_core.isolation")
    os.environ["AURA_STATE_ROOT"] = str(root)
    resolved = _real(state_root())
    if not _inside(resolved, _real(root)):
        raise StateIsolationError(
            f"asked for state under {root} and the process resolves {resolved}; "
            "a run that shares state with other runs is not the experiment its "
            "fingerprint names"
        )
    return root


def state_leaks() -> list[str]:
    """Module-level paths that still point into the root this run moved away from.

    Only that root is watched. Under the test profile the live root is
    read-only by construction, and shared model assets live there legitimately.
    """
    if _DEFAULT_ROOT is None:
        return []
    own = _real(os.environ.get("AURA_STATE_ROOT", "")) if os.environ.get("AURA_STATE_ROOT") else ""
    if own and _inside(_DEFAULT_ROOT, own):
        return []
    found: list[str] = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("core."):
            continue
        try:
            attributes = list(vars(module).items())
        except TypeError:
            continue
        for attribute, value in attributes:
            if not isinstance(value, PurePath):
                continue
            real = _real(value)
            if own and _inside(real, own):
                continue
            if _inside(real, _DEFAULT_ROOT):
                found.append(f"{name}.{attribute} -> {real}")
    return sorted(found)
