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

__all__ = ["NOT_STATE", "STATE_DIR", "StateIsolationError", "isolate_state", "state_leaks"]

#: Where a run's own state lives, under its run directory.
STATE_DIR: str = "state"

#: The root this process would have used had nothing been injected, taken the
#: first time a run is isolated. Once the override is set there is no asking
#: for it again.
_DEFAULT_ROOT: str | None = None


#: Globals that name assets rather than state, with the reason each one is
#: here. A store is forked because a run writes to it; these are read only,
#: byte-identical in every arm, and tens of gigabytes besides, so a fork would
#: copy the model weights themselves. A run that refuses over one of these is
#: refusing over where the model file lives.
NOT_STATE: frozenset[str] = frozenset(
    {
        # The checkout the process runs from. Every model path is built from it.
        "core.brain.llm.model_paths.BASE_DIR",
        "core.brain.llm.model_registry.BASE_DIR",
        # Weights and adapters. Read at load time and never written by a run.
        "core.brain.llm.model_registry.ADAPTER_PATH",
        "core.brain.llm.model_registry._CORTEX_PATH",
        "core.brain.llm.model_registry._IMPORT_MODELS_DIR",
        "core.brain.llm.model_registry._SOLVER_PATH",
        # Trained tissues shipped in the checkout, loaded only after their
        # manifest's SHA-256 checks out, and never written by a run. A whole run
        # imports the latent cortex, and the checkout sits under the live state
        # root, so each of these read as a leak and refused the report run of
        # 22 September before it started.
        "core.brain.llm.latent_cortex.neural_transition_tissue.DEFAULT_NEURAL_TRANSITION_ARTIFACT",
        "core.brain.llm.latent_cortex.systematic_neural_alu.DEFAULT_SYSTEMATIC_NEURAL_ALU_ARTIFACT",
        "core.learning.recurrent_work_memory_tissue.DEFAULT_MATHEMATICS_MEMORY_ARTIFACT",
        # The rest of what a whole run imports that points into the checkout or
        # at weights. The checkout is code wherever it sits; it is under ~/.aura
        # only because this repository lives there.
        "core.brain.llm.model_paths._SOURCE_CHECKOUT",
        "core.brain.llm.model_registry._BRAINSTEM_PATH",
        "core.config.PROJECT_ROOT",
        "core.utils.paths.PROJECT_ROOT",
        "core.utils.paths.CORE_DIR",
        # A relative path searched to read forensics written under an older
        # convention. Nothing is written there: `forensics_root` follows
        # AURA_LOG_DIR, which every run sets.
        "core.utils.paths._LEGACY_FORENSICS_RELATIVE",
    }
)


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
    from core.config import config
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
    data = _real(config.paths.data_dir)
    if not _inside(data, _real(root)):
        raise StateIsolationError(
            f"the state root moved to {root} and the data directory is still {data}; "
            "every store under it would be written outside the run"
        )
    return root


def state_leaks() -> list[str]:
    """Module-level paths that still point into the root this run moved away from.

    Only that root is watched. Under the test profile the live root is
    read-only by construction, and shared model assets live there legitimately.

    `NOT_STATE` names the globals that are assets rather than stores.
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
            if f"{name}.{attribute}" in NOT_STATE:
                continue
            real = _real(value)
            if own and _inside(real, own):
                continue
            if _inside(real, _DEFAULT_ROOT):
                found.append(f"{name}.{attribute} -> {real}")
    return sorted(found)
