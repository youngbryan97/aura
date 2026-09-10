"""Where the model artifacts live, which is not where this source lives.

The inventory and the promotion pointer belong to the running installation
rather than to one source checkout, and a linked worktree that resolved them
underneath itself found neither and reported the active cortex invalid. These
four names are the answer to that, and they were the only part of the registry
that had nothing to do with lanes, limits or identity.

Moved out because ``model_registry`` was 2,170 lines, past the ceiling above
which a new module is never grandfathered. Nothing here imports the registry,
so the registry can import this at module scope without a cycle.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "BASE_DIR",
    "get_fused_model_root",
    "get_models_dir",
    "resolve_installation_root",
]


def resolve_installation_root(checkout: Path) -> Path:
    """The installation this source is running inside, not the source itself.

    ``get_models_dir`` and ``get_fused_model_root`` both already say that the
    model inventory and the promotion pointer belong to the running
    installation rather than to one source checkout. The base directory did not
    honour that: it resolved to the directory holding this file, so a linked
    worktree looked for models and for the active cortex manifest underneath
    itself, found neither, and reported the pointer invalid.

    A linked worktree's ``.git`` is a file reading
    ``gitdir: <primary>/.git/worktrees/<name>``, so the primary checkout is
    recoverable without running git and without knowing anybody's home
    directory. A primary checkout has a ``.git`` directory and is already the
    answer.

    Falls back to the checkout whenever the marker is missing, unreadable,
    shaped differently, or names a directory that is not there -- a wrong path
    that exists is worse than the local one, and this runs at import.
    """
    marker = checkout / ".git"
    try:
        if not marker.is_file():
            return checkout
        text = marker.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return checkout
    if not text.startswith("gitdir:"):
        return checkout
    raw = text.split(":", 1)[1].strip()
    if not raw:
        return checkout
    gitdir = Path(raw)
    if not gitdir.is_absolute():
        gitdir = (checkout / gitdir).resolve()
    for parent in gitdir.parents:
        if parent.name == ".git":
            primary = parent.parent
            return primary if primary.is_dir() else checkout
    return checkout


_SOURCE_CHECKOUT = Path(__file__).resolve().parents[3]
_configured_root = str(os.getenv("AURA_ROOT", "")).strip()
BASE_DIR = (
    Path(_configured_root).expanduser()
    if _configured_root
    else resolve_installation_root(_SOURCE_CHECKOUT)
)


def get_models_dir() -> Path:
    """Return the model artifact root independently of the source checkout.

    Worktree-built desktop apps execute source from the worktree but share the
    large, immutable model inventory in the primary checkout.  Conflating those
    two roots made a valid Hugging Face repository ID get reinterpreted as a
    nonexistent path below the worktree.
    """

    configured = str(os.getenv("AURA_MODELS_DIR", "")).strip()
    return Path(configured).expanduser() if configured else BASE_DIR / "models"


def get_fused_model_root() -> Path:
    """Return the runtime-wide model promotion root.

    Promotion state belongs to the running Aura installation, not to an
    individual source worktree.  A worktree must therefore observe the same
    active manifest as the primary checkout or it can silently substitute an
    unqualified base checkpoint for the promoted cortex.
    """

    configured = str(os.getenv("AURA_FUSED_MODEL_ROOT", "")).strip()
    if configured:
        return Path(configured).expanduser()
    return BASE_DIR / "training" / "fused-model"
