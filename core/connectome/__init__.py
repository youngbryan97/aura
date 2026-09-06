"""core/connectome — Aura's own nervous system, mapped and measured.

The package is built from five bodies of work and keeps their methods rather
than their vocabulary: the H01 human cortical reconstruction, the FlyEM male
central nervous system connectome, ZAPBench, Neuroglancer, and the Potjans and
Diesmann cortical microcircuit.

Nothing here is imported at boot. Reconstruction walks the whole source tree
and recording claims a monitoring slot, so both are asked for explicitly.
"""

from __future__ import annotations

__all__ = [
    "CellClass",
    "Compartment",
    "ConnectomeSnapshot",
    "EdgeKind",
    "Unit",
    "reconstruct",
]


def __getattr__(name: str) -> object:
    if name in {"CellClass", "Compartment", "ConnectomeSnapshot", "EdgeKind", "Unit"}:
        from . import types as _types

        return getattr(_types, name)
    if name == "reconstruct":
        from .volume import reconstruct as _reconstruct

        return _reconstruct
    raise AttributeError(name)
