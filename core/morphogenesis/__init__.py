"""Morphogenetic runtime for Aura.

This package implements a bounded, enterprise-safe, self-organising layer:
cells -> tissues -> organs, all governed by resource budgets, receipts,
episodic memory, strict task ownership, and Aura's existing adaptive immunity.
"""

from .types import (
    CellLifecycle,
    CellManifest,
    CellRole,
    CellState,
    MorphogenesisConfig,
    MorphogenSignal,
    SignalKind,
)
from .field import MorphogenField
from .cell import MorphogenCell, CellTickResult
from .metabolism import MetabolismManager, ResourceSnapshot
from .organs import Organ, OrganStabilizer
from .registry import MorphogenesisRegistry
from .runtime import MorphogeneticRuntime, get_morphogenetic_runtime
from .integration import (
    build_default_cells,
    register_morphogenesis_services,
    start_morphogenesis_runtime,
)

__all__ = [
    "CellLifecycle",
    "CellManifest",
    "CellRole",
    "CellState",
    "MorphogenesisConfig",
    "MorphogenSignal",
    "SignalKind",
    "MorphogenField",
    "MorphogenCell",
    "CellTickResult",
    "MetabolismManager",
    "ResourceSnapshot",
    "Organ",
    "OrganStabilizer",
    "MorphogenesisRegistry",
    "MorphogeneticRuntime",
    "get_morphogenetic_runtime",
    "build_default_cells",
    "register_morphogenesis_services",
    "start_morphogenesis_runtime",
]
