"""Morphogenetic runtime for Aura.

A population of governed computational cells whose runtime topology is state:
cells bind, unbind, spawn, retire, specialize and route, and the shape they
take changes what the system can compute. Every change goes through one
governor that checks bounds, measures the change against the current shape,
asks Aura's governance for anything critical, and can undo the whole thing.

The layer is bounded by construction — population, replicas per capability,
spawn depth, transition rate, per-cell cooldown, reversal window, energy — and
subordinate to the existing runtime: resource budgets, receipts, episodic
memory, task ownership, and adaptive immunity.

Run the experiments with ``tools/run_morphogenesis_sandbox.py``.
"""

from .governor import MorphBounds, MorphGovernor
from .graph import EdgeType, GraphSnapshot, MorphEdge, MorphGraph
from .lineage import Lineage, LineageRecord
from .motifs import MorphMotif, MotifLibrary
from .proposal import Decision, MorphProposal, MorphTransaction, MorphTransition, TransitionKind
from .substrate import (
    LocalRuntimeSubstrate,
    SimulationSubstrate,
    SubstrateAdapter,
    SubstratePhysics,
)
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
    "Decision",
    "EdgeType",
    "GraphSnapshot",
    "Lineage",
    "LineageRecord",
    "LocalRuntimeSubstrate",
    "MorphBounds",
    "MorphEdge",
    "MorphGovernor",
    "MorphGraph",
    "MorphMotif",
    "MorphProposal",
    "MorphTransaction",
    "MorphTransition",
    "MotifLibrary",
    "SimulationSubstrate",
    "SubstrateAdapter",
    "SubstratePhysics",
    "TransitionKind",
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
