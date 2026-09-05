"""core/consciousness/caa — Production Contrastive Activation Addition

The gap between the architectural promise and the deployed system was that
steering vectors were bootstrapped from text statistics, not extracted from
actual model activations. This package closes that gap.

Components:
    VectorRegistry       — Provenance-tracked vector storage with hot-swap
    AlphaController      — Adaptive steering strength with collapse detection
    ModeCollapseDetector — Real-time generation quality monitoring
    ReadinessGate        — Gates transition from bootstrap to production
    QualityMetrics       — Vector and generation quality scoring

Integration:
    The existing AffectiveSteeringEngine in affective_steering.py creates
    and owns a ProductionCAA instance which coordinates all components.
    No separate boot sequence needed — it plugs into the existing attach()
    flow.
"""

from .vector_registry import VectorRegistry, VectorProvenance, RegisteredVector
from .alpha_controller import AlphaController, AlphaState
from .mode_collapse_detector import ModeCollapseDetector, CollapseSignal, CollapseSeverity
from .readiness_gate import ReadinessGate, ReadinessLevel
from .quality_metrics import VectorQualityReport, compute_vector_quality
from .production_caa import ProductionCAA

__all__ = [
    "VectorRegistry",
    "VectorProvenance",
    "RegisteredVector",
    "AlphaController",
    "AlphaState",
    "ModeCollapseDetector",
    "CollapseSignal",
    "CollapseSeverity",
    "ReadinessGate",
    "ReadinessLevel",
    "VectorQualityReport",
    "compute_vector_quality",
    "ProductionCAA",
]
