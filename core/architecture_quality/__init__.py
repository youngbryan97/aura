"""Architecture-quality gate for Aura.

Single-score (0..10000) architectural health metric used as a hard gate
inside the self-modification flow. A patch that degrades the score
beyond the configured tolerance is blocked at promotion time.
"""
from .scorer import (
    DependencyGraph,
    QualityScore,
    compute_metrics,
    parse_dependency_graph,
    score_codebase,
)
from .gate import (
    ArchitectureQualityGate,
    QualityReport,
    baseline_session,
    evaluate_session,
    get_installed_gate,
    install_gate,
)

__all__ = [
    "ArchitectureQualityGate",
    "DependencyGraph",
    "QualityReport",
    "QualityScore",
    "baseline_session",
    "compute_metrics",
    "evaluate_session",
    "get_installed_gate",
    "install_gate",
    "parse_dependency_graph",
    "score_codebase",
]
