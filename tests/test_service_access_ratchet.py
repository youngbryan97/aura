"""Ratchet: raw ServiceContainer.get() cross-wiring only shrinks.

July critique: 'some things still feel cross-wired through
ServiceContainer.get() rather than clean interfaces.' The clean interface is
core/runtime/service_access.py — named, typed-intent resolvers for the
load-bearing seams (orchestrator, state, LLM route, Will, inference gate,
tool execution, memory, learning stack, ...). New code resolves through the
facade; this ratchet caps the raw-get count so the cross-wiring can only
shrink. Lower the budget whenever you migrate call sites — never raise it
without the same scrutiny as widening a permission.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]

# Exact occurrence count at ratchet introduction (July 8, 2026): 1714.
# ONLY goes down. Lowered to 1709 on 2026-08-25 when llm_health_router moved
# seven inference-gate and orchestrator lookups onto the facade, and to 1703
# on 2026-09-01 when cognitive_integration_layer moved its nineteen — four
# onto named resolvers and the rest onto optional_service, which is the
# sanctioned wrapper for a seam that has no named resolver yet.
RAW_GET_BUDGET = 1703

# The facade itself is the one sanctioned wrapper around the container.
FACADE = REPO_ROOT / "core" / "runtime" / "service_access.py"

_GET_RE = re.compile(r"ServiceContainer\.get\(")


def _count_raw_gets() -> tuple[int, dict[str, int]]:
    per_file: dict[str, int] = {}
    for scope in ("core", "interface"):
        for path in (REPO_ROOT / scope).rglob("*.py"):
            hits = len(_GET_RE.findall(path.read_text(encoding="utf-8", errors="ignore")))
            if hits:
                per_file[str(path.relative_to(REPO_ROOT))] = hits
    return sum(per_file.values()), per_file


def test_raw_container_gets_do_not_grow():
    total, per_file = _count_raw_gets()
    assert total <= RAW_GET_BUDGET, (
        f"raw ServiceContainer.get() count grew to {total} (budget {RAW_GET_BUDGET}). "
        "Resolve services through core/runtime/service_access.py instead — add a "
        "named resolver there if your seam is missing. Top offenders: "
        + ", ".join(f"{f}({n})" for f, n in sorted(per_file.items(), key=lambda kv: -kv[1])[:5])
    )


def test_facade_covers_the_core_seams():
    """The seams the critique names must have first-class resolvers."""
    source = FACADE.read_text(encoding="utf-8")
    for resolver in (
        "resolve_orchestrator", "resolve_state_repository", "resolve_llm_router",
        "resolve_will", "resolve_inference_gate", "resolve_skill_router",
        "resolve_mlx_client", "resolve_memory_facade", "resolve_cognitive_engine",
        "resolve_weight_compounding", "resolve_selfplay_flywheel",
        "resolve_incident_narrator",
    ):
        assert f"def {resolver}(" in source, f"facade lost {resolver}"


def test_facade_resolvers_degrade_to_default():
    """Resolvers never raise when the container is empty — default flows out."""
    import core.container as container_mod
    from core.runtime import service_access

    original = container_mod.ServiceContainer.get
    try:
        container_mod.ServiceContainer.get = classmethod(
            lambda cls, name, default=None: default
        )
        sentinel = object()
        assert service_access.resolve_inference_gate(default=sentinel) is sentinel
        assert service_access.resolve_skill_router(default=sentinel) is sentinel
        assert service_access.resolve_weight_compounding(default=sentinel) is sentinel
        assert service_access.resolve_selfplay_flywheel(default=sentinel) is sentinel
    finally:
        container_mod.ServiceContainer.get = original
