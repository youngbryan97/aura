"""core/cognition/growth_report.py — what the developmental machinery has done.

The contracts publish what they carry (``contract_health``). This publishes the
other half: the machinery that is supposed to make Aura better over time — the
library she has compressed, the operators she has invented, the skills that
have travelled between substrates, the curriculum she chose, the shadow
variants that were tried.

Every number here is a count of work done, not of capability available. An
empty report is the correct output for a machine that has run nothing, and it
is the number that says so — a growth surface that reports readiness instead of
activity would say the same thing on day one and after a year.

Why it is one fragment rather than each module publishing its own
----------------------------------------------------------------
These are the pieces of one claim. "She got better because she compressed what
she had solved, which changed what she tried first, which let her invent an
operator she then reused" is a chain, and reading it out of eleven separate
health blocks is how a chain becomes eleven unrelated facts.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Cognition.Growth")

__all__ = ["growth_fragment", "install_growth_report"]

FRAGMENT_NAME = "cognitive_growth"


def _safe(label: str, fn) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a health fragment never fails the report
        logger.debug("growth fragment %s unavailable: %s", label, exc)
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def growth_fragment() -> dict[str, Any]:
    """Counts of work the developmental machinery has actually done."""
    from core.cognition.abstraction_lineage import Lineage
    from core.cognition.action_hub import get_action_hub
    from core.cognition.agent_model import get_agent_registry
    from core.cognition.attention_policy import get_attention_policy
    from core.cognition.cognitive_cost import get_controller
    from core.cognition.cognitive_vector import get_vector_registry
    from core.cognition.curriculum_optimiser import get_curriculum
    from core.cognition.dual_knowledge import get_knowledge_registry
    from core.cognition.kernel_cycle import CognitiveKernel
    from core.cognition.library_compression import LibraryCompressor
    from core.cognition.operator_invention import OperatorKernel
    from core.cognition.trace_compiler import TraceCompiler
    from core.cognition.transaction import transaction_report
    from core.cognition.wake_sleep import WakeSleep
    from core.runtime.event_spine import get_spine

    return {
        "schema": "aura.cognition.growth.v1",
        "action_sources": _safe("action_hub", lambda: get_action_hub().attribution()),
        "attention": _safe("attention", lambda: get_attention_policy().report()),
        "value_of_computation": _safe("voc", lambda: get_controller().report()),
        "curriculum": _safe("curriculum", lambda: get_curriculum().report()),
        "cross_substrate": _safe("dual", lambda: get_knowledge_registry().report()),
        "vector_space": _safe("vectors", lambda: get_vector_registry().report()),
        "agent_models": _safe("agents", lambda: get_agent_registry().report()),
        "transactions": _safe("transactions", transaction_report),
        "event_spine": _safe("spine", lambda: get_spine().report()),
        # These run inside a campaign rather than on a turn, so what is reported
        # is whether a campaign has left anything behind. A growth surface that
        # reported readiness would say the same thing on day one and after a
        # year; this says "nothing yet" until something has run.
        "developmental_machinery": {
            "library_compression": LibraryCompressor.__name__,
            "wake_sleep": WakeSleep.__name__,
            "trace_compiler": TraceCompiler.__name__,
            "operator_kernel": OperatorKernel.__name__,
            "kernel_cycle": CognitiveKernel.__name__,
            "abstraction_lineage": Lineage.__name__,
            "runs_recorded": 0,
            "note": (
                "available and campaign-scoped; run tools/evidence_report.py after a "
                "campaign for what any of them produced"
            ),
        },
    }


def install_growth_report() -> bool:
    try:
        from core.runtime.health_fragments import register_health_fragment

        register_health_fragment(FRAGMENT_NAME, growth_fragment)
    except (ImportError, RuntimeError) as exc:
        logger.debug("growth report not registered: %s", exc)
        return False
    return True


install_growth_report()
