"""core/cognition/contract_health.py — what the contracts are actually carrying.

A contract nobody uses is a docstring, and this repository has rediscovered
that shape often enough to have a name for it: a channel with no writer. The
canonical state envelope, the evidence packet, the procedure currency, the
impasse bus and the rest are all new here, and every one of them could sit
unused indefinitely while looking like architecture.

So they publish. This registers one health fragment that asks each contract's
own ledger what it has seen, and every number in it is a count of real traffic
rather than a capability. ``handoff_coverage`` at 0.0 means nothing is using
the envelope; ``organs_reporting`` at 1 means one organ reports impasses. Those
are the numbers that say whether any of this is load-bearing yet, and they
belong where an operator already looks rather than in a document.

Importing this module also registers the architecture invariants, which is the
only way a check next to what it protects actually runs. That is a real
dependency and not an import for its own sake: without it,
``verify("cognition")`` finds nothing to check.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Cognition.ContractHealth")

__all__ = ["contract_health_fragment", "install", "install_contract_health"]

FRAGMENT_NAME = "cognitive_contracts"


def _safe(label: str, fn) -> dict[str, Any]:
    """Ask one ledger for its numbers. A ledger that raises is reported, not fatal."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a health fragment never fails the report
        logger.debug("contract fragment %s unavailable: %s", label, exc)
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def contract_health_fragment() -> dict[str, Any]:
    """Every contract's own count of what has actually passed through it."""
    from core.cognition.action_receipt import get_receipt_ledger
    from core.cognition.architecture_invariants import architecture_report
    from core.cognition.automaticity import get_automaticity
    from core.cognition.concept_handle import get_concept_registry
    from core.cognition.cognitive_event import get_event_graph
    from core.cognition.entity_track import get_track_store
    from core.cognition.procedure import get_procedure_registry
    from core.cognition.situation import get_coordinator
    from core.cognition.substate import get_impasse_bus
    from core.evidence.state_ref import handoff_coverage
    from core.knowledge.atomspace import get_atomspace

    return {
        "schema": "aura.cognition.contracts.v1",
        "evidence": _safe("evidence", lambda: get_atomspace().evidence_report()),
        "state_handoffs": _safe("state_handoffs", handoff_coverage),
        "concepts": _safe("concepts", lambda: get_concept_registry().report()),
        "entity_tracks": _safe("entity_tracks", lambda: get_track_store().report()),
        "action_receipts": _safe("action_receipts", lambda: get_receipt_ledger().report()),
        "cognitive_events": _safe("cognitive_events", lambda: get_event_graph().report()),
        "learning_broadcast": _safe("learning_broadcast", lambda: get_coordinator().report()),
        "impasse_bus": _safe("impasse_bus", lambda: get_impasse_bus().report()),
        "procedures": _safe("procedures", lambda: get_procedure_registry().report()),
        "automaticity": _safe("automaticity", lambda: get_automaticity().report()),
        "architecture_invariants": _safe("invariants", architecture_report),
    }


def install_contract_health() -> bool:
    """Register the contract fragment. Called at import, and safe to call again."""
    return _register(FRAGMENT_NAME, contract_health_fragment)


def _register(name: str, provider) -> bool:
    try:
        from core.runtime.health_fragments import register_health_fragment

        register_health_fragment(name, provider)
    except (ImportError, RuntimeError) as exc:
        logger.debug("%s not registered: %s", name, exc)
        return False
    return True


def install() -> dict[str, bool]:
    """Register both cognitive fragments. Called from foundations at boot."""
    from core.cognition.growth_report import install_growth_report

    return {
        "contracts": install_contract_health(),
        "growth": install_growth_report(),
    }


install()
