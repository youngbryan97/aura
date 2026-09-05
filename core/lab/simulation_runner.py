"""core/lab/simulation_runner.py — Experiment Runner (real falsification).

HISTORY / WHY THIS WAS REWRITTEN: this module used to fabricate results. It computed
``stimulated = control * stimulus * noise * 1.2  # Simulate a positive effect`` — a
*guaranteed* positive effect — so the interpreter validated every hypothesis. It could
not refute anything; it was confirmation theatre that fed false "validated" beliefs
into the rest of the system. That is the most damaging kind of disconnected implementation.

It now runs a REAL experiment: the hypothesis's checkable claim is falsified by the
Frontier Discovery Engine's exact verifier (exhaustive residue checking / exact
computation). The outcome is one of proven / supported / refuted / conjecture. A claim
that cannot be reduced to an exact check is reported INCONCLUSIVE — never fabricated as
validated. No randomness, no rigged effect, no manufactured confidence.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SimulationRunner")


class SimulationRunner:
    """Runs a designed experiment as an exact falsification (no fabrication)."""

    async def run_sim(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        claim = str(
            experiment_spec.get("claim")
            or experiment_spec.get("hypothesis_statement")
            or ""
        ).strip()
        base = {
            "experiment_name": experiment_spec.get("name"),
            "hypothesis_id": experiment_spec.get("hypothesis_id"),
            "fabricated": False,
            "method": "exact_falsification",
        }
        if not claim:
            return {
                **base,
                "status": "inconclusive",
                "validated": False,
                "refuted": False,
                "inconclusive": True,
                "confidence": 0.0,
                "score": 0.0,
                "evidence": "no checkable claim supplied",
            }

        logger.info("🔬 SimulationRunner: falsifying claim '%s'", claim[:80])
        try:
            from core.discovery.frontier_discovery_engine import (
                get_frontier_discovery_engine,
            )

            assessment = get_frontier_discovery_engine().assess_claim(claim)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("simulation_runner", exc, severity="warning")
            return {
                **base,
                "status": "inconclusive",
                "validated": False,
                "refuted": False,
                "inconclusive": True,
                "confidence": 0.0,
                "score": 0.0,
                "evidence": f"falsifier unavailable: {exc}",
            }

        verdict = assessment.get("verdict", {}) or {}
        status = str(verdict.get("status", "conjecture"))
        confidence = float(verdict.get("confidence", 0.0) or 0.0)
        return {
            **base,
            "status": status,
            "validated": status in ("proven", "supported"),
            "refuted": status == "refuted",
            "inconclusive": status == "conjecture",
            "confidence": confidence,
            "score": round(confidence, 3),
            "counterexample": verdict.get("counterexample"),
            "exhaustive": bool(verdict.get("exhaustive", False)),
            "rendered": assessment.get("rendered", ""),
            "evidence": verdict.get("formal_form", ""),
        }
