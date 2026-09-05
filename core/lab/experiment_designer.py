"""core/lab/experiment_designer.py — Experiment Designer.

Designs a concrete, *checkable* experiment for a hypothesis. The experiment is the
exact claim to be falsified (carried in the ``claim`` field), which the SimulationRunner
hands to the Frontier Discovery Engine's verifier. If a checkable claim is supplied
explicitly it is used verbatim; otherwise the hypothesis statement is the claim.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.lab.hypothesis_engine import Hypothesis

logger = logging.getLogger("Aura.ExperimentDesigner")


class ExperimentDesigner:
    """Creates concrete, falsifiable experiment plans for hypotheses."""

    def design_experiment(
        self,
        hypothesis: Hypothesis,
        mined_facts: List[Dict[str, Any]],
        *,
        claim: Optional[str] = None,
    ) -> Dict[str, Any]:
        # The experiment's substance is the exact claim to falsify — prefer an explicit
        # checkable claim, then the hypothesis's own claim attr, then its statement.
        falsifiable_claim = str(
            claim or getattr(hypothesis, "claim", None) or hypothesis.statement or ""
        ).strip()
        logger.info("🧪 Designing falsification experiment for: '%s'", falsifiable_claim[:80])

        return {
            "name": f"exp_test_{hypothesis.hypothesis_id}",
            "hypothesis_id": hypothesis.hypothesis_id,
            "claim": falsifiable_claim,
            "hypothesis_statement": hypothesis.statement,
            "independent_variable": hypothesis.variables.get("independent", "x"),
            "dependent_variable": hypothesis.variables.get("dependent", "y"),
            "method": "exact_falsification",
            "steps": [
                "Reduce the hypothesis to a checkable claim",
                "Run the claim through the exact falsifier (Frontier Discovery Engine)",
                "Record proven / supported / refuted / inconclusive verdict",
                "Commit only verified survivors to belief; never fabricate validation",
            ],
            "mined_facts": list(mined_facts or []),
        }
