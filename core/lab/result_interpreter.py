"""core/lab/result_interpreter.py — Research Result Interpreter.

Reads the REAL falsification verdict (proven / supported / refuted / conjecture) the
SimulationRunner produced and maps it to a validated/refuted/inconclusive conclusion
with an honest confidence update. It no longer infers "validated" from a fabricated
effect size, and — crucially — it can now actually REFUTE a hypothesis (the old code
could only ever validate).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from core.lab.hypothesis_engine import Hypothesis

logger = logging.getLogger("Aura.ResultInterpreter")


class ResultInterpreter:
    """Interprets exact-falsification outcomes against hypotheses."""

    def interpret(
        self,
        hypothesis: Hypothesis,
        simulation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info("🎯 ResultInterpreter: analyzing hypothesis '%s'", hypothesis.hypothesis_id)

        status = str(simulation_result.get("status", "inconclusive"))
        validated = bool(simulation_result.get("validated"))
        refuted = bool(simulation_result.get("refuted"))
        verdict_conf = float(simulation_result.get("confidence", 0.0) or 0.0)
        rendered = str(simulation_result.get("rendered", "") or "")

        if validated:
            new_confidence = verdict_conf or min(1.0, hypothesis.confidence + 0.2)
            conclusion = f"Hypothesis supported by exact verification ({status}). {rendered}".strip()
        elif refuted:
            new_confidence = 0.0
            counterexample = simulation_result.get("counterexample")
            conclusion = (
                f"Hypothesis REFUTED by exact counterexample"
                + (f" (n={counterexample})" if counterexample is not None else "")
                + f". {rendered}"
            ).strip()
        else:
            # Could not be reduced to an exact check — honestly inconclusive, NOT validated.
            new_confidence = max(0.0, hypothesis.confidence - 0.05)
            conclusion = (
                "Inconclusive: the hypothesis could not be reduced to an exact check, "
                "so it is filed as unverified conjecture — not validated."
            )

        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "statement": hypothesis.statement,
            "status": status,
            "validated": validated,
            "refuted": refuted,
            "old_confidence": hypothesis.confidence,
            "new_confidence": round(new_confidence, 2),
            "method": simulation_result.get("method", "exact_falsification"),
            "fabricated": bool(simulation_result.get("fabricated", False)),
            "conclusion": conclusion,
        }
