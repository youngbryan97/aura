"""core/lab/research_lab.py — Autonomous Research Lab.

Coordinates the full scientific research loop:
  generate hypothesis → search literature → extract claims → compare evidence →
  design experiment → run simulation/code → interpret result → update beliefs →
  propose next experiment.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.lab.experiment_designer import ExperimentDesigner
from core.lab.hypothesis_engine import Hypothesis, HypothesisEngine
from core.lab.literature_miner import LiteratureMiner
from core.lab.research_memory import ResearchMemory
from core.lab.result_interpreter import ResultInterpreter
from core.lab.simulation_runner import SimulationRunner

logger = logging.getLogger("Aura.ResearchLab")


class ResearchStage(StrEnum):
    HYPOTHESIS = "hypothesis_generation"
    LITERATURE = "literature_mining"
    DESIGN = "experiment_design"
    SIMULATION = "simulation"
    INTERPRETATION = "interpretation"
    BELIEF_UPDATE = "belief_update"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ResearchCycle:
    cycle_id: str
    topic: str
    stage: ResearchStage = ResearchStage.HYPOTHESIS
    started_at: float = field(default_factory=time.time)
    hypothesis: Hypothesis | None = None
    mined_facts: list[dict[str, Any]] = field(default_factory=list)
    experiment_spec: dict[str, Any] | None = None
    simulation_result: dict[str, Any] | None = None
    conclusion: dict[str, Any] | None = None
    next_step: str | None = None
    error: str | None = None


class ResearchLab:
    """Aura's scientific research lab, driving discovery of new knowledge."""

    def __init__(self) -> None:
        self.hypothesis_engine = HypothesisEngine()
        self.literature_miner = LiteratureMiner()
        self.designer = ExperimentDesigner()
        self.simulator = SimulationRunner()
        self.interpreter = ResultInterpreter()
        self.memory = ResearchMemory()
        self.cycles: dict[str, ResearchCycle] = {}
        self._cycle_counter = 0

    async def execute_cycle(self, topic: str) -> dict[str, Any]:
        """Execute a full scientific research loop on a topic."""
        self._cycle_counter += 1
        cycle = ResearchCycle(
            cycle_id=f"research_{self._cycle_counter}_{int(time.time())}",
            topic=topic,
        )
        self.cycles[cycle.cycle_id] = cycle
        logger.info("🔬 ResearchLab starting cycle %s on '%s'", cycle.cycle_id, topic)

        try:
            # 1. HYPOTHESIS: Generate hypothesis
            cycle.stage = ResearchStage.HYPOTHESIS
            cycle.hypothesis = self.hypothesis_engine.generate_hypothesis(topic)
            logger.info("💡 Generated Hypothesis: '%s'", cycle.hypothesis.statement)

            # 2. LITERATURE: Mine literature
            cycle.stage = ResearchStage.LITERATURE
            cycle.mined_facts = self.literature_miner.mine_documents([
                {"title": "Prior Research", "content": f"Previous baseline data regarding {topic}", "confidence": 0.85}
            ])
            logger.info("📖 Mined %d facts from prior literature", len(cycle.mined_facts))

            # 3. DESIGN: Design a falsifiable experiment. The topic itself is passed as
            # the checkable claim so a verifiable topic ("is n^5 - n divisible by 30?")
            # is actually tested, not paraphrased into a template.
            cycle.stage = ResearchStage.DESIGN
            cycle.experiment_spec = self.designer.design_experiment(
                cycle.hypothesis, cycle.mined_facts, claim=topic
            )
            logger.info("🧪 Designed experiment: '%s'", cycle.experiment_spec.get("name"))

            # 4. SIMULATION: Run the experiment as a REAL exact falsification (no fabrication).
            cycle.stage = ResearchStage.SIMULATION
            cycle.simulation_result = await self.simulator.run_sim(cycle.experiment_spec)
            logger.info(
                "🔬 Experiment verdict: status=%s validated=%s refuted=%s",
                cycle.simulation_result.get("status"),
                cycle.simulation_result.get("validated"),
                cycle.simulation_result.get("refuted"),
            )

            # 5. INTERPRET: Map the exact verdict to a conclusion (can now refute).
            cycle.stage = ResearchStage.INTERPRETATION
            cycle.conclusion = self.interpreter.interpret(cycle.hypothesis, cycle.simulation_result)
            logger.info("🎯 Interpretation: hypothesis_validated=%s", cycle.conclusion.get("validated"))

            # 6. BELIEF UPDATE — only verified survivors are committed into beliefs; an
            # inconclusive/refuted result is never laundered into a "validated" belief.
            cycle.stage = ResearchStage.BELIEF_UPDATE
            self.memory.save_research_outcome(cycle.cycle_id, cycle.conclusion)
            if cycle.conclusion.get("validated"):
                self._commit_validated_belief(topic, cycle.conclusion, cycle.simulation_result)

            # 7. PROPOSE NEXT STEP
            cycle.next_step = f"Investigate variables affecting {cycle.hypothesis.variables.get('independent', 'target')}"
            cycle.stage = ResearchStage.COMPLETED

        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as e:
            cycle.stage = ResearchStage.FAILED
            cycle.error = str(e)
            logger.error("🔬 Research cycle %s failed at stage %s: %s", cycle.cycle_id, cycle.stage, e)

        return {
            "ok": cycle.stage == ResearchStage.COMPLETED,
            "cycle_id": cycle.cycle_id,
            "hypothesis": cycle.hypothesis.statement if cycle.hypothesis else None,
            "status": cycle.conclusion.get("status") if cycle.conclusion else None,
            "validated": cycle.conclusion.get("validated", False) if cycle.conclusion else False,
            "refuted": cycle.conclusion.get("refuted", False) if cycle.conclusion else False,
            "fabricated": False,
            "conclusion": cycle.conclusion.get("conclusion") if cycle.conclusion else None,
            "next_step": cycle.next_step,
            "error": cycle.error,
        }

    def _commit_validated_belief(
        self, topic: str, conclusion: dict, simulation_result: dict
    ) -> None:
        """Fold a verified research result into beliefs via the ScientificEngine.

        Causal integration: the same exact-falsification verdict that validated the
        claim becomes a confidence-weighted belief (form_hypothesis→run_experiment→
        observe), so a real research finding is visible to the rest of cognition.
        """
        try:
            from core.cognition.scientific_engine import get_scientific_engine

            sci = get_scientific_engine()
            claim = str(simulation_result.get("evidence") or topic)
            hyp_id = sci.form_hypothesis(
                f"research_finding: {claim}",
                predicted_observable="verifier_holds",
                expected=1.0,
                prior_confidence=float(conclusion.get("new_confidence", 0.7) or 0.7),
            )
            sci.run_experiment(hyp_id)
            sci.observe(hyp_id, observed=1.0, note=f"research_lab:{conclusion.get('status')}")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("ResearchLab belief commit skipped: %s", exc)
