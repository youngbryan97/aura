"""Retain complete semantic proposals and execute them before arbitration."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

from core.evidence.candidate_portfolio import CandidatePortfolioDecision, select_candidate_portfolio
from core.evidence.necessary_condition_selector import (
    NecessaryEvidenceCondition,
    build_necessary_condition_selector,
)
from core.evidence.packet import observe
from core.learning.procedure_induction import Program
from core.learning.semantic_graph_counterexamples import (
    compare_program_meanings,
    counterfactual_inputs,
)
from core.learning.semantic_program_floor import (
    compile_source_independent_program_to_floor,
    execute_semantic_floor_program,
)


@dataclass(frozen=True)
class SemanticProgramPortfolio:
    proposals: tuple[tuple[str, Program | None], ...]
    decision: CandidatePortfolioDecision
    executions: tuple[tuple[str, dict], ...]
    relations: tuple[tuple[str, str, dict], ...]

    @property
    def selected_program(self):
        return dict(self.proposals)[self.decision.selected]

    def plan_inquiries(self, *, fuel: int = 100_000):
        """Use retained disagreement witnesses to plan actual observations."""
        from core.learning.semantic_program_inquiry import plan_program_inquiries

        probes = tuple(tuple(relation["witness"]["inputs"])
                       for _, _, relation in self.relations
                       if relation.get("status") == "different" and "witness" in relation)
        return plan_program_inquiries(
            {name: program for name, program in self.proposals if program is not None},
            probes, fuel=fuel)


def select_semantic_program_portfolio(*, proposals: Mapping[str, Program | None],
                                     provenance: Mapping[str, str], public_inputs: tuple,
                                     observation_sha256: str, incumbent: str,
                                     fuel: int = 2_000_000) -> SemanticProgramPortfolio:
    """Preserve an executable incumbent and repair failures with other methods.

    Floor completion is necessary, not sufficient, for semantic correctness.
    Conflicting executable programs are retained; this policy does not claim
    to resolve their meanings or infer which answer is correct.
    """
    if (not proposals or set(proposals) != set(provenance) or incumbent not in proposals
            or type(fuel) is not int or fuel < 1):
        raise ValueError("invalid semantic portfolio")
    for identity in (observation_sha256, *provenance.values()):
        if (not isinstance(identity, str) or len(identity) != 64
                or any(x not in "0123456789abcdef" for x in identity)):
            raise ValueError("portfolio observation and methods need immutable identities")
    measurements, packets, executions = {}, {}, []
    for name, program in proposals.items():
        if program is not None and not isinstance(program, Program):
            raise ValueError("portfolio proposals must be Programs or explicit refusals")
        evidence = {"executable_program": 0.}
        execution = {"completed": False, "reason": "no_program", "result": None}
        if program is not None:
            try:
                compiled = compile_source_independent_program_to_floor(
                    program, public_inputs, provenance_receipt_sha256=provenance[name])
                result = execute_semantic_floor_program(compiled, fuel=fuel)
                execution = {"completed": True, "reason": "", "result": result.result,
                             "receipt": result.receipt}
                evidence["executable_program"] = 1.
            except (ValueError, TypeError, ArithmeticError, RuntimeError, IndexError) as exc:
                execution = {"completed": False, "reason": f"{type(exc).__name__}:{exc}",
                             "result": None}
        measurements[name] = evidence
        observation = {"method": provenance[name], "source": observation_sha256,
                       "program": program.sha() if program is not None else None,
                       "execution": execution}
        observation_identity = hashlib.sha256(json.dumps(
            observation, sort_keys=True, allow_nan=False).encode()).hexdigest()
        packets[name] = observe(1., origin=f"semantic_program_portfolio:{name}",
                                ref=observation_identity, subject=f"semantic_program:{observation_sha256}")
        executions.append((name, execution))
    selector = build_necessary_condition_selector((NecessaryEvidenceCondition(
        "executable_program", 1., "semantic_answer_requires_completed_floor_execution"),))
    decision = select_candidate_portfolio(selector, incumbent=incumbent,
                                          measurements=measurements, provenance=packets)
    relations = []
    probes = counterfactual_inputs(public_inputs, count=16)
    for left, right in combinations(proposals, 2):
        if measurements[left]["executable_program"] and measurements[right]["executable_program"]:
            relations.append((left, right, compare_program_meanings(
                proposals[left], proposals[right], probes, fuel=fuel)))
    return SemanticProgramPortfolio(tuple(proposals.items()), decision, tuple(executions), tuple(relations))
