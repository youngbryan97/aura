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
from core.evidence.packet import observe, fuse
from core.learning.procedure_induction import Program
from core.learning.semantic_program_composition import ProgramComposition, compose_semantic_programs
from core.learning.semantic_graph_counterexamples import (
    ProgramObservationCache, compare_program_meanings,
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
    composition: ProgramComposition | None = None
    source_sha256: str = ""

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
            probes, fuel=fuel, source_sha256=self.source_sha256)

    async def retain_inquiries(self, gateway, *, fuel: int = 100_000):
        """Retain unanswered distinctions without putting floor work on the loop."""
        from core.runtime.executors import off_the_loop

        inquiries = await off_the_loop(self.plan_inquiries, fuel=fuel)
        return tuple([await inquiry.retain(gateway) for inquiry in inquiries])

    async def reconcile_retained_inquiries(self, gateway, *, fuel: int = 100_000):
        """Replay durable caller-observed feedback against the unchanged proposals."""
        from core.learning.semantic_program_inquiry import ObservedProgramInquiry, ProgramInquiry
        from core.runtime.executors import off_the_loop

        pending = await gateway.snapshot(domain="semantic_inquiries")
        observed = await ObservedProgramInquiry.restore_all(gateway)
        program_shas = {name: program.sha() for name, program in self.proposals
                        if program is not None}
        feedback = []
        for record in observed:
            inquiry = record.inquiry
            if (dict(inquiry.program_shas) != program_shas
                    or inquiry.source_sha256 != self.source_sha256):
                continue
            retained = pending.get(inquiry.identity)
            if retained is None or ProgramInquiry.from_dict(retained) != inquiry:
                raise ValueError("observed inquiry lacks the matching retained distinction")
            feedback.append((inquiry, record.observed_result, record.origin, record.ref))
        return await off_the_loop(self.reconcile_inquiries, tuple(feedback), fuel=fuel)

    def reconcile_inquiry(self, inquiry, *, observed_result, origin: str, ref: str,
                          fuel: int = 100_000):
        """Select within this portfolio after independent probe feedback.

        None means the observation refutes every available candidate. Agreement
        is scoped to this probe and cannot certify unobserved behavior.
        """
        return self.reconcile_inquiries(((inquiry, observed_result, origin, ref),), fuel=fuel)

    def reconcile_inquiries(self, feedback, *, fuel: int = 100_000):
        """Require agreement with every applicable observed probe in the history."""
        from core.learning.semantic_program_inquiry import ProgramInquiry, plan_program_inquiries

        programs = {name: p.sha() for name, p in self.proposals if p is not None}
        evidence, seen = [], {}
        for inquiry, observed_result, origin, ref in feedback:
            if not isinstance(inquiry, ProgramInquiry):
                raise ValueError("feedback requires a bound program inquiry")
            if programs != dict(inquiry.program_shas):
                raise ValueError("inquiry feedback belongs to different programs")
            if inquiry.source_sha256 != self.source_sha256:
                raise ValueError("inquiry feedback belongs to a different source request")
            replay = plan_program_inquiries(
                {name: p for name, p in self.proposals if p is not None},
                (inquiry.inputs,), fuel=fuel, source_sha256=self.source_sha256,
            )
            if not replay or dict(replay[0].predictions) != dict(inquiry.predictions):
                raise ValueError("inquiry predictions do not match checked program execution")
            packets = inquiry.evidence_for(observed_result=observed_result, origin=origin, ref=ref)
            identity = (origin, ref)
            payload = json.dumps([inquiry.identity, observed_result], sort_keys=True)
            if identity in seen:
                if seen[identity] != payload:
                    raise ValueError("one observation identity cannot name conflicting feedback")
                continue
            seen[identity] = payload
            evidence.append(packets)
        if not evidence:
            return self.decision
        subject = "inquiry_history:" + hashlib.sha256(json.dumps(
            sorted((origin, ref, payload) for (origin, ref), payload in seen.items())
        ).encode()).hexdigest()
        executions = dict(self.executions)
        measurements = {
            name: {"observed_probe_agreement": min(row[name].strength for row in evidence),
                   "executable_program": float(executions[name]["completed"])}
            for name in programs
        }
        if not any(all(values.values()) for values in measurements.values()):
            return None
        selector = build_necessary_condition_selector((
            NecessaryEvidenceCondition("observed_probe_agreement", 1.,
                                       "program_must_agree_with_independent_probe"),
            NecessaryEvidenceCondition("executable_program", 1.,
                                       "semantic_answer_requires_completed_floor_execution"),
        ))
        incumbent = self.decision.selected
        if incumbent not in measurements:
            incumbent = next(iter(measurements))
        return select_candidate_portfolio(
            selector, incumbent=incumbent, measurements=measurements,
            provenance={name: fuse(tuple(row[name].with_subject(subject) for row in evidence))
                        for name in programs},
        )


def select_semantic_program_portfolio(*, proposals: Mapping[str, Program | None],
                                     provenance: Mapping[str, str], public_inputs: tuple,
                                     observation_sha256: str, incumbent: str,
                                     fuel: int = 2_000_000, composition_budget: int = 0,
                                     composition_examined_limit: int = 256) -> SemanticProgramPortfolio:
    """Preserve an executable incumbent and repair failures with other methods.

    Floor completion is necessary, not sufficient, for semantic correctness.
    Conflicting executable programs are retained; this policy does not claim
    to resolve their meanings or infer which answer is correct.
    """
    if (not proposals or set(proposals) != set(provenance) or incumbent not in proposals
            or type(fuel) is not int or fuel < 1
            or type(composition_budget) is not int or composition_budget < 0
            or type(composition_examined_limit) is not int or composition_examined_limit < 1):
        raise ValueError("invalid semantic portfolio")
    for identity in (observation_sha256, *provenance.values()):
        if (not isinstance(identity, str) or len(identity) != 64
                or any(x not in "0123456789abcdef" for x in identity)):
            raise ValueError("portfolio observation and methods need immutable identities")
    proposals, provenance = dict(proposals), dict(provenance)
    composition = None
    if composition_budget:
        composition = compose_semantic_programs(
            proposals, public_inputs, max_candidates=composition_budget,
            max_examined=composition_examined_limit,
        )
        for candidate in composition.candidates:
            if candidate.name in proposals:
                raise ValueError("derived proposal identity conflicts with an existing method")
            proposals[candidate.name] = candidate.program
            provenance[candidate.name] = candidate.provenance_sha256
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
    observation_cache = ProgramObservationCache(capacity=max(256, len(proposals) * len(probes)))
    for left, right in combinations(proposals, 2):
        if measurements[left]["executable_program"] and measurements[right]["executable_program"]:
            relations.append((left, right, compare_program_meanings(
                proposals[left], proposals[right], probes, fuel=fuel,
                observation_cache=observation_cache)))
    return SemanticProgramPortfolio(tuple(proposals.items()), decision, tuple(executions),
                                    tuple(relations), composition, observation_sha256)
