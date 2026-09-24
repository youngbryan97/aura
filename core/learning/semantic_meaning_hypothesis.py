"""Source-bound meaning hypotheses before exact program execution.

This opt-in layer preserves distinct source occurrences even when values match.
Its hypotheses come from target-blind proposals; they are not observations of
what the source meant and carry no serving authority.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from core.learning.procedure_induction import PRIMITIVES_BY_NAME, Instruction, Program
from core.learning.semantic_candidate_bank import SemanticCandidateBank
from core.learning.semantic_graph_counterexamples import (
    ProgramObservationCache,
    compare_program_meanings,
    counterfactual_inputs,
)
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_portfolio import (
    SemanticProgramPortfolio,
    select_semantic_program_portfolio,
)
from core.learning.semantic_program_ir import TokenSpan, _is_sha256


@dataclass(frozen=True, slots=True)
class SourceOccurrence:
    source_text_sha256: str
    span: TokenSpan

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_text_sha256) or not isinstance(self.span, TokenSpan):
            raise ValueError("meaning occurrence needs a source and token span")

    @property
    def identity(self) -> str:
        return _sha({"source": self.source_text_sha256, "span": self.span.to_dict()})


@dataclass(frozen=True, slots=True)
class MeaningBinding:
    role_index: int
    register: int
    mention: SourceOccurrence
    definition: SourceOccurrence | None

    def __post_init__(self) -> None:
        if (type(self.role_index) is not int or self.role_index < 0
                or type(self.register) is not int or self.register < 0
                or not isinstance(self.mention, SourceOccurrence)
                or (self.definition is not None
                    and (not isinstance(self.definition, SourceOccurrence)
                         or self.definition.source_text_sha256
                         != self.mention.source_text_sha256))):
            raise ValueError("meaning binding has invalid source or register")


@dataclass(frozen=True, slots=True)
class MeaningOperation:
    name: str
    occurrence: SourceOccurrence
    bindings: tuple[MeaningBinding, ...]

    def __post_init__(self) -> None:
        primitive = PRIMITIVES_BY_NAME.get(self.name)
        if (primitive is None
                or not isinstance(self.occurrence, SourceOccurrence)
                or not isinstance(self.bindings, tuple)
                or len(self.bindings) != primitive.arity
                or tuple(binding.role_index for binding in self.bindings)
                != tuple(range(len(self.bindings)))
                or any(binding.mention.source_text_sha256
                       != self.occurrence.source_text_sha256 for binding in self.bindings)):
            raise ValueError("meaning operation has invalid role structure")


@dataclass(frozen=True, slots=True)
class MeaningHypothesis:
    """One coherent, source-grounded proposal with uncalibrated source score."""

    source_text_sha256: str
    bank_receipt_sha256: str
    input_occurrences: tuple[SourceOccurrence, ...]
    operations: tuple[MeaningOperation, ...]
    source_score: float | None
    chart_index: int | None
    graph_index: int | None
    definition_provenance: str

    def __post_init__(self) -> None:
        if (not _is_sha256(self.source_text_sha256)
                or not _is_sha256(self.bank_receipt_sha256)
                or not self.input_occurrences or not self.operations
                or any(item.source_text_sha256 != self.source_text_sha256
                       for item in self.input_occurrences)
                or len({item.identity for item in self.input_occurrences})
                != len(self.input_occurrences)
                or any(op.occurrence.source_text_sha256 != self.source_text_sha256
                       for op in self.operations)
                or (self.source_score is not None and
                    (type(self.source_score) not in (int, float)
                     or not math.isfinite(self.source_score)))
                or self.definition_provenance not in {
                    "unavailable", "register_anchor", "optimizer_selected"}):
            raise ValueError("meaning hypothesis source or evidence differs")
        self.to_program()

    def to_program(self) -> Program:
        instructions = tuple(Instruction(op.name, tuple(binding.register
            for binding in op.bindings)) for op in self.operations)
        program = Program(len(self.input_occurrences), instructions)
        for index, instruction in enumerate(instructions):
            if any(register >= len(self.input_occurrences) + index
                   for register in instruction.args):
                raise ValueError("meaning hypothesis violates forward dataflow")
        return program

    @property
    def identity(self) -> str:
        return _sha({
            "source": self.source_text_sha256,
            "bank": self.bank_receipt_sha256,
            "inputs": [item.identity for item in self.input_occurrences],
            "operations": [{"name": op.name, "occurrence": op.occurrence.identity,
                            "bindings": [{"role": binding.role_index,
                                          "register": binding.register,
                                          "mention": binding.mention.identity,
                                          "definition": (binding.definition.identity
                                                         if binding.definition else None)}
                                         for binding in op.bindings]}
                           for op in self.operations],
        })


@dataclass(frozen=True, slots=True)
class GroundedProgramProposal:
    """A program compiled from one immutable meaning hypothesis."""

    hypothesis: MeaningHypothesis
    program: Program

    def __post_init__(self) -> None:
        if self.program != self.hypothesis.to_program():
            raise ValueError("program differs from its meaning hypothesis")

    @property
    def receipt_sha256(self) -> str:
        return _sha({"meaning": self.hypothesis.identity,
                     "program": self.program.sha(),
                     "source": self.hypothesis.source_text_sha256,
                     "bank": self.hypothesis.bank_receipt_sha256})


def compare_source_bound_meanings(
    left: GroundedProgramProposal,
    right: GroundedProgramProposal,
    public_inputs: tuple[Any, ...],
    *,
    counterfactual_count: int=32,
    seed: int=0,
    fuel: int=100000,
    observation_cache: ProgramObservationCache | None=None,
) -> dict[str, Any]:
    """Test proposed consequences without treating execution as source truth."""
    if (left.hypothesis.source_text_sha256 != right.hypothesis.source_text_sha256
            or left.hypothesis.bank_receipt_sha256 != right.hypothesis.bank_receipt_sha256):
        raise ValueError("meaning comparison requires one immutable source bank")
    if len(public_inputs) != left.program.n_inputs or left.program.n_inputs != right.program.n_inputs:
        raise ValueError("meaning comparison public input geometry differs")
    probes = counterfactual_inputs(public_inputs, count=counterfactual_count, seed=seed)
    comparison = compare_program_meanings(left.program, right.program, probes,
        fuel=fuel, observation_cache=observation_cache)
    return {
        "schema": "aura.source_bound_meaning_comparison.v1",
        "source_text_sha256": left.hypothesis.source_text_sha256,
        "bank_receipt_sha256": left.hypothesis.bank_receipt_sha256,
        "left_receipt_sha256": left.receipt_sha256,
        "right_receipt_sha256": right.receipt_sha256,
        "proposed_consequence_comparison": comparison,
        "source_interpretation_status": "unresolved",
        "claim": "program_consequences_only_not_source_interpretation_or_phenomenology",
    }


def select_source_bound_portfolio(
    proposals: Sequence[GroundedProgramProposal],
    public_inputs: tuple[Any, ...],
    *,
    incumbent_receipt_sha256: str,
    fuel: int=2_000_000,
) -> SemanticProgramPortfolio:
    """Keep source ancestry through the canonical execution and inquiry loop."""
    if not proposals:
        raise ValueError("meaning portfolio needs source-bound proposals")
    source = proposals[0].hypothesis.source_text_sha256
    bank = proposals[0].hypothesis.bank_receipt_sha256
    if any(proposal.hypothesis.source_text_sha256 != source
           or proposal.hypothesis.bank_receipt_sha256 != bank for proposal in proposals):
        raise ValueError("meaning inquiry requires one immutable source bank")
    if any(proposal.program.n_inputs != len(public_inputs) for proposal in proposals):
        raise ValueError("meaning inquiry public input geometry differs")
    names = [proposal.receipt_sha256 for proposal in proposals]
    if len(names) != len(set(names)):
        raise ValueError("meaning portfolio has duplicate proposal receipts")
    return select_semantic_program_portfolio(
        proposals={proposal.receipt_sha256: proposal.program for proposal in proposals},
        provenance={proposal.receipt_sha256: proposal.receipt_sha256 for proposal in proposals},
        public_inputs=public_inputs, observation_sha256=source,
        incumbent=incumbent_receipt_sha256, fuel=fuel,
    )


def meaning_hypotheses_from_bank(bank: SemanticCandidateBank) -> tuple[MeaningHypothesis, ...]:
    """Expose all retained proposal meanings without assigning a posterior."""
    bank.validate()
    source = bank.receipt["source_text_sha256"]
    if not _is_sha256(source):
        raise ValueError("candidate bank lacks source identity")
    inputs = tuple(SourceOccurrence(source, span) for span in bank.input_spans)
    hypotheses = []
    for candidate in bank.candidates:
        if candidate.argument_spans is None:
            continue
        operations = []
        for index, instruction in enumerate(candidate.program.instructions):
            definitions = (candidate.definition_spans[index]
                           if candidate.definition_spans is not None else None)
            bindings = tuple(MeaningBinding(
                role, register, SourceOccurrence(source, mention),
                SourceOccurrence(source, definitions[role]) if definitions else None,
            ) for role, (register, mention) in enumerate(zip(
                instruction.args, candidate.argument_spans[index], strict=True)))
            operations.append(MeaningOperation(
                instruction.op, SourceOccurrence(source, candidate.operation_spans[index]),
                bindings))
        hypothesis = MeaningHypothesis(source, bank.receipt["receipt_sha256"],
            inputs, tuple(operations), candidate.joint_score, candidate.chart_index,
            candidate.graph_index, candidate.definition_provenance)
        if hypothesis.to_program() != candidate.program:
            raise ValueError("candidate graph and meaning hypothesis differ")
        hypotheses.append(hypothesis)
    return tuple(hypotheses)
