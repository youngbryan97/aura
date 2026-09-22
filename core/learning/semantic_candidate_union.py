"""Unify independently grounded program candidates before observing a target."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from core.learning.procedure_induction import Program
from core.learning.semantic_graph_coordinates import reanchor_program_inputs
from core.learning.semantic_program_composition import (
    ProgramComposition,
    compose_semantic_programs,
)
from core.learning.semantic_program_ir import TokenSpan


@dataclass(frozen=True)
class GroundedProgram:
    """A method's proposal in its own observed input coordinates."""

    program: Program
    input_spans: tuple[TokenSpan, ...]
    source_sha256: str
    provenance_sha256: str


@dataclass(frozen=True)
class UnifiedCandidate:
    program: Program
    origins: tuple[str, ...]


@dataclass(frozen=True)
class SemanticCandidateUnion:
    source_sha256: str
    input_spans: tuple[TokenSpan, ...]
    candidates: tuple[UnifiedCandidate, ...]
    composition: ProgramComposition | None
    receipt: dict

    def validate(self) -> None:
        body = {key: value for key, value in self.receipt.items() if key != "receipt_sha256"}
        if (self.receipt.get("receipt_sha256") != _sha(body)
                or body.get("source_sha256") != self.source_sha256
                or body.get("input_spans") != [span.to_dict() for span in self.input_spans]
                or body.get("candidates") != [
                    {"program": row.program.to_dict(), "origins": row.origins}
                    for row in self.candidates
                ]):
            raise ValueError("mixed candidate receipt differs from retained programs")


class UnalignableProposalError(ValueError):
    """Independently grounded methods do not share the same source anchors."""


def _sha(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def unify_semantic_candidates(
    *, banks: Mapping[str, object], additional: Mapping[str, GroundedProgram],
    public_inputs: tuple, source_sha256: str,
    composition_budget: int = 0, composition_examined_limit: int = 256,
) -> SemanticCandidateUnion:
    """Retain all distinct proposals; source position is the common coordinate.

    The method names and receipts are observations, never correctness votes.
    This function has no access to targets, labels, or expected outputs.
    """
    if (not banks or not isinstance(public_inputs, tuple)
            or type(composition_budget) is not int or composition_budget < 0
            or type(composition_examined_limit) is not int or composition_examined_limit < 1
            or not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_sha256)
            or set(banks) & set(additional)):
        raise ValueError("mixed candidate union requires disjoint methods and finite allowances")
    for name in (*banks, *additional):
        if not isinstance(name, str) or not name:
            raise ValueError("mixed candidate methods require names")

    first = banks[sorted(banks)[0]]
    first.validate()
    common_spans = tuple(first.input_spans)
    if len(common_spans) != len(public_inputs):
        raise ValueError("mixed candidate inputs differ from common grounding")
    proposals: dict[str, Program] = {}
    origins: dict[str, list[str]] = {}
    parents: dict[str, str] = {}

    def retain(name: str, program: Program, from_spans, provenance: str) -> None:
        if not isinstance(program, Program):
            raise ValueError("mixed candidate proposal is not an executable program")
        if (not isinstance(provenance, str) or len(provenance) != 64
                or any(c not in "0123456789abcdef" for c in provenance)):
            raise ValueError("mixed candidate proposal lacks immutable provenance")
        try:
            aligned = reanchor_program_inputs(
                program, from_spans=from_spans, to_spans=common_spans,
                from_inputs=public_inputs, to_inputs=public_inputs,
            )
        except ValueError as exc:
            raise UnalignableProposalError(
                f"{name}: source anchors cannot be reconciled: {exc}") from exc
        identity = aligned.sha()
        if identity not in origins:
            proposals[name] = aligned
            origins[identity] = []
        origins[identity].append(name)
        parents[name] = provenance

    for method, bank in sorted(banks.items()):
        bank.validate()
        if bank.receipt.get("source_text_sha256") != source_sha256:
            raise ValueError("candidate bank belongs to another source request")
        provenance = bank.receipt["receipt_sha256"]
        for index, candidate in enumerate(bank.candidates):
            retain(f"{method}:{index}", candidate.program, bank.input_spans, provenance)
    for method, row in sorted(additional.items()):
        if not isinstance(row, GroundedProgram) or row.source_sha256 != source_sha256:
            raise ValueError("grounded proposal belongs to another source request")
        retain(method, row.program, row.input_spans, row.provenance_sha256)
    if not proposals:
        raise ValueError("mixed candidate union has no executable proposals")

    composition = None
    if composition_budget:
        composition = compose_semantic_programs(
            proposals, public_inputs, max_candidates=composition_budget,
            max_examined=composition_examined_limit,
        )
        for row in composition.candidates:
            retain(row.name, row.program, common_spans, row.provenance_sha256)
    representatives = {program.sha(): program for program in proposals.values()}
    candidates = tuple(UnifiedCandidate(representatives[identity], tuple(names))
                       for identity, names in origins.items())
    body = {"schema": "aura.semantic_candidate_union.v1", "source_sha256": source_sha256,
            "input_spans": [span.to_dict() for span in common_spans],
            "candidates": [{"program": row.program.to_dict(), "origins": row.origins}
                           for row in candidates],
            "provenance": parents,
            "composition": None if composition is None else {
                "examined": composition.examined,
                "search_exhausted": composition.search_exhausted,
                "candidate_count": len(composition.candidates)},
            "target_available": False, "serving_authority": False}
    result = SemanticCandidateUnion(source_sha256, common_spans, candidates, composition,
                                    {**body, "receipt_sha256": _sha(body)})
    result.validate()
    return result
