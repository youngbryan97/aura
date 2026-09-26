"""Source-supervised local decisions shared by graph training and selection.

This development scorer uses the existing request encoder, floor types and
candidate banks. It cannot generate an answer or certify intended meaning.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as functional

from core.learning.procedure_induction import PRIMITIVES_BY_NAME, Program
from core.learning.semantic_program_floor import (
    semantic_primitive_type_signature,
    semantic_program_structural_key,
)
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_request_context import RequestContextConfig, SemanticRequestContext


class AtomAlignedProgramRanker(nn.Module):
    """Score operation and typed reference factors, with direct source supervision.

    Each argument role asks the same shared pointer over the currently available
    register anchors. Earlier choices do not change that role's query. Register
    identity is a source occurrence, never its numerical value or source family.
    Complete graphs add these local log scores. They are conditional evidence
    scores, not probabilities that a whole answer is correct.
    """

    argument_evidence = False
    retain_evidence_variants = True

    def __init__(self, config: RequestContextConfig):
        super().__init__()
        self.config = config
        self.context = SemanticRequestContext(config)
        self.operations = tuple(sorted(PRIMITIVES_BY_NAME))
        self.operation_ids = {name: index for index, name in enumerate(self.operations)}
        self.max_arity = max(primitive.arity for primitive in PRIMITIVES_BY_NAME.values())
        width = config.width
        self.normalize = nn.LayerNorm(width)
        self.operation_classifier = nn.Linear(width, len(self.operations))
        self.operation = nn.Embedding(len(self.operations), width)
        self.role = nn.Embedding(self.max_arity, width)
        self.kind = nn.Embedding(2, width)
        self.query = nn.Sequential(nn.Linear(3 * width, width), nn.Tanh(),
                                   nn.Linear(width, width, bias=False))
        self.key = nn.Linear(width, width, bias=False)

    @staticmethod
    def _validate(
        features: torch.Tensor, input_spans: Sequence[TokenSpan],
        input_kinds: Sequence[str], programs: Sequence[Program], operation_spans,
    ) -> None:
        if (features.ndim != 2 or features.shape[0] == 0
                or len(input_spans) != len(input_kinds) or not input_spans or not programs
                or any(kind not in {"integer", "integer_sequence"} for kind in input_kinds)
                or len(set(input_spans)) != len(input_spans)):
            raise ValueError("atom ranker input geometry is invalid")
        if operation_spans is None or len(operation_spans) != len(programs):
            raise ValueError("atom ranker requires operation anchors")
        for span in input_spans:
            if not isinstance(span, TokenSpan):
                raise ValueError("atom ranker input anchor is not a source span")
            span.validate_bound(features.shape[0])
        for program, spans in zip(programs, operation_spans, strict=True):
            if (not isinstance(program, Program) or program.n_inputs != len(input_spans)
                    or semantic_program_structural_key(program) is None
                    or len(spans) != program.depth):
                raise ValueError("atom candidate is outside its typed graph grammar")
            kinds = list(input_kinds)
            for instruction, span in zip(program.instructions, spans, strict=True):
                signature = semantic_primitive_type_signature(instruction.op)
                if (signature is None or len(instruction.args) != len(signature[0])
                        or any(kinds[reference] != kind for reference, kind in
                               zip(instruction.args, signature[0], strict=True))):
                    raise ValueError("atom candidate argument violates its role type")
                if not isinstance(span, TokenSpan):
                    raise ValueError("atom candidate operation is not a source span")
                span.validate_bound(features.shape[0])
                kinds.append(signature[1])

    def _factors(self, tokens, input_spans, input_kinds, program, spans):
        registers = [tokens[span.start:span.end].mean(dim=0) for span in input_spans]
        kinds = list(input_kinds)
        factors = []
        for instruction, span in zip(program.instructions, spans, strict=True):
            local = tokens[span.start:span.end].mean(dim=0)
            op_id = self.operation_ids[instruction.op]
            logits = self.operation_classifier(local)
            factors.append(("operation", logits, op_id))
            signature = semantic_primitive_type_signature(instruction.op)
            for role, reference in enumerate(instruction.args):
                eligible = [index for index, kind in enumerate(kinds)
                            if kind == signature[0][role]]
                query = self.query(torch.cat((local, self.operation.weight[op_id],
                                               self.role.weight[role])))
                keys = self.key(torch.stack([
                    registers[index] + self.kind.weight[
                        0 if kinds[index] == "integer" else 1]
                    for index in eligible]))
                logits = keys @ query / math.sqrt(self.config.width)
                factors.append(("reference", logits, eligible.index(reference)))
            registers.append(local)
            kinds.append(signature[1])
        return factors

    def _encoded(self, features, *, cross_token):
        valid = torch.ones((1, len(features)), dtype=torch.bool, device=features.device)
        return self.normalize(self.context.encode_tokens(
            features.unsqueeze(0), valid, cross_token=cross_token)[0])

    def forward(
        self, features: torch.Tensor, input_spans: Sequence[TokenSpan],
        input_kinds: Sequence[str], programs: Sequence[Program],
        *, operation_spans=None, cross_token: bool = True,
    ) -> torch.Tensor:
        """Score only supplied graphs; no target or result enters this interface."""
        self._validate(features, input_spans, input_kinds, programs, operation_spans)
        tokens = self._encoded(features, cross_token=cross_token)
        scores = []
        for program, spans in zip(programs, operation_spans, strict=True):
            factors = self._factors(tokens, input_spans, input_kinds, program, spans)
            terms = [functional.log_softmax(logits, dim=0)[target]
                     + (math.log(len(logits)) if name == "reference" else 0.0)
                     for name, logits, target in factors]
            scores.append(torch.stack(terms).sum())
        return torch.stack(scores)

    def source_atom_loss(
        self, features: torch.Tensor, input_spans: Sequence[TokenSpan],
        input_kinds: Sequence[str], program: Program, *, operation_spans,
        cross_token: bool = True,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Teach the same factors from a source-labeled graph, never a held answer."""
        self._validate(features, input_spans, input_kinds, (program,), (operation_spans,))
        tokens = self._encoded(features, cross_token=cross_token)
        factors = self._factors(tokens, input_spans, input_kinds, program, operation_spans)
        losses = [-functional.log_softmax(logits, dim=0)[target]
                  for _name, logits, target in factors]
        measured = {}
        for name in ("operation", "reference"):
            rows = [(logits, target) for kind, logits, target in factors if kind == name]
            measured[name + "_total"] = len(rows)
            measured[name + "_correct"] = sum(int(logits.argmax().item() == target)
                                                for logits, target in rows)
        return torch.stack(losses).mean(), measured
