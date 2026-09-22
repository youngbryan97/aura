"""Experimental request-conditioned ranking of complete semantic programs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as functional

from core.learning.procedure_induction import PRIMITIVES_BY_NAME, Program
from core.learning.semantic_program_floor import semantic_program_structural_key
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_request_context import RequestContextConfig, SemanticRequestContext


class ContextualProgramRanker(nn.Module):
    """Score candidate graphs against one request without answer-key access.

    The request encoder runs once. Each program separately binds its input
    registers to source spans, follows its computation graph, and attends to
    source evidence at each operation. Scores compare programs for one request;
    they are not calibrated probabilities of correctness.
    """

    def __init__(self, config: RequestContextConfig):
        super().__init__()
        width = config.width
        self.config = config
        self.operations = tuple(sorted(PRIMITIVES_BY_NAME))
        self.operation_ids = {name: index for index, name in enumerate(self.operations)}
        self.context = SemanticRequestContext(config)
        self.compress = nn.Linear(config.input_width, width)
        self.kind = nn.Embedding(2, width)
        self.operation = nn.Embedding(len(self.operations), width)
        self.argument_step = nn.GRUCell(width, width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.node = nn.Linear(4 * width, width)
        self.score = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(
        self, features: torch.Tensor, input_spans: Sequence[TokenSpan],
        input_kinds: Sequence[str], programs: Sequence[Program],
        *, operation_spans: Sequence[Sequence[TokenSpan]] | None = None,
        cross_token: bool = True,
    ) -> torch.Tensor:
        """Return one differentiable score per complete candidate program."""
        if (features.ndim != 2 or features.shape[1] != self.config.input_width
                or not features.is_floating_point() or not torch.isfinite(features).all().item()
                or len(input_spans) != len(input_kinds) or not programs):
            raise ValueError("candidate ranker request geometry is invalid")
        if any(kind not in {"integer", "integer_sequence"} for kind in input_kinds):
            raise ValueError("candidate ranker input kind is unsupported")
        if len(set(input_spans)) != len(input_spans):
            raise ValueError("candidate ranker input anchors must be distinct")
        if any(not isinstance(span, TokenSpan) for span in input_spans):
            raise ValueError("candidate ranker requires measured source spans")
        for span in input_spans:
            span.validate_bound(features.shape[0])
        for program in programs:
            if (not isinstance(program, Program) or program.n_inputs != len(input_spans)
                    or semantic_program_structural_key(program) is None
                    or any(ins.op not in self.operation_ids for ins in program.instructions)):
                raise ValueError("candidate ranker program is outside its typed grammar")
        if operation_spans is not None:
            if len(operation_spans) != len(programs):
                raise ValueError("candidate operation anchors differ from programs")
            for program, spans in zip(programs, operation_spans, strict=True):
                if len(spans) != program.depth or any(not isinstance(span, TokenSpan)
                                                      for span in spans):
                    raise ValueError("candidate operation anchors differ from graph steps")
                for span in spans:
                    span.validate_bound(features.shape[0])

        valid = torch.ones((1, features.shape[0]), dtype=torch.bool, device=features.device)
        encoded = self.context(features.unsqueeze(0), valid, cross_token=cross_token)[0]
        tokens = self.compress(encoded)
        whole = tokens.mean(dim=0)
        keys = self.key(tokens)
        values = self.value(tokens)
        kinds = torch.tensor([0 if kind == "integer" else 1 for kind in input_kinds],
                             dtype=torch.long, device=features.device)
        inputs = tuple(torch.tanh(tokens[span.start:span.end].mean(dim=0) + self.kind(kinds[i]))
                       for i, span in enumerate(input_spans))
        scores = []
        for candidate_index, program in enumerate(programs):
            registers = list(inputs)
            for step, instruction in enumerate(program.instructions):
                op_id = self.operation_ids[instruction.op]
                state = self.operation.weight[op_id]
                for reference in instruction.args:
                    state = self.argument_step(registers[reference], state)
                local = (tokens[operation_spans[candidate_index][step].start:
                                operation_spans[candidate_index][step].end].mean(dim=0)
                         if operation_spans is not None else whole)
                attention = torch.softmax(keys @ self.query(state + local)
                                          / math.sqrt(self.config.width), dim=0)
                evidence = attention @ values
                registers.append(torch.tanh(self.node(torch.cat((state, evidence, local, whole)))))
            scores.append(self.score(torch.cat((registers[-1], whole))).squeeze(-1))
        return torch.stack(scores)


def candidate_set_loss(scores: torch.Tensor, correct: Sequence[bool]) -> torch.Tensor:
    """Optimize the probability mass of every independently verified solution."""
    if (scores.ndim != 1 or len(correct) != len(scores)
            or not correct or any(type(value) is not bool for value in correct)
            or not any(correct) or not torch.isfinite(scores).all().item()):
        raise ValueError("candidate loss requires a finite bank with a verified solution")
    mask = torch.tensor(correct, dtype=torch.bool, device=scores.device)
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[mask], dim=0)
