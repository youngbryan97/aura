"""Experimental request-conditioned ranking of complete semantic programs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager

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


@contextmanager
def triadic_evidence_lesion(ranker: ContextualProgramRanker):
    """Remove only the operation-conditioned mention/definition product."""
    if not ranker.argument_evidence:
        raise ValueError("triadic lesion needs an argument-evidence ranker")
    width = ranker.config.width
    if ranker.evidence_key.weight.shape[1] != 4 * width:
        raise ValueError("argument evidence geometry differs")
    original = ranker.evidence_key.weight[:, 3 * width:].detach().clone()
    try:
        with torch.no_grad():
            ranker.evidence_key.weight[:, 3 * width:].zero_()
        yield ranker
    finally:
        with torch.no_grad():
            ranker.evidence_key.weight[:, 3 * width:].copy_(original)


class ContextualProgramRanker(nn.Module):
    """Score candidate graphs against one request without answer-key access.

    The request encoder runs once. Each program separately binds its input
    registers to source spans, follows its computation graph, and attends to
    source evidence at each operation. Scores compare programs for one request;
    they are not calibrated probabilities of correctness.
    """

    def __init__(self, config: RequestContextConfig, *, identity_bindings: bool = False,
                 argument_evidence: bool = False,
                 retain_evidence_variants: bool = False):
        super().__init__()
        if (type(identity_bindings) is not bool or type(argument_evidence) is not bool
                or type(retain_evidence_variants) is not bool
                or argument_evidence and (not identity_bindings or not retain_evidence_variants)):
            raise ValueError("argument evidence requires explicit identity bindings")
        width = config.width
        self.config = config
        self.identity_bindings = identity_bindings
        self.argument_evidence = argument_evidence
        self.retain_evidence_variants = retain_evidence_variants
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
        if identity_bindings:
            self.max_arity = max(primitive.arity for primitive in PRIMITIVES_BY_NAME.values())
            self.binding_role = nn.Embedding(len(self.operations) * self.max_arity, width)
            self.binding_query = nn.Linear(width, width, bias=False)
            self.binding_key = nn.Linear(width, width, bias=False)
            nn.init.zeros_(self.binding_key.weight)
        if argument_evidence:
            self.evidence_query = nn.Linear(width, width, bias=False)
            self.evidence_key = nn.Linear(4 * width, width, bias=False)
            nn.init.zeros_(self.evidence_key.weight)

    def _span_identity(self, span: TokenSpan, *, device: torch.device,
                       dtype: torch.dtype) -> torch.Tensor:
        """Encode a source occurrence, independently of the value it denotes."""
        frequency = torch.exp(torch.arange(0, self.config.width, 2, device=device, dtype=dtype)
                              * (-math.log(10000.0) / self.config.width))
        angles = (span.start + span.end - 1) * 0.5 * frequency
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten()

    def forward(
        self, features: torch.Tensor, input_spans: Sequence[TokenSpan],
        input_kinds: Sequence[str], programs: Sequence[Program],
        *, operation_spans: Sequence[Sequence[TokenSpan]] | None = None,
        argument_spans: Sequence[Sequence[Sequence[TokenSpan]]] | None = None,
        definition_spans: Sequence[Sequence[Sequence[TokenSpan]] | None] | None = None,
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
        if self.argument_evidence:
            if argument_spans is None or len(argument_spans) != len(programs):
                raise ValueError("ranker requires runtime argument evidence for every candidate")
            if definition_spans is not None and len(definition_spans) != len(programs):
                raise ValueError("candidate definition evidence differs from programs")
            for index, program in enumerate(programs):
                for name, rows in (("argument", argument_spans[index]),
                                   ("definition", definition_spans[index]
                                    if definition_spans is not None else None)):
                    if rows is None:
                        if name == "argument":
                            raise ValueError("candidate argument evidence is missing")
                        continue
                    if len(rows) != program.depth or any(
                            len(row) != len(ins.args) for row, ins in
                            zip(rows, program.instructions, strict=True)):
                        raise ValueError(f"candidate {name} evidence differs from graph")
                    for row in rows:
                        for span in row:
                            if not isinstance(span, TokenSpan):
                                raise ValueError(f"candidate {name} evidence is not a source span")
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
        input_keys = (tuple(value + self._span_identity(span, device=features.device,
                                                        dtype=features.dtype)
                            for value, span in zip(inputs, input_spans, strict=True))
                      if self.identity_bindings else ())
        scores = []
        for candidate_index, program in enumerate(programs):
            registers = list(inputs)
            register_keys = list(input_keys)
            register_kinds = list(input_kinds)
            binding_score = tokens.new_zeros(())
            for step, instruction in enumerate(program.instructions):
                op_id = self.operation_ids[instruction.op]
                state = self.operation.weight[op_id]
                signature = semantic_primitive_type_signature(instruction.op)
                if signature is None or len(instruction.args) != len(signature[0]):
                    raise ValueError("candidate operation has no typed signature")
                local = (tokens[operation_spans[candidate_index][step].start:
                                operation_spans[candidate_index][step].end].mean(dim=0)
                         if operation_spans is not None else whole)
                for role, reference in enumerate(instruction.args):
                    eligible = [index for index, kind in enumerate(register_kinds)
                                if kind == signature[0][role]]
                    if reference not in eligible:
                        raise ValueError("candidate argument violates its role type")
                    if self.identity_bindings:
                        query = self.binding_query(state + local + self.binding_role.weight[
                            op_id * self.max_arity + role])
                        keys_for_role = self.binding_key(torch.stack(
                            [register_keys[index] for index in eligible]))
                        logits = keys_for_role @ query / math.sqrt(self.config.width)
                        binding_score = binding_score + torch.log_softmax(
                            logits, dim=0)[eligible.index(reference)] + math.log(len(eligible))
                    if self.argument_evidence:
                        mention = argument_spans[candidate_index][step][role]
                        mentioned = (tokens[mention.start:mention.end].mean(dim=0)
                                     + self._span_identity(mention, device=features.device,
                                                           dtype=features.dtype))
                        definition = (definition_spans[candidate_index][step][role]
                                      if definition_spans is not None
                                      and definition_spans[candidate_index] is not None else None)
                        if definition is None:
                            defined = register_keys[reference]
                        else:
                            defined = (tokens[definition.start:definition.end].mean(dim=0)
                                       + self._span_identity(definition, device=features.device,
                                                             dtype=features.dtype))
                        relation = torch.cat((mentioned, defined, mentioned - defined,
                                              mentioned * defined))
                        query = self.evidence_query(state + local + self.binding_role.weight[
                            op_id * self.max_arity + role])
                        binding_score = binding_score + (
                            self.evidence_key(relation) @ query / math.sqrt(self.config.width))
                    state = self.argument_step(registers[reference], state)
                attention = torch.softmax(keys @ self.query(state + local)
                                          / math.sqrt(self.config.width), dim=0)
                evidence = attention @ values
                node = torch.tanh(self.node(torch.cat((state, evidence, local, whole))))
                registers.append(node)
                register_kinds.append(signature[1])
                if self.identity_bindings:
                    anchor = (operation_spans[candidate_index][step]
                              if operation_spans is not None else TokenSpan(0, features.shape[0]))
                    register_keys.append(node + self._span_identity(
                        anchor, device=features.device, dtype=features.dtype))
            scores.append(self.score(torch.cat((registers[-1], whole))).squeeze(-1)
                          + binding_score)
        return torch.stack(scores)


def aggregate_program_scores(scores: torch.Tensor, keys: Sequence[str]) -> tuple[
        tuple[str, ...], torch.Tensor, tuple[tuple[int, ...], ...]]:
    """Average each program's evidence paths in probability space."""
    if (scores.ndim != 1 or len(keys) != len(scores) or not keys
            or any(type(key) is not str or not key for key in keys)
            or not torch.isfinite(scores).all().item()):
        raise ValueError("program aggregation requires a finite, identified candidate set")
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    values = [torch.logsumexp(scores[indices], dim=0) - math.log(len(indices))
              for indices in groups.values()]
    return tuple(groups), torch.stack(values), tuple(tuple(indices) for indices in groups.values())


def candidate_set_loss(scores: torch.Tensor, correct: Sequence[bool],
                       *, program_keys: Sequence[str] | None = None) -> torch.Tensor:
    """Optimize the probability mass of every independently verified solution."""
    if (scores.ndim != 1 or len(correct) != len(scores)
            or not correct or any(type(value) is not bool for value in correct)
            or not any(correct) or not torch.isfinite(scores).all().item()):
        raise ValueError("candidate loss requires a finite bank with a verified solution")
    mask = torch.tensor(correct, dtype=torch.bool, device=scores.device)
    if program_keys is not None:
        _keys, scores, groups = aggregate_program_scores(scores, program_keys)
        if any(len({correct[index] for index in indices}) != 1 for indices in groups):
            raise ValueError("one executable program has contradictory meaning labels")
        mask = torch.tensor([correct[indices[0]] for indices in groups],
                            dtype=torch.bool, device=scores.device)
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[mask], dim=0)
