"""Experimental program likelihood over the existing semantic floor.

The decoder reads resident token features and grounded input anchors. It
learns complete operation and register sequences, without operation spans.
It has no serving or promotion authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_floor import (
    PRIMITIVES_BY_NAME,
    semantic_primitive_type_signature,
)


@dataclass(frozen=True)
class ProgramDecoderConfig:
    input_width: int
    width: int = 64
    max_steps: int = 16
    feature_scaling: str = "none"

    def __post_init__(self):
        if (
            any(type(x) is not int or x < 1 for x in (self.input_width, self.width, self.max_steps))
            or self.width % 2
        ):
            raise ValueError("program decoder needs positive dimensions and even hidden width")
        if self.feature_scaling not in {"none", "unit_variance"}:
            raise ValueError("unknown resident feature scaling")


class SemanticProgramDecoder(nn.Module):
    """Autoregressive typed operations with pointers to grounded registers.

    Input registers hold their source-span features. A computed register holds
    the learned state at its definition. Type signatures come from the same
    floor that executes the returned Program; there is no second interpreter.
    """

    def __init__(self, config: ProgramDecoderConfig):
        super().__init__()
        self.config = config
        self.operations = tuple(sorted(PRIMITIVES_BY_NAME))
        if any(semantic_primitive_type_signature(op) is None for op in self.operations):
            raise ValueError("all decoder operations need floor type signatures")
        width = config.width
        self.project = nn.Linear(config.input_width, width)
        self.norm = nn.LayerNorm(width)
        self.initial = nn.Linear(width, width)
        self.start = nn.Parameter(torch.zeros(width))
        self.operation_embedding = nn.Embedding(len(self.operations), width)
        self.cell = nn.GRUCell(2 * width, width)
        self.attention_query = nn.Linear(width, width, bias=False)
        self.operation_logits = nn.Linear(2 * width, len(self.operations) + 1)
        self.register_query = nn.Linear(2 * width, width)
        self.result = nn.Linear(3 * width, width)

    def _encode(self, features, input_spans, input_types):
        if (
            features.ndim != 2
            or features.shape[1] != self.config.input_width
            or not len(features)
            or not features.is_floating_point()
            or not torch.isfinite(features).all().item()
        ):
            raise ValueError("program decoder requires finite resident token features")
        spans, kinds = tuple(input_spans), tuple(input_types)
        if (
            not spans
            or len(spans) != len(kinds)
            or len(set(spans)) != len(spans)
            or any(kind not in {"integer", "integer_sequence"} for kind in kinds)
        ):
            raise ValueError("program decoder requires distinct typed input anchors")
        for span in spans:
            span.validate_bound(len(features))
        if self.config.feature_scaling == "unit_variance":
            if not torch.allclose(
                features.norm(dim=-1), torch.ones_like(features[:, 0]), atol=1e-4
            ):
                raise ValueError("variance-preserving projection requires unit resident features")
            # Fan-in initialization assumes unit coordinate variance. Resident
            # features instead have unit vector norm, hence variance 1/d.
            features = features * math.sqrt(self.config.input_width)
        memory = self.project(features)
        positions = torch.arange(len(features), device=features.device, dtype=memory.dtype)
        frequencies = torch.exp(
            torch.arange(0, self.config.width, 2, device=features.device, dtype=memory.dtype)
            * (-math.log(10000.0) / self.config.width)
        )
        angles = positions[:, None] * frequencies
        memory = self.norm(memory + torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2))
        registers = [memory[span.start : span.end].mean(dim=0) for span in spans]
        return memory, registers, list(kinds)

    def _advance(self, state, previous, memory):
        attention = (memory @ self.attention_query(state) / math.sqrt(self.config.width)).softmax(
            dim=0
        )
        context = attention @ memory
        state = self.cell(torch.cat((previous, context)), state)
        return state, torch.cat((state, context))

    @staticmethod
    def _distribution(logits, allowed):
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[list(allowed)] = True
        if not mask.any().item():
            raise ValueError("typed program prefix has no continuation")
        if not torch.isfinite(logits).all().item():
            raise ValueError("program decoder produced nonfinite scores")
        return logits.masked_fill(~mask, -torch.inf).log_softmax(dim=0)

    @staticmethod
    def _choose(logits, allowed, target):
        probabilities = SemanticProgramDecoder._distribution(logits, allowed)
        selected = int(probabilities.argmax()) if target is None else target
        if type(selected) is not int or selected not in allowed:
            raise ValueError("teacher program violates the floor grammar")
        return selected, probabilities[selected]

    def _run(self, features, input_spans, input_types, target=None):
        memory, registers, kinds = self._encode(features, input_spans, input_types)
        n_inputs = len(registers)
        if target is not None and (
            target.n_inputs != n_inputs
            or not target.instructions
            or len(target.instructions) > self.config.max_steps
        ):
            raise ValueError("teacher program exceeds decoder geometry")
        state, previous = self.initial(memory.mean(dim=0)).tanh(), self.start
        instructions, terms = [], []
        eos = len(self.operations)
        for step in range(self.config.max_steps + 1):
            state, context = self._advance(state, previous, memory)
            allowed = [eos] if instructions else []
            if step < self.config.max_steps:
                allowed += [
                    index
                    for index, op in enumerate(self.operations)
                    if all(kind in kinds for kind in semantic_primitive_type_signature(op)[0])
                ]
            teacher = None
            if target is not None:
                if step == len(target.instructions):
                    teacher = eos
                else:
                    op = target.instructions[step].op
                    if op not in self.operations:
                        raise ValueError("teacher operation is outside the semantic floor")
                    teacher = self.operations.index(op)
            selected, logp = self._choose(self.operation_logits(context), allowed, teacher)
            terms.append(logp)
            if selected == eos:
                return Program(n_inputs, tuple(instructions)), torch.stack(terms).sum(), len(terms)
            operation = self.operations[selected]
            argument_types, result_type = semantic_primitive_type_signature(operation)
            embedding = self.operation_embedding.weight[selected]
            previous = embedding
            arguments = []
            if target is not None and len(target.instructions[step].args) != len(argument_types):
                raise ValueError("teacher operation has the wrong arity")
            for slot, kind in enumerate(argument_types):
                state, context = self._advance(state, previous, memory)
                logits = (
                    torch.stack(registers)
                    @ self.register_query(context)
                    / math.sqrt(self.config.width)
                )
                allowed = [index for index, value in enumerate(kinds) if value == kind]
                teacher = None if target is None else target.instructions[step].args[slot]
                register, logp = self._choose(logits, allowed, teacher)
                terms.append(logp)
                arguments.append(register)
                previous = registers[register]
            instructions.append(Instruction(operation, tuple(arguments)))
            registers.append(self.result(torch.cat((state, previous, embedding))).tanh())
            kinds.append(result_type)
            previous = registers[-1]
        raise RuntimeError("typed decoder failed to emit its required terminal symbol")

    def loss(self, features, input_spans, input_types, program):
        """Whole-program teacher-forced loss; no annotated operation spans."""
        _, logp, count = self._run(features, input_spans, input_types, program)
        return -logp / count

    def score(self, features, input_spans, input_types, program):
        """Unnormalized-by-length log probability of one complete proposal."""
        return self._run(features, input_spans, input_types, program)[1]

    @torch.no_grad()
    def decode(self, features, input_spans, input_types):
        """Greedy experimental proposal using only public request evidence."""
        program, logp, count = self._run(features, input_spans, input_types)
        return program, {"log_probability": float(logp), "tokens": count, "search": "greedy"}

    @torch.no_grad()
    def propose_beam(self, features, input_spans, input_types, *, beam_width=4,
                     max_programs=4):
        """Retain typed complete-program rivals without using an expected answer."""
        if (type(beam_width) is not int or not 1 <= beam_width <= 32
                or type(max_programs) is not int or not 1 <= max_programs <= beam_width):
            raise ValueError("program beam needs bounded positive widths")
        memory, registers, kinds = self._encode(features, input_spans, input_types)
        n_inputs = len(registers)
        initial = self.initial(memory.mean(dim=0)).tanh()
        active = [(0.0, initial, self.start, tuple(registers), tuple(kinds), ())]
        completed = []
        eos = len(self.operations)

        for step in range(self.config.max_steps + 1):
            following = []
            for score, state, previous, values, types, instructions in active:
                state, context = self._advance(state, previous, memory)
                allowed = [eos] if instructions else []
                if step < self.config.max_steps:
                    allowed.extend(index for index, operation in enumerate(self.operations)
                                   if all(kind in types for kind in
                                          semantic_primitive_type_signature(operation)[0]))
                log_probs = self._distribution(self.operation_logits(context), allowed)
                if instructions:
                    completed.append((score + float(log_probs[eos]),
                                      Program(n_inputs, instructions)))
                if step == self.config.max_steps:
                    continue
                for index in allowed:
                    if index == eos:
                        continue
                    operation = self.operations[index]
                    argument_types, result_type = semantic_primitive_type_signature(operation)
                    embedding = self.operation_embedding.weight[index]
                    partial = [(score + float(log_probs[index]), state, embedding, ())]
                    for kind in argument_types:
                        expanded = []
                        eligible = [position for position, value in enumerate(types)
                                    if value == kind]
                        for path_score, path_state, path_previous, args in partial:
                            next_state, next_context = self._advance(
                                path_state, path_previous, memory)
                            logits = (torch.stack(values) @ self.register_query(next_context)
                                      / math.sqrt(self.config.width))
                            choices = self._distribution(logits, eligible)
                            expanded.extend((path_score + float(choices[ref]), next_state,
                                             values[ref], (*args, ref)) for ref in eligible)
                        partial = sorted(expanded, key=lambda row: -row[0])[:beam_width]
                    for path_score, path_state, path_previous, args in partial:
                        result = self.result(torch.cat((path_state, path_previous,
                                                        embedding))).tanh()
                        following.append((path_score, path_state, result,
                                          (*values, result), (*types, result_type),
                                          (*instructions, Instruction(operation, args))))
            active = sorted(following, key=lambda row: -row[0])[:beam_width]
            if not active:
                break
        completed.sort(key=lambda row: (-row[0], repr(row[1])))
        return tuple((program, {"log_probability": score,
                                "tokens": 1 + sum(1 + len(ins.args)
                                                   for ins in program.instructions),
                                "beam_width": beam_width, "search": "typed_beam"})
                     for score, program in completed[:max_programs])
