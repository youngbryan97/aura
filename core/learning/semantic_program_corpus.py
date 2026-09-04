"""Program-first supervision for a learned language-to-program transducer.

The corpus owns no runtime parser.  It renders a known program into varied
language while recording the exact character spans that expressed each input,
operation, and reference.  A tokenizer adapter later projects those measured
spans into :class:`SemanticProgramIR`; no string rule participates in serving.

Evaluation splits hold out complete construction forms.  Token-level models
therefore have to transfer operation and pointer semantics to syntax they were
not trained on instead of recalling another rendering of the same template.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final, Literal

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    SemanticValue,
    TokenSpan,
    normalize_semantic_value,
)

CorpusSplit = Literal["train", "validation", "test"]

SEMANTIC_PROGRAM_CORPUS_SCHEMA: Final = "aura.semantic_program_corpus.v3"

_OPERATION_LANGUAGE: Final[dict[str, dict[str, str]]] = {
    "add": {"verb": "add", "noun": "sum"},
    "sub": {"verb": "subtract", "noun": "difference"},
    "mul": {"verb": "multiply", "noun": "product"},
    "idiv": {"verb": "integer-divide", "noun": "whole-number quotient"},
}

_SEQUENCE_TRANSFORMS: Final = (
    "unique",
    "sorted_up",
    "reversed_",
    "tail",
    "front",
)
_SEQUENCE_AGGREGATES: Final = (
    "length",
    "total",
    "largest",
    "smallest",
    "head",
    "last",
)
_SEQUENCE_BINARY_SELECTORS: Final = ("at", "count_of")
_SCALAR_CONTINUATIONS: Final = ("add", "sub", "mul", "idiv")
_SEQUENCE_OPERATION_LANGUAGE: Final[dict[str, tuple[str, ...]]] = {
    "unique": (
        "duplicate removal",
        "deduplication",
        "retaining one copy of each value",
        "removing repeated values",
        "collapsing duplicates",
        "keeping distinct values only",
        "discarding extra copies",
        "making the values unique",
        "selecting one of every value",
    ),
    "sorted_up": (
        "ascending sorting",
        "ordering from low to high",
        "increasing-order arrangement",
        "sorting upward",
        "placing values in ascending order",
        "smallest-to-largest ordering",
        "an ascending reorder",
        "low-to-high sorting",
        "increasing arrangement",
    ),
    "reversed_": (
        "order reversal",
        "reversing the order",
        "backward arrangement",
        "flipping the sequence",
        "end-to-start ordering",
        "a reverse traversal",
        "turning the order around",
        "last-to-first arrangement",
        "sequence reversal",
    ),
    "tail": (
        "dropping the first item",
        "taking everything after the first item",
        "removing the head",
        "keeping the tail",
        "discarding the initial value",
        "trimming the first entry",
        "selecting all but the first value",
        "taking the sequence tail",
        "omitting the leading item",
    ),
    "front": (
        "dropping the final item",
        "taking everything before the last item",
        "removing the last value",
        "keeping the front",
        "discarding the terminal value",
        "trimming the final entry",
        "selecting all but the last value",
        "taking the sequence front",
        "omitting the trailing item",
    ),
    "length": (
        "item counting",
        "measuring the length",
        "counting the entries",
        "finding how many values remain",
        "determining the item count",
        "measuring its size",
        "counting its members",
        "obtaining the sequence length",
        "computing the number of entries",
    ),
    "total": (
        "summation",
        "adding all values",
        "computing the total",
        "finding the sum",
        "combining the entries by addition",
        "totaling its members",
        "accumulating every value",
        "calculating the aggregate sum",
        "summing the sequence",
    ),
    "largest": (
        "maximum selection",
        "finding the largest value",
        "taking the maximum",
        "selecting the greatest entry",
        "identifying the highest value",
        "choosing its maximum member",
        "finding the top value",
        "extracting the greatest number",
        "determining the maximum",
    ),
    "smallest": (
        "minimum selection",
        "finding the smallest value",
        "taking the minimum",
        "selecting the least entry",
        "identifying the lowest value",
        "choosing its minimum member",
        "finding the bottom value",
        "extracting the least number",
        "determining the minimum",
    ),
    "head": (
        "first-item selection",
        "taking the first value",
        "selecting the head",
        "extracting the initial entry",
        "choosing the leading value",
        "reading its first member",
        "finding the value at the front",
        "obtaining the head item",
        "returning the initial value",
    ),
    "last": (
        "last-item selection",
        "taking the final value",
        "selecting the last member",
        "extracting the terminal entry",
        "choosing the trailing value",
        "reading its final member",
        "finding the value at the end",
        "obtaining the last item",
        "returning the terminal value",
    ),
}

_SEQUENCE_BINARY_OPERATION_LANGUAGE: Final[dict[str, tuple[str, ...]]] = {
    "at": (
        "index lookup",
        "position-based selection",
        "item retrieval by index",
        "indexed access",
        "selection at a numbered position",
        "reading one indexed entry",
        "retrieval from a sequence position",
        "looking up an item by index",
        "access at the given index",
    ),
    "count_of": (
        "occurrence counting",
        "frequency measurement",
        "counting matching values",
        "multiplicity calculation",
        "measuring how often a value occurs",
        "counting copies of one value",
        "finding a value's frequency",
        "tallying matching entries",
        "computing the occurrence count",
    ),
}

_SCALAR_CONTINUATION_LANGUAGE: Final[dict[str, tuple[str, ...]]] = {
    "add": (
        "plus",
        "added to",
        "increased by",
        "combined additively with",
        "summed with",
        "augmented by",
        "with the addition of",
        "plus the quantity",
        "after adding",
    ),
    "sub": (
        "minus",
        "reduced by",
        "less",
        "with the subtraction of",
        "decreased by",
        "after removing",
        "minus the quantity",
        "after subtracting",
        "with a deduction of",
    ),
    "mul": (
        "times",
        "multiplied by",
        "scaled by",
        "combined multiplicatively with",
        "with a factor of",
        "after multiplication by",
        "times the quantity",
        "using a multiplier of",
        "with the product factor",
    ),
    "idiv": (
        "integer-divided by",
        "floor-divided by",
        "divided by with the quotient rounded down using",
        "reduced to the whole-number quotient over",
        "divided without a remainder fraction by",
        "converted to the integer quotient over",
        "whole-number divided by",
        "divided by and rounded down using",
        "integrally divided by",
    ),
}


@dataclass(frozen=True, slots=True, order=True)
class CharacterSpan:
    """One non-empty half-open span in rendered source text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("semantic character span is invalid")


@dataclass(frozen=True, slots=True)
class SemanticInstructionAnnotation:
    """Gold program instruction and the language that expressed it."""

    instruction: Instruction
    operation_span: CharacterSpan
    argument_spans: tuple[CharacterSpan, ...]
    depends_on: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProgramTopology:
    """A two-step register graph independent of operations and wording."""

    topology_id: str
    first_args: tuple[int, int]
    remaining_input: int
    result_is_left: bool

    def __post_init__(self) -> None:
        if (
            not self.topology_id
            or len(set(self.first_args)) != 2
            or set(self.first_args) | {self.remaining_input} != {0, 1, 2}
        ):
            raise ValueError("semantic program topology is invalid")

    @property
    def second_args(self) -> tuple[int, int]:
        return (3, self.remaining_input) if self.result_is_left else (self.remaining_input, 3)


@dataclass(frozen=True, slots=True)
class ForkJoinTopology:
    """A three-step graph with two independent branches and one join."""

    topology_id: str
    arguments: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    def __post_init__(self) -> None:
        first, second, join = self.arguments
        if (
            not self.topology_id
            or len(set((*first, *second))) != 4
            or set((*first, *second)) != {0, 1, 2, 3}
            or set(join) != {4, 5}
        ):
            raise ValueError("semantic fork-join topology is invalid")


@dataclass(frozen=True, slots=True)
class SemanticProgramExample:
    """One immutable program-first language example."""

    example_id: str
    construction_id: str
    topology_id: str
    split: CorpusSplit
    source_text: str
    inputs: tuple[SemanticValue, ...]
    input_spans: tuple[CharacterSpan, ...]
    instructions: tuple[SemanticInstructionAnnotation, ...]
    report_value: int
    contrast_id: str
    register_definition_spans: tuple[CharacterSpan, ...] = ()
    schema: str = SEMANTIC_PROGRAM_CORPUS_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != SEMANTIC_PROGRAM_CORPUS_SCHEMA
            or not self.example_id
            or not self.construction_id
            or not self.topology_id
            or self.split not in {"train", "validation", "test"}
            or not self.source_text.strip()
            or not 1 <= len(self.inputs) <= 8
            or len(self.inputs) != len(self.input_spans)
            or not self.instructions
        ):
            raise ValueError("semantic corpus example envelope is invalid")
        if any(normalize_semantic_value(value) != value for value in self.inputs):
            raise ValueError("semantic corpus inputs are outside the exact value algebra")
        for span in self.input_spans:
            _validate_character_span(span, self.source_text)
        n_inputs = len(self.inputs)
        if self.register_definition_spans and len(self.register_definition_spans) != (
            n_inputs + len(self.instructions)
        ):
            raise ValueError("semantic corpus register definitions have wrong arity")
        for span in self.register_definition_spans:
            _validate_character_span(span, self.source_text)
        for ordinal, annotated in enumerate(self.instructions):
            output = n_inputs + ordinal
            if any(argument >= output for argument in annotated.instruction.args):
                raise ValueError("semantic corpus program is not forward SSA")
            _validate_character_span(annotated.operation_span, self.source_text)
            for span in annotated.argument_spans:
                _validate_character_span(span, self.source_text)
            if len(annotated.argument_spans) != len(annotated.instruction.args):
                raise ValueError("semantic corpus argument attribution has wrong arity")
            expected_dependencies = tuple(
                sorted(
                    argument - n_inputs
                    for argument in set(annotated.instruction.args)
                    if argument >= n_inputs
                )
            )
            if annotated.depends_on != expected_dependencies:
                raise ValueError("semantic corpus dependency attribution is invalid")
        terminal = n_inputs + len(self.instructions) - 1
        if self.report_value != terminal:
            raise ValueError("semantic corpus must report its terminal SSA value")

    @property
    def program(self) -> Program:
        return Program(
            n_inputs=len(self.inputs),
            instructions=tuple(item.instruction for item in self.instructions),
        )


class _AnnotatedText:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0
        self._spans: dict[str, CharacterSpan] = {}
        self._open_spans: dict[str, int] = {}

    def append(self, text: str, *, label: str = "") -> None:
        if not text:
            raise ValueError("semantic corpus cannot append empty text")
        start = self._length
        self._parts.append(text)
        self._length += len(text)
        if label:
            if label in self._spans:
                raise ValueError(f"semantic corpus label is not unique: {label}")
            self._spans[label] = CharacterSpan(start, self._length)

    def begin(self, label: str) -> None:
        if label in self._spans or label in self._open_spans:
            raise ValueError(f"semantic corpus label is not unique: {label}")
        self._open_spans[label] = self._length

    def finish(self, label: str) -> None:
        start = self._open_spans.pop(label)
        self._spans[label] = CharacterSpan(start, self._length)

    def span(self, label: str) -> CharacterSpan:
        return self._spans[label]

    @property
    def text(self) -> str:
        return "".join(self._parts)


def _validate_character_span(span: CharacterSpan, text: str) -> None:
    if span.end > len(text) or not text[span.start : span.end].strip():
        raise ValueError("semantic character span exceeds or misses source text")


def _example_id(
    construction_id: str,
    topology_id: str,
    first_op: str,
    second_op: str,
    inputs: tuple[int, int, int],
) -> str:
    body = f"{construction_id}|{topology_id}|{first_op}|{second_op}|{inputs}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _fork_join_example_id(
    construction_id: str,
    topology_id: str,
    operations: tuple[str, str, str],
    inputs: tuple[int, int, int, int],
) -> str:
    body = f"{construction_id}|{topology_id}|{operations}|{inputs}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _annotation(
    builder: _AnnotatedText,
    *,
    op: str,
    args: tuple[int, int],
    operation_label: str,
    argument_labels: tuple[str, str],
    depends_on: tuple[int, ...],
) -> SemanticInstructionAnnotation:
    return SemanticInstructionAnnotation(
        instruction=Instruction(op, args),
        operation_span=builder.span(operation_label),
        argument_spans=tuple(builder.span(label) for label in argument_labels),
        depends_on=depends_on,
    )


def _append_binary_verb(
    builder: _AnnotatedText,
    *,
    op: str,
    operation_label: str,
    left: tuple[str, str],
    right: tuple[str, str],
    capitalize: bool = False,
) -> tuple[str, str]:
    """Render a binary operation while preserving semantic operand order."""

    verb = _OPERATION_LANGUAGE[op]["verb"]
    builder.append(verb.capitalize() if capitalize else verb, label=operation_label)
    builder.append(" ")
    if op == "sub":
        builder.append(right[0], label=right[1])
        builder.append(" from ")
        builder.append(left[0], label=left[1])
    else:
        builder.append(left[0], label=left[1])
        builder.append(" and " if op == "add" else " by ")
        builder.append(right[0], label=right[1])
    return left[1], right[1]


def _argument(
    values: tuple[int, int, int],
    index: int,
) -> tuple[str, str]:
    return str(values[index]), f"in{index}"


def _first_arguments(
    values: tuple[int, int, int],
    topology: ProgramTopology,
) -> tuple[tuple[str, str], tuple[str, str]]:
    return (
        _argument(values, topology.first_args[0]),
        _argument(values, topology.first_args[1]),
    )


def _second_arguments(
    values: tuple[int, int, int],
    topology: ProgramTopology,
    *,
    result: tuple[str, str],
    remaining_label: str | None = None,
) -> tuple[tuple[str, str], tuple[str, str]]:
    remaining = (
        str(values[topology.remaining_input]),
        remaining_label or f"in{topology.remaining_input}",
    )
    return (result, remaining) if topology.result_is_left else (remaining, result)


def _sequential(
    first_op: str,
    second_op: str,
    values: tuple[int, int, int],
    topology: ProgramTopology,
    *,
    result_phrase: str = "that result",
    connector: str = ". Then ",
    ending: str = " using whole-number arithmetic.",
) -> tuple[_AnnotatedText, tuple[str, str, str], tuple[SemanticInstructionAnnotation, ...]]:
    builder = _AnnotatedText()
    first_left, first_right = _first_arguments(values, topology)
    first_args = _append_binary_verb(
        builder,
        op=first_op,
        operation_label="op0",
        left=first_left,
        right=first_right,
        capitalize=True,
    )
    builder.append(connector)
    second_left, second_right = _second_arguments(
        values,
        topology,
        result=(result_phrase, "ref0"),
    )
    second_args = _append_binary_verb(
        builder,
        op=second_op,
        operation_label="op1",
        left=second_left,
        right=second_right,
    )
    builder.append(ending)
    annotations = (
        _annotation(
            builder,
            op=first_op,
            args=topology.first_args,
            operation_label="op0",
            argument_labels=first_args,
            depends_on=(),
        ),
        _annotation(
            builder,
            op=second_op,
            args=topology.second_args,
            operation_label="op1",
            argument_labels=second_args,
            depends_on=(0,),
        ),
    )
    return builder, ("in0", "in1", "in2"), annotations


def _nominal_nested(
    first_op: str,
    second_op: str,
    values: tuple[int, int, int],
    topology: ProgramTopology,
) -> tuple[_AnnotatedText, tuple[str, str, str], tuple[SemanticInstructionAnnotation, ...]]:
    first = _OPERATION_LANGUAGE[first_op]
    second = _OPERATION_LANGUAGE[second_op]
    builder = _AnnotatedText()
    builder.append("Return the ")
    builder.append(second["noun"], label="op1")
    builder.append(" of ")
    first_left, first_right = _first_arguments(values, topology)

    def append_first_expression() -> None:
        builder.begin("ref0")
        builder.append("the ")
        builder.append(first["noun"], label="op0")
        builder.append(" of ")
        builder.append(first_left[0], label=first_left[1])
        builder.append(" divided by " if first_op == "idiv" else " and ")
        builder.append(first_right[0], label=first_right[1])
        builder.finish("ref0")

    remaining = _argument(values, topology.remaining_input)
    joiner = " divided by " if second_op == "idiv" else " and "
    if topology.result_is_left:
        append_first_expression()
        builder.append(joiner)
        builder.append(remaining[0], label=remaining[1])
        second_args = ("ref0", remaining[1])
    else:
        builder.append(remaining[0], label=remaining[1])
        builder.append(joiner)
        append_first_expression()
        second_args = (remaining[1], "ref0")
    builder.append(". Use integer arithmetic.")
    annotations = (
        _annotation(
            builder,
            op=first_op,
            args=topology.first_args,
            operation_label="op0",
            argument_labels=(first_left[1], first_right[1]),
            depends_on=(),
        ),
        _annotation(
            builder,
            op=second_op,
            args=topology.second_args,
            operation_label="op1",
            argument_labels=second_args,
            depends_on=(0,),
        ),
    )
    return builder, ("in0", "in1", "in2"), annotations


def _fronted_operand(
    first_op: str,
    second_op: str,
    values: tuple[int, int, int],
    topology: ProgramTopology,
    *,
    result_phrase: str = "the obtained value",
    reserved_role: str = "the earlier operand",
    connector: str = "; afterward, ",
) -> tuple[_AnnotatedText, tuple[str, str, str], tuple[SemanticInstructionAnnotation, ...]]:
    builder = _AnnotatedText()
    builder.append("Using ")
    builder.append(
        str(values[topology.remaining_input]),
        label=f"in{topology.remaining_input}",
    )
    builder.append(" as the reserved operand, first ")
    first_left, first_right = _first_arguments(values, topology)
    first_args = _append_binary_verb(
        builder,
        op=first_op,
        operation_label="op0",
        left=first_left,
        right=first_right,
    )
    builder.append(connector)
    second_left, second_right = _second_arguments(
        values,
        topology,
        result=(result_phrase, "ref0"),
        remaining_label="ref_remaining",
    )
    second_args = _append_binary_verb(
        builder,
        op=second_op,
        operation_label="op1",
        left=(
            result_phrase if second_left[1] == "ref0" else reserved_role,
            second_left[1],
        ),
        right=(
            result_phrase if second_right[1] == "ref0" else reserved_role,
            second_right[1],
        ),
    )
    builder.append(". Return the integer result.")
    annotations = (
        _annotation(
            builder,
            op=first_op,
            args=topology.first_args,
            operation_label="op0",
            argument_labels=first_args,
            depends_on=(),
        ),
        _annotation(
            builder,
            op=second_op,
            args=topology.second_args,
            operation_label="op1",
            argument_labels=second_args,
            depends_on=(0,),
        ),
    )
    return builder, ("in0", "in1", "in2"), annotations


def _reverse_clause(
    first_op: str,
    second_op: str,
    values: tuple[int, int, int],
    topology: ProgramTopology,
    *,
    result_phrase: str = "the intermediate quantity",
    bridge: str = ", obtain that quantity: ",
    ending: str = ". Report the whole-number result.",
) -> tuple[_AnnotatedText, tuple[str, str, str], tuple[SemanticInstructionAnnotation, ...]]:
    builder = _AnnotatedText()
    builder.append("Before you ")
    second_left, second_right = _second_arguments(
        values,
        topology,
        result=(result_phrase, "ref0"),
    )
    second_args = _append_binary_verb(
        builder,
        op=second_op,
        operation_label="op1",
        left=second_left,
        right=second_right,
    )
    builder.append(bridge)
    first_left, first_right = _first_arguments(values, topology)
    first_args = _append_binary_verb(
        builder,
        op=first_op,
        operation_label="op0",
        left=first_left,
        right=first_right,
    )
    builder.append(ending)
    annotations = (
        _annotation(
            builder,
            op=first_op,
            args=topology.first_args,
            operation_label="op0",
            argument_labels=first_args,
            depends_on=(),
        ),
        _annotation(
            builder,
            op=second_op,
            args=topology.second_args,
            operation_label="op1",
            argument_labels=second_args,
            depends_on=(0,),
        ),
    )
    return builder, ("in0", "in1", "in2"), annotations


def _fork_join(
    operations: tuple[str, str, str],
    values: tuple[int, int, int, int],
    topology: ForkJoinTopology,
    *,
    opening: str,
    branch_bridge: str,
    join_bridge: str,
    first_reference: str,
    second_reference: str,
    ending: str,
) -> tuple[
    _AnnotatedText,
    tuple[str, str, str, str],
    tuple[SemanticInstructionAnnotation, ...],
]:
    builder = _AnnotatedText()
    builder.append(opening)
    first_args = topology.arguments[0]
    first_labels = _append_binary_verb(
        builder,
        op=operations[0],
        operation_label="op0",
        left=(str(values[first_args[0]]), f"in{first_args[0]}"),
        right=(str(values[first_args[1]]), f"in{first_args[1]}"),
    )
    builder.append(f", naming it {first_reference}")
    builder.append(branch_bridge)
    second_args = topology.arguments[1]
    second_labels = _append_binary_verb(
        builder,
        op=operations[1],
        operation_label="op1",
        left=(str(values[second_args[0]]), f"in{second_args[0]}"),
        right=(str(values[second_args[1]]), f"in{second_args[1]}"),
    )
    builder.append(f", naming it {second_reference}")
    builder.append(join_bridge)
    join_args = topology.arguments[2]
    references = {
        4: (first_reference, "ref0"),
        5: (second_reference, "ref1"),
    }
    join_labels = _append_binary_verb(
        builder,
        op=operations[2],
        operation_label="op2",
        left=references[join_args[0]],
        right=references[join_args[1]],
    )
    builder.append(ending)
    annotations = (
        _annotation(
            builder,
            op=operations[0],
            args=first_args,
            operation_label="op0",
            argument_labels=first_labels,
            depends_on=(),
        ),
        _annotation(
            builder,
            op=operations[1],
            args=second_args,
            operation_label="op1",
            argument_labels=second_labels,
            depends_on=(),
        ),
        _annotation(
            builder,
            op=operations[2],
            args=join_args,
            operation_label="op2",
            argument_labels=join_labels,
            depends_on=(0, 1),
        ),
    )
    input_labels = tuple(f"in{index}" for index in range(4))
    source_order = tuple(sorted(input_labels, key=lambda label: builder.span(label).start))
    return builder, source_order, annotations


_CONSTRUCTIONS: Final = {
    "sequential_result_then": ("train", _sequential),
    "sequential_intermediate_after": (
        "train",
        partial(
            _sequential,
            result_phrase="the intermediate value",
            connector=". After that, ",
        ),
    ),
    "nominal_nested": ("train", _nominal_nested),
    "fronted_obtained_afterward": ("train", _fronted_operand),
    "reverse_intermediate_before": ("train", _reverse_clause),
    "sequential_obtained_afterward": (
        "validation",
        partial(
            _sequential,
            result_phrase="the obtained value",
            connector=". Afterward, ",
        ),
    ),
    "fronted_intermediate_then": (
        "validation",
        partial(
            _fronted_operand,
            result_phrase="the intermediate value",
            connector="; then, ",
        ),
    ),
    "reverse_result_prior": (
        "test",
        partial(
            _reverse_clause,
            result_phrase="that result",
            bridge=", first produce it: ",
        ),
    ),
    "fronted_result_then": (
        "test",
        partial(
            _fronted_operand,
            result_phrase="that result",
            connector="; then, ",
        ),
    ),
}

_TOPOLOGIES: Final = (
    ProgramTopology("left_01_then_2", (0, 1), 2, True),
    ProgramTopology("left_12_then_0", (1, 2), 0, True),
    ProgramTopology("1_then_right_02", (0, 2), 1, False),
    ProgramTopology("0_then_right_12", (1, 2), 0, False),
)

_FORK_JOIN_TOPOLOGIES: Final = (
    ForkJoinTopology("pair_01_23_join_45", ((0, 1), (2, 3), (4, 5))),
    ForkJoinTopology("pair_02_13_join_54", ((0, 2), (1, 3), (5, 4))),
    ForkJoinTopology("pair_12_03_join_45", ((1, 2), (0, 3), (4, 5))),
    ForkJoinTopology("pair_23_01_join_54", ((2, 3), (0, 1), (5, 4))),
    ForkJoinTopology("pair_02_13_join_45", ((0, 2), (1, 3), (4, 5))),
    ForkJoinTopology("pair_03_12_join_54", ((0, 3), (1, 2), (5, 4))),
    ForkJoinTopology("pair_01_23_join_54", ((0, 1), (2, 3), (5, 4))),
    ForkJoinTopology("pair_13_02_join_45", ((1, 3), (0, 2), (4, 5))),
    ForkJoinTopology("pair_12_03_join_54", ((1, 2), (0, 3), (5, 4))),
)

_FORK_JOIN_CONSTRUCTIONS: Final = (
    (
        "fork_first_separate_finally",
        "train",
        partial(
            _fork_join,
            opening="First, ",
            branch_bridge=". Separately, ",
            join_bridge=". Finally, ",
            first_reference="the first result",
            second_reference="the second result",
            ending=". Return the exact integer.",
        ),
    ),
    (
        "fork_begin_independently_combine",
        "train",
        partial(
            _fork_join,
            opening="Begin by ",
            branch_bridge=". Independently, ",
            join_bridge=". Combine them: ",
            first_reference="result alpha",
            second_reference="result beta",
            ending=". Report the whole-number answer.",
        ),
    ),
    (
        "fork_compute_apart_then",
        "train",
        partial(
            _fork_join,
            opening="Compute ",
            branch_bridge=". Apart from that, ",
            join_bridge=". Then, ",
            first_reference="the earlier value",
            second_reference="the separate value",
            ending=". Give the integer result.",
        ),
    ),
    (
        "fork_prepare_also_resolve",
        "validation",
        partial(
            _fork_join,
            opening="Prepare one branch: ",
            branch_bridge=". Also prepare another: ",
            join_bridge=". Resolve the two branches by ",
            first_reference="branch one",
            second_reference="branch two",
            ending=". Return that number.",
        ),
    ),
    (
        "fork_form_in_parallel_merge",
        "validation",
        partial(
            _fork_join,
            opening="Form ",
            branch_bridge=". In parallel, ",
            join_bridge=". Merge those values: ",
            first_reference="the former result",
            second_reference="the latter result",
            ending=". Use integer arithmetic.",
        ),
    ),
    (
        "fork_derive_elsewhere_finish",
        "validation",
        partial(
            _fork_join,
            opening="Derive ",
            branch_bridge=". Elsewhere, ",
            join_bridge=". Finish by ",
            first_reference="the primary result",
            second_reference="the auxiliary result",
            ending=". Report the exact value.",
        ),
    ),
    (
        "fork_establish_separately_reconcile",
        "test",
        partial(
            _fork_join,
            opening="Establish ",
            branch_bridge=". Separately establish ",
            join_bridge=". Reconcile them: ",
            first_reference="the initial branch",
            second_reference="the other branch",
            ending=". Return the integer.",
        ),
    ),
    (
        "fork_obtain_independently_conclude",
        "test",
        partial(
            _fork_join,
            opening="Obtain ",
            branch_bridge=". Independently obtain ",
            join_bridge=". Conclude by ",
            first_reference="value one",
            second_reference="value two",
            ending=". Give the final number.",
        ),
    ),
    (
        "fork_produce_aside_join",
        "test",
        partial(
            _fork_join,
            opening="Produce ",
            branch_bridge=". Aside from it, produce ",
            join_bridge=". Join the results: ",
            first_reference="the first quantity",
            second_reference="the second quantity",
            ending=". Report the integer result.",
        ),
    ),
)


def build_semantic_program_corpus(
    *,
    seed: int = 271828,
    examples_per_operation_pair: int = 2,
) -> tuple[SemanticProgramExample, ...]:
    """Return deterministic examples with construction-disjoint splits."""

    if examples_per_operation_pair < 1:
        raise ValueError("semantic corpus needs at least one example per operation pair")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    operations = tuple(_OPERATION_LANGUAGE)
    for construction_id, (split, renderer) in _CONSTRUCTIONS.items():
        for topology in _TOPOLOGIES:
            sample_values = tuple(
                (
                    rng.randint(60, 97),
                    rng.randint(20, 49),
                    rng.randint(2, 9),
                )
                for _ in range(examples_per_operation_pair)
            )
            for first_op in operations:
                for second_op in operations:
                    for sample_index, values in enumerate(sample_values):
                        builder, input_labels, annotations = renderer(
                            first_op,
                            second_op,
                            values,
                            topology,
                        )
                        contrast_id = (
                            f"{construction_id}:{topology.topology_id}:{sample_index}:{values}"
                        )
                        examples.append(
                            SemanticProgramExample(
                                example_id=_example_id(
                                    construction_id,
                                    topology.topology_id,
                                    first_op,
                                    second_op,
                                    values,
                                ),
                                construction_id=construction_id,
                                topology_id=topology.topology_id,
                                split=split,
                                source_text=builder.text,
                                inputs=values,
                                input_spans=tuple(builder.span(label) for label in input_labels),
                                instructions=annotations,
                                report_value=4,
                                contrast_id=contrast_id,
                            )
                        )
    return tuple(examples)


def build_semantic_program_fork_join_corpus(
    *,
    seed: int = 1618033,
    examples_per_operation_triple: int = 1,
    source_order_registers: bool = False,
) -> tuple[SemanticProgramExample, ...]:
    """Return a construction- and topology-disjoint three-step corpus.

    The historical corpus preserves generator register identities for exact
    evidence replay.  The source-order variant makes input identity observable:
    register ``i`` is the ``i``th input mention in the source text.
    """

    if examples_per_operation_triple < 1:
        raise ValueError("fork-join corpus needs at least one example per operation triple")
    rng = random.Random(seed)
    operations = tuple(_OPERATION_LANGUAGE)
    examples: list[SemanticProgramExample] = []
    for construction_index, (construction_id, split, renderer) in enumerate(
        _FORK_JOIN_CONSTRUCTIONS
    ):
        topology = _FORK_JOIN_TOPOLOGIES[construction_index]
        sample_values = tuple(
            (
                rng.randint(80, 97),
                rng.randint(40, 59),
                rng.randint(15, 29),
                rng.randint(2, 9),
            )
            for _ in range(examples_per_operation_triple)
        )
        for first_op in operations:
            for second_op in operations:
                for join_op in operations:
                    operation_tuple = (first_op, second_op, join_op)
                    for sample_index, values in enumerate(sample_values):
                        examples.append(
                            _build_fork_join_example(
                                construction_id=construction_id,
                                split=split,
                                renderer=renderer,
                                topology=topology,
                                operations=operation_tuple,
                                values=values,
                                sample_index=sample_index,
                                source_order_registers=source_order_registers,
                            )
                        )
    return tuple(examples)


def _build_fork_join_example(
    *,
    construction_id: str,
    split: CorpusSplit,
    renderer: Callable[
        ...,
        tuple[
            _AnnotatedText,
            tuple[str, str, str, str],
            tuple[SemanticInstructionAnnotation, ...],
        ],
    ],
    topology: ForkJoinTopology,
    operations: tuple[str, str, str],
    values: tuple[int, int, int, int],
    sample_index: int,
    source_order_registers: bool,
) -> SemanticProgramExample:
    builder, input_labels, annotations = renderer(operations, values, topology)
    if source_order_registers:
        input_order = tuple(int(label.removeprefix("in")) for label in input_labels)
        if set(input_order) != {0, 1, 2, 3}:
            raise AssertionError("fork-join renderer did not expose every input once")
        register_map = {
            old_register: new_register for new_register, old_register in enumerate(input_order)
        }
        corpus_values = tuple(values[index] for index in input_order)
        corpus_annotations = tuple(
            SemanticInstructionAnnotation(
                instruction=Instruction(
                    item.instruction.op,
                    tuple(
                        register_map.get(argument, argument) for argument in item.instruction.args
                    ),
                ),
                operation_span=item.operation_span,
                argument_spans=item.argument_spans,
                depends_on=item.depends_on,
            )
            for item in annotations
        )
        corpus_input_labels = input_labels
    else:
        corpus_values = values
        corpus_annotations = annotations
        corpus_input_labels = tuple(f"in{index}" for index in range(4))
    contrast_id = f"{construction_id}:{topology.topology_id}:{sample_index}:{corpus_values}"
    return SemanticProgramExample(
        example_id=_fork_join_example_id(
            construction_id,
            topology.topology_id,
            operations,
            corpus_values,
        ),
        construction_id=construction_id,
        topology_id=topology.topology_id,
        split=split,
        source_text=builder.text,
        inputs=corpus_values,
        input_spans=tuple(builder.span(label) for label in corpus_input_labels),
        instructions=corpus_annotations,
        report_value=6,
        contrast_id=contrast_id,
    )


def build_semantic_program_fork_join_factorial_corpus(
    *,
    seed: int = 2718281,
    examples_per_cell: int = 1,
) -> tuple[SemanticProgramExample, ...]:
    """Cross wording, graph topology, and balanced primitive coverage."""

    if examples_per_cell < 1:
        raise ValueError("factorial fork-join corpus needs at least one sample per cell")
    rng = random.Random(seed)
    operations = tuple(_OPERATION_LANGUAGE)
    operation_cover = tuple(
        (first, second, operations[(first_index + second_index) % len(operations)])
        for first_index, first in enumerate(operations)
        for second_index, second in enumerate(operations)
    )
    examples: list[SemanticProgramExample] = []
    for construction_id, split, renderer in _FORK_JOIN_CONSTRUCTIONS:
        for topology in _FORK_JOIN_TOPOLOGIES:
            sample_values = tuple(
                (
                    rng.randint(80, 97),
                    rng.randint(40, 59),
                    rng.randint(15, 29),
                    rng.randint(2, 9),
                )
                for _ in range(examples_per_cell)
            )
            for operation_tuple in operation_cover:
                for sample_index, values in enumerate(sample_values):
                    examples.append(
                        _build_fork_join_example(
                            construction_id=construction_id,
                            split=split,
                            renderer=renderer,
                            topology=topology,
                            operations=operation_tuple,
                            values=values,
                            sample_index=sample_index,
                            source_order_registers=True,
                        )
                    )
    return tuple(examples)


def _sequence_chain_example_id(
    construction_id: str,
    first_op: str,
    second_op: str,
    values: tuple[int, ...],
    sample_index: int,
) -> str:
    body = f"{construction_id}|{first_op}|{second_op}|{values}|{sample_index}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _render_sequence_chain(
    *,
    construction_index: int,
    first_op: str,
    second_op: str,
    values: tuple[int, ...],
) -> tuple[str, CharacterSpan, tuple[SemanticInstructionAnnotation, ...]]:
    builder = _AnnotatedText()
    first_phrase = _SEQUENCE_OPERATION_LANGUAGE[first_op][construction_index]
    second_phrase = _SEQUENCE_OPERATION_LANGUAGE[second_op][construction_index]
    register_name = (
        "interim",
        "prepared",
        "derived",
        "working set",
        "transformed list",
        "intermediate sequence",
        "updated values",
        "resulting series",
        "processed sequence",
    )[construction_index]
    input_text = "[" + ", ".join(str(value) for value in values) + "]"

    def append_input() -> None:
        builder.append(input_text, label="input")

    def append_first_operation() -> None:
        builder.append(first_phrase, label="operation:0")

    def append_second_operation() -> None:
        builder.append(second_phrase, label="operation:1")

    def define_result() -> None:
        builder.append(register_name, label="result:0")

    def reference_result() -> None:
        builder.append(register_name, label="argument:1:0")

    if construction_index == 0:
        builder.append("Perform ")
        append_first_operation()
        builder.append(" on ")
        append_input()
        builder.append(", and call the output ")
        define_result()
        builder.append(". Then perform ")
        append_second_operation()
        builder.append(" on ")
        reference_result()
        builder.append(".")
    elif construction_index == 1:
        builder.append("Starting from ")
        append_input()
        builder.append(", use ")
        append_first_operation()
        builder.append(" to produce ")
        define_result()
        builder.append("; afterward use ")
        append_second_operation()
        builder.append(" on ")
        reference_result()
        builder.append(".")
    elif construction_index == 2:
        builder.append("After ")
        append_first_operation()
        builder.append(" is applied to ")
        append_input()
        builder.append(", name that sequence ")
        define_result()
        builder.append(". Return what ")
        append_second_operation()
        builder.append(" produces from ")
        reference_result()
        builder.append(".")
    elif construction_index == 3:
        builder.append("Take ")
        append_input()
        builder.append(" through ")
        append_first_operation()
        builder.append("; the new sequence is ")
        define_result()
        builder.append(". From ")
        reference_result()
        builder.append(", obtain the result by ")
        append_second_operation()
        builder.append(".")
    elif construction_index == 4:
        builder.append("For the values ")
        append_input()
        builder.append(", first carry out ")
        append_first_operation()
        builder.append(" and bind the outcome as ")
        define_result()
        builder.append(". Next evaluate ")
        reference_result()
        builder.append(" with ")
        append_second_operation()
        builder.append(".")
    elif construction_index == 5:
        builder.append("Transform ")
        append_input()
        builder.append(" via ")
        append_first_operation()
        builder.append(". Let ")
        define_result()
        builder.append(" denote that transformation; compute ")
        append_second_operation()
        builder.append(" over ")
        reference_result()
        builder.append(".")
    elif construction_index == 6:
        builder.append("Use ")
        append_first_operation()
        builder.append(" to turn ")
        append_input()
        builder.append(" into ")
        define_result()
        builder.append(". The final scalar comes from ")
        append_second_operation()
        builder.append(" on ")
        reference_result()
        builder.append(".")
    elif construction_index == 7:
        builder.append("Given ")
        append_input()
        builder.append(", apply ")
        append_first_operation()
        builder.append(" and call the outcome ")
        define_result()
        builder.append(". Evaluate ")
        reference_result()
        builder.append(" afterward through ")
        append_second_operation()
        builder.append(".")
    elif construction_index == 8:
        builder.append("Begin with ")
        append_input()
        builder.append(" and execute ")
        append_first_operation()
        builder.append("; refer to the result as ")
        define_result()
        builder.append(". Finish by ")
        append_second_operation()
        builder.append(" on ")
        reference_result()
        builder.append(".")
    else:  # pragma: no cover - private caller pins the construction range
        raise ValueError("sequence construction index is invalid")

    input_span = builder.span("input")
    instructions = (
        SemanticInstructionAnnotation(
            instruction=Instruction(first_op, (0,)),
            operation_span=builder.span("operation:0"),
            argument_spans=(input_span,),
            depends_on=(),
        ),
        SemanticInstructionAnnotation(
            instruction=Instruction(second_op, (1,)),
            operation_span=builder.span("operation:1"),
            argument_spans=(builder.span("argument:1:0"),),
            depends_on=(0,),
        ),
    )
    return builder.text, input_span, instructions


def build_semantic_program_sequence_corpus(
    *,
    seed: int = 1414213,
    examples_per_operation_pair: int = 2,
) -> tuple[SemanticProgramExample, ...]:
    """Build a typed sequence family with construction-held-out language."""

    if examples_per_operation_pair < 1:
        raise ValueError("sequence corpus needs at least one sample per operation pair")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    for construction_index in range(9):
        construction_id = f"sequence-construction-{construction_index}"
        split: CorpusSplit = (
            "train"
            if construction_index < 3
            else "validation"
            if construction_index < 6
            else "test"
        )
        for first_op in _SEQUENCE_TRANSFORMS:
            for second_op in _SEQUENCE_AGGREGATES:
                for sample_index in range(examples_per_operation_pair):
                    values = [rng.randint(1, 30) for _ in range(rng.randint(5, 8))]
                    values[-1] = values[0]
                    rng.shuffle(values)
                    public_sequence = tuple(values)
                    source_text, input_span, instructions = _render_sequence_chain(
                        construction_index=construction_index,
                        first_op=first_op,
                        second_op=second_op,
                        values=public_sequence,
                    )
                    example_id = _sequence_chain_example_id(
                        construction_id,
                        first_op,
                        second_op,
                        public_sequence,
                        sample_index,
                    )
                    examples.append(
                        SemanticProgramExample(
                            example_id=example_id,
                            construction_id=construction_id,
                            topology_id="unary-sequence-chain",
                            split=split,
                            source_text=source_text,
                            inputs=(public_sequence,),
                            input_spans=(input_span,),
                            instructions=instructions,
                            report_value=2,
                            contrast_id=hashlib.sha256(
                                f"sequence|{first_op}|{second_op}|{public_sequence}".encode()
                            ).hexdigest()[:24],
                        )
                    )
    return tuple(examples)


def _sequence_binary_example_id(
    construction_id: str,
    first_op: str,
    second_op: str,
    inputs: tuple[tuple[int, ...], int, int],
    sample_index: int,
) -> str:
    body = f"{construction_id}|{first_op}|{second_op}|{inputs}|{sample_index}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _render_sequence_binary_chain(
    *,
    construction_index: int,
    first_op: str,
    second_op: str,
    values: tuple[int, ...],
    selector: int,
    adjustment: int,
) -> tuple[
    str,
    tuple[CharacterSpan, CharacterSpan, CharacterSpan],
    tuple[SemanticInstructionAnnotation, SemanticInstructionAnnotation],
]:
    builder = _AnnotatedText()
    first_phrase = _SEQUENCE_BINARY_OPERATION_LANGUAGE[first_op][construction_index]
    second_phrase = _SCALAR_CONTINUATION_LANGUAGE[second_op][construction_index]
    register_name = (
        "selected quantity",
        "lookup result",
        "derived count",
        "intermediate number",
        "retrieved value",
        "computed scalar",
        "selection result",
        "working number",
        "obtained quantity",
    )[construction_index]
    input_text = "[" + ", ".join(str(value) for value in values) + "]"

    def append_sequence() -> None:
        builder.append(input_text, label="input:0")

    def append_selector() -> None:
        builder.append(str(selector), label="input:1")

    def append_adjustment() -> None:
        builder.append(str(adjustment), label="input:2")

    def append_first_operation() -> None:
        builder.append(first_phrase, label="operation:0")

    def define_result() -> None:
        builder.append(register_name, label="result:0")

    def reference_result() -> None:
        builder.append(register_name, label="argument:1:0")

    def append_second_expression() -> None:
        reference_result()
        builder.append(" ")
        builder.append(second_phrase, label="operation:1")
        builder.append(" ")
        append_adjustment()

    if construction_index == 0:
        builder.append("For ")
        append_sequence()
        builder.append(", perform ")
        append_first_operation()
        builder.append(" using selector ")
        append_selector()
        builder.append(", and call the output ")
        define_result()
        builder.append(". Compute ")
        append_second_expression()
        builder.append(".")
    elif construction_index == 1:
        builder.append("Starting with ")
        append_sequence()
        builder.append(", use ")
        append_first_operation()
        builder.append(" for selector ")
        append_selector()
        builder.append("; name its output ")
        define_result()
        builder.append(". Then evaluate ")
        append_second_expression()
        builder.append(".")
    elif construction_index == 2:
        builder.append("Take ")
        append_sequence()
        builder.append(" through ")
        append_first_operation()
        builder.append(" with selector ")
        append_selector()
        builder.append(". Let ")
        define_result()
        builder.append(" denote the scalar produced; return ")
        append_second_expression()
        builder.append(".")
    elif construction_index == 3:
        builder.append("Given ")
        append_sequence()
        builder.append(", apply ")
        append_first_operation()
        builder.append(" at selector ")
        append_selector()
        builder.append(" and bind the answer as ")
        define_result()
        builder.append(". Finish with ")
        append_second_expression()
        builder.append(".")
    elif construction_index == 4:
        builder.append("On the values ")
        append_sequence()
        builder.append(", carry out ")
        append_first_operation()
        builder.append(" for selector ")
        append_selector()
        builder.append("; refer to that output as ")
        define_result()
        builder.append(". The final number is ")
        append_second_expression()
        builder.append(".")
    elif construction_index == 5:
        builder.append("Begin from ")
        append_sequence()
        builder.append(" and execute ")
        append_first_operation()
        builder.append(" with selector ")
        append_selector()
        builder.append(". Label the outcome ")
        define_result()
        builder.append(", then calculate ")
        append_second_expression()
        builder.append(".")
    elif construction_index == 6:
        builder.append("Use ")
        append_first_operation()
        builder.append(" on ")
        append_sequence()
        builder.append(" with selector ")
        append_selector()
        builder.append(" to obtain ")
        define_result()
        builder.append(". Report ")
        append_second_expression()
        builder.append(".")
    elif construction_index == 7:
        builder.append("From ")
        append_sequence()
        builder.append(", run ")
        append_first_operation()
        builder.append(" for selector ")
        append_selector()
        builder.append(" and retain the result as ")
        define_result()
        builder.append(". Next compute ")
        append_second_expression()
        builder.append(".")
    elif construction_index == 8:
        builder.append("Process ")
        append_sequence()
        builder.append(" by ")
        append_first_operation()
        builder.append(" using selector ")
        append_selector()
        builder.append(". Call what it returns ")
        define_result()
        builder.append("; the requested result is ")
        append_second_expression()
        builder.append(".")
    else:  # pragma: no cover - private caller pins the construction range
        raise ValueError("sequence binary construction index is invalid")

    input_spans = tuple(builder.span(f"input:{index}") for index in range(3))
    instructions = (
        SemanticInstructionAnnotation(
            instruction=Instruction(first_op, (0, 1)),
            operation_span=builder.span("operation:0"),
            argument_spans=(input_spans[0], input_spans[1]),
            depends_on=(),
        ),
        SemanticInstructionAnnotation(
            instruction=Instruction(second_op, (3, 2)),
            operation_span=builder.span("operation:1"),
            argument_spans=(builder.span("argument:1:0"), input_spans[2]),
            depends_on=(0,),
        ),
    )
    return builder.text, input_spans, instructions


def build_semantic_program_sequence_binary_corpus(
    *,
    seed: int = 2236067,
    examples_per_operation_pair: int = 2,
) -> tuple[SemanticProgramExample, ...]:
    """Build mixed sequence/scalar programs with binary lookup semantics."""

    if examples_per_operation_pair < 1:
        raise ValueError("sequence binary corpus needs at least one sample per operation pair")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    for construction_index in range(9):
        construction_id = f"sequence-binary-construction-{construction_index}"
        split: CorpusSplit = (
            "train"
            if construction_index < 3
            else "validation"
            if construction_index < 6
            else "test"
        )
        for sample_index in range(examples_per_operation_pair):
            selector = rng.randint(1, 4)
            values = [rng.randint(1, 20) for _ in range(rng.randint(6, 8))]
            values[rng.randrange(len(values))] = selector
            public_sequence = tuple(values)
            adjustment = rng.randint(2, 7)
            inputs = (public_sequence, selector, adjustment)
            contrast_id = hashlib.sha256(
                f"sequence-binary|{construction_id}|{inputs}|{sample_index}".encode()
            ).hexdigest()[:24]
            for first_op in _SEQUENCE_BINARY_SELECTORS:
                for second_op in _SCALAR_CONTINUATIONS:
                    source_text, input_spans, instructions = _render_sequence_binary_chain(
                        construction_index=construction_index,
                        first_op=first_op,
                        second_op=second_op,
                        values=public_sequence,
                        selector=selector,
                        adjustment=adjustment,
                    )
                    examples.append(
                        SemanticProgramExample(
                            example_id=_sequence_binary_example_id(
                                construction_id,
                                first_op,
                                second_op,
                                inputs,
                                sample_index,
                            ),
                            construction_id=construction_id,
                            topology_id="binary-sequence-to-scalar-chain",
                            split=split,
                            source_text=source_text,
                            inputs=inputs,
                            input_spans=input_spans,
                            instructions=instructions,
                            report_value=4,
                            contrast_id=contrast_id,
                        )
                    )
    return tuple(examples)


def _sequence_cataphoric_example_id(
    construction_id: str,
    first_op: str,
    second_op: str,
    inputs: tuple[tuple[int, ...], int, int],
    sample_index: int,
) -> str:
    body = f"sequence-cataphoric|{construction_id}|{first_op}|{second_op}|{inputs}|{sample_index}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _render_sequence_cataphoric_chain(
    *,
    construction_index: int,
    first_op: str,
    second_op: str,
    values: tuple[int, ...],
    selector: int,
    adjustment: int,
) -> tuple[
    str,
    tuple[CharacterSpan, CharacterSpan, CharacterSpan],
    tuple[SemanticInstructionAnnotation, SemanticInstructionAnnotation],
]:
    """Render a dependency before its producing operation is mentioned."""

    builder = _AnnotatedText()
    first_phrase = _SEQUENCE_BINARY_OPERATION_LANGUAGE[first_op][construction_index]
    second_phrase = _SCALAR_CONTINUATION_LANGUAGE[second_op][construction_index]
    result_names = (
        "the pending entry",
        "the result to be found",
        "the forthcoming tally",
        "the value obtained later",
        "the yet-unknown scalar",
        "the subsequently computed number",
        "the later selection result",
        "the value established next",
        "the eventual lookup output",
    )
    result_name = result_names[construction_index]
    input_text = "[" + ", ".join(str(value) for value in values) + "]"

    def append_sequence() -> None:
        builder.append(input_text, label="input:0")

    def append_selector() -> None:
        builder.append(str(selector), label="input:1")

    def append_adjustment() -> None:
        builder.append(str(adjustment), label="input:2")

    def append_first_operation() -> None:
        builder.append(first_phrase, label="operation:0")

    def append_second_operation() -> None:
        builder.append(second_phrase, label="operation:1")

    def reference_result() -> None:
        builder.append(result_name, label="argument:result")

    result_first = construction_index % 2 == 0

    def append_dependent_expression() -> None:
        if result_first:
            reference_result()
            builder.append(" ")
            append_second_operation()
            builder.append(" ")
            append_adjustment()
        else:
            append_adjustment()
            builder.append(" ")
            append_second_operation()
            builder.append(" ")
            reference_result()

    if construction_index == 0:
        builder.append("Before reporting ")
        append_dependent_expression()
        builder.append(", first produce ")
        builder.append(result_name)
        builder.append(" by performing ")
        append_first_operation()
        builder.append(" on ")
        append_sequence()
        builder.append(" with selector ")
        append_selector()
        builder.append(".")
    elif construction_index == 1:
        builder.append("To eventually evaluate ")
        append_dependent_expression()
        builder.append(", obtain ")
        builder.append(result_name)
        builder.append(" afterward: use ")
        append_first_operation()
        builder.append(" for ")
        append_sequence()
        builder.append(" at selector ")
        append_selector()
        builder.append(".")
    elif construction_index == 2:
        builder.append("The requested scalar is ")
        append_dependent_expression()
        builder.append("; derive ")
        builder.append(result_name)
        builder.append(" first through ")
        append_first_operation()
        builder.append(" over ")
        append_sequence()
        builder.append(" using selector ")
        append_selector()
        builder.append(".")
    elif construction_index == 3:
        builder.append("Prior to calculating ")
        append_dependent_expression()
        builder.append(", establish ")
        builder.append(result_name)
        builder.append(": apply ")
        append_first_operation()
        builder.append(" to ")
        append_sequence()
        builder.append(" with selector ")
        append_selector()
        builder.append(".")
    elif construction_index == 4:
        builder.append("Although the final form is ")
        append_dependent_expression()
        builder.append(", begin by making ")
        builder.append(result_name)
        builder.append(" via ")
        append_first_operation()
        builder.append(" on ")
        append_sequence()
        builder.append(" for selector ")
        append_selector()
        builder.append(".")
    elif construction_index == 5:
        builder.append("Resolve ")
        append_dependent_expression()
        builder.append(" only after computing ")
        builder.append(result_name)
        builder.append(" with ")
        append_first_operation()
        builder.append(" from ")
        append_sequence()
        builder.append(" at selector ")
        append_selector()
        builder.append(".")
    elif construction_index == 6:
        builder.append("Before you return ")
        append_dependent_expression()
        builder.append(", determine ")
        builder.append(result_name)
        builder.append(" by running ")
        append_first_operation()
        builder.append(" on ")
        append_sequence()
        builder.append(" with selector ")
        append_selector()
        builder.append(".")
    elif construction_index == 7:
        builder.append("The eventual answer takes ")
        append_dependent_expression()
        builder.append("; before that, set ")
        builder.append(result_name)
        builder.append(" using ")
        append_first_operation()
        builder.append(" over ")
        append_sequence()
        builder.append(" for selector ")
        append_selector()
        builder.append(".")
    elif construction_index == 8:
        builder.append("Plan to finish with ")
        append_dependent_expression()
        builder.append(", but first form ")
        builder.append(result_name)
        builder.append(" through ")
        append_first_operation()
        builder.append(" applied to ")
        append_sequence()
        builder.append(" at selector ")
        append_selector()
        builder.append(".")
    else:  # pragma: no cover - private caller pins the construction range
        raise ValueError("sequence cataphoric construction index is invalid")

    input_spans = tuple(builder.span(f"input:{index}") for index in range(3))
    second_args = (3, 2) if result_first else (2, 3)
    second_spans = (
        (builder.span("argument:result"), input_spans[2])
        if result_first
        else (input_spans[2], builder.span("argument:result"))
    )
    instructions = (
        SemanticInstructionAnnotation(
            instruction=Instruction(first_op, (0, 1)),
            operation_span=builder.span("operation:0"),
            argument_spans=(input_spans[0], input_spans[1]),
            depends_on=(),
        ),
        SemanticInstructionAnnotation(
            instruction=Instruction(second_op, second_args),
            operation_span=builder.span("operation:1"),
            argument_spans=second_spans,
            depends_on=(0,),
        ),
    )
    return builder.text, input_spans, instructions


def build_semantic_program_sequence_cataphoric_corpus(
    *,
    seed: int = 2653589,
    examples_per_operation_pair: int = 2,
) -> tuple[SemanticProgramExample, ...]:
    """Build mixed-type programs whose textual and causal orders differ."""

    if examples_per_operation_pair < 1:
        raise ValueError("sequence cataphoric corpus needs at least one sample")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    for construction_index in range(9):
        construction_id = f"sequence-cataphoric-{construction_index}"
        split: CorpusSplit = (
            "train"
            if construction_index < 3
            else "validation"
            if construction_index < 6
            else "test"
        )
        for sample_index in range(examples_per_operation_pair):
            selector = rng.randint(1, 4)
            values = [rng.randint(1, 20) for _ in range(rng.randint(6, 8))]
            values[rng.randrange(len(values))] = selector
            public_sequence = tuple(values)
            adjustment = rng.randint(2, 7)
            inputs = (public_sequence, selector, adjustment)
            contrast_id = hashlib.sha256(
                f"sequence-cataphoric|{construction_id}|{inputs}|{sample_index}".encode()
            ).hexdigest()[:24]
            for first_op in _SEQUENCE_BINARY_SELECTORS:
                for second_op in _SCALAR_CONTINUATIONS:
                    source_text, input_spans, instructions = _render_sequence_cataphoric_chain(
                        construction_index=construction_index,
                        first_op=first_op,
                        second_op=second_op,
                        values=public_sequence,
                        selector=selector,
                        adjustment=adjustment,
                    )
                    examples.append(
                        SemanticProgramExample(
                            example_id=_sequence_cataphoric_example_id(
                                construction_id,
                                first_op,
                                second_op,
                                inputs,
                                sample_index,
                            ),
                            construction_id=construction_id,
                            topology_id="cataphoric-sequence-to-scalar-chain",
                            split=split,
                            source_text=source_text,
                            inputs=inputs,
                            input_spans=input_spans,
                            instructions=instructions,
                            report_value=4,
                            contrast_id=contrast_id,
                        )
                    )
    return tuple(examples)


def _sequence_reserved_alias_example_id(
    construction_id: str,
    first_op: str,
    second_op: str,
    inputs: tuple[tuple[int, ...], int, int],
    sample_index: int,
) -> str:
    body = f"reserved-alias|{construction_id}|{first_op}|{second_op}|{inputs}|{sample_index}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _render_sequence_reserved_alias_chain(
    *,
    construction_index: int,
    first_op: str,
    second_op: str,
    values: tuple[int, ...],
    selector: int,
    adjustment: int,
    role_bound: bool = False,
) -> tuple[
    str,
    tuple[CharacterSpan, CharacterSpan, CharacterSpan],
    tuple[SemanticInstructionAnnotation, SemanticInstructionAnnotation],
    tuple[CharacterSpan, ...],
]:
    """Render an input alias independently of arithmetic-family language."""

    builder = _AnnotatedText()
    first_phrase = _SEQUENCE_BINARY_OPERATION_LANGUAGE[first_op][construction_index]
    second_phrase = _SCALAR_CONTINUATION_LANGUAGE[second_op][construction_index]
    if role_bound:
        reserved_definitions = (
            "the baseline quantity",
            "the retained scalar",
            "the auxiliary adjustment",
            "the spare value",
            "the initial amount",
            "the carried quantity",
            "the fixed scalar",
            "the side amount",
            "the supporting value",
        )
        reserved_references = (
            "the prior baseline",
            "that kept scalar",
            "the same adjustment",
            "the spare quantity",
            "that initial amount",
            "the carried quantity",
            "the fixed scalar",
            "that side amount",
            "the supporting value",
        )
    else:
        reserved_definitions = (
            "the label offset",
            "the name side value",
            "the title fixed amount",
            "the label held quantity",
            "the name saved scalar",
            "the title adjustment",
            "the label constant term",
            "the name carried value",
            "the title extra quantity",
        )
        reserved_references = (
            "the labeled offset",
            "that side value",
            "the fixed amount",
            "the held quantity",
            "that saved scalar",
            "the named adjustment",
            "the constant term",
            "the carried value",
            "that extra quantity",
        )
    result_names = (
        "chosen entry",
        "lookup output",
        "derived tally",
        "selected number",
        "retrieved scalar",
        "computed entry",
        "selection output",
        "working scalar",
        "obtained number",
    )
    input_text = "[" + ", ".join(str(value) for value in values) + "]"

    def append_sequence() -> None:
        builder.append(input_text, label="input:0")

    def append_selector() -> None:
        builder.append(str(selector), label="input:1")

    def append_reserved_input() -> None:
        builder.append(str(adjustment), label="input:2")

    def append_first_operation() -> None:
        builder.append(first_phrase, label="operation:0")

    def append_second_operation() -> None:
        builder.append(second_phrase, label="operation:1")

    def define_result() -> None:
        builder.append(result_names[construction_index], label="result:0")

    def reference_result() -> None:
        builder.append(result_names[construction_index], label="argument:result")

    def reference_reserved() -> None:
        builder.append(
            reserved_references[construction_index],
            label="argument:reserved",
        )

    def append_reserved_definition() -> None:
        builder.begin("definition:reserved")
        append_reserved_input()
        builder.append(" as " if role_bound else " under ")
        builder.append(reserved_definitions[construction_index])
        builder.finish("definition:reserved")

    if construction_index == 0:
        builder.append("Put ")
        append_reserved_definition()
        builder.append(" aside. For ")
        append_sequence()
        builder.append(", perform ")
        append_first_operation()
        builder.append(" using selector ")
        append_selector()
        builder.append(", calling the output ")
        define_result()
        builder.append(". Compute ")
    elif construction_index == 1:
        builder.append("Keep ")
        append_reserved_definition()
        builder.append(". Starting with ")
        append_sequence()
        builder.append(", use ")
        append_first_operation()
        builder.append(" for selector ")
        append_selector()
        builder.append(" and name its output ")
        define_result()
        builder.append(". Evaluate ")
    elif construction_index == 2:
        builder.append("Store ")
        append_reserved_definition()
        builder.append(". Take ")
        append_sequence()
        builder.append(" through ")
        append_first_operation()
        builder.append(" with selector ")
        append_selector()
        builder.append("; call the scalar ")
        define_result()
        builder.append(". Return ")
    elif construction_index == 3:
        builder.append("Designate ")
        append_reserved_definition()
        builder.append(". Given ")
        append_sequence()
        builder.append(", apply ")
        append_first_operation()
        builder.append(" at selector ")
        append_selector()
        builder.append(" and bind the answer as ")
        define_result()
        builder.append(". Finish with ")
    elif construction_index == 4:
        builder.append("Let ")
        append_reserved_definition()
        builder.append(" be kept. On ")
        append_sequence()
        builder.append(", carry out ")
        append_first_operation()
        builder.append(" for selector ")
        append_selector()
        builder.append("; refer to the output as ")
        define_result()
        builder.append(". The final number is ")
    elif construction_index == 5:
        builder.append("Retain ")
        append_reserved_definition()
        builder.append(". Begin from ")
        append_sequence()
        builder.append(" and execute ")
        append_first_operation()
        builder.append(" with selector ")
        append_selector()
        builder.append(". Label the outcome ")
        define_result()
        builder.append(", then calculate ")
    elif construction_index == 6:
        builder.append("Set ")
        append_reserved_definition()
        builder.append(" apart. Use ")
        append_first_operation()
        builder.append(" on ")
        append_sequence()
        builder.append(" with selector ")
        append_selector()
        builder.append(" to obtain ")
        define_result()
        builder.append(". Report ")
    elif construction_index == 7:
        builder.append("Mark ")
        append_reserved_definition()
        builder.append(". From ")
        append_sequence()
        builder.append(", run ")
        append_first_operation()
        builder.append(" for selector ")
        append_selector()
        builder.append(" and retain the result as ")
        define_result()
        builder.append(". Next compute ")
    elif construction_index == 8:
        builder.append("Hold ")
        append_reserved_definition()
        builder.append(". Process ")
        append_sequence()
        builder.append(" by ")
        append_first_operation()
        builder.append(" using selector ")
        append_selector()
        builder.append(". Call what it returns ")
        define_result()
        builder.append("; the requested result is ")
    else:  # pragma: no cover - private caller pins the construction range
        raise ValueError("sequence reserved-alias construction index is invalid")

    reserved_first = bool(construction_index % 2)
    if reserved_first:
        reference_reserved()
        builder.append(" ")
        append_second_operation()
        builder.append(" ")
        reference_result()
        second_args = (2, 3)
        second_spans = (
            builder.span("argument:reserved"),
            builder.span("argument:result"),
        )
    else:
        reference_result()
        builder.append(" ")
        append_second_operation()
        builder.append(" ")
        reference_reserved()
        second_args = (3, 2)
        second_spans = (
            builder.span("argument:result"),
            builder.span("argument:reserved"),
        )
    builder.append(".")

    input_spans = tuple(builder.span(f"input:{index}") for index in range(3))
    instructions = (
        SemanticInstructionAnnotation(
            instruction=Instruction(first_op, (0, 1)),
            operation_span=builder.span("operation:0"),
            argument_spans=(input_spans[0], input_spans[1]),
            depends_on=(),
        ),
        SemanticInstructionAnnotation(
            instruction=Instruction(second_op, second_args),
            operation_span=builder.span("operation:1"),
            argument_spans=second_spans,
            depends_on=(0,),
        ),
    )
    register_definition_spans = (
        input_spans[0],
        input_spans[1],
        builder.span("definition:reserved"),
        builder.span("result:0"),
        builder.span("operation:1"),
    )
    return builder.text, input_spans, instructions, register_definition_spans


def build_semantic_program_sequence_reserved_alias_corpus(
    *,
    seed: int = 2449489,
    examples_per_operation_pair: int = 2,
) -> tuple[SemanticProgramExample, ...]:
    """Build sequence programs whose fronted scalar is referenced by alias."""

    if examples_per_operation_pair < 1:
        raise ValueError("sequence reserved-alias corpus needs at least one sample")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    for construction_index in range(9):
        construction_id = f"sequence-reserved-alias-{construction_index}"
        split: CorpusSplit = (
            "train"
            if construction_index < 3
            else "validation"
            if construction_index < 6
            else "test"
        )
        for sample_index in range(examples_per_operation_pair):
            selector = rng.randint(1, 4)
            values = [rng.randint(1, 20) for _ in range(rng.randint(6, 8))]
            values[rng.randrange(len(values))] = selector
            public_sequence = tuple(values)
            adjustment = rng.randint(2, 7)
            inputs = (public_sequence, selector, adjustment)
            contrast_id = hashlib.sha256(
                f"sequence-reserved-alias|{construction_id}|{inputs}|{sample_index}".encode()
            ).hexdigest()[:24]
            for first_op in _SEQUENCE_BINARY_SELECTORS:
                for second_op in _SCALAR_CONTINUATIONS:
                    source_text, input_spans, instructions, _definition_spans = (
                        _render_sequence_reserved_alias_chain(
                            construction_index=construction_index,
                            first_op=first_op,
                            second_op=second_op,
                            values=public_sequence,
                            selector=selector,
                            adjustment=adjustment,
                        )
                    )
                    examples.append(
                        SemanticProgramExample(
                            example_id=_sequence_reserved_alias_example_id(
                                construction_id,
                                first_op,
                                second_op,
                                inputs,
                                sample_index,
                            ),
                            construction_id=construction_id,
                            topology_id="aliased-input-sequence-to-scalar-chain",
                            split=split,
                            source_text=source_text,
                            inputs=inputs,
                            input_spans=input_spans,
                            instructions=instructions,
                            report_value=4,
                            contrast_id=contrast_id,
                        )
                    )
    return tuple(examples)


def build_semantic_program_sequence_role_binding_corpus(
    *,
    seed: int = 2828427,
    examples_per_operation_pair: int = 2,
) -> tuple[SemanticProgramExample, ...]:
    """Build non-arithmetic programs with implicit functional-role binding."""

    if examples_per_operation_pair < 1:
        raise ValueError("sequence role-binding corpus needs at least one sample")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    for construction_index in range(9):
        construction_id = f"sequence-role-binding-{construction_index}"
        split: CorpusSplit = (
            "train"
            if construction_index < 3
            else "validation"
            if construction_index < 6
            else "test"
        )
        for sample_index in range(examples_per_operation_pair):
            selector = rng.randint(1, 4)
            values = [rng.randint(1, 20) for _ in range(rng.randint(6, 8))]
            values[rng.randrange(len(values))] = selector
            public_sequence = tuple(values)
            adjustment = rng.randint(2, 7)
            inputs = (public_sequence, selector, adjustment)
            contrast_id = hashlib.sha256(
                f"sequence-role-binding|{construction_id}|{inputs}|{sample_index}".encode()
            ).hexdigest()[:24]
            for first_op in _SEQUENCE_BINARY_SELECTORS:
                for second_op in _SCALAR_CONTINUATIONS:
                    (
                        source_text,
                        input_spans,
                        instructions,
                        register_definition_spans,
                    ) = _render_sequence_reserved_alias_chain(
                        construction_index=construction_index,
                        first_op=first_op,
                        second_op=second_op,
                        values=public_sequence,
                        selector=selector,
                        adjustment=adjustment,
                        role_bound=True,
                    )
                    examples.append(
                        SemanticProgramExample(
                            example_id=_sequence_reserved_alias_example_id(
                                construction_id,
                                first_op,
                                second_op,
                                inputs,
                                sample_index,
                            ),
                            construction_id=construction_id,
                            topology_id="role-bound-input-sequence-to-scalar-chain",
                            split=split,
                            source_text=source_text,
                            inputs=inputs,
                            input_spans=input_spans,
                            instructions=instructions,
                            report_value=4,
                            contrast_id=contrast_id,
                            register_definition_spans=register_definition_spans,
                        )
                    )
    return tuple(examples)


_NATURAL_SCALAR_DOMAINS: Final = (
    ("warehouse", "arrivals", "returns", "scale", "reserve"),
    ("project dashboard", "opened issues", "closed issues", "weight", "holdback"),
    ("event desk", "advance tickets", "door tickets", "room factor", "staff reserve"),
    ("energy monitor", "solar units", "grid units", "conversion factor", "safety reserve"),
    ("kitchen", "first batch", "second batch", "portion factor", "sample reserve"),
    ("research ledger", "confirmed cases", "new cases", "study weight", "audit reserve"),
    ("transit board", "northbound riders", "southbound riders", "route factor", "service reserve"),
    ("studio schedule", "morning minutes", "afternoon minutes", "billing factor", "setup reserve"),
)

_NATURAL_SEQUENCE_DOMAINS: Final = (
    ("warehouse", "shelf counts", "zero-based shelf index", "target count", "scale", "reserve"),
    (
        "project dashboard",
        "issue counts by sprint",
        "zero-based sprint index",
        "target issue count",
        "weight",
        "holdback",
    ),
    (
        "event desk",
        "attendance by session",
        "zero-based session index",
        "target attendance",
        "room factor",
        "staff reserve",
    ),
    (
        "energy monitor",
        "hourly readings",
        "zero-based reading index",
        "target reading",
        "conversion factor",
        "safety reserve",
    ),
    (
        "kitchen",
        "batch sizes",
        "zero-based batch index",
        "target batch size",
        "portion factor",
        "sample reserve",
    ),
    (
        "research ledger",
        "case counts by cohort",
        "zero-based cohort index",
        "target case count",
        "study weight",
        "audit reserve",
    ),
    (
        "transit board",
        "riders by route",
        "zero-based route index",
        "target rider count",
        "route factor",
        "service reserve",
    ),
    (
        "studio schedule",
        "minutes by booking",
        "zero-based booking index",
        "target duration",
        "billing factor",
        "setup reserve",
    ),
)

_NATURAL_SOURCE_SCALAR_DOMAINS: Final = (
    ("shipping desk", "received parcels", "dispatched parcels", "batch factor"),
    ("support queue", "opened tickets", "resolved tickets", "priority factor"),
    ("clinic board", "morning visits", "afternoon visits", "staffing factor"),
    ("factory line", "first shift units", "second shift units", "yield factor"),
    ("library desk", "checked-out books", "returned books", "shelving factor"),
    ("farm ledger", "north field crates", "south field crates", "packing factor"),
    ("network console", "accepted packets", "dropped packets", "routing factor"),
    ("theater office", "matinee seats", "evening seats", "pricing factor"),
)

_NATURAL_SOURCE_SEQUENCE_DOMAINS: Final = (
    (
        "shipping desk",
        "parcel counts by bay",
        "zero-based bay index",
        "target parcel count",
        "batch factor",
    ),
    (
        "support queue",
        "ticket counts by hour",
        "zero-based hour index",
        "target ticket count",
        "priority factor",
    ),
    (
        "clinic board",
        "visits by room",
        "zero-based room index",
        "target visit count",
        "staffing factor",
    ),
    (
        "factory line",
        "units by station",
        "zero-based station index",
        "target unit count",
        "yield factor",
    ),
    (
        "library desk",
        "books by cart",
        "zero-based cart index",
        "target book count",
        "shelving factor",
    ),
    (
        "farm ledger",
        "crates by row",
        "zero-based row index",
        "target crate count",
        "packing factor",
    ),
    (
        "network console",
        "packets by route",
        "zero-based route index",
        "target packet count",
        "routing factor",
    ),
    (
        "theater office",
        "seats by section",
        "zero-based section index",
        "target seat count",
        "pricing factor",
    ),
)

_NATURAL_ALIAS_SOURCE_SCALAR_DOMAINS: Final = (
    ("weather station", "primary readings", "backup readings", "calibration factor"),
    ("bakery ledger", "morning loaves", "evening loaves", "tray factor"),
    ("rail depot", "inbound cars", "outbound cars", "consist factor"),
    ("school office", "enrolled students", "withdrawn students", "class factor"),
    ("data center", "active jobs", "queued jobs", "shard factor"),
    ("pharmacy shelf", "stocked packs", "dispensed packs", "case factor"),
    ("garden log", "north seedlings", "south seedlings", "bed factor"),
    ("radio schedule", "live minutes", "recorded minutes", "slot factor"),
)

_NATURAL_ALIAS_SOURCE_SEQUENCE_DOMAINS: Final = (
    (
        "weather station",
        "readings by sensor",
        "sensor position",
        "target reading",
        "calibration factor",
    ),
    ("bakery ledger", "loaves by rack", "rack position", "target loaf count", "tray factor"),
    ("rail depot", "cars by track", "track position", "target car count", "consist factor"),
    (
        "school office",
        "students by class",
        "class position",
        "target student count",
        "class factor",
    ),
    ("data center", "jobs by shard", "shard position", "target job count", "shard factor"),
    ("pharmacy shelf", "packs by cabinet", "cabinet position", "target pack count", "case factor"),
    ("garden log", "seedlings by bed", "bed position", "target seedling count", "bed factor"),
    ("radio schedule", "minutes by slot", "slot position", "target minute count", "slot factor"),
)

_NATURAL_ALIAS_SOURCE_NAMES: Final = (
    ("accumulated quantity", "completed quantity"),
    ("carry amount", "settled amount"),
    ("bridge value", "resolved value"),
    ("stage output", "closing output"),
    ("working quantity", "finalized quantity"),
    ("derived quantity", "concluded quantity"),
    ("transient amount", "terminal amount"),
    ("retained result", "finished result"),
)

_NATURAL_IDENTITY_SOURCE_SCALAR_DOMAINS: Final = (
    ("power dispatch", "available megawatts", "committed megawatts", "reserve factor"),
    ("kitchen inventory", "prepared portions", "served portions", "batch factor"),
    ("freight terminal", "arriving pallets", "departing pallets", "load factor"),
    ("registrar table", "active records", "closed records", "filing factor"),
    ("render farm", "finished frames", "pending frames", "node factor"),
    ("medical supply room", "received kits", "issued kits", "carton factor"),
    ("greenhouse table", "sprouted trays", "planted trays", "bench factor"),
    ("broadcast console", "scheduled segments", "aired segments", "channel factor"),
)

_NATURAL_IDENTITY_SOURCE_SEQUENCE_DOMAINS: Final = (
    ("power dispatch", "megawatts by feeder", "feeder index", "target load", "reserve factor"),
    ("kitchen inventory", "portions by station", "station index", "target portion", "batch factor"),
    ("freight terminal", "pallets by dock", "dock index", "target pallet count", "load factor"),
    ("registrar table", "records by drawer", "drawer index", "target record count", "filing factor"),
    ("render farm", "frames by node", "node index", "target frame count", "node factor"),
    ("medical supply room", "kits by cabinet", "cabinet index", "target kit count", "carton factor"),
    ("greenhouse table", "trays by bench", "bench index", "target tray count", "bench factor"),
    ("broadcast console", "segments by channel", "channel index", "target segment count", "channel factor"),
)

_NATURAL_IDENTITY_SOURCE_NAMES: Final = (
    ("running balance", "reported balance"),
    ("intermediate tally", "published tally"),
    ("carried measure", "delivered measure"),
    ("provisional count", "recorded count"),
    ("working total", "rendered total"),
    ("derived stock", "final stock"),
    ("current yield", "returned yield"),
    ("combined duration", "broadcast duration"),
)

_NATURAL_REPLICATION_SCALAR_DOMAINS: Final = (
    ("observatory log", "first exposure count", "second exposure count", "gain", "offset"),
    ("museum archive", "catalogued objects", "loaned objects", "batch size", "reserve"),
    ("harbor control", "inbound containers", "outbound containers", "crane factor", "buffer"),
    ("laboratory inventory", "stored samples", "used samples", "assay factor", "control"),
    ("hotel desk", "weekday bookings", "weekend bookings", "rate factor", "allowance"),
    ("water utility", "north meter units", "south meter units", "conversion", "baseline"),
    ("publishing queue", "accepted pages", "revised pages", "print factor", "holdback"),
    ("sports venue", "lower seats", "upper seats", "section factor", "staff block"),
)

_NATURAL_REPLICATION_SEQUENCE_DOMAINS: Final = (
    (
        "observatory log",
        "exposures by detector",
        "detector position",
        "target exposure",
        "gain",
        "offset",
    ),
    (
        "museum archive",
        "objects by gallery",
        "gallery position",
        "target object count",
        "batch size",
        "reserve",
    ),
    (
        "harbor control",
        "containers by berth",
        "berth position",
        "target container count",
        "crane factor",
        "buffer",
    ),
    (
        "laboratory inventory",
        "samples by freezer",
        "freezer position",
        "target sample count",
        "assay factor",
        "control",
    ),
    (
        "hotel desk",
        "bookings by floor",
        "floor position",
        "target booking count",
        "rate factor",
        "allowance",
    ),
    (
        "water utility",
        "units by district",
        "district position",
        "target meter value",
        "conversion",
        "baseline",
    ),
    (
        "publishing queue",
        "pages by edition",
        "edition position",
        "target page count",
        "print factor",
        "holdback",
    ),
    (
        "sports venue",
        "seats by gate",
        "gate position",
        "target seat count",
        "section factor",
        "staff block",
    ),
)

_NATURAL_REPLICATION_SURFACES: Final = (
    (
        "Use the {domain} record",
        "Begin with",
        "Call its output the first result",
        "Next",
        "Call that output the second result",
        "Finish with",
        "Return the final integer",
    ),
    (
        "Read the {domain} report",
        "First compute",
        "Label the result the subtotal",
        "Afterward",
        "Name that output the adjusted total",
        "Last compute",
        "What integer results",
    ),
    (
        "Take the values from the {domain}",
        "Stage one is to",
        "The resulting amount is the interim value",
        "Stage two is to",
        "The new amount is the revised value",
        "Stage three is to",
        "Give the final integer",
    ),
    (
        "Work from the {domain} entry",
        "Initially",
        "Refer to its output as the provisional value",
        "Then",
        "Refer to that output as the combined value",
        "Finally",
        "Report the resulting integer",
    ),
)

_NATURAL_SCALAR_CHAINS: Final = (
    ("add", "mul", "sub"),
    ("sub", "add", "mul"),
    ("mul", "idiv", "add"),
    ("add", "sub", "idiv"),
    ("sub", "mul", "add"),
    ("mul", "add", "sub"),
    ("add", "idiv", "mul"),
    ("idiv", "sub", "mul"),
)


def _append_natural_binary_operation(
    builder: _AnnotatedText,
    *,
    op: str,
    ordinal: int,
    left_text: str,
    left_label: str,
    right_text: str,
    right_label: str,
) -> None:
    """Render one ordinary-language binary clause with exact semantic roles."""

    operation_label = f"natural:operation:{ordinal}"
    if op == "add":
        builder.append("add", label=operation_label)
        builder.append(" ")
        builder.append(left_text, label=left_label)
        builder.append(" and ")
        builder.append(right_text, label=right_label)
    elif op == "sub":
        builder.append("subtract", label=operation_label)
        builder.append(" ")
        builder.append(right_text, label=right_label)
        builder.append(" from ")
        builder.append(left_text, label=left_label)
    elif op == "mul":
        builder.append("multiply", label=operation_label)
        builder.append(" ")
        builder.append(left_text, label=left_label)
        builder.append(" by ")
        builder.append(right_text, label=right_label)
    elif op == "idiv":
        builder.append("whole-number divide", label=operation_label)
        builder.append(" ")
        builder.append(left_text, label=left_label)
        builder.append(" by ")
        builder.append(right_text, label=right_label)
    else:  # pragma: no cover - callers use the declared scalar vocabulary
        raise ValueError("natural procedure scalar operation is unsupported")


def _natural_three_step_example(
    *,
    schema_kind: str,
    domain_index: int,
    sample_index: int,
    inputs: tuple[SemanticValue, ...],
    operations: tuple[str, str, str],
) -> SemanticProgramExample:
    """Render a domain request whose three-step chain was absent from fitting."""

    if schema_kind == "scalar_linear_three":
        domain, first_name, second_name, third_name, fourth_name = _NATURAL_SCALAR_DOMAINS[
            domain_index
        ]
    else:
        domain, first_name, index_name, target_name, third_name, fourth_name = (
            _NATURAL_SEQUENCE_DOMAINS[domain_index]
        )
        second_name = index_name if schema_kind == "lookup_linear_three" else target_name
    builder = _AnnotatedText()
    builder.append(f"For the {domain}, the recorded inputs are ")
    input_names = (first_name, second_name, third_name, fourth_name)
    for index, (name, value) in enumerate(zip(input_names, inputs, strict=True)):
        if index:
            builder.append(", " if index < 3 else ", and ")
        builder.append(name)
        builder.append(" ")
        rendered = (
            "[" + ", ".join(str(item) for item in value) + "]"
            if isinstance(value, tuple)
            else str(value)
        )
        builder.append(rendered, label=f"natural:input:{index}")
    builder.append(". First, ")

    instructions: list[SemanticInstructionAnnotation] = []
    if schema_kind == "scalar_linear_three":
        _append_natural_binary_operation(
            builder,
            op=operations[0],
            ordinal=0,
            left_text=first_name,
            left_label="natural:argument:0:0",
            right_text=second_name,
            right_label="natural:argument:0:1",
        )
    elif schema_kind == "lookup_linear_three":
        builder.append("select the item at", label="natural:operation:0")
        builder.append(" ")
        builder.append(second_name, label="natural:argument:0:1")
        builder.append(" in ")
        builder.append(first_name, label="natural:argument:0:0")
    elif schema_kind == "count_linear_three":
        builder.append("count", label="natural:operation:0")
        builder.append(" how often ")
        builder.append(second_name, label="natural:argument:0:1")
        builder.append(" occurs in ")
        builder.append(first_name, label="natural:argument:0:0")
    else:  # pragma: no cover - builder owns the schema inventory
        raise ValueError("natural procedure schema is unsupported")
    instructions.append(
        SemanticInstructionAnnotation(
            instruction=Instruction(operations[0], (0, 1)),
            operation_span=builder.span("natural:operation:0"),
            argument_spans=tuple(
                builder.span(f"natural:argument:0:{position}") for position in range(2)
            ),
            depends_on=(),
        )
    )

    builder.append(", and call that the running figure. Next, ")
    _append_natural_binary_operation(
        builder,
        op=operations[1],
        ordinal=1,
        left_text="the running figure",
        left_label="natural:argument:1:0",
        right_text=third_name,
        right_label="natural:argument:1:1",
    )
    instructions.append(
        SemanticInstructionAnnotation(
            instruction=Instruction(operations[1], (4, 2)),
            operation_span=builder.span("natural:operation:1"),
            argument_spans=tuple(
                builder.span(f"natural:argument:1:{position}") for position in range(2)
            ),
            depends_on=(0,),
        )
    )

    builder.append(", calling the result the revised figure. Finally, ")
    _append_natural_binary_operation(
        builder,
        op=operations[2],
        ordinal=2,
        left_text="the revised figure",
        left_label="natural:argument:2:0",
        right_text=fourth_name,
        right_label="natural:argument:2:1",
    )
    builder.append(". What is the final value?")
    instructions.append(
        SemanticInstructionAnnotation(
            instruction=Instruction(operations[2], (5, 3)),
            operation_span=builder.span("natural:operation:2"),
            argument_spans=tuple(
                builder.span(f"natural:argument:2:{position}") for position in range(2)
            ),
            depends_on=(1,),
        )
    )

    construction_id = f"natural-{schema_kind}-{domain_index}"
    identity = f"{construction_id}|{sample_index}|{inputs}|{operations}|{builder.text}"
    return SemanticProgramExample(
        example_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        construction_id=construction_id,
        topology_id=schema_kind,
        split="validation" if (domain_index + sample_index) % 2 == 0 else "test",
        source_text=builder.text,
        inputs=inputs,
        input_spans=tuple(builder.span(f"natural:input:{index}") for index in range(4)),
        instructions=tuple(instructions),
        report_value=6,
        contrast_id=hashlib.sha256(
            f"natural|{schema_kind}|{domain_index}|{sample_index}".encode("ascii")
        ).hexdigest()[:24],
    )


def _natural_two_step_source_example(
    *,
    schema_kind: str,
    domain_index: int,
    sample_index: int,
    inputs: tuple[SemanticValue, ...],
    operations: tuple[str, str],
) -> SemanticProgramExample:
    """Render reusable natural relations without exposing the target graph."""

    if schema_kind == "scalar_linear_two":
        domain, first_name, second_name, third_name = _NATURAL_SOURCE_SCALAR_DOMAINS[domain_index]
    else:
        domain, first_name, index_name, target_name, third_name = _NATURAL_SOURCE_SEQUENCE_DOMAINS[
            domain_index
        ]
        second_name = index_name if schema_kind == "lookup_linear_two" else target_name
    builder = _AnnotatedText()
    builder.append(f"For the {domain}, the recorded inputs are ")
    input_names = (first_name, second_name, third_name)
    for index, (name, value) in enumerate(zip(input_names, inputs, strict=True)):
        if index:
            builder.append(", " if index < 2 else ", and ")
        builder.begin(f"natural:definition:{index}")
        builder.append(name)
        builder.append(" ")
        rendered = (
            "[" + ", ".join(str(item) for item in value) + "]"
            if isinstance(value, tuple)
            else str(value)
        )
        builder.append(rendered, label=f"natural:input:{index}")
        builder.finish(f"natural:definition:{index}")
    builder.append(". First, ")

    builder.begin("natural:definition:3")
    if schema_kind == "scalar_linear_two":
        _append_natural_binary_operation(
            builder,
            op=operations[0],
            ordinal=0,
            left_text=first_name,
            left_label="natural:argument:0:0",
            right_text=second_name,
            right_label="natural:argument:0:1",
        )
    elif schema_kind == "lookup_linear_two":
        builder.append("select the item at", label="natural:operation:0")
        builder.append(" ")
        builder.append(second_name, label="natural:argument:0:1")
        builder.append(" in ")
        builder.append(first_name, label="natural:argument:0:0")
    elif schema_kind == "count_linear_two":
        builder.append("count", label="natural:operation:0")
        builder.append(" how often ")
        builder.append(second_name, label="natural:argument:0:1")
        builder.append(" occurs in ")
        builder.append(first_name, label="natural:argument:0:0")
    else:  # pragma: no cover - builder owns the schema inventory
        raise ValueError("natural source procedure schema is unsupported")
    first_instruction = SemanticInstructionAnnotation(
        instruction=Instruction(operations[0], (0, 1)),
        operation_span=builder.span("natural:operation:0"),
        argument_spans=tuple(
            builder.span(f"natural:argument:0:{position}") for position in range(2)
        ),
        depends_on=(),
    )
    builder.append(", and call that the running figure")
    builder.finish("natural:definition:3")
    builder.append(". Then, ")

    builder.begin("natural:definition:4")
    _append_natural_binary_operation(
        builder,
        op=operations[1],
        ordinal=1,
        left_text="the running figure",
        left_label="natural:argument:1:0",
        right_text=third_name,
        right_label="natural:argument:1:1",
    )
    builder.finish("natural:definition:4")
    builder.append(". What is the final value?")
    second_instruction = SemanticInstructionAnnotation(
        instruction=Instruction(operations[1], (3, 2)),
        operation_span=builder.span("natural:operation:1"),
        argument_spans=tuple(
            builder.span(f"natural:argument:1:{position}") for position in range(2)
        ),
        depends_on=(0,),
    )
    construction_id = f"natural-source-{schema_kind}-{domain_index}"
    identity = f"{construction_id}|{sample_index}|{inputs}|{operations}|{builder.text}"
    split: CorpusSplit = (
        "train" if domain_index < 4 else "validation" if domain_index < 6 else "test"
    )
    return SemanticProgramExample(
        example_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        construction_id=construction_id,
        topology_id=schema_kind,
        split=split,
        source_text=builder.text,
        inputs=inputs,
        input_spans=tuple(builder.span(f"natural:input:{index}") for index in range(3)),
        instructions=(first_instruction, second_instruction),
        report_value=4,
        contrast_id=hashlib.sha256(
            f"natural-source|{schema_kind}|{domain_index}|{sample_index}".encode("ascii")
        ).hexdigest()[:24],
        register_definition_spans=tuple(
            builder.span(f"natural:definition:{index}") for index in range(5)
        ),
    )


def build_semantic_program_natural_source_corpus(
    *,
    seed: int = 2718281,
    examples_per_schema_domain: int = 1,
) -> tuple[SemanticProgramExample, ...]:
    """Teach natural local relations on schemas and domains outside the target."""

    if examples_per_schema_domain < 1:
        raise ValueError("natural source corpus needs a sample in every schema-domain cell")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    schemas = ("scalar_linear_two", "lookup_linear_two", "count_linear_two")
    for schema_index, schema_kind in enumerate(schemas):
        for domain_index in range(len(_NATURAL_SOURCE_SCALAR_DOMAINS)):
            for sample_index in range(examples_per_schema_domain):
                chain = _NATURAL_SCALAR_CHAINS[
                    (schema_index * 3 + domain_index + sample_index) % len(_NATURAL_SCALAR_CHAINS)
                ]
                operations = (chain[0], chain[1])
                if schema_kind == "scalar_linear_two":
                    inputs: tuple[SemanticValue, ...] = (
                        rng.randint(120, 940),
                        rng.randint(11, 89),
                        rng.randint(2, 9),
                    )
                else:
                    selector = rng.randint(1, 5)
                    values = [rng.randint(10, 80) for _ in range(7)]
                    if schema_kind == "count_linear_two":
                        wanted = rng.randint(3, 9)
                        values[1] = wanted
                        values[4] = wanted
                        first_op = "count_of"
                        second_input = wanted
                    else:
                        first_op = "at"
                        second_input = selector
                    inputs = (tuple(values), second_input, rng.randint(2, 9))
                    operations = (first_op, operations[1])
                examples.append(
                    _natural_two_step_source_example(
                        schema_kind=schema_kind,
                        domain_index=domain_index,
                        sample_index=sample_index,
                        inputs=inputs,
                        operations=operations,
                    )
                )
    return tuple(examples)


def _append_natural_alias_definition(
    builder: _AnnotatedText,
    *,
    alias: str,
    label: str,
    construction_index: int,
    terminal: bool,
) -> None:
    prefix = (
        (", and store the result as ", ", storing the result as "),
        (". Let ", ". Let "),
        ("; retain that value under the name ", "; retain that value under the name "),
        (". Record its output as ", ". Record its output as "),
    )[construction_index % 4][int(terminal)]
    builder.append(prefix)
    builder.append(alias, label=label)
    if construction_index % 4 == 1:
        builder.append(" to denote that output")


def _natural_alias_source_example(
    *,
    schema_kind: str,
    domain_index: int,
    sample_index: int,
    inputs: tuple[SemanticValue, ...],
    operations: tuple[str, str],
    identity_only: bool = False,
) -> SemanticProgramExample:
    """Render alias-local register supervision outside every target domain."""

    scalar_domains = (
        _NATURAL_IDENTITY_SOURCE_SCALAR_DOMAINS
        if identity_only
        else _NATURAL_ALIAS_SOURCE_SCALAR_DOMAINS
    )
    sequence_domains = (
        _NATURAL_IDENTITY_SOURCE_SEQUENCE_DOMAINS
        if identity_only
        else _NATURAL_ALIAS_SOURCE_SEQUENCE_DOMAINS
    )
    aliases = _NATURAL_IDENTITY_SOURCE_NAMES if identity_only else _NATURAL_ALIAS_SOURCE_NAMES
    if schema_kind == "scalar_alias_linear_two":
        domain, first_name, second_name, third_name = scalar_domains[domain_index]
    else:
        domain, first_name, index_name, target_name, third_name = sequence_domains[domain_index]
        second_name = index_name if schema_kind == "lookup_alias_linear_two" else target_name
    intermediate_alias, terminal_alias = aliases[domain_index]
    builder = _AnnotatedText()
    builder.append(f"In the {domain}, use ")
    input_names = (first_name, second_name, third_name)
    for index, (name, value) in enumerate(zip(input_names, inputs, strict=True)):
        if index:
            builder.append(", " if index < 2 else ", and ")
        builder.begin(f"natural-alias:definition:{index}")
        builder.append(name, label=f"natural-alias:identity:{index}")
        builder.append(" ")
        rendered = (
            "[" + ", ".join(str(item) for item in value) + "]"
            if isinstance(value, tuple)
            else str(value)
        )
        builder.append(rendered, label=f"natural-alias:input:{index}")
        builder.finish(f"natural-alias:definition:{index}")
    builder.append(". First, ")

    if schema_kind == "scalar_alias_linear_two":
        _append_natural_binary_operation(
            builder,
            op=operations[0],
            ordinal=0,
            left_text=first_name,
            left_label="natural:argument:0:0",
            right_text=second_name,
            right_label="natural:argument:0:1",
        )
    elif schema_kind == "lookup_alias_linear_two":
        builder.append("select the item at", label="natural:operation:0")
        builder.append(" ")
        builder.append(second_name, label="natural:argument:0:1")
        builder.append(" in ")
        builder.append(first_name, label="natural:argument:0:0")
    elif schema_kind == "count_alias_linear_two":
        builder.append("count", label="natural:operation:0")
        builder.append(" how often ")
        builder.append(second_name, label="natural:argument:0:1")
        builder.append(" occurs in ")
        builder.append(first_name, label="natural:argument:0:0")
    else:  # pragma: no cover - builder owns the schema inventory
        raise ValueError("natural alias source procedure schema is unsupported")
    first_instruction = SemanticInstructionAnnotation(
        instruction=Instruction(operations[0], (0, 1)),
        operation_span=builder.span("natural:operation:0"),
        argument_spans=tuple(
            builder.span(f"natural:argument:0:{position}") for position in range(2)
        ),
        depends_on=(),
    )
    _append_natural_alias_definition(
        builder,
        alias=intermediate_alias,
        label="natural-alias:definition:3",
        construction_index=domain_index,
        terminal=False,
    )
    builder.append(". Next, ")
    _append_natural_binary_operation(
        builder,
        op=operations[1],
        ordinal=1,
        left_text=intermediate_alias,
        left_label="natural:argument:1:0",
        right_text=third_name,
        right_label="natural:argument:1:1",
    )
    second_instruction = SemanticInstructionAnnotation(
        instruction=Instruction(operations[1], (3, 2)),
        operation_span=builder.span("natural:operation:1"),
        argument_spans=tuple(
            builder.span(f"natural:argument:1:{position}") for position in range(2)
        ),
        depends_on=(0,),
    )
    _append_natural_alias_definition(
        builder,
        alias=terminal_alias,
        label="natural-alias:definition:4",
        construction_index=domain_index,
        terminal=True,
    )
    builder.append(f". Return the {terminal_alias}.")
    source_kind = "natural-identity-source" if identity_only else "natural-alias-source"
    construction_id = f"{source_kind}-{schema_kind}-{domain_index}"
    identity = f"{construction_id}|{sample_index}|{inputs}|{operations}|{builder.text}"
    split: CorpusSplit = (
        "train" if domain_index < 4 else "validation" if domain_index < 6 else "test"
    )
    return SemanticProgramExample(
        example_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        construction_id=construction_id,
        topology_id=schema_kind,
        split=split,
        source_text=builder.text,
        inputs=inputs,
        input_spans=tuple(builder.span(f"natural-alias:input:{index}") for index in range(3)),
        instructions=(first_instruction, second_instruction),
        report_value=4,
        contrast_id=hashlib.sha256(
            f"{source_kind}|{schema_kind}|{domain_index}|{sample_index}".encode("ascii")
        ).hexdigest()[:24],
        register_definition_spans=tuple(
            (
                builder.span(f"natural-alias:identity:{index}")
                if identity_only and index < 3
                else builder.span(f"natural-alias:definition:{index}")
            )
            for index in range(5)
        ),
    )


def build_semantic_program_natural_alias_source_corpus(
    *,
    seed: int = 1618034,
    examples_per_schema_domain: int = 1,
) -> tuple[SemanticProgramExample, ...]:
    """Teach computed-register binding from local aliases in ordinary prose."""

    if examples_per_schema_domain < 1:
        raise ValueError("natural alias source corpus needs every schema-domain cell")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    schemas = (
        "scalar_alias_linear_two",
        "lookup_alias_linear_two",
        "count_alias_linear_two",
    )
    for schema_index, schema_kind in enumerate(schemas):
        for domain_index in range(len(_NATURAL_ALIAS_SOURCE_SCALAR_DOMAINS)):
            for sample_index in range(examples_per_schema_domain):
                chain = _NATURAL_SCALAR_CHAINS[
                    (schema_index * 5 + domain_index + sample_index) % len(_NATURAL_SCALAR_CHAINS)
                ]
                operations = (chain[0], chain[1])
                if schema_kind == "scalar_alias_linear_two":
                    inputs: tuple[SemanticValue, ...] = (
                        rng.randint(150, 980),
                        rng.randint(13, 97),
                        rng.randint(2, 11),
                    )
                else:
                    values = [rng.randint(12, 88) for _ in range(7)]
                    if schema_kind == "count_alias_linear_two":
                        wanted = rng.randint(3, 11)
                        values[2] = wanted
                        values[6] = wanted
                        first_op = "count_of"
                        second_input = wanted
                    else:
                        first_op = "at"
                        second_input = rng.randint(0, len(values) - 1)
                    inputs = (tuple(values), second_input, rng.randint(2, 11))
                    operations = (first_op, operations[1])
                examples.append(
                    _natural_alias_source_example(
                        schema_kind=schema_kind,
                        domain_index=domain_index,
                        sample_index=sample_index,
                        inputs=inputs,
                        operations=operations,
                    )
                )
    return tuple(examples)


def build_semantic_program_natural_identity_source_corpus(
    *,
    seed: int = 2236067,
    examples_per_schema_domain: int = 1,
) -> tuple[SemanticProgramExample, ...]:
    """Teach register identities independently of exact values and operations."""

    if examples_per_schema_domain < 1:
        raise ValueError("natural identity source corpus needs every schema-domain cell")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    schemas = (
        "scalar_alias_linear_two",
        "lookup_alias_linear_two",
        "count_alias_linear_two",
    )
    for schema_index, schema_kind in enumerate(schemas):
        for domain_index in range(len(_NATURAL_IDENTITY_SOURCE_SCALAR_DOMAINS)):
            for sample_index in range(examples_per_schema_domain):
                chain = _NATURAL_SCALAR_CHAINS[
                    (schema_index * 7 + domain_index + sample_index) % len(_NATURAL_SCALAR_CHAINS)
                ]
                operations = (chain[0], chain[1])
                if schema_kind == "scalar_alias_linear_two":
                    inputs: tuple[SemanticValue, ...] = (
                        rng.randint(130, 990),
                        rng.randint(17, 99),
                        rng.randint(2, 13),
                    )
                else:
                    values = [rng.randint(14, 96) for _ in range(7)]
                    if schema_kind == "count_alias_linear_two":
                        wanted = rng.randint(4, 13)
                        values[0] = wanted
                        values[5] = wanted
                        first_op = "count_of"
                        second_input = wanted
                    else:
                        first_op = "at"
                        second_input = rng.randint(0, len(values) - 1)
                    inputs = (tuple(values), second_input, rng.randint(2, 13))
                    operations = (first_op, operations[1])
                examples.append(
                    _natural_alias_source_example(
                        schema_kind=schema_kind,
                        domain_index=domain_index,
                        sample_index=sample_index,
                        inputs=inputs,
                        operations=operations,
                        identity_only=True,
                    )
                )
    return tuple(examples)


def build_semantic_program_natural_request_corpus(
    *,
    seed: int = 3141592,
    examples_per_schema_domain: int = 1,
) -> tuple[SemanticProgramExample, ...]:
    """Build heterogeneous requests over wholly withheld three-step schemas.

    The frozen v14 tissue saw scalar three-input/two-step chains and four-input
    fork/join graphs. It did not see a four-input three-step linear graph, nor
    typed lookup/count variants of that graph. Domain nouns and values vary
    independently of those schemas.
    """

    if examples_per_schema_domain < 1:
        raise ValueError("natural request corpus needs a sample in every schema-domain cell")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    schemas = ("scalar_linear_three", "lookup_linear_three", "count_linear_three")
    for schema_index, schema_kind in enumerate(schemas):
        for domain_index in range(len(_NATURAL_SCALAR_DOMAINS)):
            for sample_index in range(examples_per_schema_domain):
                operations = _NATURAL_SCALAR_CHAINS[
                    (schema_index * 3 + domain_index + sample_index) % len(_NATURAL_SCALAR_CHAINS)
                ]
                if schema_kind == "scalar_linear_three":
                    inputs: tuple[SemanticValue, ...] = (
                        rng.randint(120, 940),
                        rng.randint(11, 89),
                        rng.randint(2, 9),
                        rng.randint(2, 17),
                    )
                else:
                    selector = rng.randint(1, 5)
                    values = [rng.randint(10, 80) for _ in range(7)]
                    if schema_kind == "count_linear_three":
                        wanted = rng.randint(3, 9)
                        values[1] = wanted
                        values[4] = wanted
                        first_op = "count_of"
                        second_input = wanted
                    else:
                        first_op = "at"
                        second_input = selector
                    inputs = (
                        tuple(values),
                        second_input,
                        rng.randint(2, 9),
                        rng.randint(2, 17),
                    )
                    operations = (first_op, operations[1], operations[2])
                examples.append(
                    _natural_three_step_example(
                        schema_kind=schema_kind,
                        domain_index=domain_index,
                        sample_index=sample_index,
                        inputs=inputs,
                        operations=operations,
                    )
                )
    return tuple(examples)


def _natural_replication_example(
    *,
    schema_kind: str,
    domain_index: int,
    sample_index: int,
    inputs: tuple[SemanticValue, ...],
    operations: tuple[str, str, str],
) -> SemanticProgramExample:
    """Render a preregistered request outside the source and development language."""

    if schema_kind == "scalar_linear_three":
        domain, first_name, second_name, third_name, fourth_name = (
            _NATURAL_REPLICATION_SCALAR_DOMAINS[domain_index]
        )
    else:
        domain, first_name, index_name, target_name, third_name, fourth_name = (
            _NATURAL_REPLICATION_SEQUENCE_DOMAINS[domain_index]
        )
        second_name = index_name if schema_kind == "lookup_linear_three" else target_name
    (
        opening,
        first_intro,
        first_alias_clause,
        second_intro,
        second_alias_clause,
        third_intro,
        question,
    ) = _NATURAL_REPLICATION_SURFACES[sample_index % len(_NATURAL_REPLICATION_SURFACES)]
    first_alias = ("first result", "subtotal", "interim value", "provisional value")[
        sample_index % 4
    ]
    second_alias = ("second result", "adjusted total", "revised value", "combined value")[
        sample_index % 4
    ]
    builder = _AnnotatedText()
    builder.append(opening.format(domain=domain))
    builder.append(": ")
    input_names = (first_name, second_name, third_name, fourth_name)
    for index, (name, value) in enumerate(zip(input_names, inputs, strict=True)):
        if index:
            builder.append("; ")
        builder.append(name)
        builder.append(" = ")
        rendered = (
            "[" + ", ".join(str(item) for item in value) + "]"
            if isinstance(value, tuple)
            else str(value)
        )
        builder.append(rendered, label=f"replication:input:{index}")
    builder.append(f". {first_intro} ")

    if schema_kind == "scalar_linear_three":
        _append_natural_binary_operation(
            builder,
            op=operations[0],
            ordinal=0,
            left_text=first_name,
            left_label="replication:argument:0:0",
            right_text=second_name,
            right_label="replication:argument:0:1",
        )
    elif schema_kind == "lookup_linear_three":
        builder.append("read the value at", label="natural:operation:0")
        builder.append(" ")
        builder.append(second_name, label="replication:argument:0:1")
        builder.append(" from ")
        builder.append(first_name, label="replication:argument:0:0")
    elif schema_kind == "count_linear_three":
        builder.append("count", label="natural:operation:0")
        builder.append(" the entries equal to ")
        builder.append(second_name, label="replication:argument:0:1")
        builder.append(" within ")
        builder.append(first_name, label="replication:argument:0:0")
    else:  # pragma: no cover - builder owns the schema inventory
        raise ValueError("natural replication schema is unsupported")
    first_instruction = SemanticInstructionAnnotation(
        instruction=Instruction(operations[0], (0, 1)),
        operation_span=builder.span("natural:operation:0"),
        argument_spans=tuple(
            builder.span(f"replication:argument:0:{position}") for position in range(2)
        ),
        depends_on=(),
    )
    builder.append(f". {first_alias_clause}. {second_intro} ")
    _append_natural_binary_operation(
        builder,
        op=operations[1],
        ordinal=1,
        left_text=first_alias,
        left_label="replication:argument:1:0",
        right_text=third_name,
        right_label="replication:argument:1:1",
    )
    second_instruction = SemanticInstructionAnnotation(
        instruction=Instruction(operations[1], (4, 2)),
        operation_span=builder.span("natural:operation:1"),
        argument_spans=tuple(
            builder.span(f"replication:argument:1:{position}") for position in range(2)
        ),
        depends_on=(0,),
    )
    builder.append(f". {second_alias_clause}. {third_intro} ")
    _append_natural_binary_operation(
        builder,
        op=operations[2],
        ordinal=2,
        left_text=second_alias,
        left_label="replication:argument:2:0",
        right_text=fourth_name,
        right_label="replication:argument:2:1",
    )
    third_instruction = SemanticInstructionAnnotation(
        instruction=Instruction(operations[2], (5, 3)),
        operation_span=builder.span("natural:operation:2"),
        argument_spans=tuple(
            builder.span(f"replication:argument:2:{position}") for position in range(2)
        ),
        depends_on=(1,),
    )
    builder.append(f". {question}?")
    construction_id = f"natural-replication-{schema_kind}-{domain_index}-{sample_index % 4}"
    identity = f"{construction_id}|{sample_index}|{inputs}|{operations}|{builder.text}"
    return SemanticProgramExample(
        example_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        construction_id=construction_id,
        topology_id=schema_kind,
        split="validation" if (domain_index + sample_index) % 2 == 0 else "test",
        source_text=builder.text,
        inputs=inputs,
        input_spans=tuple(builder.span(f"replication:input:{index}") for index in range(4)),
        instructions=(first_instruction, second_instruction, third_instruction),
        report_value=6,
        contrast_id=hashlib.sha256(
            f"natural-replication|{schema_kind}|{domain_index}|{sample_index}".encode("ascii")
        ).hexdigest()[:24],
    )


def build_semantic_program_natural_replication_corpus(
    *,
    seed: int = 1732051,
    examples_per_schema_domain: int = 4,
) -> tuple[SemanticProgramExample, ...]:
    """Build the fresh preregistered natural transfer replication."""

    if examples_per_schema_domain < 1:
        raise ValueError("natural replication needs a sample in every schema-domain cell")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    schemas = ("scalar_linear_three", "lookup_linear_three", "count_linear_three")
    for schema_index, schema_kind in enumerate(schemas):
        for domain_index in range(len(_NATURAL_REPLICATION_SCALAR_DOMAINS)):
            for sample_index in range(examples_per_schema_domain):
                operations = _NATURAL_SCALAR_CHAINS[
                    (schema_index * 5 + domain_index + sample_index) % len(_NATURAL_SCALAR_CHAINS)
                ]
                if schema_kind == "scalar_linear_three":
                    inputs: tuple[SemanticValue, ...] = (
                        rng.randint(100_000_000, 900_000_000),
                        rng.randint(10_000_000, 90_000_000),
                        rng.randint(11, 999),
                        rng.randint(1_001, 99_999),
                    )
                else:
                    values = [rng.randint(10_000_000, 900_000_000) for _ in range(7)]
                    if schema_kind == "count_linear_three":
                        wanted = rng.randint(10_000_000, 900_000_000)
                        values[1] = wanted
                        values[5] = wanted
                        first_op = "count_of"
                        second_input = wanted
                    else:
                        first_op = "at"
                        second_input = rng.randint(0, len(values) - 1)
                    inputs = (
                        tuple(values),
                        second_input,
                        rng.randint(11, 999),
                        rng.randint(1_001, 99_999),
                    )
                    operations = (first_op, operations[1], operations[2])
                examples.append(
                    _natural_replication_example(
                        schema_kind=schema_kind,
                        domain_index=domain_index,
                        sample_index=sample_index,
                        inputs=inputs,
                        operations=operations,
                    )
                )
    return tuple(examples)


def _character_to_token_span(
    span: CharacterSpan,
    offsets: Sequence[tuple[int, int]],
) -> TokenSpan:
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < span.end and end > span.start
    ]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("tokenizer offsets do not preserve a semantic source span")
    covered_start = min(offsets[index][0] for index in indices)
    covered_end = max(offsets[index][1] for index in indices)
    if covered_start > span.start or covered_end < span.end:
        raise ValueError("tokenizer offsets do not cover a semantic source span")
    return TokenSpan(indices[0], indices[-1] + 1)


def project_register_definition_spans(
    example: SemanticProgramExample,
    *,
    offset_mapping: Sequence[tuple[int, int]],
) -> tuple[TokenSpan, ...]:
    """Project role-bearing register definitions without changing value grounding."""

    character_spans = example.register_definition_spans or (
        *example.input_spans,
        *(instruction.operation_span for instruction in example.instructions),
    )
    return tuple(_character_to_token_span(span, offset_mapping) for span in character_spans)


def project_example_to_ir(
    example: SemanticProgramExample,
    *,
    source_token_ids: Sequence[int],
    offset_mapping: Sequence[tuple[int, int]],
    model_basis_receipt_sha256: str,
    transducer_receipt_sha256: str,
) -> SemanticProgramIR:
    """Project measured character attribution onto one tokenizer basis."""

    if len(source_token_ids) != len(offset_mapping):
        raise ValueError("token ids and source offsets differ in length")
    instructions = tuple(
        SemanticIRInstruction(
            op=item.instruction.op,
            args=item.instruction.args,
            operation_span=_character_to_token_span(
                item.operation_span,
                offset_mapping,
            ),
            argument_spans=tuple(
                _character_to_token_span(span, offset_mapping) for span in item.argument_spans
            ),
            depends_on=item.depends_on,
        )
        for item in example.instructions
    )
    return SemanticProgramIR(
        source_token_ids=tuple(source_token_ids),
        source_text_sha256=hashlib.sha256(example.source_text.encode("utf-8")).hexdigest(),
        input_spans=tuple(
            _character_to_token_span(span, offset_mapping) for span in example.input_spans
        ),
        instructions=instructions,
        report_value=example.report_value,
        model_basis_receipt_sha256=model_basis_receipt_sha256,
        transducer_receipt_sha256=transducer_receipt_sha256,
    )


__all__ = [
    "CharacterSpan",
    "ForkJoinTopology",
    "ProgramTopology",
    "SEMANTIC_PROGRAM_CORPUS_SCHEMA",
    "SemanticInstructionAnnotation",
    "SemanticProgramExample",
    "build_semantic_program_corpus",
    "build_semantic_program_fork_join_factorial_corpus",
    "build_semantic_program_fork_join_corpus",
    "build_semantic_program_natural_alias_source_corpus",
    "build_semantic_program_natural_identity_source_corpus",
    "build_semantic_program_natural_request_corpus",
    "build_semantic_program_natural_replication_corpus",
    "build_semantic_program_natural_source_corpus",
    "build_semantic_program_sequence_binary_corpus",
    "build_semantic_program_sequence_cataphoric_corpus",
    "build_semantic_program_sequence_corpus",
    "build_semantic_program_sequence_reserved_alias_corpus",
    "build_semantic_program_sequence_role_binding_corpus",
    "project_register_definition_spans",
    "project_example_to_ir",
]
