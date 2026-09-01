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
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final, Literal

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    TokenSpan,
)

CorpusSplit = Literal["train", "validation", "test"]

SEMANTIC_PROGRAM_CORPUS_SCHEMA: Final = "aura.semantic_program_corpus.v3"

_OPERATION_LANGUAGE: Final[dict[str, dict[str, str]]] = {
    "add": {"verb": "add", "noun": "sum"},
    "sub": {"verb": "subtract", "noun": "difference"},
    "mul": {"verb": "multiply", "noun": "product"},
    "idiv": {"verb": "integer-divide", "noun": "whole-number quotient"},
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
        return (
            (3, self.remaining_input)
            if self.result_is_left
            else (self.remaining_input, 3)
        )


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
    inputs: tuple[int, ...]
    input_spans: tuple[CharacterSpan, ...]
    instructions: tuple[SemanticInstructionAnnotation, ...]
    report_value: int
    contrast_id: str
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
        for span in self.input_spans:
            _validate_character_span(span, self.source_text)
        n_inputs = len(self.inputs)
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
    source_order = tuple(
        sorted(input_labels, key=lambda label: builder.span(label).start)
    )
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
                            f"{construction_id}:{topology.topology_id}:"
                            f"{sample_index}:{values}"
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
                                input_spans=tuple(
                                    builder.span(label) for label in input_labels
                                ),
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
                        builder, input_labels, annotations = renderer(
                            operation_tuple,
                            values,
                            topology,
                        )
                        if source_order_registers:
                            input_order = tuple(
                                int(label.removeprefix("in"))
                                for label in input_labels
                            )
                            if set(input_order) != {0, 1, 2, 3}:
                                raise AssertionError(
                                    "fork-join renderer did not expose every input once"
                                )
                            register_map = {
                                old_register: new_register
                                for new_register, old_register in enumerate(input_order)
                            }
                            corpus_values = tuple(values[index] for index in input_order)
                            corpus_annotations = tuple(
                                SemanticInstructionAnnotation(
                                    instruction=Instruction(
                                        item.instruction.op,
                                        tuple(
                                            register_map.get(argument, argument)
                                            for argument in item.instruction.args
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
                        contrast_id = (
                            f"{construction_id}:{topology.topology_id}:"
                            f"{sample_index}:{corpus_values}"
                        )
                        examples.append(
                            SemanticProgramExample(
                                example_id=_fork_join_example_id(
                                    construction_id,
                                    topology.topology_id,
                                    operation_tuple,
                                    corpus_values,
                                ),
                                construction_id=construction_id,
                                topology_id=topology.topology_id,
                                split=split,
                                source_text=builder.text,
                                inputs=corpus_values,
                                input_spans=tuple(
                                    builder.span(label) for label in corpus_input_labels
                                ),
                                instructions=corpus_annotations,
                                report_value=6,
                                contrast_id=contrast_id,
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
                _character_to_token_span(span, offset_mapping)
                for span in item.argument_spans
            ),
            depends_on=item.depends_on,
        )
        for item in example.instructions
    )
    return SemanticProgramIR(
        source_token_ids=tuple(source_token_ids),
        source_text_sha256=hashlib.sha256(
            example.source_text.encode("utf-8")
        ).hexdigest(),
        input_spans=tuple(
            _character_to_token_span(span, offset_mapping)
            for span in example.input_spans
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
    "build_semantic_program_fork_join_corpus",
    "project_example_to_ir",
]
