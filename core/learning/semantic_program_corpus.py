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
from typing import Final, Literal

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    TokenSpan,
)

CorpusSplit = Literal["train", "validation", "test"]

SEMANTIC_PROGRAM_CORPUS_SCHEMA: Final = "aura.semantic_program_corpus.v2"

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
    builder.append(". Then ")
    second_left, second_right = _second_arguments(
        values,
        topology,
        result=("that result", "ref0"),
    )
    second_args = _append_binary_verb(
        builder,
        op=second_op,
        operation_label="op1",
        left=second_left,
        right=second_right,
    )
    builder.append(" using whole-number arithmetic.")
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
) -> tuple[_AnnotatedText, tuple[str, str, str], tuple[SemanticInstructionAnnotation, ...]]:
    builder = _AnnotatedText()
    builder.append("Using ")
    builder.append(
        str(values[topology.remaining_input]),
        label=f"in{topology.remaining_input}",
    )
    builder.append(" as the later operand, first ")
    first_left, first_right = _first_arguments(values, topology)
    first_args = _append_binary_verb(
        builder,
        op=first_op,
        operation_label="op0",
        left=first_left,
        right=first_right,
    )
    builder.append("; afterward, ")
    second_left, second_right = _second_arguments(
        values,
        topology,
        result=("the obtained value", "ref0"),
        remaining_label="ref_remaining",
    )
    second_args = _append_binary_verb(
        builder,
        op=second_op,
        operation_label="op1",
        left=(
            "the obtained value" if second_left[1] == "ref0" else "that earlier operand",
            second_left[1],
        ),
        right=(
            "the obtained value" if second_right[1] == "ref0" else "that earlier operand",
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
) -> tuple[_AnnotatedText, tuple[str, str, str], tuple[SemanticInstructionAnnotation, ...]]:
    builder = _AnnotatedText()
    builder.append("Before you ")
    second_left, second_right = _second_arguments(
        values,
        topology,
        result=("the intermediate quantity", "ref0"),
    )
    second_args = _append_binary_verb(
        builder,
        op=second_op,
        operation_label="op1",
        left=second_left,
        right=second_right,
    )
    builder.append(", obtain that quantity: ")
    first_left, first_right = _first_arguments(values, topology)
    first_args = _append_binary_verb(
        builder,
        op=first_op,
        operation_label="op0",
        left=first_left,
        right=first_right,
    )
    builder.append(". Report the whole-number result.")
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


_CONSTRUCTIONS: Final = {
    "sequential": ("train", _sequential),
    "nominal_nested": ("train", _nominal_nested),
    "fronted_operand": ("validation", _fronted_operand),
    "reverse_clause": ("test", _reverse_clause),
}

_TOPOLOGIES: Final = (
    ProgramTopology("left_01_then_2", (0, 1), 2, True),
    ProgramTopology("left_12_then_0", (1, 2), 0, True),
    ProgramTopology("1_then_right_02", (0, 2), 1, False),
    ProgramTopology("0_then_right_12", (1, 2), 0, False),
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
    "ProgramTopology",
    "SEMANTIC_PROGRAM_CORPUS_SCHEMA",
    "SemanticInstructionAnnotation",
    "SemanticProgramExample",
    "build_semantic_program_corpus",
    "project_example_to_ir",
]
