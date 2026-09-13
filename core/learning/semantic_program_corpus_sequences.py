"""Sequence programs, and the four ways this corpus says the same one.

A chain read left to right; the same chain with its operands fronted; one where
the name arrives before the thing it names; one that reuses a name the language
has already reserved. Each renderer writes the instruction and the annotation in
the same pass, so the span the corpus claims a word covers is the span the
renderer actually wrote. Building them apart is how an annotation drifts off its
own text and nobody notices for a month.
"""
from __future__ import annotations

import hashlib
import random

from core.learning.procedure_induction import Instruction

from .semantic_program_corpus import (
    _SCALAR_CONTINUATION_LANGUAGE,
    _SCALAR_CONTINUATIONS,
    _SEQUENCE_AGGREGATES,
    _SEQUENCE_BINARY_OPERATION_LANGUAGE,
    _SEQUENCE_BINARY_SELECTORS,
    _SEQUENCE_OPERATION_LANGUAGE,
    _SEQUENCE_TRANSFORMS,
    CharacterSpan,
    CorpusSplit,
    SemanticInstructionAnnotation,
    SemanticProgramExample,
    _AnnotatedText,
)


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
