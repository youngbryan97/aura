"""Programs asked for the way a person would ask for them.

The sequence corpora state the computation in a register language. These state
it in ordinary sentences over ordinary domains — a source to read, an alias
standing in for it, a request that names neither directly — and keep every
annotation aligned to the words that carry it, which is the only thing that
makes the pairing checkable at all.
"""
from __future__ import annotations

import hashlib
import random
from typing import Final

from core.learning.procedure_induction import Instruction
from core.learning.semantic_program_ir import (
    SemanticValue,
)

from .semantic_program_corpus import (
    _NATURAL_ALIAS_SOURCE_NAMES,
    _NATURAL_ALIAS_SOURCE_SCALAR_DOMAINS,
    _NATURAL_ALIAS_SOURCE_SEQUENCE_DOMAINS,
    _NATURAL_IDENTITY_SOURCE_NAMES,
    _NATURAL_IDENTITY_SOURCE_SCALAR_DOMAINS,
    _NATURAL_IDENTITY_SOURCE_SEQUENCE_DOMAINS,
    _NATURAL_SCALAR_CHAINS,
    _NATURAL_SCALAR_DOMAINS,
    _NATURAL_SOURCE_SCALAR_DOMAINS,
    CorpusSplit,
    SemanticInstructionAnnotation,
    SemanticProgramExample,
    _AnnotatedText,
    _append_natural_binary_operation,
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
