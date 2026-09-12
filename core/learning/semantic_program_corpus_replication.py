"""Programs that do the same work more than once.

Plain replication, a branch that replicates down one side only, and a weave
where two sequences advance together. A transducer that has learned the shape
of a program rather than the instance it was shown either holds here or falls
over here, which is what these are for.
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
    _NATURAL_BRANCH_CHAINS,
    _NATURAL_BRANCH_REPLICATION_SCALAR_DOMAINS,
    _NATURAL_BRANCH_REPLICATION_SEQUENCE_DOMAINS,
    _NATURAL_BRANCH_REPLICATION_SURFACES,
    _NATURAL_REPLICATION_SCALAR_DOMAINS,
    _NATURAL_REPLICATION_SURFACES,
    _NATURAL_SCALAR_CHAINS,
    _NATURAL_WEAVE_CHAINS,
    _NATURAL_WEAVE_REPLICATION_SURFACES,
    NATURAL_WEAVE_REPLICATION_DOMAINS,
    SemanticInstructionAnnotation,
    SemanticProgramExample,
    _AnnotatedText,
    _append_natural_binary_operation,
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


_NATURAL_WEAVE_REPLICATION_SCALAR_DOMAINS: Final = (
    (
        "geothermal dispatch",
        "intake flow",
        "return flow",
        "turbine reserve",
        "storage reserve",
        "correction",
        "trim",
    ),
    (
        "library conservation",
        "treated folios",
        "queued folios",
        "drying reserve",
        "binding reserve",
        "inspection adjustment",
        "catalogue trim",
    ),
    (
        "drone depot",
        "outbound units",
        "returned units",
        "charging reserve",
        "repair reserve",
        "route adjustment",
        "dispatch trim",
    ),
    (
        "orchard survey",
        "north trees",
        "south trees",
        "sampling reserve",
        "mapping reserve",
        "boundary adjustment",
        "survey trim",
    ),
    (
        "ceramics kiln",
        "glazed pieces",
        "bisque pieces",
        "shelf reserve",
        "cooling reserve",
        "firing adjustment",
        "kiln trim",
    ),
    (
        "marine laboratory",
        "surface samples",
        "deep samples",
        "freezer reserve",
        "assay reserve",
        "salinity adjustment",
        "lab trim",
    ),
    (
        "transit workshop",
        "inspected axles",
        "serviced axles",
        "parts reserve",
        "bay reserve",
        "schedule adjustment",
        "workshop trim",
    ),
    (
        "concert production",
        "floor tickets",
        "balcony tickets",
        "access reserve",
        "staffing reserve",
        "seating adjustment",
        "production trim",
    ),
)


_NATURAL_WEAVE_REPLICATION_SEQUENCE_DOMAINS: Final = (
    (
        "geothermal dispatch",
        "readings by well",
        "well position",
        "target reading",
        "turbine reserve",
        "storage reserve",
        "correction",
        "trim",
    ),
    (
        "library conservation",
        "folios by shelf",
        "shelf position",
        "target folio",
        "drying reserve",
        "binding reserve",
        "inspection adjustment",
        "catalogue trim",
    ),
    (
        "drone depot",
        "units by pad",
        "pad position",
        "target unit",
        "charging reserve",
        "repair reserve",
        "route adjustment",
        "dispatch trim",
    ),
    (
        "orchard survey",
        "trees by block",
        "block position",
        "target tree count",
        "sampling reserve",
        "mapping reserve",
        "boundary adjustment",
        "survey trim",
    ),
    (
        "ceramics kiln",
        "pieces by rack",
        "rack position",
        "target piece count",
        "shelf reserve",
        "cooling reserve",
        "firing adjustment",
        "kiln trim",
    ),
    (
        "marine laboratory",
        "samples by station",
        "station position",
        "target sample",
        "freezer reserve",
        "assay reserve",
        "salinity adjustment",
        "lab trim",
    ),
    (
        "transit workshop",
        "axles by lift",
        "lift position",
        "target axle count",
        "parts reserve",
        "bay reserve",
        "schedule adjustment",
        "workshop trim",
    ),
    (
        "concert production",
        "tickets by section",
        "section position",
        "target ticket count",
        "access reserve",
        "staffing reserve",
        "seating adjustment",
        "production trim",
    ),
)


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


def _natural_branch_replication_example(
    *,
    schema_kind: str,
    domain_index: int,
    sample_index: int,
    inputs: tuple[SemanticValue, ...],
    operations: tuple[str, str, str, str],
) -> SemanticProgramExample:
    """Render an evaluation-only branch, merge, and terminal procedure."""

    if schema_kind == "scalar_branch_merge_four":
        domain, first_name, second_name, third_name, fourth_name, fifth_name = (
            _NATURAL_BRANCH_REPLICATION_SCALAR_DOMAINS[domain_index]
        )
    else:
        (
            domain,
            first_name,
            index_name,
            target_name,
            third_name,
            fourth_name,
            fifth_name,
        ) = _NATURAL_BRANCH_REPLICATION_SEQUENCE_DOMAINS[domain_index]
        second_name = index_name if schema_kind == "lookup_branch_merge_four" else target_name
    (
        opening,
        first_intro,
        first_alias_clause,
        second_intro,
        second_alias_clause,
        merge_intro,
        merge_alias_clause,
        terminal_intro,
        question,
        first_alias,
        second_alias,
        merge_alias,
    ) = _NATURAL_BRANCH_REPLICATION_SURFACES[
        sample_index % len(_NATURAL_BRANCH_REPLICATION_SURFACES)
    ]
    builder = _AnnotatedText()
    builder.append(opening.format(domain=domain))
    builder.append(": input definitions are ")
    input_names = (first_name, second_name, third_name, fourth_name, fifth_name)
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
        builder.append(rendered, label=f"branch-replication:input:{index}")
    builder.append(f". {first_intro} ")

    if schema_kind == "scalar_branch_merge_four":
        _append_natural_binary_operation(
            builder,
            op=operations[0],
            ordinal=0,
            left_text=first_name,
            left_label="branch-replication:argument:0:0",
            right_text=second_name,
            right_label="branch-replication:argument:0:1",
        )
    elif schema_kind == "lookup_branch_merge_four":
        builder.append("retrieve", label="natural:operation:0")
        builder.append(" the entry at ")
        builder.append(second_name, label="branch-replication:argument:0:1")
        builder.append(" in ")
        builder.append(first_name, label="branch-replication:argument:0:0")
    elif schema_kind == "count_branch_merge_four":
        builder.append("count", label="natural:operation:0")
        builder.append(" occurrences of ")
        builder.append(second_name, label="branch-replication:argument:0:1")
        builder.append(" in ")
        builder.append(first_name, label="branch-replication:argument:0:0")
    else:  # pragma: no cover - builder owns the schema inventory
        raise ValueError("natural branch replication schema is unsupported")
    instructions = [
        SemanticInstructionAnnotation(
            instruction=Instruction(operations[0], (0, 1)),
            operation_span=builder.span("natural:operation:0"),
            argument_spans=tuple(
                builder.span(f"branch-replication:argument:0:{position}")
                for position in range(2)
            ),
            depends_on=(),
        )
    ]

    builder.append(f". {first_alias_clause}. {second_intro} ")
    _append_natural_binary_operation(
        builder,
        op=operations[1],
        ordinal=1,
        left_text=third_name,
        left_label="branch-replication:argument:1:0",
        right_text=fourth_name,
        right_label="branch-replication:argument:1:1",
    )
    instructions.append(
        SemanticInstructionAnnotation(
            instruction=Instruction(operations[1], (2, 3)),
            operation_span=builder.span("natural:operation:1"),
            argument_spans=tuple(
                builder.span(f"branch-replication:argument:1:{position}")
                for position in range(2)
            ),
            depends_on=(),
        )
    )

    builder.append(f". {second_alias_clause}. {merge_intro} ")
    _append_natural_binary_operation(
        builder,
        op=operations[2],
        ordinal=2,
        left_text=first_alias,
        left_label="branch-replication:argument:2:0",
        right_text=second_alias,
        right_label="branch-replication:argument:2:1",
    )
    instructions.append(
        SemanticInstructionAnnotation(
            instruction=Instruction(operations[2], (5, 6)),
            operation_span=builder.span("natural:operation:2"),
            argument_spans=tuple(
                builder.span(f"branch-replication:argument:2:{position}")
                for position in range(2)
            ),
            depends_on=(0, 1),
        )
    )

    builder.append(f". {merge_alias_clause}. {terminal_intro} ")
    _append_natural_binary_operation(
        builder,
        op=operations[3],
        ordinal=3,
        left_text=merge_alias,
        left_label="branch-replication:argument:3:0",
        right_text=fifth_name,
        right_label="branch-replication:argument:3:1",
    )
    instructions.append(
        SemanticInstructionAnnotation(
            instruction=Instruction(operations[3], (7, 4)),
            operation_span=builder.span("natural:operation:3"),
            argument_spans=tuple(
                builder.span(f"branch-replication:argument:3:{position}")
                for position in range(2)
            ),
            depends_on=(2,),
        )
    )
    builder.append(f". {question}?")
    construction_id = (
        f"natural-branch-replication-{schema_kind}-{domain_index}-{sample_index % 2}"
    )
    identity = f"{construction_id}|{sample_index}|{inputs}|{operations}|{builder.text}"
    return SemanticProgramExample(
        example_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        construction_id=construction_id,
        topology_id=schema_kind,
        split="validation" if (domain_index + sample_index) % 2 == 0 else "test",
        source_text=builder.text,
        inputs=inputs,
        input_spans=tuple(
            builder.span(f"branch-replication:input:{index}") for index in range(5)
        ),
        instructions=tuple(instructions),
        report_value=8,
        contrast_id=hashlib.sha256(
            f"natural-branch-replication|{schema_kind}|{domain_index}|{sample_index}".encode(
                "ascii"
            )
        ).hexdigest()[:24],
    )


def build_semantic_program_natural_branch_replication_corpus(
    *,
    seed: int = 2718281828,
    examples_per_schema_domain: int = 2,
) -> tuple[SemanticProgramExample, ...]:
    """Build the preregistered fresh branch-and-merge transfer corpus."""

    if examples_per_schema_domain < 1:
        raise ValueError("natural branch replication needs every schema-domain cell")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    schemas = (
        "scalar_branch_merge_four",
        "lookup_branch_merge_four",
        "count_branch_merge_four",
    )
    for schema_index, schema_kind in enumerate(schemas):
        for domain_index in range(len(_NATURAL_BRANCH_REPLICATION_SCALAR_DOMAINS)):
            for sample_index in range(examples_per_schema_domain):
                operations = _NATURAL_BRANCH_CHAINS[
                    (schema_index * 5 + domain_index + sample_index)
                    % len(_NATURAL_BRANCH_CHAINS)
                ]
                if schema_kind == "scalar_branch_merge_four":
                    inputs: tuple[SemanticValue, ...] = (
                        rng.randint(100_000, 900_000),
                        rng.randint(10_000, 90_000),
                        rng.randint(1_000, 9_000),
                        rng.randint(100, 900),
                        rng.randint(2, 97),
                    )
                else:
                    values = [rng.randint(100_000, 900_000) for _ in range(7)]
                    if schema_kind == "count_branch_merge_four":
                        wanted = rng.randint(100_000, 900_000)
                        values[0] = wanted
                        values[4] = wanted
                        first_op = "count_of"
                        second_input = wanted
                    else:
                        first_op = "at"
                        second_input = rng.randint(0, len(values) - 1)
                    inputs = (
                        tuple(values),
                        second_input,
                        rng.randint(1_000, 9_000),
                        rng.randint(100, 900),
                        rng.randint(2, 97),
                    )
                    operations = (first_op, operations[1], operations[2], operations[3])
                examples.append(
                    _natural_branch_replication_example(
                        schema_kind=schema_kind,
                        domain_index=domain_index,
                        sample_index=sample_index,
                        inputs=inputs,
                        operations=operations,
                    )
                )
    return tuple(examples)


def _natural_weave_replication_example(
    *,
    schema_kind: str,
    domain_index: int,
    sample_index: int,
    inputs: tuple[SemanticValue, ...],
    operations: tuple[str, str, str, str, str],
) -> SemanticProgramExample:
    """Render the preregistered branch, extension, merge, and terminal graph."""
    if schema_kind == "scalar_branch_weave_five":
        domain, *input_names = _NATURAL_WEAVE_REPLICATION_SCALAR_DOMAINS[domain_index]
    else:
        (
            domain,
            first_name,
            index_name,
            target_name,
            third_name,
            fourth_name,
            fifth_name,
            sixth_name,
        ) = _NATURAL_WEAVE_REPLICATION_SEQUENCE_DOMAINS[domain_index]
        second_name = index_name if schema_kind == "lookup_branch_weave_five" else target_name
        input_names = [
            first_name,
            second_name,
            third_name,
            fourth_name,
            fifth_name,
            sixth_name,
        ]
    (
        opening,
        first_intro,
        first_alias_clause,
        second_intro,
        second_alias_clause,
        extend_intro,
        extend_alias_clause,
        merge_intro,
        merge_alias_clause,
        terminal_intro,
        question,
        first_alias,
        second_alias,
        extended_alias,
        merge_alias,
    ) = _NATURAL_WEAVE_REPLICATION_SURFACES[sample_index % len(_NATURAL_WEAVE_REPLICATION_SURFACES)]
    builder = _AnnotatedText()
    builder.append(opening.format(domain=domain))
    builder.append(": inputs are ")
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
        builder.append(rendered, label=f"weave-replication:input:{index}")
    builder.append(f". {first_intro} ")

    if schema_kind == "scalar_branch_weave_five":
        _append_natural_binary_operation(
            builder,
            op=operations[0],
            ordinal=0,
            left_text=input_names[0],
            left_label="weave-replication:argument:0:0",
            right_text=input_names[1],
            right_label="weave-replication:argument:0:1",
        )
    elif schema_kind == "lookup_branch_weave_five":
        builder.append("retrieve", label="natural:operation:0")
        builder.append(" the entry at ")
        builder.append(input_names[1], label="weave-replication:argument:0:1")
        builder.append(" in ")
        builder.append(input_names[0], label="weave-replication:argument:0:0")
    elif schema_kind == "count_branch_weave_five":
        builder.append("count", label="natural:operation:0")
        builder.append(" occurrences of ")
        builder.append(input_names[1], label="weave-replication:argument:0:1")
        builder.append(" in ")
        builder.append(input_names[0], label="weave-replication:argument:0:0")
    else:  # pragma: no cover - builder owns the schema inventory
        raise ValueError("natural weave replication schema is unsupported")
    instructions = [
        SemanticInstructionAnnotation(
            instruction=Instruction(operations[0], (0, 1)),
            operation_span=builder.span("natural:operation:0"),
            argument_spans=tuple(
                builder.span(f"weave-replication:argument:0:{position}") for position in range(2)
            ),
            depends_on=(),
        )
    ]

    clauses = (
        (
            first_alias_clause,
            second_intro,
            operations[1],
            input_names[2],
            input_names[3],
            (2, 3),
            (),
        ),
        (
            second_alias_clause,
            extend_intro,
            operations[2],
            first_alias,
            input_names[4],
            (6, 4),
            (0,),
        ),
        (
            extend_alias_clause,
            merge_intro,
            operations[3],
            extended_alias,
            second_alias,
            (8, 7),
            (1, 2),
        ),
        (
            merge_alias_clause,
            terminal_intro,
            operations[4],
            merge_alias,
            input_names[5],
            (9, 5),
            (3,),
        ),
    )
    for ordinal, (
        preceding_alias_clause,
        intro,
        operation,
        left_text,
        right_text,
        arguments,
        dependencies,
    ) in enumerate(clauses, start=1):
        builder.append(f". {preceding_alias_clause}. {intro} ")
        _append_natural_binary_operation(
            builder,
            op=operation,
            ordinal=ordinal,
            left_text=left_text,
            left_label=f"weave-replication:argument:{ordinal}:0",
            right_text=right_text,
            right_label=f"weave-replication:argument:{ordinal}:1",
        )
        instructions.append(
            SemanticInstructionAnnotation(
                instruction=Instruction(operation, arguments),
                operation_span=builder.span(f"natural:operation:{ordinal}"),
                argument_spans=tuple(
                    builder.span(f"weave-replication:argument:{ordinal}:{position}")
                    for position in range(2)
                ),
                depends_on=dependencies,
            )
        )
    builder.append(f". {question}?")
    construction_id = f"natural-weave-replication-{schema_kind}-{domain_index}-{sample_index % 2}"
    identity = f"{construction_id}|{sample_index}|{inputs}|{operations}|{builder.text}"
    return SemanticProgramExample(
        example_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        construction_id=construction_id,
        topology_id=schema_kind,
        split="validation" if (domain_index + sample_index) % 2 == 0 else "test",
        source_text=builder.text,
        inputs=inputs,
        input_spans=tuple(builder.span(f"weave-replication:input:{index}") for index in range(6)),
        instructions=tuple(instructions),
        report_value=10,
        contrast_id=hashlib.sha256(
            f"natural-weave-replication|{schema_kind}|{domain_index}|{sample_index}".encode("ascii")
        ).hexdigest()[:24],
    )


def build_semantic_program_natural_weave_replication_corpus(
    *,
    seed: int = 3141592653,
    examples_per_schema_domain: int = 2,
) -> tuple[SemanticProgramExample, ...]:
    """Build the preregistered unseen six-input, five-step transfer corpus."""
    if examples_per_schema_domain < 1:
        raise ValueError("natural weave replication needs every schema-domain cell")
    rng = random.Random(seed)
    examples: list[SemanticProgramExample] = []
    schemas = (
        "scalar_branch_weave_five",
        "lookup_branch_weave_five",
        "count_branch_weave_five",
    )
    for schema_index, schema_kind in enumerate(schemas):
        for domain_index in range(len(NATURAL_WEAVE_REPLICATION_DOMAINS)):
            for sample_index in range(examples_per_schema_domain):
                operations = _NATURAL_WEAVE_CHAINS[
                    (schema_index * 5 + domain_index + sample_index) % len(_NATURAL_WEAVE_CHAINS)
                ]
                if schema_kind == "scalar_branch_weave_five":
                    inputs: tuple[SemanticValue, ...] = (
                        rng.randint(100_000, 900_000),
                        rng.randint(10_000, 90_000),
                        rng.randint(1_000, 9_000),
                        rng.randint(100, 900),
                        rng.randint(2, 97),
                        rng.randint(2, 31),
                    )
                else:
                    values = [rng.randint(100_000, 900_000) for _ in range(7)]
                    if schema_kind == "count_branch_weave_five":
                        wanted = rng.randint(100_000, 900_000)
                        values[0] = wanted
                        values[4] = wanted
                        first_op = "count_of"
                        second_input = wanted
                    else:
                        first_op = "at"
                        second_input = rng.randint(0, len(values) - 1)
                    inputs = (
                        tuple(values),
                        second_input,
                        rng.randint(1_000, 9_000),
                        rng.randint(100, 900),
                        rng.randint(2, 97),
                        rng.randint(2, 31),
                    )
                    operations = (
                        first_op,
                        operations[1],
                        operations[2],
                        operations[3],
                        operations[4],
                    )
                examples.append(
                    _natural_weave_replication_example(
                        schema_kind=schema_kind,
                        domain_index=domain_index,
                        sample_index=sample_index,
                        inputs=inputs,
                        operations=operations,
                    )
                )
    return tuple(examples)
