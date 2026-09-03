from __future__ import annotations

import pytest

from core.learning.semantic_program_corpus import (
    build_semantic_program_corpus,
    build_semantic_program_fork_join_corpus,
    build_semantic_program_fork_join_factorial_corpus,
    build_semantic_program_natural_request_corpus,
    build_semantic_program_natural_source_corpus,
    build_semantic_program_sequence_binary_corpus,
    build_semantic_program_sequence_cataphoric_corpus,
    build_semantic_program_sequence_corpus,
    build_semantic_program_sequence_reserved_alias_corpus,
    build_semantic_program_sequence_role_binding_corpus,
    project_example_to_ir,
    project_register_definition_spans,
)


def _character_offsets(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((index, index + 1) for index in range(len(text)))


def test_corpus_splits_hold_out_complete_constructions() -> None:
    examples = build_semantic_program_corpus(examples_per_operation_pair=2)
    constructions_by_split = {
        split: {item.construction_id for item in examples if item.split == split}
        for split in ("train", "validation", "test")
    }

    assert constructions_by_split == {
        "train": {
            "fronted_obtained_afterward",
            "nominal_nested",
            "reverse_intermediate_before",
            "sequential_intermediate_after",
            "sequential_result_then",
        },
        "validation": {
            "fronted_intermediate_then",
            "sequential_obtained_afterward",
        },
        "test": {"fronted_result_then", "reverse_result_prior"},
    }
    assert not (constructions_by_split["train"] & constructions_by_split["validation"])
    assert not constructions_by_split["train"] & constructions_by_split["test"]


def test_every_split_covers_all_operation_compositions() -> None:
    examples = build_semantic_program_corpus(examples_per_operation_pair=1)
    expected = {
        (first, second)
        for first in ("add", "sub", "mul", "idiv")
        for second in ("add", "sub", "mul", "idiv")
    }

    for split in ("train", "validation", "test"):
        observed = {
            (item.instructions[0].instruction.op, item.instructions[1].instruction.op)
            for item in examples
            if item.split == split
        }
        assert observed == expected


def test_every_split_covers_distinct_register_graphs() -> None:
    examples = build_semantic_program_corpus(examples_per_operation_pair=1)
    expected = {
        ((0, 1), (3, 2)),
        ((1, 2), (3, 0)),
        ((0, 2), (1, 3)),
        ((1, 2), (0, 3)),
    }

    for split in ("train", "validation", "test"):
        observed = {
            tuple(item.instruction.args for item in example.instructions)
            for example in examples
            if example.split == split
        }
        assert observed == expected


def test_program_first_examples_execute_the_annotated_program() -> None:
    examples = build_semantic_program_corpus(examples_per_operation_pair=2)

    for example in examples:
        assert isinstance(example.program.run(example.inputs), int)
        assert example.report_value == len(example.inputs) + len(example.instructions) - 1


def test_noncommutative_language_preserves_operand_order() -> None:
    examples = build_semantic_program_corpus(examples_per_operation_pair=1)

    sequential_sub = next(
        item
        for item in examples
        if item.construction_id == "sequential_result_then"
        and item.instructions[0].instruction.op == "sub"
        and item.instructions[1].instruction.op == "sub"
    )
    reverse_division = next(
        item
        for item in examples
        if item.construction_id == "reverse_result_prior"
        and item.instructions[1].instruction.op == "idiv"
    )

    assert "Subtract " in sequential_sub.source_text
    assert " from " in sequential_sub.source_text
    assert "subtract " in sequential_sub.source_text
    assert "integer-divide that result by" in reverse_division.source_text


def test_contrast_groups_change_operations_over_identical_inputs() -> None:
    examples = build_semantic_program_corpus(examples_per_operation_pair=2)
    groups: dict[str, list] = {}
    for example in examples:
        groups.setdefault(example.contrast_id, []).append(example)

    assert groups
    for group in groups.values():
        assert len(group) == 16
        assert len({item.inputs for item in group}) == 1
        assert len({item.program.describe() for item in group}) == 16


def test_character_attribution_projects_to_valid_token_ir() -> None:
    example = build_semantic_program_corpus(examples_per_operation_pair=1)[-1]
    offsets = _character_offsets(example.source_text)
    ir = project_example_to_ir(
        example,
        source_token_ids=tuple(range(len(offsets))),
        offset_mapping=offsets,
        model_basis_receipt_sha256="a" * 64,
        transducer_receipt_sha256="b" * 64,
    )

    assert ir.to_program() == example.program
    assert ir.lower(example.inputs).program_sha == example.program.sha()
    assert ir.receipt()["expected_answer_available"] is False


def test_projection_refuses_missing_or_noncontiguous_offsets() -> None:
    example = build_semantic_program_corpus(examples_per_operation_pair=1)[0]
    offsets = list(_character_offsets(example.source_text))

    with pytest.raises(ValueError, match="differ in length"):
        project_example_to_ir(
            example,
            source_token_ids=(1,),
            offset_mapping=offsets,
            model_basis_receipt_sha256="a" * 64,
            transducer_receipt_sha256="b" * 64,
        )

    first_input = example.input_spans[0]
    offsets[first_input.start] = (0, 0)
    with pytest.raises(ValueError, match="offsets"):
        project_example_to_ir(
            example,
            source_token_ids=tuple(range(len(offsets))),
            offset_mapping=offsets,
            model_basis_receipt_sha256="a" * 64,
            transducer_receipt_sha256="b" * 64,
        )


def test_corpus_is_deterministic_and_seeded() -> None:
    first = build_semantic_program_corpus(seed=9, examples_per_operation_pair=1)
    replay = build_semantic_program_corpus(seed=9, examples_per_operation_pair=1)
    changed = build_semantic_program_corpus(seed=10, examples_per_operation_pair=1)

    assert first == replay
    assert tuple(item.inputs for item in first) != tuple(item.inputs for item in changed)


def test_fork_join_corpus_holds_out_wording_and_graphs() -> None:
    examples = build_semantic_program_fork_join_corpus()
    constructions = {
        split: {item.construction_id for item in examples if item.split == split}
        for split in ("train", "validation", "test")
    }
    topologies = {
        split: {item.topology_id for item in examples if item.split == split}
        for split in ("train", "validation", "test")
    }

    assert len(examples) == 576
    assert {split: len(values) for split, values in constructions.items()} == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }
    assert all(
        not constructions[left] & constructions[right] and not topologies[left] & topologies[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    )


def test_fork_join_corpus_covers_every_operation_triple_per_split() -> None:
    examples = build_semantic_program_fork_join_corpus()
    expected = {
        (first, second, third)
        for first in ("add", "sub", "mul", "idiv")
        for second in ("add", "sub", "mul", "idiv")
        for third in ("add", "sub", "mul", "idiv")
    }

    for split in ("train", "validation", "test"):
        observed = {
            tuple(item.instruction.op for item in example.instructions)
            for example in examples
            if example.split == split
        }
        assert observed == expected


def test_fork_join_programs_are_exact_and_all_steps_are_load_bearing() -> None:
    examples = build_semantic_program_fork_join_corpus(source_order_registers=True)

    for example in examples:
        assert len(example.inputs) == 4
        assert len(example.instructions) == 3
        assert list(example.input_spans) == sorted(
            example.input_spans,
            key=lambda span: span.start,
        )
        assert example.instructions[0].depends_on == ()
        assert example.instructions[1].depends_on == ()
        assert example.instructions[2].depends_on == (0, 1)
        for instruction in example.instructions[:2]:
            for register, span in zip(
                instruction.instruction.args,
                instruction.argument_spans,
                strict=True,
            ):
                assert example.input_spans[register] == span
        assert isinstance(example.program.run(example.inputs), int)
        offsets = _character_offsets(example.source_text)
        ir = project_example_to_ir(
            example,
            source_token_ids=tuple(range(len(offsets))),
            offset_mapping=offsets,
            model_basis_receipt_sha256="a" * 64,
            transducer_receipt_sha256="b" * 64,
        )
        assert ir.to_program() == example.program
        assert ir.receipt()["all_steps_causally_load_bearing"] is True


def test_factorial_fork_join_separates_wording_from_graph_topology() -> None:
    examples = build_semantic_program_fork_join_factorial_corpus()

    assert len(examples) == 1296
    assert {
        split: sum(item.split == split for item in examples)
        for split in (
            "train",
            "validation",
            "test",
        )
    } == {"train": 432, "validation": 432, "test": 432}
    for construction in {item.construction_id for item in examples}:
        selected = [item for item in examples if item.construction_id == construction]
        assert len({item.topology_id for item in selected}) == 9
        for position in range(3):
            support = {item.instructions[position].instruction.op for item in selected}
            assert support == {"add", "sub", "mul", "idiv"}
    assert all(
        list(item.input_spans) == sorted(item.input_spans, key=lambda span: span.start)
        for item in examples
    )


def test_sequence_corpus_holds_out_complete_constructions() -> None:
    examples = build_semantic_program_sequence_corpus()
    constructions = {
        split: {item.construction_id for item in examples if item.split == split}
        for split in ("train", "validation", "test")
    }

    assert len(examples) == 540
    assert {split: sum(item.split == split for item in examples) for split in constructions} == {
        "train": 180,
        "validation": 180,
        "test": 180,
    }
    assert all(
        not constructions[left] & constructions[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    )


def test_sequence_corpus_covers_every_typed_composition_per_split() -> None:
    examples = build_semantic_program_sequence_corpus(examples_per_operation_pair=1)
    expected = {
        (first, second)
        for first in ("unique", "sorted_up", "reversed_", "tail", "front")
        for second in ("length", "total", "largest", "smallest", "head", "last")
    }

    for split in ("train", "validation", "test"):
        observed = {
            tuple(item.instruction.op for item in example.instructions)
            for example in examples
            if example.split == split
        }
        assert observed == expected


def test_sequence_programs_are_exact_and_both_steps_are_load_bearing() -> None:
    examples = build_semantic_program_sequence_corpus()

    assert len({example.example_id for example in examples}) == len(examples)
    for example in examples:
        assert len(example.inputs) == 1
        assert isinstance(example.inputs[0], tuple)
        assert len(example.inputs[0]) >= 5
        assert len(set(example.inputs[0])) < len(example.inputs[0])
        assert len(example.instructions) == 2
        assert tuple(len(item.instruction.args) for item in example.instructions) == (1, 1)
        assert example.instructions[0].depends_on == ()
        assert example.instructions[1].depends_on == (0,)
        assert example.instructions[0].instruction.args == (0,)
        assert example.instructions[1].instruction.args == (1,)
        assert type(example.program.run(example.inputs)) is int
        assert (
            example.instructions[0].operation_span.start
            < example.instructions[1].argument_spans[0].start
        )
        for instruction in example.instructions:
            assert example.source_text[
                instruction.operation_span.start : instruction.operation_span.end
            ]
        offsets = _character_offsets(example.source_text)
        ir = project_example_to_ir(
            example,
            source_token_ids=tuple(range(len(offsets))),
            offset_mapping=offsets,
            model_basis_receipt_sha256="a" * 64,
            transducer_receipt_sha256="b" * 64,
        )
        assert ir.to_program() == example.program
        assert ir.receipt()["all_steps_causally_load_bearing"] is True


def test_sequence_binary_corpus_holds_out_constructions_and_covers_operations() -> None:
    examples = build_semantic_program_sequence_binary_corpus(examples_per_operation_pair=2)
    constructions = {
        split: {item.construction_id for item in examples if item.split == split}
        for split in ("train", "validation", "test")
    }
    expected_operations = {
        (first, second) for first in ("at", "count_of") for second in ("add", "sub", "mul", "idiv")
    }

    assert len(examples) == 144
    assert {split: len(values) for split, values in constructions.items()} == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }
    assert all(
        not constructions[left] & constructions[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    )
    for split in constructions:
        assert {
            tuple(item.instruction.op for item in example.instructions)
            for example in examples
            if example.split == split
        } == expected_operations


def test_sequence_binary_programs_preserve_mixed_value_and_pointer_semantics() -> None:
    examples = build_semantic_program_sequence_binary_corpus()

    assert len({example.example_id for example in examples}) == len(examples)
    for example in examples:
        sequence, selector, adjustment = example.inputs
        assert isinstance(sequence, tuple)
        assert selector in sequence
        assert 0 <= selector < len(sequence)
        assert type(adjustment) is int and adjustment > 0
        assert tuple(item.instruction.args for item in example.instructions) == (
            (0, 1),
            (3, 2),
        )
        assert example.instructions[0].depends_on == ()
        assert example.instructions[1].depends_on == (0,)
        assert type(example.program.run(example.inputs)) is int
        ir = project_example_to_ir(
            example,
            source_token_ids=tuple(range(len(example.source_text))),
            offset_mapping=_character_offsets(example.source_text),
            model_basis_receipt_sha256="a" * 64,
            transducer_receipt_sha256="b" * 64,
        )
        assert ir.to_program() == example.program


def test_sequence_binary_corpus_is_deterministic_and_seeded() -> None:
    first = build_semantic_program_sequence_binary_corpus(seed=11)
    replay = build_semantic_program_sequence_binary_corpus(seed=11)
    changed = build_semantic_program_sequence_binary_corpus(seed=12)

    assert first == replay
    assert tuple(item.inputs for item in first) != tuple(item.inputs for item in changed)


def test_sequence_cataphoric_corpus_separates_text_order_from_execution_order() -> None:
    examples = build_semantic_program_sequence_cataphoric_corpus(examples_per_operation_pair=2)

    assert len(examples) == 144
    assert {
        split: sum(item.split == split for item in examples)
        for split in ("train", "validation", "test")
    } == {"train": 48, "validation": 48, "test": 48}
    assert {
        len({item.construction_id for item in examples if item.split == split})
        for split in ("train", "validation", "test")
    } == {3}
    for example in examples:
        first, second = example.instructions
        assert second.operation_span.end <= first.operation_span.start
        assert first.instruction.args == (0, 1)
        assert second.instruction.args in {(3, 2), (2, 3)}
        assert second.depends_on == (0,)
        assert example.topology_id == "cataphoric-sequence-to-scalar-chain"
        assert type(example.program.run(example.inputs)) is int
        ir = project_example_to_ir(
            example,
            source_token_ids=tuple(range(len(example.source_text))),
            offset_mapping=_character_offsets(example.source_text),
            model_basis_receipt_sha256="a" * 64,
            transducer_receipt_sha256="b" * 64,
        )
        assert ir.to_program() == example.program


def test_sequence_cataphoric_corpus_is_deterministic_and_seeded() -> None:
    first = build_semantic_program_sequence_cataphoric_corpus(seed=19)
    replay = build_semantic_program_sequence_cataphoric_corpus(seed=19)
    changed = build_semantic_program_sequence_cataphoric_corpus(seed=20)

    assert first == replay
    assert tuple(item.inputs for item in first) != tuple(item.inputs for item in changed)


def test_sequence_reserved_alias_corpus_teaches_input_aliases_without_arithmetic_topology() -> None:
    examples = build_semantic_program_sequence_reserved_alias_corpus(examples_per_operation_pair=2)
    constructions = {
        split: {item.construction_id for item in examples if item.split == split}
        for split in ("train", "validation", "test")
    }
    expected_operations = {
        (first, second) for first in ("at", "count_of") for second in ("add", "sub", "mul", "idiv")
    }

    assert len(examples) == 144
    assert {split: len(values) for split, values in constructions.items()} == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }
    assert all(
        not constructions[left] & constructions[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    )
    for split in constructions:
        assert {
            tuple(item.instruction.op for item in example.instructions)
            for example in examples
            if example.split == split
        } == expected_operations
    for example in examples:
        first, second = example.instructions
        assert first.instruction.args == (0, 1)
        assert second.instruction.args in {(3, 2), (2, 3)}
        reserved_position = second.instruction.args.index(2)
        reserved_reference = second.argument_spans[reserved_position]
        assert reserved_reference != example.input_spans[2]
        assert not (
            reserved_reference.start < example.input_spans[2].end
            and example.input_spans[2].start < reserved_reference.end
        )
        assert second.depends_on == (0,)
        assert type(example.program.run(example.inputs)) is int
        ir = project_example_to_ir(
            example,
            source_token_ids=tuple(range(len(example.source_text))),
            offset_mapping=_character_offsets(example.source_text),
            model_basis_receipt_sha256="a" * 64,
            transducer_receipt_sha256="b" * 64,
        )
        assert ir.to_program() == example.program


def test_sequence_reserved_alias_corpus_is_deterministic_and_seeded() -> None:
    first = build_semantic_program_sequence_reserved_alias_corpus(seed=17)
    replay = build_semantic_program_sequence_reserved_alias_corpus(seed=17)
    changed = build_semantic_program_sequence_reserved_alias_corpus(seed=18)

    assert first == replay
    assert tuple(item.inputs for item in first) != tuple(item.inputs for item in changed)


def test_sequence_role_binding_corpus_teaches_implicit_input_roles() -> None:
    examples = build_semantic_program_sequence_role_binding_corpus(examples_per_operation_pair=2)

    assert len(examples) == 144
    assert {
        split: len({item.construction_id for item in examples if item.split == split})
        for split in ("train", "validation", "test")
    } == {"train": 3, "validation": 3, "test": 3}
    for example in examples:
        first, second = example.instructions
        assert example.topology_id == "role-bound-input-sequence-to-scalar-chain"
        assert first.instruction.args == (0, 1)
        assert second.instruction.args in {(3, 2), (2, 3)}
        reserved_position = second.instruction.args.index(2)
        assert second.argument_spans[reserved_position] != example.input_spans[2]
        assert "reserved operand" not in example.source_text.casefold()
        assert "earlier operand" not in example.source_text.casefold()
        assert len(example.register_definition_spans) == 5
        reserved_definition = example.register_definition_spans[2]
        assert reserved_definition.start == example.input_spans[2].start
        assert reserved_definition.end > example.input_spans[2].end
        assert " as " in example.source_text[reserved_definition.start : reserved_definition.end]
        assert type(example.program.run(example.inputs)) is int
        ir = project_example_to_ir(
            example,
            source_token_ids=tuple(range(len(example.source_text))),
            offset_mapping=_character_offsets(example.source_text),
            model_basis_receipt_sha256="a" * 64,
            transducer_receipt_sha256="b" * 64,
        )
        assert ir.to_program() == example.program
        definitions = project_register_definition_spans(
            example,
            offset_mapping=_character_offsets(example.source_text),
        )
        assert definitions[2].start == ir.input_spans[2].start
        assert definitions[2].end > ir.input_spans[2].end


def test_sequence_role_binding_corpus_is_deterministic_and_seeded() -> None:
    first = build_semantic_program_sequence_role_binding_corpus(seed=23)
    replay = build_semantic_program_sequence_role_binding_corpus(seed=23)
    changed = build_semantic_program_sequence_role_binding_corpus(seed=24)

    assert first == replay
    assert tuple(item.inputs for item in first) != tuple(item.inputs for item in changed)


def test_natural_request_corpus_withholds_three_complete_linear_schemas() -> None:
    examples = build_semantic_program_natural_request_corpus()

    assert len(examples) == 24
    assert {item.topology_id for item in examples} == {
        "scalar_linear_three",
        "lookup_linear_three",
        "count_linear_three",
    }
    assert {item.split for item in examples} == {"validation", "test"}
    assert all(len(item.inputs) == 4 and len(item.instructions) == 3 for item in examples)
    assert all(
        tuple(step.instruction.args for step in item.instructions) == ((0, 1), (4, 2), (5, 3))
        for item in examples
    )
    assert all(item.program.run(item.inputs) is not None for item in examples)


def test_natural_request_corpus_varies_domain_independently_of_schema() -> None:
    examples = build_semantic_program_natural_request_corpus()
    by_schema = {
        schema: [item for item in examples if item.topology_id == schema]
        for schema in {item.topology_id for item in examples}
    }

    assert all(len(items) == 8 for items in by_schema.values())
    assert all(len({item.construction_id for item in items}) == 8 for items in by_schema.values())
    assert len({item.source_text for item in examples}) == len(examples)
    assert build_semantic_program_natural_request_corpus() == examples
    assert build_semantic_program_natural_request_corpus(seed=3141593) != examples


def test_natural_source_teaches_shallow_relations_outside_target_domains() -> None:
    source = build_semantic_program_natural_source_corpus()
    target = build_semantic_program_natural_request_corpus()

    assert len(source) == 24
    assert {item.topology_id for item in source} == {
        "scalar_linear_two",
        "lookup_linear_two",
        "count_linear_two",
    }
    assert {
        split: sum(item.split == split for item in source)
        for split in ("train", "validation", "test")
    } == {"train": 12, "validation": 6, "test": 6}
    assert all(len(item.inputs) == 3 and len(item.instructions) == 2 for item in source)
    assert all(len(item.register_definition_spans) == 5 for item in source)
    assert all(
        tuple(step.instruction.args for step in item.instructions) == ((0, 1), (3, 2))
        for item in source
    )
    assert not {item.source_text.split(",", 1)[0] for item in source} & {
        item.source_text.split(",", 1)[0] for item in target
    }
    assert build_semantic_program_natural_source_corpus() == source
    assert build_semantic_program_natural_source_corpus(seed=2718282) != source


def test_natural_definition_envelopes_are_runtime_representable() -> None:
    examples = (
        *build_semantic_program_natural_source_corpus(),
        *build_semantic_program_natural_request_corpus(),
    )

    for example in examples:
        for index, input_span in enumerate(example.input_spans):
            definition = example.register_definition_spans[index]
            assert definition.start <= input_span.start
            assert definition.end == input_span.end
        for step, instruction in enumerate(example.instructions):
            definition = example.register_definition_spans[len(example.inputs) + step]
            assert definition.start == instruction.operation_span.start
            assert definition.end >= instruction.operation_span.end
