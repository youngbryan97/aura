from __future__ import annotations

from dataclasses import replace

from core.learning.semantic_program_evaluation import (
    coefficient_lesion,
    evaluate_semantic_program_transducer,
    label_permuted_training_examples,
    shuffle_hidden_tokens,
)
from core.learning.semantic_program_transducer import fit_semantic_program_transducer
from tests.test_semantic_program_transducer import _example, _training


def _battery():
    return [
        *_training(),
        *[
            _example(
                first,
                second,
                topology,
                split="test",
                order=(8, 6, 4, 2, 0, 7, 5, 3, 1),
            )
            for topology in range(4)
            for first in ("add", "sub", "mul", "idiv")
            for second in ("add", "sub", "mul", "idiv")
        ],
    ]


def test_evaluation_separates_semantics_from_source_attribution() -> None:
    examples = _battery()
    model = fit_semantic_program_transducer(examples)

    treatment = evaluate_semantic_program_transducer(model, examples, split="test")
    grounding_lesion = evaluate_semantic_program_transducer(
        model,
        examples,
        split="test",
        arm="token_binding_lesion",
        hidden_transform=shuffle_hidden_tokens,
    )

    assert treatment.program_exact == treatment.total == 64
    assert treatment.full_ir_exact == treatment.total
    assert treatment.answer_exact == treatment.total
    assert treatment.answer_emitted == treatment.total
    assert grounding_lesion.full_ir_exact < treatment.full_ir_exact


def test_coefficient_lesion_removes_the_learned_gain() -> None:
    examples = _battery()
    model = fit_semantic_program_transducer(examples)
    lesion = coefficient_lesion(model)

    result = evaluate_semantic_program_transducer(
        lesion,
        examples,
        split="test",
        arm="coefficient_lesion",
    )

    assert result.program_exact < 64
    assert result.full_ir_exact == 0


def test_label_permutation_changes_every_training_program() -> None:
    examples = _battery()
    permuted = label_permuted_training_examples(examples, seed=7)

    original_training = [item for item in examples if item.split == "train"]
    permuted_training = [item for item in permuted if item.split == "train"]
    assert len(original_training) == len(permuted_training)
    assert all(
        original.ir.to_program() != changed.ir.to_program()
        for original, changed in zip(original_training, permuted_training, strict=True)
    )
    assert [item for item in permuted if item.split == "test"] == [
        item for item in examples if item.split == "test"
    ]


def test_label_permutation_deranges_repeated_program_classes() -> None:
    distinct = []
    for item in _training():
        if all(item.ir.to_program() != seen.ir.to_program() for seen in distinct):
            distinct.append(item)
        if len(distinct) == 3:
            break
    repeated = [
        replace(item, construction_id=f"duplicate-{group}-{copy}")
        for group, item in enumerate(distinct)
        for copy in range(12)
    ]

    permuted = label_permuted_training_examples(repeated, seed=11)

    assert len(permuted) == len(repeated)
    assert all(
        original.ir.to_program() != changed.ir.to_program()
        for original, changed in zip(repeated, permuted, strict=True)
    )
