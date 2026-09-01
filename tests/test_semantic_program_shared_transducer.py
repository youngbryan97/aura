from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from core.learning.semantic_input_grounding import (
    SemanticInputGroundingContract,
    SequenceTokenFormat,
    semantic_input_grounding_contract_from_dict,
    semantic_input_grounding_contract_from_tokenizer,
)
from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    TokenSpan,
)
from core.learning.semantic_program_shared_campaign import (
    run_shared_semantic_program_campaign_from_examples,
)
from core.learning.semantic_program_shared_evaluation import (
    evaluate_shared_semantic_program_transducer,
)
from core.learning.semantic_program_shared_transducer import (
    _POINTER_HARD_NEGATIVES,
    _pointer_training_indices,
    fit_shared_semantic_program_transducer,
    shared_semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_transducer import (
    LinearPointerHead,
    SemanticTransducerTrainingExample,
)

_BASIS = "a" * 64
_TOKENIZER_IDENTITY = "c" * 64
_MAX_ROLES = (
    *(f"input:{index}" for index in range(4)),
    *(f"operation:{step}" for step in range(3)),
    *(f"argument:{step}:{position}" for step in range(3) for position in range(2)),
)
_OPS = ("add", "sub")
_REGISTER_LABELS = (
    "input:0",
    "input:1",
    "input:2",
    "input:3",
    "result:0",
    "result:1",
    "result:2",
)


class _CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)


def _grounding() -> SemanticInputGroundingContract:
    return SemanticInputGroundingContract(
        tokenizer_identity_sha256=_TOKENIZER_IDENTITY,
        digit_token_ids=tuple(range(10, 20)),
        positive_integer_prefixes=((),),
        negative_integer_prefixes=((99,),),
        integer_suffixes=((),),
        empty_sequence_variants=((91, 93),),
        sequence_formats=(
            SequenceTokenFormat(
                positive_prefix=(91,),
                negative_prefix=(91, 99),
                positive_separator=(44,),
                negative_separator=(44, 99),
                suffixes=((93,),),
                singleton_suffixes=((93,),),
            ),
        ),
    )


def _shared_example(
    *,
    three_steps: bool,
    variant: int,
    split: str,
) -> SemanticTransducerTrainingExample:
    input_count = 4 if three_steps else 3
    arguments = ((0, 1), (4, 2), (5, 3)) if three_steps else ((0, 1), (3, 2))
    operations = tuple(_OPS[(variant + step) % len(_OPS)] for step in range(len(arguments)))
    roles = (
        *(f"input:{index}" for index in range(input_count)),
        *(f"operation:{step}" for step in range(len(arguments))),
        *(f"argument:{step}:{position}" for step in range(len(arguments)) for position in range(2)),
    )
    shift = variant % 3
    positions = {
        role: TokenSpan(
            2 + ((2 * index + shift) % (2 * len(roles))),
            3 + ((2 * index + shift) % (2 * len(roles))),
        )
        for index, role in enumerate(roles)
    }
    instructions = tuple(
        SemanticIRInstruction(
            op=operations[step],
            args=args,
            operation_span=positions[f"operation:{step}"],
            argument_spans=tuple(positions[f"argument:{step}:{position}"] for position in range(2)),
            depends_on=tuple(
                sorted(argument - input_count for argument in set(args) if argument >= input_count)
            ),
        )
        for step, args in enumerate(arguments)
    )
    token_count = 2 * len(_MAX_ROLES) + 4
    source_token_ids = list(range(500, 500 + token_count))
    public_inputs = (9, 4, 2, 1) if three_steps else (9, 4, 2)
    grounding = _grounding()
    for index, value in enumerate(public_inputs):
        source_token_ids[positions[f"input:{index}"].start] = grounding.digit_token_ids[value]
    ir = SemanticProgramIR(
        source_token_ids=tuple(source_token_ids),
        source_text_sha256=hashlib.sha256(f"{three_steps}:{variant}:{split}".encode()).hexdigest(),
        input_spans=tuple(positions[f"input:{index}"] for index in range(input_count)),
        instructions=instructions,
        report_value=input_count + len(instructions) - 1,
        model_basis_receipt_sha256=_BASIS,
        transducer_receipt_sha256="b" * 64,
    )
    width = len(_MAX_ROLES) + len(_OPS) + len(_REGISTER_LABELS) + 2 + 1
    hidden = np.zeros((token_count, width), dtype=np.float32)
    hidden[:, -3 + int(three_steps)] = 1.0
    hidden[:, -1] = 0.5
    for role in roles:
        position = positions[role].start
        hidden[position, _MAX_ROLES.index(role)] = 4.0
    for step, operation in enumerate(operations):
        position = positions[f"operation:{step}"].start
        hidden[position, len(_MAX_ROLES) + _OPS.index(operation)] = 4.0
    register_offset = len(_MAX_ROLES) + len(_OPS)
    for input_index in range(input_count):
        position = positions[f"input:{input_index}"].start
        register = _REGISTER_LABELS.index(f"input:{input_index}")
        hidden[position, register_offset + register] = 4.0
    for step in range(len(arguments)):
        position = positions[f"operation:{step}"].start
        register = _REGISTER_LABELS.index(f"result:{step}")
        hidden[position, register_offset + register] = 4.0
    for step, args in enumerate(arguments):
        for argument_position, register in enumerate(args):
            role = (
                f"input:{register}"
                if register < input_count
                else f"result:{register - input_count}"
            )
            position = positions[f"argument:{step}:{argument_position}"].start
            hidden[position, register_offset + _REGISTER_LABELS.index(role)] = 4.0
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    packed_hidden = np.concatenate((hidden, hidden, hidden), axis=1) / np.sqrt(3.0)
    return SemanticTransducerTrainingExample(
        ir=ir,
        hidden_states=packed_hidden,
        split=split,
        construction_id=f"shared-{int(three_steps)}-{variant}",
        topology_id=f"chain-{len(arguments)}",
        public_inputs=public_inputs,
        hidden_channels=(
            "input_token_embedding",
            "middle_causal_hidden",
            "final_causal_hidden",
        ),
        hidden_channel_widths=(width, width, width),
        tokenizer_identity_sha256=_TOKENIZER_IDENTITY,
    )


def _examples() -> tuple[SemanticTransducerTrainingExample, ...]:
    return tuple(
        _shared_example(three_steps=three_steps, variant=variant, split=split)
        for three_steps in (False, True)
        for split, variants in (
            ("train", range(8)),
            ("validation", range(8, 10)),
            ("test", range(10, 12)),
        )
        for variant in variants
    )


def test_shared_transducer_infers_geometry_and_program_without_a_router() -> None:
    examples = _examples()
    model = fit_shared_semantic_program_transducer(examples, input_grounding=_grounding())
    replay = shared_semantic_program_transducer_from_dict(model.to_dict())

    result = evaluate_shared_semantic_program_transducer(
        replay,
        examples,
        split="test",
    )

    assert result.total == 4
    assert result.geometry_exact == result.total
    assert result.program_exact == result.total
    assert result.answer_exact == result.total
    assert model.geometry_contract["geometry_labels_available_to_decode"] is False
    assert model.geometry_contract["register_encoding"] == "role_relative_v1"
    assert model.geometry_contract["max_span_tokens"] >= 1
    assert replay.to_dict() == model.to_dict()


def test_shared_pointer_can_use_a_contract_span_wider_than_the_legacy_limit() -> None:
    hidden = np.zeros((32, 3), dtype=np.float32)
    hidden[:, 0] = 1.0
    pointer = LinearPointerHead(np.zeros(3), 0.0, np.zeros(3), 0.0)

    legacy = pointer.decode_candidates(hidden, limit=1024)
    widened = pointer.decode_candidates(hidden, limit=1024, max_span_tokens=32)

    assert TokenSpan(0, 32) not in {span for span, _score in legacy}
    assert TokenSpan(0, 32) in {span for span, _score in widened}


def test_shared_transducer_lesion_removes_the_geometry_program_gain() -> None:
    examples = _examples()
    model = fit_shared_semantic_program_transducer(examples, input_grounding=_grounding())

    treatment = evaluate_shared_semantic_program_transducer(
        model,
        examples,
        split="test",
    )
    lesion = evaluate_shared_semantic_program_transducer(
        model.coefficient_lesion(),
        examples,
        split="test",
        arm="coefficient_lesion",
    )

    assert treatment.program_exact == treatment.total
    assert lesion.program_exact < treatment.program_exact


def test_shared_pointer_fit_is_bounded_and_keeps_the_positive_boundary() -> None:
    example = _shared_example(three_steps=True, variant=0, split="train")
    role = "argument:1:0"
    span = example.ir.instructions[1].argument_spans[0]

    for end, expected in ((False, span.start), (True, span.end - 1)):
        indices = _pointer_training_indices(example, role, end=end)
        assert indices[0] == expected
        assert expected not in indices[1:]
        assert len(indices[1:]) <= _POINTER_HARD_NEGATIVES
        assert len(indices) == len(set(indices))


def test_tokenizer_grounding_round_trips_signed_scalars_and_sequences() -> None:
    tokenizer = _CharacterTokenizer()
    contract = semantic_input_grounding_contract_from_tokenizer(
        tokenizer,
        tokenizer_identity_sha256=_TOKENIZER_IDENTITY,
    )
    replay = semantic_input_grounding_contract_from_dict(contract.to_dict())
    text = "values: -12, [3, -14, 5], (), and 57."
    tokens = tuple(tokenizer.encode(text))

    for value in (-12, (3, -14, 5), (), 57):
        spans = replay.candidate_spans(tokens, value)
        assert spans
        expected = {str(value)}
        if isinstance(value, tuple):
            expected.add(str(list(value)))
        assert any(
            tokenizer.decode(list(tokens[span.start : span.end])).strip(" ,.")
            in expected
            for span in spans
        )
    assert replay.to_dict() == contract.to_dict()


def test_shared_fit_rejects_a_tokenizer_basis_mismatch() -> None:
    examples = list(_examples())
    item = examples[-1]
    examples[-1] = SemanticTransducerTrainingExample(
        ir=item.ir,
        hidden_states=item.hidden_states,
        split=item.split,
        construction_id=item.construction_id,
        topology_id=item.topology_id,
        public_inputs=item.public_inputs,
        hidden_channels=item.hidden_channels,
        hidden_channel_widths=item.hidden_channel_widths,
        tokenizer_identity_sha256="d" * 64,
    )

    with np.testing.assert_raises_regex(ValueError, "neural bases differ"):
        fit_shared_semantic_program_transducer(
            examples,
            input_grounding=_grounding(),
        )


def test_shared_campaign_records_family_controls_without_a_router(monkeypatch) -> None:
    examples = _examples()
    by_family = {
        "short": tuple(item for item in examples if item.ir.n_inputs == 3),
        "long": tuple(item for item in examples if item.ir.n_inputs == 4),
    }
    compatibility = {
        "target_training_session_basis_sha256": _BASIS,
        "receipt_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        "core.learning.semantic_program_shared_campaign."
        "establish_semantic_training_representation_compatibility",
        lambda _manifests: compatibility,
    )
    def bind(grouped, *, compatibility):
        assert compatibility["target_training_session_basis_sha256"] == _BASIS
        return tuple(
            replace(item, construction_id=f"{family}:{item.construction_id}")
            for family in sorted(grouped)
            for item in grouped[family]
        )

    monkeypatch.setattr(
        "core.learning.semantic_program_shared_campaign."
        "bind_training_examples_to_shared_representation",
        bind,
    )

    result = run_shared_semantic_program_campaign_from_examples(
        by_family,
        manifests={
            "short": {"manifest_sha256": "1" * 64},
            "long": {"manifest_sha256": "2" * 64},
        },
        input_grounding=_grounding(),
    )

    assert result.report["shared_model_count"] == 1
    assert result.report["family_router_present"] is False
    assert set(result.report["families"]) == {"short", "long"}
    assert result.report["arms"]["treatment:test"]["program_exact"] == 4
    assert result.report["arms"]["coefficient_lesion:test"]["program_exact"] < 4
    assert result.report["paired_program_controls"][
        "coefficient_lesion:test"
    ]["treatment_only"] > 0
