from __future__ import annotations

import copy
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from core.learning.semantic_input_grounding import (
    SemanticInputGroundingContract,
    SequenceTokenFormat,
    semantic_input_grounding_contract_from_dict,
    semantic_input_grounding_contract_from_tokenizer,
)
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_compositional_campaign import (
    diagnose_compositional_definition_relations,
    diagnose_compositional_transfer_lesions,
)
from core.learning.semantic_program_compositional_transducer import (
    _LOCAL_DEFINITION_CANDIDATE_STRATEGY,
    DirectionalRelationHead,
    _best_penalized_operation_chart,
    _definition_span_candidates,
    _directional_relation_feature,
    _mention_invariant_relation_evidence,
    _operation_chart,
    _operation_chart_candidates,
    _operation_order,
    _OperationNode,
    _register_definition_candidates,
    compositional_semantic_program_transducer_from_dict,
    fit_compositional_semantic_program_transducer,
)
from core.learning.semantic_program_compositional_transducer import (
    _POINTER_HARD_NEGATIVES as _COMPOSITIONAL_POINTER_HARD_NEGATIVES,
)
from core.learning.semantic_program_floor_verification import (
    SEMANTIC_PROGRAM_FLOOR_VERIFICATION_SOURCES,
    verify_semantic_program_floor_equivalence,
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
from core.learning.semantic_program_shared_verification import (
    SEMANTIC_PROGRAM_SHARED_VERIFICATION_SOURCES,
    verify_shared_semantic_program_campaign,
)
from core.learning.semantic_program_transducer import (
    LinearPointerHead,
    LinearPointerSequenceScores,
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


def test_compositional_transducer_assembles_typed_atoms_without_a_geometry_head() -> None:
    examples = _examples()
    model = fit_compositional_semantic_program_transducer(
        examples,
        input_grounding=_grounding(),
    )
    replay = compositional_semantic_program_transducer_from_dict(model.to_dict())

    treatment = evaluate_shared_semantic_program_transducer(
        replay,
        examples,
        split="test",
    )
    coefficient = evaluate_shared_semantic_program_transducer(
        replay.coefficient_lesion(),
        examples,
        split="test",
        arm="coefficient_lesion",
    )
    dependency = evaluate_shared_semantic_program_transducer(
        replay.dependency_lesion(),
        examples,
        split="test",
        arm="dependency_lesion",
    )
    register_use = evaluate_shared_semantic_program_transducer(
        replay.register_use_lesion(),
        examples,
        split="test",
        arm="register_use_lesion",
    )

    assert treatment.total == 4
    assert treatment.geometry_exact == treatment.total
    assert treatment.operation_exact == treatment.total
    assert treatment.program_exact >= 2
    assert treatment.answer_exact >= 2
    assert coefficient.program_exact < treatment.program_exact
    assert dependency.program_exact < treatment.program_exact
    assert register_use.total == treatment.total
    assert register_use.arm == "register_use_lesion"
    assert model.training_receipt["global_geometry_classifier_present"] is False
    assert model.training_receipt["step_indexed_heads_present"] is False
    assert model.training_receipt["family_router_present"] is False
    assert (
        model.training_receipt["relation_score_contract"]
        == "mention_invariant_conditional_tissue_v1"
    )
    proposal_fit = model.training_receipt["argument_proposal_fit"]
    assert (
        model.training_receipt["argument_role_contract"]
        == "semantic_and_pointer_proposal_product_v1"
    )
    assert proposal_fit["positive_rows"] > 0
    assert proposal_fit["pointer_hard_negative_rows"] > 0
    assert proposal_fit["hard_negative_limit"] == _COMPOSITIONAL_POINTER_HARD_NEGATIVES
    selected_proposal_scales = [row for row in proposal_fit["scale_selection"] if row["selected"]]
    assert len(selected_proposal_scales) == 1
    assert selected_proposal_scales[0]["proposal_scale"] == model.argument_proposal_scale
    assert model.register_use_contract.to_dict() == {
        "input_min_uses": 1,
        "input_max_uses": 1,
        "intermediate_min_uses": 1,
        "intermediate_max_uses": 1,
        "distinct_arguments": True,
    }
    assert model.training_receipt["definition_pointer_scale_selection"]
    assert replay.chart_beam_lesion().operation_chart_beam == 1
    assert (
        np.count_nonzero(replay.relation_tissue_lesion().definition_relation_head.query_projection)
        == 0
    )
    proposal_lesion = replay.argument_proposal_lesion()
    assert proposal_lesion.argument_proposal_scale == replay.argument_proposal_scale
    assert all(
        np.count_nonzero(head.weight) == 0 and head.bias == 0.0
        for head in proposal_lesion.argument_proposal_heads
    )
    assert replay.to_dict() == model.to_dict()


def test_compositional_transducer_binds_learned_type_limits_to_its_receipt() -> None:
    model = fit_compositional_semantic_program_transducer(
        _examples(),
        input_grounding=_grounding(),
    )
    payload = copy.deepcopy(model.to_dict())
    payload["max_argument_span_tokens_by_type"]["integer"] += 1

    with pytest.raises(ValueError, match="envelope"):
        compositional_semantic_program_transducer_from_dict(payload)

    payload = copy.deepcopy(model.to_dict())
    payload["register_use_contract"]["input_max_uses"] += 1

    with pytest.raises(ValueError, match="envelope"):
        compositional_semantic_program_transducer_from_dict(payload)

    payload = copy.deepcopy(model.to_dict())
    payload["operation_chart_beam"] += 1

    with pytest.raises(ValueError, match="envelope"):
        compositional_semantic_program_transducer_from_dict(payload)


def test_compositional_transducer_replays_the_frozen_v13_candidate_geometry() -> None:
    model = fit_compositional_semantic_program_transducer(
        _examples(),
        input_grounding=_grounding(),
    )
    payload = copy.deepcopy(model.to_dict())
    payload["schema"] = "aura.semantic_program_transducer.v13"
    payload.pop("definition_candidate_strategy")
    receipt = payload["training_receipt"]
    receipt["schema"] = "aura.semantic_program_transducer_receipt.v13"
    receipt.pop("definition_candidate_strategy")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = _sha(body)

    replay = compositional_semantic_program_transducer_from_dict(payload)

    assert replay.definition_candidate_strategy == "anchored_envelope_v1"
    assert replay.to_dict() == payload


def test_compositional_graph_orders_dependencies_instead_of_prose_position() -> None:
    nodes = (
        _OperationNode(TokenSpan(2, 3), "add", 1.0, 1.0, 1.0),
        _OperationNode(TokenSpan(10, 11), "mul", 1.0, 1.0, 1.0),
        _OperationNode(TokenSpan(14, 15), "sub", 1.0, 1.0, 1.0),
    )

    assert _operation_order(((1,), (), (0, 1)), nodes, require_connected=True) == (
        1,
        0,
        2,
    )
    assert _operation_order(((1,),), nodes, require_connected=False) == (0,)
    assert _operation_order(((1,), (0,)), nodes, require_connected=False) is None
    assert _operation_order(((), (), ()), nodes, require_connected=True) is None


def test_compositional_relation_preserves_reference_direction() -> None:
    reference = np.asarray([1.0, 0.0], dtype=np.float32)
    definition = np.asarray([0.0, 1.0], dtype=np.float32)

    forward = _directional_relation_feature(reference, definition)
    reverse = _directional_relation_feature(definition, reference)

    np.testing.assert_array_equal(forward[:4], reverse[:4])
    np.testing.assert_array_equal(forward[4:], -reverse[4:])


def test_local_definition_candidates_stay_inside_their_operation_clause() -> None:
    anchors = (TokenSpan(1, 2), TokenSpan(5, 6), TokenSpan(12, 13))
    pointer = LinearPointerSequenceScores(
        np.zeros(20, dtype=np.float32),
        np.zeros(20, dtype=np.float32),
    )

    candidates = _register_definition_candidates(
        anchors,
        input_count=1,
        token_count=20,
        max_span_tokens=12,
        pointer_scores=pointer,
        strategy=_LOCAL_DEFINITION_CANDIDATE_STRATEGY,
    )

    assert any(span.start >= anchors[1].end for span in candidates[1])
    assert all(span.end <= anchors[2].start for span in candidates[1])
    assert all(span.start == anchors[1].start for span in candidates[1][:1])


def test_compositional_relation_tissue_scores_cross_feature_identity() -> None:
    head = DirectionalRelationHead(
        np.zeros(6, dtype=np.float32),
        0.0,
        0.0,
        np.eye(2, dtype=np.float32),
        np.eye(2, dtype=np.float32),
    )
    reference = np.asarray([1.0, 0.0], dtype=np.float32)

    assert head.score(reference, reference) == 1.0
    assert head.score(reference, np.asarray([0.0, 1.0], dtype=np.float32)) == 0.0


def test_compositional_relation_tissue_cannot_inflate_mention_evidence() -> None:
    base = (1.0, 0.5, -2.0)
    unchanged = _mention_invariant_relation_evidence(base, base)
    redirected = _mention_invariant_relation_evidence(base, (-2.0, 4.0, 1.0))

    np.testing.assert_allclose(unchanged, tuple(-np.logaddexp(0.0, -x) for x in base))
    assert max(redirected) == pytest.approx(max(unchanged))
    assert int(np.argmax(redirected)) == 1


def test_compositional_relation_tissue_is_bound_to_the_receipt() -> None:
    model = fit_compositional_semantic_program_transducer(
        _examples(),
        input_grounding=_grounding(),
    )
    relation_fit = model.training_receipt["relation_tissue_fit"]

    assert relation_fit["algorithm"] == "minibatch_adamw_cross_entropy_v1"
    assert relation_fit["selection_objective"] == "minimum_validation_cross_entropy"
    selected = [row for row in relation_fit["validation_selection"] if row["selected"]]
    assert len(selected) == 1
    assert selected[0]["validation_cross_entropy"] == min(
        row["validation_cross_entropy"] for row in relation_fit["validation_selection"]
    )

    payload = copy.deepcopy(model.to_dict())
    payload["definition_relation_head"]["query_projection"][0][0] += 1.0
    with pytest.raises(ValueError, match="envelope"):
        compositional_semantic_program_transducer_from_dict(payload)

    payload = copy.deepcopy(model.to_dict())
    payload["argument_proposal_heads"][0]["weight"][0] += 1.0
    with pytest.raises(ValueError, match="envelope"):
        compositional_semantic_program_transducer_from_dict(payload)

    payload = copy.deepcopy(model.to_dict())
    payload["argument_proposal_scale"] += 0.125
    with pytest.raises(ValueError, match="envelope"):
        compositional_semantic_program_transducer_from_dict(payload)


def test_compositional_definition_envelope_always_contains_its_anchor() -> None:
    anchor = TokenSpan(8, 10)

    assert _definition_span_candidates(
        anchor,
        token_count=10,
        max_span_tokens=1,
    ) == (anchor,)
    assert _definition_span_candidates(
        anchor,
        token_count=12,
        max_span_tokens=4,
        direction="left",
    ) == (TokenSpan(6, 10), TokenSpan(7, 10), anchor)
    assert _definition_span_candidates(
        anchor,
        token_count=12,
        max_span_tokens=4,
        direction="right",
    ) == (anchor, TokenSpan(8, 11), TokenSpan(8, 12))


def test_compositional_calibration_treats_a_missing_chart_as_a_refusal() -> None:
    assert (
        _best_penalized_operation_chart(
            ((float("-inf"), ()),),
            penalty=0.0,
        )
        == ()
    )


def test_compositional_kbest_chart_preserves_the_greedy_chart_then_falls_back() -> None:
    nodes = (
        _OperationNode(TokenSpan(0, 1), "add", 5.0, 5.0, 1.0),
        _OperationNode(TokenSpan(0, 2), "sub", 4.0, 4.0, 1.0),
        _OperationNode(TokenSpan(3, 4), "add", 3.0, 3.0, 1.0),
    )

    charts = _operation_chart_candidates(
        nodes,
        max_steps=2,
        length_penalty=0.0,
        limit=3,
    )

    assert charts[0] == _operation_chart(nodes, max_steps=2, length_penalty=0.0)
    assert charts[1] == (nodes[1], nodes[2])


def test_pointer_sequence_scores_reuse_one_validated_hidden_sequence() -> None:
    hidden = np.eye(3, dtype=np.float32)
    pointer = LinearPointerHead(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        0.5,
        np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
        -0.5,
    )
    scores = pointer.score_sequence(hidden)

    assert scores.score_span(TokenSpan(0, 1)) == pointer.score_span(
        hidden,
        TokenSpan(0, 1),
    )
    assert scores.decode_candidates(limit=4, max_span_tokens=2) == pointer.decode_candidates(
        hidden,
        limit=4,
        max_span_tokens=2,
    )


def test_compositional_relation_diagnostic_separates_runtime_and_oracle_spans() -> None:
    examples = _examples()
    model = fit_compositional_semantic_program_transducer(
        examples,
        input_grounding=_grounding(),
    )

    report = diagnose_compositional_definition_relations(model, examples)

    assert report["gold_reference_spans_available"] is True
    assert report["gold_definition_spans_available_to_runtime_arm"] is False
    assert report["expected_answers_available"] is False
    assert report["serving_authority"] is False
    assert report["splits"]["test"]["total"] == 20
    assert report["splits"]["test"]["runtime_top1"] <= 20
    assert report["splits"]["test"]["oracle_top1"] <= 20
    assert sum(row["total"] for row in report["splits"]["test"]["by_slot"].values()) == 20


def test_compositional_lesion_diagnostic_replays_without_refitting() -> None:
    examples = _examples()
    model = fit_compositional_semantic_program_transducer(
        examples,
        input_grounding=_grounding(),
    )

    report = diagnose_compositional_transfer_lesions(model, examples)

    assert report["fit_or_refit_calls"] == 0
    assert report["expected_answers_available_to_decode"] is False
    assert set(report["evaluated_arms"]) == set(report["arms"])
    assert report["arms"]["treatment"]["test"]["total"] == 4
    assert (
        report["arms"]["coefficient_lesion"]["test"]["program_exact"]
        < report["arms"]["treatment"]["test"]["program_exact"]
    )

    focused = diagnose_compositional_transfer_lesions(
        model,
        examples,
        arm_names=("treatment", "argument_proposal_lesion"),
    )
    assert focused["evaluated_arms"] == [
        "treatment",
        "argument_proposal_lesion",
    ]
    assert set(focused["arms"]) == set(focused["evaluated_arms"])

    with pytest.raises(ValueError, match="requires treatment"):
        diagnose_compositional_transfer_lesions(
            model,
            examples,
            arm_names=("argument_proposal_lesion",),
        )


def test_shared_transducer_programs_replay_on_the_universal_floor() -> None:
    examples = _examples()
    model = fit_shared_semantic_program_transducer(examples, input_grounding=_grounding())
    sources = {relative: "d" * 64 for relative in SEMANTIC_PROGRAM_FLOOR_VERIFICATION_SOURCES}

    report = verify_semantic_program_floor_equivalence(
        model,
        examples,
        feature_manifest_sha256s={"synthetic": "e" * 64},
        source_sha256s=sources,
    )

    assert report["verified"] is True
    assert report["test_total"] == 4
    assert report["accepted"] == 4
    assert report["agreements"] == 4
    assert report["value_agreements"] == 4
    assert report["refusal_agreements"] == 0
    assert report["primitive_coverage"]["complete"] is True
    assert report["fit_or_refit_calls"] == 0
    assert report["expected_answers_available"] is False


def test_floor_verifier_rejects_an_incomplete_source_inventory() -> None:
    examples = _examples()
    model = fit_shared_semantic_program_transducer(examples, input_grounding=_grounding())

    with pytest.raises(ValueError, match="source inventory"):
        verify_semantic_program_floor_equivalence(
            model,
            examples,
            feature_manifest_sha256s={"synthetic": "e" * 64},
            source_sha256s={},
        )


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
            tokenizer.decode(list(tokens[span.start : span.end])).strip(" ,.") in expected
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
    assert result.report["paired_program_controls"]["coefficient_lesion:test"]["treatment_only"] > 0


def test_shared_verifier_replays_frozen_arms_and_rejects_tampering(monkeypatch) -> None:
    examples = _examples()
    by_family = {
        "short": tuple(item for item in examples if item.ir.n_inputs == 3),
        "long": tuple(item for item in examples if item.ir.n_inputs == 4),
    }
    compatibility = {
        "target_training_session_basis_sha256": _BASIS,
        "receipt_sha256": "f" * 64,
    }

    def bind(grouped, *, compatibility):
        assert compatibility["target_training_session_basis_sha256"] == _BASIS
        return tuple(
            replace(item, construction_id=f"{family}:{item.construction_id}")
            for family in sorted(grouped)
            for item in grouped[family]
        )

    for module in (
        "core.learning.semantic_program_shared_campaign",
        "core.learning.semantic_program_shared_verification",
    ):
        monkeypatch.setattr(
            f"{module}.establish_semantic_training_representation_compatibility",
            lambda _manifests: compatibility,
        )
        monkeypatch.setattr(
            f"{module}.bind_training_examples_to_shared_representation",
            bind,
        )
    manifests = {
        "short": {"manifest_sha256": "1" * 64},
        "long": {"manifest_sha256": "2" * 64},
    }
    result = run_shared_semantic_program_campaign_from_examples(
        by_family,
        manifests=manifests,
        input_grounding=_grounding(),
    )

    class _Bundle:
        def __init__(self, manifest, examples):
            self.manifest = manifest
            self.examples = examples

    monkeypatch.setattr(
        "core.learning.semantic_program_shared_verification.training_examples_from_feature_bundle",
        lambda bundle: bundle.examples,
    )
    bundles = {family: _Bundle(manifests[family], by_family[family]) for family in by_family}
    sources = {
        relative: hashlib.sha256(relative.encode()).hexdigest()
        for relative in SEMANTIC_PROGRAM_SHARED_VERIFICATION_SOURCES
    }
    verification = verify_shared_semantic_program_campaign(
        bundles,
        stored_model_payload=result.model.to_dict(),
        stored_report=result.report,
        source_sha256s=sources,
    )
    assert verification["verified"] is True
    assert verification["test_program_exact"] == 4

    tampered = copy.deepcopy(result.report)
    tampered["arms"]["treatment:test"]["program_exact"] -= 1
    body = {key: value for key, value in tampered.items() if key != "report_sha256"}
    tampered["report_sha256"] = _sha(body)
    with np.testing.assert_raises_regex(ValueError, "replayed arms differ"):
        verify_shared_semantic_program_campaign(
            bundles,
            stored_model_payload=result.model.to_dict(),
            stored_report=tampered,
            source_sha256s=sources,
        )
