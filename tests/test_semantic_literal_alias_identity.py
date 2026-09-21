"""Literal formatting cannot change a grounded input into another register."""

from dataclasses import replace

import pytest

from core.learning.semantic_input_grounding import semantic_input_grounding_contract_from_tokenizer
from core.learning.semantic_program_ir import TokenSpan
from tests.test_semantic_program_shared_transducer import _CharacterTokenizer, _examples, _grounding


@pytest.fixture(scope="module")
def grammar():
    return semantic_input_grounding_contract_from_tokenizer(_CharacterTokenizer(), tokenizer_identity_sha256="a" * 64)


@pytest.mark.parametrize("value,text", [(78, "78"), (-12, "-12"), ((3, -2), "[3, -2]"), ((), "[]")])
def test_all_grammar_forms_of_an_occurrence_keep_its_identity(grammar, value, text):
    source = "x " + text + ". y"
    tokens = tuple(map(ord, source))
    anchor = TokenSpan(2, 2 + len(text))
    bindings = grammar.literal_alias_bindings(tokens, (value,), (anchor,))
    assert bindings[anchor] == (0,)
    assert bindings[TokenSpan(1, anchor.end)] == (0,)
    assert bindings[TokenSpan(1, anchor.end + 1)] == (0,)
    assert all(owner == (0,) and span in grammar.candidate_spans(tokens, value) for span, owner in bindings.items())
    assert TokenSpan(0, anchor.end) not in bindings


def test_equal_values_at_distinct_occurrences_do_not_merge(grammar):
    tokens = tuple(map(ord, "78 and 78."))
    bindings = grammar.literal_alias_bindings(tokens, (78, 78), (TokenSpan(0, 2), TokenSpan(7, 9)))
    assert bindings[TokenSpan(0, 2)] == (0,)
    assert bindings[TokenSpan(6, 9)] == (1,)
    assert bindings[TokenSpan(7, 10)] == (1,)
    assert TokenSpan(0, 9) not in bindings


def test_registered_occurrence_canary():
    from core.learning.semantic_input_grounding import literal_alias_occurrence_identity

    assert literal_alias_occurrence_identity() == []


def test_linguistic_expression_containing_literal_is_not_a_literal_alias(grammar):
    tokens = tuple(map(ord, "the value before 78"))
    assert grammar.literal_alias_bindings(tokens, (78,), (TokenSpan(17, 19),)) == {
        TokenSpan(16, 19): (0,), TokenSpan(17, 19): (0,)}


def test_partial_digit_match_does_not_cross_input_occurrences(grammar):
    tokens = tuple(map(ord, "178 plus 78"))
    bindings = grammar.literal_alias_bindings(tokens, (178, 78), (TokenSpan(0, 3), TokenSpan(9, 11)))
    assert TokenSpan(1, 3) not in bindings
    assert bindings[TokenSpan(8, 11)] == (1,)


@pytest.mark.parametrize("tokens,values,anchors", [
    ((ord("7"), ord("8")), (7,), (TokenSpan(0, 2),)),
    ((ord("7"),), (), (TokenSpan(0, 1),)),
    ((True,), (7,), (TokenSpan(0, 1),)),
])
def test_unbound_or_invalid_source_cannot_claim_literal_identity(grammar, tokens, values, anchors):
    with pytest.raises(ValueError, match="literal alias"):
        grammar.literal_alias_bindings(tokens, values, anchors)


@pytest.fixture(scope="module")
def parent():
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_compositional_transducer import (
        fit_compositional_semantic_program_transducer,
    )

    grounding = replace(_grounding(), positive_integer_prefixes=((), (777,)))
    model = fit_compositional_semantic_program_transducer(_examples(), input_grounding=grounding)
    model = model.with_joint_definition_graph().with_categorical_relation_scores()
    bounds = {kind: max(2, limit) for kind, limit in model.max_argument_span_tokens_by_type.items()}
    body = {key: value for key, value in model.training_receipt.items() if key != "receipt_sha256"}
    body["argument_span_bounds"] = bounds
    return replace(model, max_argument_span_tokens_by_type=bounds,
        training_receipt={**body, "receipt_sha256": _sha(body)})


def test_policy_roundtrip_keeps_coefficients_and_binds_new_behavior(parent):
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )

    candidate = parent.with_literal_grammar_identities()
    assert candidate._coefficient_body() == parent._coefficient_body()
    assert candidate.receipt_sha256 != parent.receipt_sha256
    assert "argument_literal_identity" not in parent.training_receipt
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).to_dict() == candidate.to_dict()


def test_runtime_chart_uses_grammar_identity_not_only_exact_anchor(parent, monkeypatch):
    from core.learning import semantic_program_transducer_fitting as fitting

    item = next(item for item in _examples() if item.split == "train")
    anchor = item.ir.input_spans[0]
    alias = TokenSpan(anchor.start - 1, anchor.end)
    tokens = list(item.ir.source_token_ids)
    tokens[alias.start] = 777
    original = fitting._argument_proposals_by_operation
    def proposals(*args, **kwargs):
        return tuple((*rows, (alias, 30.)) for rows in original(*args, **kwargs))
    monkeypatch.setattr(fitting, "_argument_proposals_by_operation", proposals)
    charts = []
    for model in (parent, parent.with_literal_grammar_identities()):
        captured = []
        fitting._assign_typed_arguments(model=model, hidden=item.hidden_states, inputs=item.public_inputs,
            input_spans=item.ir.input_spans, source_token_ids=tuple(tokens),
            operation_nodes=tuple(fitting._OperationNode(ins.operation_span, ins.op, 0., 0., 1.) for ins in item.ir.instructions),
            argument_pointer_scores=model.argument_pointer.score_sequence(item.hidden_states),
            chart_observer=captured.append, build_only=True)
        assert captured
        charts.append(captured[0])
    def alias_owners(chart):
        return {register for node in chart.options for slot in node for _, register, span in slot if span == alias}
    assert alias_owners(charts[0]) - {0}
    assert alias_owners(charts[1]) == {0}


def test_activated_policy_requires_tokens_before_any_chart_work(parent):
    from core.learning import semantic_program_transducer_fitting as fitting

    item = next(item for item in _examples() if item.split == "train")
    with pytest.raises(ValueError, match="source token sequence"):
        fitting._assign_typed_arguments(model=parent.with_literal_grammar_identities(), hidden=item.hidden_states,
            inputs=item.public_inputs, input_spans=item.ir.input_spans, operation_nodes=(),
            argument_pointer_scores=parent.argument_pointer.score_sequence(item.hidden_states))


def test_ordinary_decode_and_offline_score_share_the_token_contract(parent):
    from core.learning.semantic_graph_trial import _observe

    item = next(item for item in _examples() if item.split == "validation")
    result = _observe(parent.with_literal_grammar_identities(), item)
    assert result["accepted"] and result["source_grounding_aligned"]
    assert result["annotated_graph_feasible"]
