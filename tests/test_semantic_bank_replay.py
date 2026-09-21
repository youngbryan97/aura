"""A frozen bank separates parameter ranking from hypothesis generation."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_bank_replay import rescore_semantic_candidate_bank
from core.learning.semantic_candidate_bank import SemanticCandidate
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_ir import TokenSpan
from tests.test_semantic_relation_graph_learning import model_examples


@pytest.fixture(scope="module")
def fixture():
    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    item = examples[0]
    kwargs = dict(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=model.model_basis_sha256)
    bank = model.decode_candidates(**kwargs, max_charts=2, max_graphs_per_chart=3)
    return model, bank, kwargs


def test_forward_reference_attribution_follows_program_not_chart_order():
    from core.learning.semantic_candidate_bank import _candidate_span_order
    assert _candidate_span_order() == ()
    nodes = (SimpleNamespace(operation="sub", span=TokenSpan(2, 3)),
             SimpleNamespace(operation="add", span=TokenSpan(9, 10)))
    candidate = SemanticCandidate.from_argument_graph(nodes, ((3, 0), (0, 1)),
        n_inputs=2, joint_score=1., chart_index=0, graph_index=0)
    assert candidate.program == Program(2, (Instruction("add", (0, 1)), Instruction("sub", (2, 0))))
    assert candidate.operation_spans == (nodes[1].span, nodes[0].span)
    assert candidate.program.run((3, 7)) == 7


def test_every_same_model_bank_score_replays(fixture):
    model, bank, kwargs = fixture
    report = rescore_semantic_candidate_bank(bank, model, **kwargs)
    assert report["scoring_complete"] and report["rows"]
    for row, candidate in zip(report["rows"], bank.candidates, strict=True):
        assert row["status"] == "scored"
        assert row["score"] == pytest.approx(row["operation_score"] + row["argument_score"])
        if candidate.joint_score is not None:
            assert row["score"] == pytest.approx(candidate.joint_score, abs=1e-4)
    assert not report["candidate_generation_repeated"]
    assert not report["source_annotations_available"] and not report["expected_answer_available"]
    assert not report["serving_authority"]
    assert report["receipt_sha256"] == _sha({k: v for k, v in report.items() if k != "receipt_sha256"})


def test_second_scorer_keeps_bank_identity_without_generating_candidates(fixture, monkeypatch):
    model, bank, kwargs = fixture
    other = model._with_coefficients(operation_head=replace(model.operation_head, heads=tuple(
        replace(head, bias=head.bias + np.arange(len(head.bias)) * .01)
        for head in model.operation_head.heads)))
    def prohibited(*args, **options):
        raise AssertionError("fixed-bank replay must not decode new programs")
    monkeypatch.setattr(type(model), "decode", prohibited)
    left = rescore_semantic_candidate_bank(bank, model, **kwargs)
    right = rescore_semantic_candidate_bank(bank, other, **kwargs)
    assert left["bank_receipt_sha256"] == right["bank_receipt_sha256"]
    assert left["scorer_transducer_receipt_sha256"] != right["scorer_transducer_receipt_sha256"]
    assert [r["candidate_sha256"] for r in left["rows"]] == [r["candidate_sha256"] for r in right["rows"]]
    assert any(a["score"] != b["score"] for a, b in zip(left["rows"], right["rows"], strict=True))


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_allowance(fixture, value):
    model, bank, kwargs = fixture
    with pytest.raises(ValueError, match="allowance"):
        rescore_semantic_candidate_bank(bank, model, **kwargs, solve_time_limit_s=value)


def test_old_mixed_span_order_receipt_cannot_be_reinterpreted(fixture):
    model, bank, kwargs = fixture
    body = {k: v for k, v in bank.receipt.items() if k != "receipt_sha256"}
    body["schema"] = "aura.semantic_candidate_bank.v1"
    old = replace(bank, receipt={**body, "receipt_sha256": _sha(body)})
    with pytest.raises(ValueError, match="execution-ordered"):
        rescore_semantic_candidate_bank(old, model, **kwargs)


@pytest.mark.parametrize("changed", ["basis", "source", "hidden", "tokens", "inputs"])
def test_observation_and_basis_are_bound(fixture, changed):
    model, bank, kwargs = fixture
    options = dict(kwargs)
    if changed == "basis":
        options["model_basis_sha256"] = "f" * 64
    elif changed == "source":
        options["source_text_sha256"] = "f" * 64
    elif changed == "hidden":
        options["hidden_states"] = -options["hidden_states"]
    elif changed == "tokens":
        options["source_token_ids"] = (*options["source_token_ids"][:-1], 99999)
    else:
        options["public_inputs"] = tuple(7 if type(x) is int else (7,) for x in options["public_inputs"])
    with pytest.raises(ValueError, match="differs"):
        rescore_semantic_candidate_bank(bank, model, **options)


def test_incomplete_scoring_does_not_declare_a_bank_winner(fixture, monkeypatch):
    from core.learning import semantic_bank_replay as replay
    from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
    model, bank, kwargs = fixture
    original = replay.score_annotated_graph
    calls = 0
    def limited(*args, **options):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ArgumentOptimizationIncompleteError("interrupted test solve")
        return original(*args, **options)
    monkeypatch.setattr(replay, "score_annotated_graph", limited)
    report = replay.rescore_semantic_candidate_bank(bank, model, **kwargs)
    assert not report["scoring_complete"] and report["selected_index"] is None
    assert report["best_observed_index"] is not None
    assert report["rows"][0]["status"] == "incomplete"


def test_same_models_produce_equal_two_by_two_replay():
    from core.learning.semantic_bank_replay import compare_semantic_candidate_banks

    model, examples = model_examples()
    item = next(row for row in examples if row.split == 'train')
    report = compare_semantic_candidate_banks(model, model, item, max_charts=1, max_graphs_per_chart=1)
    assert report['candidate_overlap']['parent_only'] == report['candidate_overlap']['candidate_only'] == []
    assert report['labels_used_after_both_banks_completed']
    for replay in report['replays'].values():
        assert replay['status'] == 'measured'
        assert replay['scorers']['parent'] == replay['scorers']['candidate']
    assert not report['learning_performed']
    assert not report['fresh_transfer_claim']


def test_sealed_test_cannot_enter_bank_comparison():
    from core.learning.semantic_bank_replay import compare_semantic_candidate_banks

    model, examples = model_examples()
    with pytest.raises(ValueError, match='development'):
        compare_semantic_candidate_banks(model, model, replace(examples[0], split='test'))
