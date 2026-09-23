"""Candidate coverage uses deployed builders and keeps selection and evidence apart."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.learning.semantic_failure_diagnosis import diagnose_semantic_candidate_bank
from core.learning.semantic_candidate_bank import SemanticCandidate
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import SemanticTransductionOutcome
from tests.test_semantic_relation_graph_learning import model_examples


@pytest.fixture(scope="module")
def fixture():
    model, examples = model_examples()
    item = examples[0]
    kwargs = dict(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=model.model_basis_sha256)
    bank = model.decode_candidates(**kwargs, max_charts=2, max_graphs_per_chart=4)
    return model, item, kwargs, bank


def completed_fixture_bank(bank, complete):
    body = {key: value for key, value in bank.receipt.items() if key != "receipt_sha256"}
    body.update(search_complete=complete, operation_inventory_exhausted=complete,
                operation_search_complete=complete,
                charts=[{**row, "search_complete": complete} for row in body["charts"]])
    return replace(bank, receipt={**body, "receipt_sha256": _sha(body)})


def test_runtime_selection_and_coefficients_are_unchanged(fixture):
    model, item, kwargs, bank = fixture
    before = model.receipt_sha256
    ordinary = model.decode(**kwargs)
    assert bank.selected.ir == ordinary.ir
    assert bank.selected.pointer_scores == ordinary.pointer_scores
    assert bank.candidates[0].program == ordinary.ir.to_program()
    assert bank.receipt["charts"]
    assert model.receipt_sha256 == before
    assert not bank.receipt["selection_changed"]
    assert not bank.receipt["expected_answer_available"]
    assert not bank.receipt["source_annotations_available"]
    assert not bank.receipt["serving_authority"]
    assert bank.receipt["schema"] == "aura.semantic_candidate_bank.v3"
    assert bank.candidates[0].argument_spans == tuple(
        ins.argument_spans for ins in ordinary.ir.instructions)
    assert bank.candidates[0].definition_provenance == "register_anchor"
    assert bank.candidates[0].definition_spans is not None
    assert all(candidate.argument_spans is not None for candidate in bank.candidates)


def test_graph_evidence_follows_topological_execution_order():
    nodes = (SimpleNamespace(operation="sub", span=TokenSpan(2, 3)),
             SimpleNamespace(operation="add", span=TokenSpan(9, 10)))
    mentions = ((TokenSpan(3, 4), TokenSpan(4, 5)),
                (TokenSpan(6, 7), TokenSpan(7, 8)))
    definitions = ((TokenSpan(0, 1), TokenSpan(1, 2)),
                   (TokenSpan(10, 11), TokenSpan(11, 12)))
    candidate = SemanticCandidate.from_argument_graph(
        nodes, ((3, 0), (0, 1)), n_inputs=2, joint_score=1.,
        chart_index=0, graph_index=0, argument_spans=mentions,
        definition_spans=definitions)
    assert candidate.operation_spans == (nodes[1].span, nodes[0].span)
    assert candidate.argument_spans == (mentions[1], mentions[0])
    assert candidate.definition_spans == (definitions[1], definitions[0])
    assert candidate.definition_provenance == "optimizer_selected"
    assert candidate.to_dict()["argument_spans"][0][0] == mentions[1][0].to_dict()


def test_search_limits_do_not_prove_absence(fixture):
    *_, bank = fixture
    assert not bank.receipt["search_complete"]
    assert bank.receipt["limit_reason"]


@pytest.mark.parametrize("execution,emission,stage", [
    (None, None, "execution_unmeasured"), (False, None, "execution"),
    (True, None, "emission_unmeasured"), (True, False, "emission"), (True, True, "success"),
])
def test_unmeasured_is_not_success(fixture, execution, emission, stage):
    _, item, _, bank = fixture
    result = diagnose_semantic_candidate_bank(bank, item,
        execution_correct=execution, emission_correct=emission)
    assert result["selected_semantic_status"] == "equivalent"
    assert result["correct_reachable"] is True
    assert result["failure_stage"] == stage


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_bank_allowances(fixture, value):
    model, _, kwargs, _ = fixture
    with pytest.raises(ValueError, match="allowances"):
        model.decode_candidates(**kwargs, max_charts=value)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_solve_allowances(fixture, value):
    model, _, kwargs, _ = fixture
    with pytest.raises(ValueError, match="solve allowance"):
        model.decode_candidates(**kwargs, solve_time_limit_s=value)


def test_no_diagnostic_rescue_of_wrong_model_identity(fixture):
    model, item, kwargs, _ = fixture
    bank = model.decode_candidates(**{**kwargs, "model_basis_sha256": "f" * 64})
    assert not bank.candidates and bank.selected.ir is None
    assert bank.receipt["selected_refusal"] == "model_basis_mismatch"
    with pytest.raises(ValueError, match="source identity"):
        diagnose_semantic_candidate_bank(bank, item)


def test_same_text_hash_cannot_reuse_other_hidden_observation(fixture):
    _, item, _, bank = fixture
    with pytest.raises(ValueError, match="observation differs"):
        diagnose_semantic_candidate_bank(bank, replace(item, hidden_states=-item.hidden_states))


def test_downstream_truthiness_is_rejected(fixture):
    _, item, _, bank = fixture
    with pytest.raises(ValueError, match="booleans"):
        diagnose_semantic_candidate_bank(bank, item, execution_correct="incorrect")


@pytest.mark.parametrize("complete", [False, True])
def test_unknown_equivalence_is_not_unreachable(fixture, monkeypatch, complete):
    _, item, _, bank = fixture
    monkeypatch.setattr("core.learning.semantic_failure_diagnosis.compare_program_meanings",
                        lambda *_args, **_kwargs: {"status": "unknown"})
    result = diagnose_semantic_candidate_bank(completed_fixture_bank(bank, complete), item)
    assert result["failure_stage"] == "verification_unknown"
    assert result["correct_reachable"] is None


@pytest.mark.parametrize("complete,stage", [(False, "incomplete_search"), (True, "reachability")])
def test_exhaustion_is_required_for_unreachability(fixture, monkeypatch, complete, stage):
    _, item, _, bank = fixture
    monkeypatch.setattr("core.learning.semantic_failure_diagnosis.compare_program_meanings",
                        lambda *_args, **_kwargs: {"status": "different"})
    result = diagnose_semantic_candidate_bank(completed_fixture_bank(bank, complete), item)
    assert result["failure_stage"] == stage
    assert result["correct_reachable"] is (False if complete else None)


def test_correct_candidate_outranked_is_selection_failure(fixture, monkeypatch):
    _, item, _, bank = fixture
    selected = bank.selected.ir.to_program().sha()
    assert any(candidate.program.sha() != selected for candidate in bank.candidates)
    monkeypatch.setattr("core.learning.semantic_failure_diagnosis.compare_program_meanings",
        lambda _target, candidate, *_args, **_kwargs:
            {"status": "different" if candidate.sha() == selected else "equivalent"})
    result = diagnose_semantic_candidate_bank(bank, item)
    assert result["failure_stage"] == "selection"
    assert result["correct_reachable"] is True
    assert result["operation_view"]["equivalent_candidates"] > 0
    assert result["operation_view"]["selected_spans"] == [
        span.to_dict() for span in bank.candidates[0].operation_spans]
    assert result["operation_view"]["diagnostic_only"] is True


def test_operation_view_diagnosis_separates_source_chart_coverage(fixture, monkeypatch):
    _, item, _, bank = fixture
    selected = bank.selected.ir.to_program().sha()
    monkeypatch.setattr("core.learning.semantic_failure_diagnosis.compare_program_meanings",
        lambda _target, candidate, *_args, **_kwargs:
            {"status": "different" if candidate.sha() == selected else "equivalent"})
    result = diagnose_semantic_candidate_bank(bank, item)
    source_spans = tuple(ins.operation_span for ins in item.ir.instructions)
    expected = sum(candidate.program.sha() != selected
                   and candidate.operation_spans == source_spans
                   and tuple(ins.op for ins in candidate.program.instructions) ==
                   tuple(ins.op for ins in item.ir.instructions)
                   for candidate in bank.candidates)
    assert result["operation_view"]["equivalent_exact_source_spans"] == expected
    assert result["failure_stage"] == "selection"


def test_receipt_mutation_cannot_certify_exhaustion(fixture):
    _, item, _, bank = fixture
    with pytest.raises(ValueError, match="integrity"):
        diagnose_semantic_candidate_bank(replace(bank,
            receipt={**bank.receipt, "search_complete": True}), item)


def test_rehashed_completion_still_needs_component_evidence(fixture):
    _, item, _, bank = fixture
    body = {key: value for key, value in bank.receipt.items() if key != "receipt_sha256"}
    body["search_complete"] = True
    with pytest.raises(ValueError, match="exhausted search"):
        diagnose_semantic_candidate_bank(replace(bank,
            receipt={**body, "receipt_sha256": _sha(body)}), item)


def test_candidate_payload_cannot_be_substituted(fixture):
    _, item, _, bank = fixture
    with pytest.raises(ValueError, match="payload"):
        diagnose_semantic_candidate_bank(replace(bank, candidates=bank.candidates[:1]), item)


def test_progress_covers_selection_and_graph_search(fixture):
    model, _, kwargs, _ = fixture
    progress = []
    bank = model.decode_candidates(**kwargs, max_charts=2, max_graphs_per_chart=1,
                                   progress=progress.append)
    bank.validate()
    assert progress[0] == {"stage": "ordinary_decode"}
    assert {row["stage"] for row in progress} >= {
        "candidate_chart", "candidate_graph", "candidate_retained"}


@pytest.mark.parametrize("refusal", ["decode_search_budget_exhausted",
                                     "argument_optimizer_budget_exhausted",
                                     "argument_chart_construction_budget_exhausted",
                                     "argument_optimizer_status:1"])
def test_search_interruption_retains_diagnostic_alternatives_without_answer_rescue(
        fixture, monkeypatch, refusal):
    model, item, kwargs, _ = fixture
    monkeypatch.setattr(type(model), "decode", lambda self, **_kwargs:
                        SemanticTransductionOutcome(None, refusal, {}, {}, search_interrupted=True))
    bank = model.decode_candidates(**kwargs, max_charts=2, max_graphs_per_chart=2)
    bank.validate()
    assert bank.selected.ir is None
    assert bank.receipt["selected_refusal"] == refusal
    assert bank.receipt["selected_search_interrupted"] is True
    assert bank.receipt["selected_program_sha256"] is None
    assert bank.receipt["selection_changed"] is False
    assert bank.candidates
    diagnosis = diagnose_semantic_candidate_bank(bank, item)
    assert diagnosis["selected_semantic_status"] == "unavailable"
    assert diagnosis["failure_stage"] in {"selection", "incomplete_search",
                                          "verification_unknown", "reachability"}


def test_pre_search_refusal_does_not_run_diagnostic_search(fixture, monkeypatch):
    model, _, kwargs, _ = fixture
    monkeypatch.setattr(type(model), "decode", lambda self, **_kwargs:
                        SemanticTransductionOutcome(None, "input_value_not_grounded:0", {}, {}))
    bank = model.decode_candidates(**kwargs, max_charts=2, max_graphs_per_chart=2)
    bank.validate()
    assert bank.candidates == ()
    assert bank.receipt["limit_reason"] == "ordinary_decode_unavailable"
    assert bank.receipt["selected_search_interrupted"] is False


def test_one_incomplete_chart_does_not_discard_later_charts(fixture, monkeypatch):
    from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
    from core.learning import semantic_candidate_bank as candidate_bank

    model, _, kwargs, _ = fixture
    original = candidate_bank._assign_typed_arguments
    calls = 0

    def interrupt_first(**arguments):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ArgumentOptimizationIncompleteError("argument_chart_construction_budget_exhausted")
        return original(**arguments)

    monkeypatch.setattr(candidate_bank, "_assign_typed_arguments", interrupt_first)
    bank = model.decode_candidates(**kwargs, max_charts=2, max_graphs_per_chart=2)
    bank.validate()
    assert bank.receipt["charts"][0]["interruption"] == (
        "argument_chart_construction_budget_exhausted")
    assert bank.receipt["charts"][0]["search_complete"] is False
    assert bank.receipt["search_complete"] is False
    assert calls >= 2
