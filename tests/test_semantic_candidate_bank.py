"""Candidate coverage uses deployed builders and keeps selection and evidence apart."""

from dataclasses import replace

import pytest

from core.learning.semantic_failure_diagnosis import diagnose_semantic_candidate_bank
from core.learning.semantic_program_campaign import _sha
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
