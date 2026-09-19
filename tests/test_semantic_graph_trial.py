"""Small trials keep source fitting, validation and release authority separate."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.learning.semantic_graph_trial import _observe, run_semantic_graph_trial, select_trial_examples
from tests.test_semantic_relation_graph_learning import model_examples


@pytest.fixture(scope="module")
def source():
    model, examples = model_examples()
    return model.with_joint_operation_argument_scores().with_source_ordered_definitions(), examples


def test_trial_selection_is_order_independent_and_split_bound(source):
    _, examples = source
    selected = select_trial_examples(examples, split="train", count=4)
    reordered = select_trial_examples(tuple(reversed(examples)), split="train", count=4)
    assert [row.ir.source_text_sha256 for row in selected] == [row.ir.source_text_sha256 for row in reordered]
    assert len(selected) == 4 and all(row.split == "train" for row in selected)
    assert len({len(row.ir.instructions) for row in selected}) == 2


@pytest.mark.parametrize("split,count", [("test", 4), ("train", 0), ("train", True)])
def test_invalid_trial_cohorts_are_not_silently_substituted(source, split, count):
    with pytest.raises(ValueError):
        select_trial_examples(source[1], split=split, count=count)


def test_same_source_cannot_enter_both_sides_of_trial(source):
    model, examples = source
    with pytest.raises(ValueError, match="overlap"):
        run_semantic_graph_trial(model, (examples[0], replace(examples[0], split="validation")))


def test_small_trial_executes_training_and_independent_replay_without_promotion(source):
    model, examples = source
    result = run_semantic_graph_trial(model, examples, training_count=2, validation_count=2,
                                     training_pool_count=4, steps=2, max_charts=2,
                                     operation_retention_count=4)
    assert result["parent"] == model.receipt_sha256
    assert not set(result["training_sources"]) & set(result["validation_sources"])
    assert {row["source_text_sha256"] for row in result["mining"]} == set(result["training_sources"])
    assert all(row["selected_decode"]["negative_origin"] == "runtime_decode" for row in result["mining"])
    assert not result["validation_used_for_fit"] and result["test_examples_used"] == 0
    assert len(result["operation_retention_sources"]) == 4
    assert set(result["training_sources"]) <= set(result["operation_retention_sources"])
    assert not set(result["validation_sources"]) & set(result["operation_retention_sources"])
    assert result["summaries"]["train"]["total"] == result["summaries"]["validation"]["total"] == 2
    assert len(result["before"]) == len(result["after"]) == 4
    assert result["fit"]["retained_positive_regressions"] == 0
    assert not result["promotion_allowed"] and not result["serving_authority"]
    assert not result["fresh_transfer_claim"]
    assert "operation_retention_search_incomplete" in result["blockers"]
    assert not result["larger_development_run_ready"]
    pool = result["training_pool_observations"]
    assert len(pool) == 4 and all(row["split"] == "train" for row in pool)
    expected = sorted(pool, key=lambda row: row["semantic_status"] not in {"different", "decode_refused"})[:2]
    assert result["training_sources"] == [row["source_text_sha256"] for row in expected]
    groups = result["constraint_groups"]
    assert sum(row["count"] for row in groups.values()) == result["fit"]["pairs"]
    assert sum(row["initial_wrong_or_tied"] for row in groups.values()) == result["fit"]["initial_wrong_or_tied"]


def test_constraint_attribution_cannot_turn_missing_or_duplicate_evidence_into_success():
    from core.learning.semantic_graph_trial import _constraint_group_summary

    assert _constraint_group_summary({"labels": [0]}, {}) is None
    fit = {"initial_margins": [-.2, .5], "stored_margins": [.2, .5], "required_margin": .1}
    result = _constraint_group_summary({"labels": [0], "bindings": [1]}, fit)
    assert result["labels"]["initial_wrong_or_tied"] == 1
    assert result["bindings"]["initial_squared_deficit"] == 0.
    with pytest.raises(ValueError, match="partition"):
        _constraint_group_summary({"labels": [0], "bindings": [0]}, fit)


@pytest.mark.parametrize("pool", [0, True, 1])
def test_training_acquisition_requires_a_sufficient_pool(source, pool):
    with pytest.raises(ValueError, match="training pool"):
        run_semantic_graph_trial(*source, training_count=2, training_pool_count=pool)


def test_decode_refusal_is_a_measured_completion_failure_not_unknown_semantics(source):
    class RefusingDecoder:
        def decode(self, **kwargs):
            return SimpleNamespace(ir=None, refusal="typed_argument_chart_empty")

    row = _observe(RefusingDecoder(), source[1][0])
    assert row["semantic_status"] == "decode_refused"
    assert not row["accepted"]
    assert row["refusal"] == "typed_argument_chart_empty"
    assert row["source_grounding_aligned"] is None
    assert row["annotated_graph_feasible"] is None
    assert "target_reachable" not in row


def test_selected_error_is_retained_even_outside_the_chart_probe(source, monkeypatch):
    import core.learning.semantic_graph_trial as trial
    from core.learning.semantic_relation_graph_learning import RelationGraphContrast

    selected_error = RelationGraphContrast((), (), -1.)
    seen = []
    def selected(model, item, **kwargs):
        seen.append((item.split, kwargs))
        return selected_error, {"status": "counterexample", "negative_origin": "runtime_decode"}

    def limited_search(*args, **kwargs):
        return (), {"status": "no_witnessed_competitor", "operation_search_complete": False, "charts": []}

    def fit(model, constraints, **kwargs):
        assert any(row is selected_error for row in constraints)
        return model, {"stored_wrong_or_tied": 1}

    monkeypatch.setattr(trial, "mine_runtime_graph_contrast", selected)
    monkeypatch.setattr(trial, "mine_runtime_graph_constraints", limited_search)
    monkeypatch.setattr(trial, "fit_complete_graph_constraints", fit)
    result = trial.run_semantic_graph_trial(*source, training_count=1, validation_count=1,
                                            max_charts=1, learn_operation_pointer=True)
    assert seen == [("train", {"learn_arguments": True, "learn_operation_pointer": True})]
    assert result["schema"] == "aura.semantic_graph_trial.v3"
    assert "retained_constraints_unsatisfied" in result["blockers"]


def test_unreplayed_selected_error_prevents_readiness(source, monkeypatch):
    import core.learning.semantic_graph_trial as trial

    monkeypatch.setattr(trial, "mine_runtime_graph_contrast", lambda *args, **kwargs:
                        (None, {"status": "runtime_score_replay_differs"}))
    result = trial.run_semantic_graph_trial(*source, training_count=1, validation_count=1,
                                            steps=1, max_charts=1)
    assert "selected_decode_constraint_unavailable" in result["blockers"]
    assert not result["larger_development_run_ready"]
