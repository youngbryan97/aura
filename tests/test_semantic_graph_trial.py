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
                                     training_pool_count=4, steps=2, max_charts=2)
    assert result["parent"] == model.receipt_sha256
    assert not set(result["training_sources"]) & set(result["validation_sources"])
    assert {row["source_text_sha256"] for row in result["mining"]} == set(result["training_sources"])
    assert not result["validation_used_for_fit"] and result["test_examples_used"] == 0
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
