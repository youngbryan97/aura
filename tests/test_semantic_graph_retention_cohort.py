"""Active mining cannot silently narrow or contaminate source retention."""

from dataclasses import replace

import pytest

from core.learning.semantic_joint_graph_learning import refit_compositional_joint_graphs
from tests.test_semantic_relation_graph_learning import model_examples


@pytest.fixture(scope="module")
def source():
    model, examples = model_examples()
    return (model.with_joint_operation_argument_scores().with_source_ordered_definitions(),
            tuple(x for x in examples if x.split == "train"),
            tuple(x for x in examples if x.split == "validation"))


def test_mining_subset_preserves_broader_source_supervision(source):
    model, training, validation = source
    candidate = refit_compositional_joint_graphs(model, (training[0], *validation),
        source_retention_examples=training, rounds=1, steps=1)
    receipt = candidate.training_receipt["joint_graph_refit"]
    assert receipt["training_examples"] == 1
    assert receipt["source_retention_examples"] == len(training)
    assert receipt["source_operations"] == sum(len(x.ir.instructions) for x in training)
    assert len(receipt["rounds"][0]["records"]) == 1
    assert not receipt["validation_used_for_fit"]
    assert receipt["all_source_decisions_correct"] is None
    assert receipt["source_graph_replays"] == []


@pytest.mark.parametrize("kind", ["empty", "validation", "test", "missing", "duplicate", "changed"])
def test_retention_rejects_wrong_split_missing_or_changed_observations(source, kind):
    model, training, validation = source
    retained = {"empty": (), "validation": (validation[0],),
                "test": (replace(training[0], split="test"),),
                "missing": training[1:], "duplicate": (*training, training[0]),
                "changed": (replace(training[0], public_inputs=tuple(100 for _ in training[0].public_inputs)),
                            *training[1:])}[kind]
    with pytest.raises(ValueError):
        refit_compositional_joint_graphs(model, (training[0], *validation),
            source_retention_examples=retained, rounds=1, steps=1)


@pytest.mark.parametrize("kind", ["empty", "validation", "test", "duplicate"])
def test_graph_replay_does_not_admit_heldout_or_duplicate_rows(source, kind):
    from core.learning.semantic_joint_graph_learning import replay_source_graph_retention

    model, training, validation = source
    examples = {"empty": (), "validation": validation,
                "test": (replace(training[0], split="test"),),
                "duplicate": (training[0], training[0])}[kind]
    with pytest.raises(ValueError, match="unique source training"):
        replay_source_graph_retention(model, examples)


def test_graph_replay_measures_real_source_decisions(source):
    from core.learning.semantic_joint_graph_learning import replay_source_graph_retention

    model, training, _ = source
    pairs, receipt = replay_source_graph_retention(model, training[:2])
    assert receipt["observed_count"] == 2
    assert receipt["counterexample_count"] == len(pairs)
    assert receipt["equivalent_count"] + len(pairs) + receipt["unresolved_count"] == 2
    assert receipt["all_source_decisions_correct"] == (receipt["equivalent_count"] == 2)
    assert receipt["model"] == model.receipt_sha256
    assert not receipt["validation_used"] and receipt["test_examples_used"] == 0


def test_unknown_graph_replay_is_not_success(source, monkeypatch):
    from core.learning import semantic_joint_graph_learning as learning

    model, training, _ = source
    options = []

    def unresolved(_model, item, **kwargs):
        options.append(kwargs)
        return None, {"status": "runtime_decode_unavailable", "source_text_sha256": item.ir.source_text_sha256}

    monkeypatch.setattr(learning, "mine_runtime_graph_contrast", unresolved)
    pairs, receipt = learning.replay_source_graph_retention(model, training, solve_time_limit_s=1.5)
    assert not pairs and receipt["unresolved_count"] == len(training)
    assert not receipt["all_source_decisions_correct"]
    assert all(row["decode_time_limit_s"] == 1.5 for row in options)


def test_outside_mining_regression_enters_next_fit_and_final_replay(source, monkeypatch):
    from core.learning import semantic_joint_graph_learning as learning
    from core.learning import semantic_graph_constraints as fitting
    from core.learning import semantic_runtime_graph_retention as mining

    model, training, validation = source
    phase, batches, seen = [0], [], []
    outside = training[-1].ir.source_text_sha256
    witness = object()

    def observe(candidate, item, **kwargs):
        source_id = item.ir.source_text_sha256
        seen.append((phase[0], source_id))
        wrong = phase[0] == 1 and source_id == outside
        return (witness if wrong else None), {"status": "counterexample" if wrong else "equivalent",
                                              "source_text_sha256": source_id}

    def update(candidate, rows, **kwargs):
        batches.append(tuple(rows))
        phase[0] += 1
        return candidate._with_coefficients(operation_length_penalty=candidate.operation_length_penalty + .01), {}

    monkeypatch.setattr(learning, "mine_runtime_graph_contrast", observe)
    monkeypatch.setattr(learning, "mine_source_binding_constraint", lambda *a, **kw: (None, {}))
    monkeypatch.setattr(mining, "mine_runtime_graph_constraints", lambda *a, **kw: ((), {}))
    monkeypatch.setattr(fitting, "fit_complete_graph_constraints", update)
    candidate = learning.refit_compositional_joint_graphs(model, (training[0], *validation),
        source_retention_examples=training, source_graph_retention=True, constraint_learning=True,
        learn_arguments=True, rounds=2, steps=1)
    receipt = candidate.training_receipt["joint_graph_refit"]
    assert witness not in batches[0] and witness in batches[1]
    assert len(receipt["source_graph_replays"]) == 3
    assert [row["receipt"]["counterexample_count"] for row in receipt["source_graph_replays"]] == [0, 1, 0]
    assert receipt["all_source_decisions_correct"] is True
    for index in range(3):
        assert {source_id for phase_id, source_id in seen if phase_id == index} == {
            item.ir.source_text_sha256 for item in training}
    assert not receipt["validation_used_for_fit"]


def test_final_round_regression_cannot_be_reported_as_retained(source, monkeypatch):
    from core.learning import semantic_joint_graph_learning as learning

    model, training, validation = source
    monkeypatch.setattr(learning, "replay_source_graph_retention", lambda *a, **kw: ((), {
        "all_source_decisions_correct": False, "unresolved_count": len(training)}))
    candidate = learning.refit_compositional_joint_graphs(model, (training[0], *validation),
        source_retention_examples=training, source_graph_retention=True, constraint_learning=True,
        rounds=1, steps=1)
    assert candidate.training_receipt["joint_graph_refit"]["all_source_decisions_correct"] is False


@pytest.mark.parametrize("option", [True, "yes", 1])
def test_graph_retention_requires_explicit_constrained_fit(source, option):
    model, training, validation = source
    with pytest.raises(ValueError, match="source graph retention"):
        refit_compositional_joint_graphs(model, (*training, *validation), source_graph_retention=option)
