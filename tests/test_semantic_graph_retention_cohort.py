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
